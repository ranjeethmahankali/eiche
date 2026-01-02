use super::{
    JitCompiler, fast_math,
    interval::{self, Constants},
};
use crate::{
    BinaryOp::*,
    Error,
    Node::{self, *},
    TernaryOp::*,
    Tree, Value,
};
use inkwell::{
    AddressSpace, OptimizationLevel,
    basic_block::BasicBlock,
    builder::Builder,
    llvm_sys::core::LLVMBuildFreeze,
    module::Module,
    types::{BasicType, IntType},
    values::{
        AsValueRef, BasicValue, BasicValueEnum, FunctionValue, IntValue, PhiValue, PointerValue,
    },
};
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    ffi::{CStr, CString, c_void},
    ops::Range,
};

use super::{JitContext, NumberType};

type NativePruningIntervalFunc = unsafe extern "C" fn(
    *const c_void, // Inputs
    *mut c_void,   // Outputs
    *mut u32,      // Signals - tells future pruned evaluations how to skip and jump.
);

type NativeSingleFunc = unsafe extern "C" fn(
    *const c_void, // Inputs,
    *mut c_void,   // Outputs,
    *const u32,    // Signals,
);

pub type NativeSimdFunc = unsafe extern "C" fn(
    *const c_void, // Inputs
    *mut c_void,   // Outputs
    *const u32,    // Signals,
    u64,           // Number of evals.
);

pub type NativeIntervalFunc = unsafe extern "C" fn(
    *const c_void, // Inputs
    *mut c_void,   // Outputs
    *const u32,    // Signals,
);

impl Tree {
    pub fn jit_compile_pruner<'ctx, T: NumberType>(
        &self,
        context: &'ctx JitContext,
        params: &str,
        pruning_threshold: usize,
    ) -> Result<(), Error> {
        let blocks = make_blocks(make_interrupts(&self, pruning_threshold)?, self.len())?;
        let (code_map, merge_map) = reverse_lookup(&blocks);
        if !self.is_scalar() {
            // Only support scalar output trees.
            return Err(Error::TypeMismatch);
        }
        let ranges = interval::compute_ranges(self)?;
        let func_name = context.new_func_name::<T>(Some("interval"));
        let context = &context.inner;
        let compiler = JitCompiler::new(context)?;
        let builder = &compiler.builder;
        let interval_type = T::jit_type(context).vec_type(2);
        let ptr_type = context.ptr_type(AddressSpace::default());
        let mut constants = Constants::create::<T>(context);
        let fn_type = context
            .void_type()
            .fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
        let function = compiler.module.add_function(&func_name, fn_type, None);
        compiler.set_attributes(function, context)?;
        builder.position_at_end(context.append_basic_block(function, "entry"));
        let mut regs = Vec::<BasicValueEnum>::with_capacity(self.len());
        let mut bbs: Box<[BasicBlock]> = blocks
            .iter()
            .enumerate()
            .map(|(bi, block)| {
                context.append_basic_block(
                    function,
                    &format!(
                        "{}_bb_{bi}",
                        match block {
                            Block::Code(_) => "code",
                            Block::Branch(_) => "branch",
                            Block::Merge(_) => "merge",
                        }
                    ),
                )
            })
            .collect();
        let (signal_ptrs, block_signal_map) = init_signal_ptrs(
            &blocks,
            function
                .get_nth_param(2)
                .ok_or(Error::JitCompilationError(
                    "Cannot read output address".to_string(),
                ))?
                .into_pointer_value(),
            context.i32_type(),
            builder,
            &mut constants,
        )?;
        if let Some(first) = bbs.first() {
            builder.build_unconditional_branch(*first)?;
        }
        let (phis, phi_map) = init_merge_phi(&blocks, builder, &bbs, interval_type)?;
        let mut notifications = Vec::<Notification>::new();
        let mut merge_list = Vec::<Incoming>::new();
        let mut notified = HashSet::<(usize, u32)>::new();
        for (bi, block) in blocks.iter().enumerate() {
            builder.position_at_end(bbs[bi]);
            match block {
                Block::Code(range) => {
                    for (index, node) in self.nodes()[range.clone()].iter().copied().enumerate() {
                        let reg = interval::build_op::<T>(
                            interval::BuildArgs {
                                nodes: self.nodes(),
                                params,
                                ranges: ranges.as_ref(),
                                regs: &regs,
                                constants: &mut constants,
                                interval_type,
                                function,
                                node,
                                index,
                            },
                            builder,
                            &compiler.module,
                        )?;
                        regs.push(fast_math(reg));
                        // We may have created new blocks when building the op. So we overwrite the basic block.
                        if let Some(bb) = builder.get_insert_block() {
                            bbs[bi] = bb;
                        }
                    }
                    // Notify upstream.
                    for Notification {
                        src_inst,
                        dst_signal,
                        signal,
                        kind,
                    } in notifications.iter().filter(|n| n.src_inst == range.end - 1)
                    {
                        let ci = *code_map
                            .get(&src_inst)
                            .expect("Code map is not complete. This is a bug.");
                        let bb = bbs[ci];
                        builder.position_at_end(bb);
                        let first = notified.insert((*dst_signal, *signal));
                        bbs[ci] = build_notify(
                            self.node(*src_inst).clone(),
                            *src_inst,
                            *kind,
                            *signal,
                            signal_ptrs[*dst_signal],
                            first,
                            &regs,
                            builder,
                            &compiler.module,
                            &mut constants,
                            function,
                        )?;
                    }
                    // Clear the processed notifications.
                    notifications.retain(|n| n.src_inst != range.end - 1);
                }
                Block::Branch(jumps) => {
                    let si = block_signal_map[bi];
                    let signal = builder
                        .build_load(
                            context.i32_type(),
                            signal_ptrs[si],
                            &format!("load_signal_{si}"),
                        )?
                        .into_int_value();
                    let mut cases = Vec::<(IntValue, BasicBlock)>::new();
                    let mut prev_target = usize::MAX;
                    let mut prev_val = None;
                    let mut index = 0u32;
                    for jump in jumps.iter() {
                        if prev_target != jump.target || prev_val != Some(jump.alternate.clone()) {
                            // New case.
                            index += 1;
                            let (case_bb, incoming_bb) = match jump.alternate {
                                Alternate::None => {
                                    let mbi = merge_map
                                        .get(&jump.target)
                                        .expect("We must find a match. This is a bug.");
                                    (bbs[*mbi], bbs[bi])
                                }
                                Alternate::Node(_) | Alternate::Constant(_) => {
                                    let bb = context.append_basic_block(
                                        function,
                                        &format!("branch_{bi}_case_{si}_bb"),
                                    );
                                    (bb, bb)
                                }
                            };
                            cases.push((constants.int_32(index, false), case_bb));
                            merge_list.push(Incoming {
                                target: jump.target,
                                basic_block: incoming_bb,
                                alternate: jump.alternate.clone(),
                            });
                            prev_target = jump.target;
                            prev_val = Some(jump.alternate.clone());
                        }
                        notifications.push(Notification {
                            src_inst: jump.owner,
                            dst_signal: si,
                            signal: index,
                            kind: jump.trigger,
                        });
                    }
                    builder.build_switch(signal, bbs[bi + 1], &cases)?;
                }
                Block::Merge(target) => {
                    build_merges(
                        &phis,
                        &phi_map,
                        &mut merge_list,
                        &merge_map,
                        &bbs,
                        interval_type.get_poison(),
                        builder,
                        &mut regs,
                        &mut constants,
                        interval::build_const,
                        *target,
                    )?;
                }
            }
        }
        // Branch out of all code blocks.
        let mut done = vec![false; bbs.len()];
        for ci in code_map.values() {
            if let Some(next) = bbs.get(ci + 1)
                && !std::mem::replace(&mut done[*ci], true)
            {
                builder.position_at_end(bbs[*ci]);
                builder.build_unconditional_branch(*next)?;
            }
        }
        if let Some(last) = bbs.last() {
            builder.position_at_end(*last);
        }
        // Compile instructions to copy the outputs to the out argument.
        let outputs = function
            .get_nth_param(1)
            .ok_or(Error::JitCompilationError(
                "Cannot read output address".to_string(),
            ))?
            .into_pointer_value();
        for (i, reg) in regs[(self.len() - self.num_roots())..].iter().enumerate() {
            // # SAFETY: This is unit tested a lot. If this fails, we segfault.
            let dst = unsafe {
                builder.build_gep(
                    interval_type,
                    outputs,
                    &[constants.int_32(i as u32, false)],
                    &format!("output_ptr_{i}"),
                )?
            };
            let store_inst = builder.build_store(dst, reg.into_vector_value())?;
            /*
            Rust arrays only guarantee alignment with the size of T, where as
            LLVM load / store instructions expect alignment with the vector size
            (double the size of T). This mismatch can cause a segfault. So we
            manually set the alignment for this store instruction.
            */
            store_inst
                .set_alignment(std::mem::size_of::<T>() as u32)
                .map_err(|e| {
                    Error::JitCompilationError(format!(
                        "Cannot set alignment when storing output: {e}"
                    ))
                })?;
        }
        builder.build_return(None)?;

        compiler.module.print_to_stderr();

        compiler.run_passes("mem2reg,instcombine,reassociate,gvn,simplifycfg,adce,instcombine")?;
        let engine = compiler
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .map_err(|_| Error::CannotCreateJitModule)?;
        // SAFETY: The signature is correct, and well tested. The function
        // pointer should never be invalidated, because we allocated a dedicated
        // execution engine, with it's own block of executable memory, that will
        // live as long as the function wrapper lives.
        // let func = unsafe { engine.get_function(&func_name)? };
        todo!("Do proper branching and set the outputs.");
    }
}

fn build_merges<'ctx, T: BasicValue<'ctx> + Copy>(
    phis: &Box<[PhiValue<'ctx>]>,
    phi_map: &Box<[usize]>,
    merge_list: &mut Vec<Incoming<'ctx>>,
    merge_map: &HashMap<usize, usize>,
    bbs: &[BasicBlock<'ctx>],
    poison: T,
    builder: &'ctx Builder,
    regs: &mut [BasicValueEnum<'ctx>],
    constants: &mut Constants<'ctx>,
    build_const_fn: impl Fn(&mut Constants<'ctx>, Value) -> BasicValueEnum<'ctx>,
    target: usize,
) -> Result<(), Error> {
    let bi = merge_map
        .get(&target)
        .copied()
        .expect("This is a bug. We must find a block index here.");
    let phi = phis[phi_map[bi]];
    for Incoming {
        target: _,
        basic_block,
        alternate,
    } in merge_list.iter().filter(|m| m.target == target)
    {
        let val = match alternate {
            Alternate::None => poison.as_basic_value_enum(),
            Alternate::Node(ni) => {
                builder.position_at_end(*basic_block);
                builder.build_unconditional_branch(bbs[bi])?;
                regs[*ni]
            }
            Alternate::Constant(value) => {
                builder.position_at_end(*basic_block);
                let out = build_const_fn(constants, *value);
                builder.build_unconditional_branch(bbs[bi])?;
                out
            }
        };
        phi.add_incoming(&[(&val as &dyn BasicValue, *basic_block)]);
    }
    // Default path when nothing is pruned.
    phi.add_incoming(&[(&regs[target], bbs[bi - 1])]);
    builder.position_at_end(bbs[bi]);
    regs[target] = build_freeze(builder, phi, &format!("phi_{bi}_freeze"));
    if let Some(next_bb) = bbs.get(bi + 1) {
        builder.build_unconditional_branch(*next_bb)?;
    }
    // Clean up.
    merge_list.retain(|m| m.target != target);
    Ok(())
}

fn build_notify<'ctx>(
    node: Node,
    index: usize,
    trigger: PruneKind,
    signal: u32,
    signal_ptr: PointerValue<'ctx>,
    first: bool,
    regs: &[BasicValueEnum<'ctx>],
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &mut Constants<'ctx>,
    function: FunctionValue<'ctx>,
) -> Result<BasicBlock<'ctx>, Error> {
    let cond = match node {
        Binary(op, lhs, rhs) => {
            let interval::InequalityFlags {
                either_empty,
                strictly_before,
                strictly_after,
                touching,
            } = interval::build_interval_inequality_flags(
                regs[lhs].into_vector_value(),
                regs[rhs].into_vector_value(),
                builder,
                module,
                constants,
                index,
            )?;
            let before = builder.build_or(
                strictly_before,
                builder
                    .build_extract_element(
                        touching,
                        constants.int_32(1, false),
                        &format!("extract_touching_left"),
                    )?
                    .into_int_value(),
                &format!("before_check"),
            )?;
            let after = builder.build_or(
                strictly_after,
                builder
                    .build_extract_element(
                        touching,
                        constants.int_32(0, false),
                        &format!("extract_touching_left"),
                    )?
                    .into_int_value(),
                &format!("before_check"),
            )?;
            match (op, trigger) {
                (Min, PruneKind::Right)
                | (Max, PruneKind::Left)
                | (LessOrEqual, PruneKind::AlwaysTrue)
                | (Greater, PruneKind::AlwaysFalse)
                | (GreaterOrEqual, PruneKind::AlwaysFalse) => before,
                (Min, PruneKind::Left)
                | (Max, PruneKind::Right)
                | (GreaterOrEqual, PruneKind::AlwaysTrue)
                | (Less, PruneKind::AlwaysFalse)
                | (LessOrEqual, PruneKind::AlwaysFalse) => after,
                (Less, PruneKind::AlwaysTrue) => strictly_before,
                (Greater, PruneKind::AlwaysTrue) => strictly_after,
                _ => unreachable!("Invalid node / trigger combo. This is a bug."),
            }
        }
        Ternary(op, _, _, _) => todo!(),
        _ => unreachable!("Unsupported node type. This is a bug."),
    };
    let then_bb = module
        .get_context()
        .append_basic_block(function, &format!("notify_block_{index}"));
    let merge_bb = module
        .get_context()
        .append_basic_block(function, &format!("notify_merge_{index}"));
    builder.build_conditional_branch(cond, then_bb, merge_bb)?;
    builder.position_at_end(then_bb);
    let signal = constants.int_32(signal, false);
    let signal = if first {
        signal
    } else {
        let old = builder
            .build_load(
                module.get_context().i32_type(),
                signal_ptr,
                &format!("notify_combine_load_{index}"),
            )?
            .into_int_value();
        builder.build_and(old, signal, &format!("notify_combine_{index}"))?
    };
    builder.build_store(signal_ptr, signal)?;
    builder.build_unconditional_branch(merge_bb)?;
    Ok(merge_bb)
}

fn build_freeze<'ctx>(
    builder: &'ctx Builder,
    phi: PhiValue<'ctx>,
    name: &str,
) -> BasicValueEnum<'ctx> {
    /// This function takes in a Rust string and either:
    ///
    /// A) Finds a terminating null byte in the Rust string and can reference it directly like a C string.
    ///
    /// B) Finds no null byte and allocates a new C string based on the input Rust string.
    pub(crate) fn to_c_str(mut s: &str) -> Cow<'_, CStr> {
        if s.is_empty() {
            s = "\0";
        }

        // Start from the end of the string as it's the most likely place to find a null byte
        if !s.chars().rev().any(|ch| ch == '\0') {
            return Cow::from(CString::new(s).expect("unreachable since null bytes are checked"));
        }

        unsafe { Cow::from(CStr::from_ptr(s.as_ptr() as *const _)) }
    }
    let c_string = to_c_str(name);
    unsafe {
        BasicValueEnum::new(LLVMBuildFreeze(
            builder.as_mut_ptr(),
            phi.as_value_ref(),
            c_string.as_ptr(),
        ))
    }
}

fn reverse_lookup(blocks: &[Block]) -> (HashMap<usize, usize>, HashMap<usize, usize>) {
    let mut code_map = HashMap::default();
    let mut merge_map = HashMap::default();
    for (bi, block) in blocks.iter().enumerate() {
        match block {
            Block::Code(range) => {
                code_map.extend(range.clone().map(|inst| (inst, bi)));
            }
            Block::Branch(_) => {}
            Block::Merge(after) => {
                merge_map.insert(*after, bi);
            }
        };
    }
    (code_map, merge_map)
}

#[derive(Debug)]
struct Incoming<'ctx> {
    target: usize,
    basic_block: BasicBlock<'ctx>,
    alternate: Alternate,
}

#[derive(Debug)]
struct Notification {
    src_inst: usize,
    dst_signal: usize,
    signal: u32,
    kind: PruneKind,
}

fn init_merge_phi<'ctx, TPhi: BasicType<'ctx> + Copy>(
    blocks: &[Block],
    builder: &'ctx Builder,
    bbs: &[BasicBlock<'ctx>],
    out_type: TPhi,
) -> Result<(Box<[PhiValue<'ctx>]>, Box<[usize]>), Error> {
    let original_bb = builder.get_insert_block();
    let (phis, indices) = blocks.iter().zip(bbs.iter()).try_fold(
        (Vec::<PhiValue>::new(), Vec::<usize>::new()),
        |(mut phis, mut indices), (block, bb)| -> Result<(Vec<PhiValue>, Vec<usize>), Error> {
            match block {
                Block::Code(_) | Block::Branch(_) => {
                    indices.push(usize::MAX);
                    Ok((phis, indices))
                }
                Block::Merge(_) => {
                    let index = phis.len();
                    builder.position_at_end(*bb);
                    let phi = builder.build_phi(out_type, &format!("merge_phi_{index}"))?;
                    phis.push(phi);
                    indices.push(index);
                    Ok((phis, indices))
                }
            }
        },
    )?;
    if let Some(bb) = original_bb {
        builder.position_at_end(bb);
    }
    Ok((phis.into_boxed_slice(), indices.into_boxed_slice()))
}

fn init_signal_ptrs<'ctx>(
    blocks: &[Block],
    signals: PointerValue<'ctx>,
    i32_type: IntType<'ctx>,
    builder: &'ctx Builder,
    constants: &mut Constants<'ctx>,
) -> Result<(Box<[PointerValue<'ctx>]>, Box<[usize]>), Error> {
    let (ptrs, indices) = blocks.iter().try_fold(
        (Vec::<PointerValue>::new(), Vec::<usize>::new()),
        |(mut ptrs, mut indices), block| -> Result<(Vec<PointerValue>, Vec<usize>), Error> {
            match block {
                Block::Code(_) | Block::Merge(_) => {
                    indices.push(usize::MAX);
                    Ok((ptrs, indices))
                }
                Block::Branch(_) => {
                    let index = ptrs.len();
                    indices.push(index);
                    let ptr = unsafe {
                        builder.build_gep(
                            i32_type,
                            signals,
                            &[constants.int_32(index as u32, false)],
                            &format!("signal_ptr_{}", index),
                        )?
                    };
                    ptrs.push(ptr);
                    Ok((ptrs, indices))
                }
            }
        },
    )?;
    Ok((ptrs.into_boxed_slice(), indices.into_boxed_slice()))
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd)]
enum Alternate {
    None,
    Node(usize),
    Constant(Value),
}

impl Ord for Alternate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        match (self, other) {
            (Alternate::None, Alternate::None) => Equal,
            (Alternate::None, Alternate::Node(_))
            | (Alternate::None, Alternate::Constant(_))
            | (Alternate::Node(_), Alternate::Constant(_)) => Less,
            (Alternate::Node(_), Alternate::None)
            | (Alternate::Constant(_), Alternate::None)
            | (Alternate::Constant(_), Alternate::Node(_)) => Greater,
            (Alternate::Node(a), Alternate::Node(b)) => a.cmp(&b),
            (Alternate::Constant(a), Alternate::Constant(b)) => match a.partial_cmp(&b) {
                Some(cmp) => cmp,
                None => Equal,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Interrupt {
    Jump {
        before_node: usize,
        target: usize,
        alternate: Alternate,
        owner: usize,
        trigger: PruneKind,
    },
    Land {
        after_node: usize,
    },
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
enum PruneKind {
    None = 0,
    Left = 1,
    Right = 2,
    AlwaysTrue = 3,
    AlwaysFalse = 4,
}

fn make_interrupts(tree: &Tree, threshold: usize) -> Result<Box<[Interrupt]>, Error> {
    let (tree, ndom) = tree.control_dependence_sorted()?;
    let mut interrupts = Vec::<Interrupt>::with_capacity(tree.len() / 2);
    let mut land_map: Vec<Option<usize>> = vec![None; tree.len()];
    for (ni, node) in tree.nodes().iter().enumerate() {
        match node {
            Binary(Min | Max, lhs, rhs) => {
                let ldom = ndom[*lhs];
                let rdom = ndom[*rhs];
                let lskip = ldom > threshold;
                let rskip = rdom > threshold;
                if lskip {
                    push_land(&mut interrupts, *lhs, &mut land_map);
                    interrupts.push(Interrupt::Jump {
                        before_node: *lhs - ldom,
                        target: *lhs,
                        alternate: Alternate::None,
                        owner: ni,
                        trigger: PruneKind::Left,
                    });
                }
                if rskip {
                    push_land(&mut interrupts, *rhs, &mut land_map);
                    interrupts.push(Interrupt::Jump {
                        before_node: *rhs - rdom,
                        target: *rhs,
                        alternate: Alternate::None,
                        owner: ni,
                        trigger: PruneKind::Right,
                    });
                }
                if lskip || rskip {
                    push_land(&mut interrupts, ni, &mut land_map);
                    if lskip {
                        interrupts.push(Interrupt::Jump {
                            before_node: ni,
                            target: ni,
                            alternate: Alternate::Node(*rhs),
                            owner: ni,
                            trigger: PruneKind::Left,
                        });
                    }
                    if rskip {
                        interrupts.push(Interrupt::Jump {
                            before_node: ni,
                            target: ni,
                            alternate: Alternate::Node(*lhs),
                            owner: ni,
                            trigger: PruneKind::Right,
                        });
                    }
                }
            }
            Binary(Less | LessOrEqual | Greater | GreaterOrEqual, _lhs, _rhs) => {
                let dom = ndom[ni];
                let start = ni - dom;
                if dom > threshold {
                    // push_land(dst, after_node, land_map);
                }
            }
            Ternary(Choose, _cond, tt, ff) => {
                let ttdom = ndom[*tt];
                let ffdom = ndom[*ff];
                let tskip = ttdom > threshold;
                let fskip = ffdom > threshold;
                if tskip {}
                if fskip {}
                if tskip || fskip {}
                todo!();
            }
            _ => continue,
        }
    }
    interrupts.sort_by(|a, b| -> std::cmp::Ordering {
        match (a, b) {
            (
                Interrupt::Jump {
                    before_node: lbn,
                    target: lt,
                    alternate: la,
                    owner: lo,
                    ..
                },
                Interrupt::Jump {
                    before_node: rbn,
                    target: rt,
                    alternate: ra,
                    owner: ro,
                    ..
                },
            ) => (lbn, std::cmp::Reverse(lt), la, lo).cmp(&(rbn, std::cmp::Reverse(rt), ra, ro)),
            (Interrupt::Jump { before_node, .. }, Interrupt::Land { after_node }) => {
                (before_node, 0).cmp(&(after_node, 1))
            }
            (Interrupt::Land { after_node }, Interrupt::Jump { before_node, .. }) => {
                (after_node, 1).cmp(&(before_node, 0))
            }
            (Interrupt::Land { after_node: la }, Interrupt::Land { after_node: ra }) => la.cmp(&ra),
        }
    });
    Ok(interrupts.into_boxed_slice())
}

#[derive(Clone, Debug)]
struct Jump {
    before_node: usize,
    target: usize,
    alternate: Alternate,
    owner: usize,
    trigger: PruneKind,
}

#[derive(Debug)]
enum Block {
    Code(Range<usize>),
    Branch(Vec<Jump>),
    Merge(usize),
}

fn make_blocks(interrupts: Box<[Interrupt]>, n_nodes: usize) -> Result<Box<[Block]>, Error> {
    enum PartialBlock {
        Code(usize),
        Branch(Vec<Jump>),
    }
    let (mut blocks, last) = interrupts.into_iter().enumerate().fold(
        (Vec::<Block>::new(), PartialBlock::Code(0)),
        |(mut blocks, partial), (i, current)| match (partial, current) {
            (
                PartialBlock::Code(start),
                Interrupt::Jump {
                    before_node,
                    target,
                    alternate,
                    owner,
                    trigger,
                },
            ) => {
                if start < before_node {
                    blocks.push(Block::Code(start..before_node));
                }
                (
                    blocks,
                    PartialBlock::Branch(vec![Jump {
                        before_node,
                        target,
                        alternate,
                        owner,
                        trigger,
                    }]),
                )
            }
            (PartialBlock::Code(start), Interrupt::Land { after_node }) => {
                if start < after_node + 1 {
                    blocks.push(Block::Code(start..(after_node + 1)));
                }
                blocks.push(Block::Merge(after_node));
                (blocks, PartialBlock::Code(after_node + 1))
            }
            (
                PartialBlock::Branch(mut jumps),
                Interrupt::Jump {
                    before_node,
                    target,
                    alternate,
                    owner,
                    trigger,
                },
            ) => {
                debug_assert!(
                    match jumps.last() {
                        Some(prev) => prev.before_node == before_node,
                        None => true,
                    },
                    "Consecutive jumps without a land block is invalid."
                );
                jumps.push(Jump {
                    before_node,
                    target,
                    alternate,
                    owner,
                    trigger,
                });
                (blocks, PartialBlock::Branch(jumps))
            }
            (PartialBlock::Branch(jumps), Interrupt::Land { after_node }) => {
                let start = match jumps.last() {
                    Some(j) => j.before_node,
                    None => unreachable!("Invalid interrupts. This is a bug."),
                };
                blocks.push(Block::Branch(jumps));
                blocks.push(Block::Code(start..(after_node + 1)));
                blocks.push(Block::Merge(after_node));
                (blocks, PartialBlock::Code(after_node + 1))
            }
        },
    );
    match last {
        PartialBlock::Code(start) if start < n_nodes => {
            blocks.push(Block::Code(start..n_nodes));
        }
        PartialBlock::Branch(_) => {
            unreachable!("Last block should not be a branch block. This is a bug")
        }
        _ => {}
    };
    Ok(blocks.into_boxed_slice())
}

fn push_land(dst: &mut Vec<Interrupt>, after_node: usize, land_map: &mut [Option<usize>]) {
    let mapped = &mut land_map[after_node];
    if let None = *mapped {
        let idx = dst.len();
        dst.push(Interrupt::Land { after_node });
        *mapped = Some(idx);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::deftree;

    #[test]
    fn t_min_tiny() {
        let tree = Tree::from_nodes(
            vec![
                Symbol('x'),                  // 0
                Constant(Value::Scalar(1.0)), // 1
                Binary(Add, 0, 1),            // 2
                Symbol('y'),                  // 3
                Constant(Value::Scalar(2.0)), // 4
                Binary(Add, 3, 4),            // 5
                Binary(Min, 2, 5),            // 6
            ],
            (1, 1),
        )
        .expect("Cannot create tree");
        let context = JitContext::default();
        let eval = tree.jit_compile_pruner::<f32>(&context, "xy", 0).unwrap();
        assert!(false);
    }

    #[test]
    fn t_min_nested() {
        let tree = Tree::from_nodes(
            vec![
                Symbol('x'),                  // 0
                Constant(Value::Scalar(1.0)), // 1
                Binary(Add, 0, 1),            // 2
                Symbol('y'),                  // 3
                Constant(Value::Scalar(2.0)), // 4
                Binary(Add, 3, 4),            // 5
                Binary(Min, 2, 5),            // 6
                Symbol('z'),                  // 7
                Constant(Value::Scalar(3.0)), // 8
                Binary(Add, 7, 8),            // 9
                Binary(Min, 6, 9),            // 10
            ],
            (1, 1),
        )
        .unwrap();
        let cfg = make_interrupts(&tree, 0).expect("Unable to build control flow");
        dbg!(&cfg);
        assert_eq!(cfg, todo!());
    }

    #[test]
    fn t_choose_tiny() {
        let tree = Tree::from_nodes(
            vec![
                Symbol('x'),                  // 0
                Symbol('a'),                  // 1
                Constant(Value::Scalar(1.0)), // 2
                Binary(Add, 1, 2),            // 3
                Binary(Less, 0, 3),           // 4
                Symbol('y'),                  // 5
                Constant(Value::Scalar(2.0)), // 6
                Binary(Add, 5, 6),            // 7
                Symbol('z'),                  // 8
                Constant(Value::Scalar(3.0)), // 9
                Binary(Add, 8, 9),            // 10
                Ternary(Choose, 4, 7, 10),    // 11
            ],
            (1, 1),
        )
        .unwrap();
        let cfg = make_interrupts(&tree, 0).expect("Unable to build control flow");
        dbg!(tree.nodes(), cfg);
        assert!(false);
    }
}
