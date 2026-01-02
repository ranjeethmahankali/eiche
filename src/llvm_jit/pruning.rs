use super::{
    JitCompiler, build_vec_unary_intrinsic, fast_math,
    interval::{self, Constants},
    single,
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
    execution_engine::JitFunction,
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
    marker::PhantomData,
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

type NativeSimdFunc = unsafe extern "C" fn(
    *const c_void, // Inputs
    *mut c_void,   // Outputs
    *const u32,    // Signals,
    u64,           // Number of evals.
);

type NativeIntervalFunc = unsafe extern "C" fn(
    *const c_void, // Inputs
    *mut c_void,   // Outputs
    *const u32,    // Signals,
);

pub struct JitPruner<'ctx, T: NumberType> {
    func: JitFunction<'ctx, NativePruningIntervalFunc>,
    n_inputs: usize,
    n_outputs: usize,
    n_signals: usize,
    params: String,
    blocks: Box<[Block]>,
    merge_block_map: HashMap<usize, usize>,
    block_signal_map: Box<[usize]>,
    tree: Tree,
    phantom: PhantomData<T>,
}

pub struct JitPruningFn<'ctx, T: NumberType> {
    func: JitFunction<'ctx, NativeSingleFunc>,
    n_inputs: usize,
    n_outputs: usize,
    n_signals: usize,
    _phantom: PhantomData<T>,
}

impl<'ctx, T: NumberType> JitPruningFn<'ctx, T> {
    pub fn run(&self, inputs: &[T], outputs: &mut [T], signals: &[u32]) -> Result<(), Error> {
        if inputs.len() != self.n_inputs {
            return Err(Error::InputSizeMismatch(inputs.len(), self.n_inputs));
        } else if outputs.len() != self.n_outputs {
            return Err(Error::OutputSizeMismatch(outputs.len(), self.n_outputs));
        } else if signals.len() != self.n_signals {
            return Err(Error::OutputSizeMismatch(signals.len(), self.n_signals));
        }
        unsafe { self.run_unchecked(inputs, outputs, signals) }
        Ok(())
    }

    pub unsafe fn run_unchecked(&self, inputs: &[T], outputs: &mut [T], signals: &[u32]) {
        unsafe {
            self.func.call(
                inputs.as_ptr().cast(),
                outputs.as_mut_ptr().cast(),
                signals.as_ptr().cast(),
            )
        }
    }
}

impl<'ctx, T: NumberType> JitPruner<'ctx, T> {
    fn run(
        &self,
        inputs: &[[T; 2]],
        outputs: &mut [[T; 2]],
        signals: &mut [u32],
    ) -> Result<(), Error> {
        if inputs.len() != self.n_inputs {
            return Err(Error::InputSizeMismatch(inputs.len(), self.n_inputs));
        } else if outputs.len() != self.n_outputs {
            return Err(Error::OutputSizeMismatch(outputs.len(), self.n_outputs));
        } else if signals.len() != self.n_signals {
            return Err(Error::OutputSizeMismatch(signals.len(), self.n_signals));
        }
        unsafe { self.run_unchecked(inputs, outputs, signals) }
        Ok(())
    }

    pub unsafe fn run_unchecked(
        &self,
        inputs: &[[T; 2]],
        outputs: &mut [[T; 2]],
        signals: &mut [u32],
    ) {
        unsafe {
            self.func.call(
                inputs.as_ptr().cast(),
                outputs.as_mut_ptr().cast(),
                signals.as_mut_ptr().cast(),
            )
        }
    }

    pub fn compile_single_func(
        &self,
        context: &'ctx JitContext,
    ) -> Result<JitPruningFn<'ctx, T>, Error> {
        // No need to check if the tree has a valid scalar output, because we
        // already checked that when we made this pruner.
        let func_name = context.new_func_name::<T>(Some("pruning"));
        let context = &context.inner;
        let compiler = JitCompiler::new(context)?;
        let builder = &compiler.builder;
        let flt_type = T::jit_type(context);
        let ptr_type = context.ptr_type(AddressSpace::default());
        let mut constants = Constants::create::<T>(context);
        let fn_type = context
            .void_type()
            .fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
        let function = compiler.module.add_function(&func_name, fn_type, None);
        compiler.set_attributes(function, context)?;
        builder.position_at_end(context.append_basic_block(function, "entry"));
        let mut regs = Vec::<BasicValueEnum>::with_capacity(self.tree.len());
        let mut bbs: Box<[BasicBlock]> = self
            .blocks
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
        let signals_arg = function
            .get_nth_param(2)
            .ok_or(Error::JitCompilationError(
                "Cannot read output address".to_string(),
            ))?
            .into_pointer_value();
        let signal_ptrs: Result<Vec<PointerValue>, inkwell::builder::BuilderError> = (0..self
            .n_signals)
            .map(|si| unsafe {
                builder.build_gep(
                    context.i32_type(),
                    signals_arg,
                    &[constants.int_32(si as u32, false)],
                    &format!("signal_ptr_{}", si),
                )
            })
            .collect();
        let signal_ptrs = signal_ptrs?.into_boxed_slice();
        if let Some(first) = bbs.first() {
            builder.build_unconditional_branch(*first)?;
        }
        let (phis, phi_map) = init_merge_phi(&self.blocks, builder, &bbs, flt_type)?;
        let mut merge_list = Vec::<Incoming>::new();
        for (bi, block) in self.blocks.iter().enumerate() {
            builder.position_at_end(bbs[bi]);
            match block {
                Block::Code(range) => {
                    for (index, node) in
                        self.tree.nodes()[range.clone()].iter().copied().enumerate()
                    {
                        let reg = single::build_op(
                            single::BuildArgs {
                                nodes: self.tree.nodes(),
                                params: &self.params,
                                float_type: flt_type,
                                function,
                                regs: &regs,
                                node,
                                index,
                            },
                            builder,
                            &compiler.module,
                        )?;
                        regs.push(reg);
                        // We may have created new blocks when building the op. So we overwrite the basic block.
                        if let Some(bb) = builder.get_insert_block() {
                            bbs[bi] = bb;
                        }
                    }
                    if let Some(next_bb) = bbs.get(bi + 1) {
                        builder.build_unconditional_branch(*next_bb)?;
                    }
                }
                Block::Branch(jumps) => {
                    let si = self.block_signal_map[bi];
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
                                    let mbi = self
                                        .merge_block_map
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
                    }
                    builder.build_switch(signal, bbs[bi + 1], &cases)?;
                }
                Block::Merge(target) => {
                    build_merges(
                        &phis,
                        &phi_map,
                        &mut merge_list,
                        &self.merge_block_map,
                        &bbs,
                        flt_type.get_poison(),
                        builder,
                        &mut regs,
                        &mut constants,
                        interval::build_const,
                        *target,
                    )?;
                }
            }
        }
        // Copy the outputs.
        let outputs = function
            .get_nth_param(1)
            .ok_or(Error::JitCompilationError(
                "Cannot write to outputs".to_string(),
            ))?
            .into_pointer_value();
        for (i, reg) in regs[(self.tree.len() - self.tree.num_roots())..]
            .iter()
            .enumerate()
        {
            // SAFETY: GEP can segfault if the index is out of bounds. The
            // offset calculation looks pretty solid, and is thoroughly tested.
            let dst = unsafe {
                builder.build_gep(
                    flt_type,
                    outputs,
                    &[context.i64_type().const_int(i as u64, false)],
                    &format!("output_{i}"),
                )?
            };
            builder.build_store(dst, *reg)?;
        }
        builder.build_return(None)?;
        // TODO: Copied these passes from single.rs. Find out what is optimal for pruning single evals.
        compiler.run_passes("mem2reg,instcombine,reassociate,gvn,instcombine,slp-vectorizer,instcombine,simplifycfg,adce")?;
        let engine = compiler
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .map_err(|_| Error::CannotCreateJitModule)?;
        // SAFETY: The signature is correct, and well tested. The function
        // pointer should never be invalidated, because we allocated a dedicated
        // execution engine, with it's own block of executable memory, that will
        // live as long as the function wrapper lives.
        let func = unsafe { engine.get_function(&func_name)? };
        Ok(JitPruningFn {
            func,
            n_inputs: self.n_inputs,
            n_outputs: self.n_outputs,
            n_signals: self.n_signals,
            _phantom: PhantomData,
        })
    }
}

impl Tree {
    pub fn jit_compile_pruner<'ctx, T: NumberType>(
        &'ctx self,
        context: &'ctx JitContext,
        params: &str,
        pruning_threshold: usize,
    ) -> Result<JitPruner<'ctx, T>, Error> {
        if !self.is_scalar() {
            // Only support scalar output trees.
            return Err(Error::TypeMismatch);
        }
        let (tree, ndom) = self.control_dependence_sorted()?;
        compile_pruner_impl(tree, context, ndom, params, pruning_threshold)
    }
}

fn compile_pruner_impl<'ctx, T: NumberType>(
    tree: Tree,
    context: &'ctx JitContext,
    ndom: Box<[usize]>,
    params: &str,
    pruning_threshold: usize,
) -> Result<JitPruner<'ctx, T>, Error> {
    let blocks = make_blocks(
        make_interrupts(&tree, &ndom, pruning_threshold)?,
        tree.len(),
    )?;
    let (code_block_map, merge_block_map) = reverse_lookup(&blocks);
    let ranges = interval::compute_ranges(&tree)?;
    let func_name = context.new_func_name::<T>(Some("pruner"));
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
    let mut regs = Vec::<BasicValueEnum>::with_capacity(tree.len());
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
                for (index, node) in tree.nodes()[range.clone()].iter().copied().enumerate() {
                    let reg = interval::build_op::<T>(
                        interval::BuildArgs {
                            nodes: tree.nodes(),
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
                    let ci = *code_block_map
                        .get(&src_inst)
                        .expect("Code map is not complete. This is a bug.");
                    let bb = bbs[ci];
                    builder.position_at_end(bb);
                    let first = notified.insert((*dst_signal, *signal));
                    bbs[ci] = build_notify(
                        tree.node(*src_inst).clone(),
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
                                let mbi = merge_block_map
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
                    &merge_block_map,
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
    assert!(
        merge_list.is_empty(),
        "All merges should be processed by now. This is a bug otherwise"
    );
    // Branch out of all code blocks.
    let mut done = vec![false; bbs.len()];
    for ci in code_block_map.values() {
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
    for (i, reg) in regs[(tree.len() - tree.num_roots())..].iter().enumerate() {
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
                Error::JitCompilationError(format!("Cannot set alignment when storing output: {e}"))
            })?;
    }
    builder.build_return(None)?;
    compiler.run_passes("mem2reg,instcombine,reassociate,gvn,simplifycfg,adce,instcombine")?;
    let engine = compiler
        .module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .map_err(|_| Error::CannotCreateJitModule)?;
    // SAFETY: The signature is correct, and well tested. The function
    // pointer should never be invalidated, because we allocated a dedicated
    // execution engine, with it's own block of executable memory, that will
    // live as long as the function wrapper lives.
    let func = unsafe { engine.get_function(&func_name)? };
    Ok(JitPruner {
        func,
        n_inputs: params.len(),
        n_outputs: tree.num_roots(),
        n_signals: signal_ptrs.len(),
        params: params.to_string(),
        blocks,
        merge_block_map,
        block_signal_map,
        tree,
        phantom: PhantomData,
    })
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
            let not_empty = builder.build_not(either_empty, &format!("not_empty_{index}"))?;
            let before = builder.build_or(
                strictly_before,
                builder
                    .build_extract_element(
                        touching,
                        constants.int_32(1, false),
                        &format!("extract_touching_left_{index}"),
                    )?
                    .into_int_value(),
                &format!("before_check_{index}"),
            )?;
            let after = builder.build_or(
                strictly_after,
                builder
                    .build_extract_element(
                        touching,
                        constants.int_32(0, false),
                        &format!("extract_touching_left_{index}"),
                    )?
                    .into_int_value(),
                &format!("before_check_{index}"),
            )?;
            let strictly_before = builder.build_and(
                not_empty,
                strictly_before,
                &format!("not_empty_combine_flags_{index}"),
            )?;
            let strictly_after = builder.build_and(
                not_empty,
                strictly_after,
                &format!("not_empty_combine_flags_{index}"),
            )?;
            let before = builder.build_and(
                not_empty,
                before,
                &format!("not_empty_combine_flags_{index}"),
            )?;
            let after = builder.build_and(
                not_empty,
                after,
                &format!("not_empty_combine_flags_{index}"),
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
        Ternary(Choose, cond, _, _) => match trigger {
            PruneKind::Right => build_vec_unary_intrinsic(
                builder,
                module,
                "llvm.vector.reduce.and.*",
                &format!("all_true_checK"),
                regs[cond].into_vector_value(),
            )?
            .into_int_value(),
            PruneKind::Left => build_vec_unary_intrinsic(
                builder,
                module,
                "llvm.vector.reduce.and.*",
                &format!("all_false_check_{index}"),
                builder.build_not(
                    regs[cond].into_vector_value(),
                    &format!("always_false_check_flip_{index}"),
                )?,
            )?
            .into_int_value(),
            _ => unreachable!("Invalid node / trigger combo. This is a bug."),
        },
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
    // 0 is reserved for cases that are not pruned. We don't need a variant for that here.
    Left = 1,
    Right = 2,
    AlwaysTrue = 3,
    AlwaysFalse = 4,
}

fn make_interrupts(
    tree: &Tree,
    ndom: &[usize],
    threshold: usize,
) -> Result<Box<[Interrupt]>, Error> {
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
                if dom > threshold {
                    push_land(&mut interrupts, ni, &mut land_map);
                    interrupts.push(Interrupt::Jump {
                        before_node: ni - dom,
                        target: ni,
                        alternate: Alternate::Constant(Value::Bool(true)),
                        owner: ni,
                        trigger: PruneKind::AlwaysTrue,
                    });
                    interrupts.push(Interrupt::Jump {
                        before_node: ni - dom,
                        target: ni,
                        alternate: Alternate::Constant(Value::Bool(false)),
                        owner: ni,
                        trigger: PruneKind::AlwaysFalse,
                    });
                }
            }
            Ternary(Choose, _cond, tt, ff) => {
                let ttdom = ndom[*tt];
                let ffdom = ndom[*ff];
                let tskip = ttdom > threshold;
                let fskip = ffdom > threshold;
                if tskip {
                    push_land(&mut interrupts, *tt, &mut land_map);
                    interrupts.push(Interrupt::Jump {
                        before_node: *tt - ttdom,
                        target: *tt,
                        alternate: Alternate::None,
                        owner: ni,
                        trigger: PruneKind::Left,
                    });
                }
                if fskip {
                    push_land(&mut interrupts, *ff, &mut land_map);
                    interrupts.push(Interrupt::Jump {
                        before_node: *ff - ffdom,
                        target: *ff,
                        alternate: Alternate::None,
                        owner: ni,
                        trigger: PruneKind::Right,
                    });
                }
                if tskip || fskip {
                    push_land(&mut interrupts, ni, &mut land_map);
                    if tskip {
                        interrupts.push(Interrupt::Jump {
                            before_node: ni,
                            target: ni,
                            alternate: Alternate::Node(*ff),
                            owner: ni,
                            trigger: PruneKind::Left,
                        });
                    }
                    if fskip {
                        interrupts.push(Interrupt::Jump {
                            before_node: ni,
                            target: ni,
                            alternate: Alternate::Node(*tt),
                            owner: ni,
                            trigger: PruneKind::Right,
                        });
                    }
                }
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
            ) => match jumps.last().map(|j| j.before_node) {
                Some(prev) if prev != before_node => {
                    blocks.push(Block::Branch(jumps));
                    blocks.push(Block::Code(prev..before_node));
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
                _ => {
                    jumps.push(Jump {
                        before_node,
                        target,
                        alternate,
                        owner,
                        trigger,
                    });
                    (blocks, PartialBlock::Branch(jumps))
                }
            },
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
    use crate::{assert_float_eq, deftree};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn format_interleave_interrupts(tree: &Tree) -> String {
        use std::fmt::Write;
        let (tree, ndom) = tree.control_dependence_sorted().unwrap();
        let interrupts = make_interrupts(&tree, &ndom, 3).unwrap();
        // Print the interrupts interleaved with nodes.
        let mut i = 0usize;
        let mut out = String::new();
        writeln!(out, "").unwrap();
        for interrupt in interrupts {
            match interrupt {
                Interrupt::Jump {
                    before_node,
                    target,
                    alternate,
                    owner,
                    trigger,
                } => {
                    if before_node > i {
                        let range = i..before_node;
                        for (ni, node) in range.clone().zip(tree.nodes()[range].iter()) {
                            writeln!(out, "\t{ni}: {node}").unwrap();
                        }
                    }
                    writeln!(
                        out,
                        "Jump({before_node}, {target}, {alternate:?}, {owner}, {trigger:?})"
                    )
                    .unwrap();
                    i = before_node;
                }
                Interrupt::Land { after_node } => {
                    if after_node >= i {
                        let range = i..=after_node;
                        for (ni, node) in range.clone().zip(tree.nodes()[range].iter()) {
                            writeln!(out, "\t{ni}: {node}").unwrap();
                        }
                    }
                    writeln!(out, "Land({after_node})").unwrap();
                    i = after_node + 1;
                }
            }
        }
        out
    }

    #[test]
    fn t_pruning_two_circles() {
        let tree = deftree!(min
                            (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5)
                            (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5))
        .unwrap();
        let ctx = JitContext::default();
        let eval = tree.jit_compile_pruner::<f64>(&ctx, "xy", 8).unwrap();
        assert_eq!(eval.n_signals, 3);
        assert_eq!(eval.n_inputs, 2);
        assert_eq!(eval.n_outputs, 1);
        // Prune the RHS with an interval to the left of the origin.
        let mut outputs = [[f64::NAN; 2]];
        let mut signals = [0u32; 3];
        eval.run(&[[-2.0, -1.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_eq!(&signals, &[0, 1, 1]);
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        // Reset and test the other side of the origin.
        signals.fill(0u32);
        outputs[0].fill(f64::NAN);
        eval.run(&[[1.0, 2.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        assert_eq!(&signals, &[1, 0, 2]);
    }

    #[test]
    fn t_pruning_choose_two_circles() {
        let tree = deftree!(if (< 'x 0)
                            (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5)
                            (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5))
        .unwrap();
        let ctx = JitContext::default();
        let eval = tree.jit_compile_pruner::<f64>(&ctx, "xy", 8).unwrap();
        assert_eq!(eval.n_signals, 3);
        assert_eq!(eval.n_inputs, 2);
        assert_eq!(eval.n_outputs, 1);
        // Prune the RHS with an interval to the left of the origin.
        let mut outputs = [[f64::NAN; 2]];
        let mut signals = [0u32; 3];
        eval.run(&[[-2.0, -1.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_eq!(&signals, &[0, 1, 1]);
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        // Reset and test the other side of the origin.
        signals.fill(0u32);
        outputs[0].fill(f64::NAN);
        eval.run(&[[1.0, 2.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        assert_eq!(&signals, &[1, 0, 2]);
    }

    #[test]
    fn t_pruning_three_circles() {
        let tree = deftree!(min
                            (- (sqrt (+ (pow 'x 2) (pow (- 'y 1) 2))) 1.5)
                            (min
                             (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5)
                             (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5)))
        .unwrap();
        let ctx = JitContext::default();
        let eval = tree.jit_compile_pruner::<f64>(&ctx, "xy", 8).unwrap();
        assert_eq!(eval.n_signals, 5);
        assert_eq!(eval.n_inputs, 2);
        assert_eq!(eval.n_outputs, 1);
        // Prune the RHS with an interval to the left of the origin.
        let mut outputs = [[f64::NAN; 2]];
        let mut signals = [0u32; 5];
        eval.run(&[[-2.0, -1.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_eq!(&signals, &[0, 0, 1, 1, 0]);
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        // Reset and test the other side of the origin.
        signals.fill(0u32);
        outputs[0].fill(f64::NAN);
        eval.run(&[[1.0, 2.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        assert_eq!(&signals, &[0, 2, 0, 2, 0]);
    }

    #[test]
    fn t_pruning_two_circles_compacted() {
        let tree = deftree!(min
                            (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5)
                            (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5))
        .unwrap()
        .compacted()
        .unwrap();
        let ctx = JitContext::default();
        // Because the tree is compacted, more nodes are shared, and the min
        // nodes dominate fewer nodes. So we lower the pruning threshold for this test.
        let eval = tree.jit_compile_pruner::<f64>(&ctx, "xy", 3).unwrap();
        assert_eq!(eval.n_signals, 3);
        assert_eq!(eval.n_inputs, 2);
        assert_eq!(eval.n_outputs, 1);
        // Prune the RHS with an interval to the left of the origin.
        let mut outputs = [[f64::NAN; 2]];
        let mut signals = [0u32; 3];
        eval.run(&[[-2.0, -1.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_eq!(&signals, &[0, 1, 1]);
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        // Reset and test the other side of the origin.
        signals.fill(0u32);
        outputs[0].fill(f64::NAN);
        eval.run(&[[1.0, 2.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        assert_eq!(&signals, &[1, 0, 2]);
    }

    #[test]
    fn t_pruning_three_circles_compacted() {
        let tree = deftree!(min
                            (- (sqrt (+ (pow 'x 2) (pow (- 'y 1) 2))) 1.5)
                            (min
                             (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5)
                             (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5)))
        .unwrap()
        .compacted()
        .unwrap();
        let ctx = JitContext::default();
        let eval = tree.jit_compile_pruner::<f64>(&ctx, "xy", 3).unwrap();
        assert_eq!(eval.n_signals, 6);
        assert_eq!(eval.n_inputs, 2);
        assert_eq!(eval.n_outputs, 1);
        // Prune the RHS with an interval to the left of the origin.
        let mut outputs = [[f64::NAN; 2]];
        let mut signals = [0u32; 6];
        eval.run(&[[-2.0, -1.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_eq!(&signals, &[0, 0, 1, 1, 0, 0]);
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        // Reset and test the other side of the origin.
        signals.fill(0u32);
        outputs[0].fill(f64::NAN);
        eval.run(&[[1.0, 2.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        assert_eq!(&signals, &[0, 1, 0, 2, 0, 0]);
    }

    fn check_pruned_eval_impl<T: NumberType>(
        tree: &Tree,
        pruning_threshold: usize,
        vardata: &[(char, f64, f64)],
        eps: f64,
    ) {
        let params: String = vardata.iter().map(|(c, _, _)| *c).collect();
        let context = JitContext::default();
        let interval_eval = tree
            .jit_compile_interval::<T>(&context, &params)
            .expect("Unable to compile interval eval");
        let eval = tree
            .jit_compile::<T>(&context, &params)
            .expect("Unable to compile single eval");
        let pruner = tree
            .jit_compile_pruner::<T>(&context, &params, pruning_threshold)
            .expect("Unable to compile a JIT pruner");
        let pruned_eval = pruner
            .compile_single_func(&context)
            .expect("Unable to compile a pruned single eval");
        let mut rng = StdRng::seed_from_u64(42);
        let n_outputs = tree.num_roots();
        const N_INTERVALS: usize = 32;
        const N_QUERIES: usize = 32;
        // Reusable buffers for evaluations.
        let mut interval = Vec::<[T; 2]>::new();
        let mut iout_pruned = Vec::<[T; 2]>::new();
        let mut iout = Vec::<[T; 2]>::new();
        let mut sample = Vec::<T>::new();
        let mut out = Vec::<T>::new();
        let mut out_pruned = Vec::<T>::new();
        let mut signals = Vec::<u32>::new();
        for _ in 0..N_INTERVALS {
            // Sample a random interval.
            interval.clear();
            interval.extend(vardata.iter().map(|(_, lo, hi)| {
                let mut bounds = [0, 1].map(|_| lo + rng.random::<f64>() * (hi - lo));
                if bounds[0] > bounds[1] {
                    bounds.swap(0, 1);
                }
                bounds.map(|b| T::from_f64(b))
            }));
            // Ensure the pruner and interval eval report the same value.
            iout_pruned.clear();
            iout_pruned.resize(n_outputs, [T::nan(); 2]);
            signals.clear();
            signals.resize(pruner.n_signals, 0u32);
            pruner
                .run(&interval, &mut iout_pruned, &mut signals)
                .unwrap();
            // Now the actual interval evaluator we trust.
            iout.clear();
            iout.resize(n_outputs, [T::nan(); 2]);
            interval_eval.run(&interval, &mut iout).unwrap();
            // Compare intervals.
            for (i, j) in iout.iter().zip(iout_pruned.iter()) {
                for (i, j) in i.iter().zip(j.iter()) {
                    assert_float_eq!(i.to_f64(), j.to_f64(), eps);
                }
            }
            // Now sample that interval and compare pruned and un-pruned evaluations.
            for _ in 0..N_QUERIES {
                sample.clear();
                sample.extend(interval.iter().map(|i| {
                    T::from_f64(
                        i[0].to_f64() + rng.random::<f64>() * (i[1].to_f64() - i[0].to_f64()),
                    )
                }));
                out_pruned.clear();
                out_pruned.resize(n_outputs, T::nan());
                pruned_eval.run(&sample, &mut out_pruned, &signals).unwrap();
                out.clear();
                out.resize(n_outputs, T::nan());
                eval.run(&sample, &mut out).unwrap();
                for (i, j) in out_pruned.iter().zip(out.iter()) {
                    assert_float_eq!(i.to_f64(), j.to_f64(), eps);
                }
            }
        }
    }

    fn check_pruned_eval(
        tree: Tree,
        pruning_threshold: usize,
        vardata: &[(char, f64, f64)],
        eps: f64,
    ) {
        check_pruned_eval_impl::<f32>(&tree, pruning_threshold, vardata, eps);
        check_pruned_eval_impl::<f64>(&tree, pruning_threshold, vardata, eps);
    }

    #[test]
    fn t_two_circles() {
        check_pruned_eval(
            deftree!(min
                            (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5)
                            (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5))
            .unwrap()
            .compacted()
            .unwrap(),
            3,
            &[('x', -10.0, 10.0), ('y', -10.0, 10.0)],
            1e-16,
        );
        check_pruned_eval(
            deftree!(min
                            (- (sqrt (+ (pow 'x 2) (pow (- 'y 1) 2))) 1.5)
                            (min
                             (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5)
                             (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5)))
            .unwrap()
            .compacted()
            .unwrap(),
            3,
            &[('x', -10.0, 10.0), ('y', -10.0, 10.0)],
            1e-16,
        );
    }

    #[test]
    fn t_choose_two_circles() {
        check_pruned_eval(
            deftree!(if (< 'x 0)
                            (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5)
                            (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5))
            .unwrap(),
            3,
            &[('x', -10.0, 10.0), ('y', -10.0, 10.0)],
            1e-16,
        );
    }

    #[test]
    fn t_interval_tree_2() {
        check_pruned_eval(
            deftree!(
                (max (min
                      (- (sqrt (+ (+ (pow (- 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 2.75)
                      (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 4.))
                 (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (+ 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 5.25))
            )
                .unwrap(),
            8,
            &[('x', -10., 10.), ('y', -9., 10.), ('z', -11., 12.)],
            1e-5,
        );
    }

    #[test]
    fn t_interval_trees_concat_1() {
        check_pruned_eval(
            deftree!(concat
                     (/ (pow (log (+ (sin 'x) 2.)) 3.) (+ (cos 'x) 2.))
                     (+ 'x 'y)
                     ((max (min
                            (- (sqrt (+ (+ (pow (- 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 2.75)
                            (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 4.))
                       (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (+ 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 5.25))
            )).unwrap(), 8,
            &[('x', -10., 10.), ('y', -9., 10.), ('z', -11., 12.)],
            1e-8,
        );
    }

    #[test]
    fn t_interval_choose() {
        check_pruned_eval(
            deftree!(if (> 'x 0) 'x (- 'x)).unwrap(),
            0,
            &[('x', -10., 10.)],
            1e-16,
        );
        check_pruned_eval(
            deftree!(if (< 'x 0) (- 'x) 'x).unwrap(),
            0,
            &[('x', -10., 10.)],
            1e-16,
        );
    }

    #[test]
    fn t_jit_interval_boolean_ops() {
        check_pruned_eval(
            deftree!(if (and (> 'x 0) (< 'y 5)) (- 'x 2.) (+ 'y 1.5)).unwrap(),
            0,
            &[('x', -2., 3.), ('y', 2., 7.)],
            1e-16,
        );
        check_pruned_eval(
            deftree!(if (or (> 'x 5) (< 'y 0)) (- 'x 2.) (+ 'y 1.5)).unwrap(),
            0,
            &[('x', -2., 3.), ('y', 2., 7.)],
            1e-16,
        );
        check_pruned_eval(
            deftree!(if (not (> 'x 0)) (- 'x 2.) 1.5).unwrap(),
            0,
            &[('x', -5., 5.)],
            1e-16,
        );
    }
}
