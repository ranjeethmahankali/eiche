use super::{
    JitCompiler, fast_math,
    interval::{self, Constants},
};
use crate::{BinaryOp::*, Error, Node::*, TernaryOp::*, Tree, Value};
use inkwell::{
    AddressSpace,
    basic_block::BasicBlock,
    builder::Builder,
    types::IntType,
    values::{BasicValue, BasicValueEnum, PointerValue},
};
use std::{ffi::c_void, ops::Range};

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
        let mut bbs: Box<[BasicBlock]> = (0..blocks.len())
            .map(|bi| context.append_basic_block(function, &format!("bb_{bi}")))
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
                }
                Block::Branch(range) => {
                    let si = block_signal_map[bi];
                    let signal = builder
                        .build_load(
                            context.i32_type(),
                            signal_ptrs[si],
                            &format!("load_signal_{si}"),
                        )?
                        .into_int_value();
                    todo!();
                }
                Block::Merge(_) => todo!(),
            }
        }
        todo!();
    }
}

fn build_cases<'ctx>(jumps: &[Jump], dst: &mut Vec<(&dyn BasicValue, BasicBlock)>) {}

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
                if dom > threshold {}
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
    let mut numbered: Vec<(usize, Interrupt)> = interrupts.iter().cloned().enumerate().collect();
    numbered.sort_by(|(_, a), (_, b)| -> std::cmp::Ordering {
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
            ) => {
                let (lt, rt) = match (&interrupts[*lt], &interrupts[*rt]) {
                    (Interrupt::Land { after_node: la }, Interrupt::Land { after_node: ra }) => {
                        (la, ra)
                    }
                    _ => unreachable!("This is a bug"),
                };
                (lbn, std::cmp::Reverse(lt), la, lo).cmp(&(rbn, std::cmp::Reverse(rt), ra, ro))
            }
            (Interrupt::Jump { before_node, .. }, Interrupt::Land { after_node }) => {
                (before_node, 0).cmp(&(after_node, 1))
            }
            (Interrupt::Land { after_node }, Interrupt::Jump { before_node, .. }) => {
                (after_node, 1).cmp(&(before_node, 0))
            }
            (Interrupt::Land { after_node: la }, Interrupt::Land { after_node: ra }) => la.cmp(&ra),
        }
    });
    let idxmap = numbered.iter().enumerate().fold(
        vec![0usize; numbered.len()],
        |mut idxmap, (inew, (iold, _))| {
            idxmap[*iold] = inew;
            idxmap
        },
    );
    interrupts.clear();
    interrupts.extend(numbered.drain(..).map(|(_, i)| match i {
        Interrupt::Jump {
            before_node,
            target,
            alternate,
            owner,
            trigger,
        } => Interrupt::Jump {
            before_node,
            target: idxmap[target],
            alternate,
            owner,
            trigger,
        },
        Interrupt::Land { after_node } => Interrupt::Land { after_node },
    }));
    Ok(interrupts.into_boxed_slice())
}

struct Jump {
    before_node: usize,
    target: usize,
    alternate: Alternate,
    owner: usize,
    trigger: PruneKind,
}

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
                Symbol('x'),
                Constant(Value::Scalar(1.0)),
                Binary(Add, 0, 1),
                Symbol('y'),
                Constant(Value::Scalar(2.0)),
                Binary(Add, 3, 4),
                Binary(Min, 2, 5),
            ],
            (1, 1),
        )
        .expect("Cannot create tree");
        let cfg = make_interrupts(&tree, 0).expect("Cannot compute control flow");
        dbg!(&cfg);
        assert_eq!(cfg, todo!(),);
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
