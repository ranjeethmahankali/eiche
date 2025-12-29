use inkwell::{
    AddressSpace, IntPredicate,
    basic_block::BasicBlock,
    builder::Builder,
    values::{BasicValue, BasicValueEnum, IntValue, PointerValue},
};

use super::{JitContext, NumberType, interval, simd_array, single};
use crate::{
    BinaryOp::*,
    Error,
    Node::*,
    TernaryOp::*,
    Tree, Value,
    llvm_jit::{JitCompiler, interval::BuildArgs},
    tree::is_node_scalar,
};
use std::{collections::HashMap, ffi::c_void, marker::PhantomData, ops::Range};

/*
This is a vague sketch of how this should work. I am writing this before I write
the code, so this may not all be true depending on how things play out.

First do a dominator sort of the nodes.

Ops where pruning make sense declare their inputs as potential candidates for pruning:
- Min
- Max
- Less
- LessOrEqual
- Greater
- GreaterOrequal
- Choose

Of these condidates, only nodes that dominate more nodes than a threshold should
be considered for pruning.

Now we have the list of nodes we want to prune. Divide the nodes up into blocks
so that each node + it's dominated subrange has a branch block before and after
it. And the selector node, i.e. of the ops listed above, i.e. the parent of the
prunable node should be part of a merge block. This can probably be
done by just walking over the dom-sorted nodes in one pass.

Side note for clarity on how this maps to LLVM blocks: `Branch` is a LLVM switch
whose cases are integers from zero to n. The merge block is actually two LLVM
basic blocks: first to conditionally run the selector/parent node (e.g. Min /
Max) if the inputs weren't pruned, and a second block to create a phi value that
combines all the possibilities.

Maybe in a separate pass, or maybe in the same pass as above... The list of
blocks should be populated with data. The branch blocks should know all the
cases and target blocks. The merge blocks should know their token (explained
later) and the incoming branches.

Once this entire datastructure, i.e. a list of cross referencing blocks is
built, that can be used to compile LLVM functions that use instruction pruning
to skip instructions based on interval evaluations.
 */

#[derive(Debug)]
pub enum PruningType {
    None,
    Left,
    Right,
    AlwaysTrue,
    AlwaysFalse,
}

#[derive(Debug)]
pub struct Listener {
    branch: usize,
    case: u32,
    prune: PruningType,
}

#[derive(Debug)]
pub struct Incoming {
    block: usize,
    output: Option<usize>,
}

#[derive(Debug, Default)]
pub struct Case {
    target_block: usize,
    output: Option<Value>,
}

#[derive(Debug)]
pub enum Block {
    Branch {
        cases: Vec<Case>,
    },
    Code {
        instructions: Range<usize>,
    },
    Merge {
        listeners: Vec<Listener>,
        incoming: Vec<Incoming>,
        selector_node: usize,
    },
}

pub fn make_blocks(tree: &Tree, threshold: usize) -> Result<Box<[Block]>, Error> {
    let (tree, ndom) = tree.control_dependence_sorted()?;
    debug_assert_eq!(
        tree.len(),
        ndom.len(),
        "This should never happen, it is a bug in control dependence sorting"
    );
    let (splits, is_selector) = make_layout(&tree, threshold, &ndom);
    let (mut blocks, inst) = splits.iter().fold(
        (Vec::<Block>::new(), 0usize),
        |(mut blocks, mut inst), split| {
            let (pos, block) = match split {
                Split::Branch(p) => (
                    *p,
                    Some(Block::Branch {
                        cases: Default::default(),
                    }),
                ),
                Split::Merge(p) => (
                    *p,
                    Some(Block::Merge {
                        listeners: Default::default(),
                        incoming: Default::default(),
                        selector_node: *p - 1,
                    }),
                ),
                Split::Direct(p) => (*p, None),
            };
            assert!(pos >= inst, "This is a bug");
            if pos > inst {
                blocks.push(Block::Code {
                    instructions: inst..pos,
                });
                inst = pos;
            }
            if let Some(block) = block {
                blocks.push(block);
            }
            (blocks, inst)
        },
    );
    // Push any remaining instructions as the last code block.
    if inst < tree.len() {
        blocks.push(Block::Code {
            instructions: inst..tree.len(),
        });
    }
    // Build index maps for later use.
    let (branch_map, merge_map, code_map, _) = blocks.iter().enumerate().fold(
        (
            HashMap::<usize, usize>::new(),
            HashMap::<usize, usize>::new(),
            HashMap::<usize, usize>::new(),
            0usize,
        ),
        |(mut bmap, mut mmap, mut cmap, mut inst), (bi, block)| {
            match block {
                Block::Branch { .. } => bmap.insert(inst, bi),
                Block::Code { instructions } => {
                    debug_assert_eq!(
                        inst, instructions.start,
                        "This should never break. This is a bug."
                    );
                    let old = std::mem::replace(&mut inst, instructions.end);
                    cmap.insert(old, bi)
                }
                Block::Merge { .. } => mmap.insert(inst, bi),
            };
            (bmap, mmap, cmap, inst)
        },
    );
    // Build jumps and links between blocks.
    let mut blocks = blocks.into_boxed_slice();
    // First build the trivial links between consecutive blocks. This represents
    // the code flow when nothing is pruned.
    for bi in 0..(blocks.len() - 1) {
        let (left, right) = blocks.split_at_mut(bi + 1);
        let (left, right) = (&mut left[bi], &mut right[0]);
        match (left, right) {
            (Block::Branch { .. }, Block::Branch { .. })
            | (Block::Branch { .. }, Block::Merge { .. })
            | (Block::Merge { .. }, Block::Merge { .. }) => {
                unreachable!(
                    "Two merge / branch blocks should never occur consecutively. This is a bug"
                )
            }
            (
                Block::Code { instructions },
                Block::Merge {
                    listeners: _,
                    incoming,
                    ..
                },
            ) => incoming.push(Incoming {
                block: bi,
                output: Some(instructions.end - 1),
            }),
            // The default case i.e. '0' in every branch just goes to the next block.
            (Block::Branch { cases, .. }, Block::Code { .. }) => {
                cases.push(Case {
                    target_block: bi + 1,
                    output: None,
                });
            }
            // All other code blocks and merge blocks unconditionally branch to the next block.
            (Block::Code { .. }, _) | (Block::Merge { .. }, _) => continue,
        }
    }
    // Now build jumps and links for cases when instructions get pruned.
    for (ni, node) in tree
        .nodes()
        .iter()
        .zip(is_selector)
        .enumerate()
        .filter_map(|(ni, (node, flag))| if flag { Some((ni, node)) } else { None })
    {
        match node {
            Constant(_) | Symbol(_) | Unary(_, _) => {
                unreachable!("This should never happen. This is a bug")
            }
            Binary(op, lhs, rhs) => match op {
                Add | Subtract | Multiply | Divide | Pow | Remainder | Equal | NotEqual | And
                | Or => unreachable!("This should never happen. This is a bug"),
                Min | Max => {
                    let ldom = ndom[*lhs];
                    let rdom = ndom[*rhs];
                    let lstart = *lhs - ldom;
                    let rstart = *rhs - rdom;
                    let c1 = code_map.get(&(&lstart)).copied().expect("This is a bug");
                    let c2 = code_map.get(&rstart).copied().expect("This is a bug");
                    let b3 = branch_map.get(&ni).copied().expect("This is a bug");
                    let c3 = code_map.get(&ni).copied().expect("This is a bug");
                    let merge = merge_map.get(&(ni + 1)).copied().expect("This is a bug");
                    match (ldom > threshold, rdom > threshold) {
                        (true, true) => {
                            // branch | ldom, lhs | branch | rdom, rhs | branch | op | merge
                            let b1 = branch_map.get(&lstart).copied().expect("This is a bug");
                            let b2 = branch_map.get(&rstart).copied().expect("This is a bug");
                            link_bin_op_both_prunable(BlockGroup::new(
                                &mut blocks,
                                [b1, c1, b2, c2, b3, c3, merge],
                            ));
                        }
                        (true, false) => {
                            // branch | ldom, lhs | rdom, rhs | branch | op | merge
                            let b1 = branch_map.get(&lstart).copied().expect("This is a bug");
                            link_bin_op_left_prunable(BlockGroup::new(
                                &mut blocks,
                                [b1, c1, c2, b3, c3, merge],
                            ));
                        }
                        (false, true) => {
                            // | ldom, lhs | branch | rdom, rhs | branch | op | merge
                            let b2 = branch_map.get(&rstart).copied().expect("This is a bug");
                            link_bin_op_right_prunable(BlockGroup::new(
                                &mut blocks,
                                [c1, b2, c2, b3, c3, merge],
                            ));
                        }
                        (false, false) => unreachable!(
                            "We only iterate over selector nodes, this should never happen."
                        ),
                    }
                }
                Less | LessOrEqual | Greater | GreaterOrEqual => {
                    let n = ndom[ni];
                    debug_assert!(n > threshold, "This invariant should always hold.");
                    // branch | cond-dom, cond | merge
                    let start = ni - n;
                    let branch = branch_map.get(&start).copied().expect("This is a bug");
                    let code = code_map.get(&start).copied().expect("This is a bug");
                    let merge = merge_map.get(&(ni + 1)).copied().expect("This is a bug");
                    link_cond(BlockGroup::new(&mut blocks, [branch, code, merge]));
                }
            },
            Ternary(op, _, lhs, rhs) => match op {
                Choose => {
                    // Code duplication with this being the same as Min / Max. But this is OK for now.
                    let ldom = ndom[*lhs];
                    let rdom = ndom[*rhs];
                    let lstart = *lhs - ldom;
                    let rstart = *rhs - rdom;
                    let c1 = code_map.get(&(&lstart)).copied().expect("This is a bug");
                    let c2 = code_map.get(&rstart).copied().expect("This is a bug");
                    let b3 = branch_map.get(&ni).copied().expect("This is a bug");
                    let c3 = code_map.get(&ni).copied().expect("This is a bug");
                    let merge = merge_map.get(&(ni + 1)).copied().expect("This is a bug");
                    match (ldom > threshold, rdom > threshold) {
                        (true, true) => {
                            // branch | ldom, lhs | branch | rdom, rhs | branch | op | merge
                            let b1 = branch_map.get(&lstart).copied().expect("This is a bug");
                            let b2 = branch_map.get(&rstart).copied().expect("This is a bug");
                            link_bin_op_both_prunable(BlockGroup::new(
                                &mut blocks,
                                [b1, c1, b2, c2, b3, c3, merge],
                            ));
                        }
                        (true, false) => {
                            // branch | ldom, lhs | rdom, rhs | branch | op | merge
                            let b1 = branch_map.get(&lstart).copied().expect("This is a bug");
                            link_bin_op_left_prunable(BlockGroup::new(
                                &mut blocks,
                                [b1, c1, c2, b3, c3, merge],
                            ));
                        }
                        (false, true) => {
                            // | ldom, lhs | branch | rdom, rhs | branch | op | merge
                            let b2 = branch_map.get(&rstart).copied().expect("This is a bug");
                            link_bin_op_right_prunable(BlockGroup::new(
                                &mut blocks,
                                [c1, b2, c2, b3, c3, merge],
                            ));
                        }
                        (false, false) => unreachable!(
                            "We only iterate over selector nodes, this should never happen."
                        ),
                    }
                }
            },
        }
    }
    Ok(blocks)
}

fn link_cond(blocks: BlockGroup<'_, 3>) {
    let [branch, _code, merge] = blocks.indices;
    if let [
        Block::Branch { cases },
        Block::Code { instructions },
        Block::Merge {
            listeners,
            incoming,
            selector_node,
        },
    ] = blocks.blocks
    {
        debug_assert_eq!(
            *selector_node,
            instructions.end - 1,
            "This invariant should always hold"
        );
        /* The path is the same whether this gets pruned with a True or
         * False. The only thing that changes is the constant value used in
         * place of the condition op.

        bleft | cond-dom, cond | merge
          │                        ↑
          └────────────────────────┘
         */
        // If true:
        link_jump(
            branch,
            cases,
            merge,
            listeners,
            PruningType::AlwaysTrue,
            Some(Value::Bool(true)),
        );
        link_jump(
            branch,
            cases,
            merge,
            listeners,
            PruningType::AlwaysFalse,
            Some(Value::Bool(false)),
        );
        incoming.push(Incoming {
            block: branch,
            output: None,
        });
    } else {
        unreachable!("Wrong types of block. This is a bug");
    }
}

fn link_bin_op_both_prunable(blocks: BlockGroup<'_, 7>) {
    let [bleft, _, bright, cright, bop, _, merge] = blocks.indices;
    if let [
        Block::Branch {
            cases: left_cases, ..
        },
        Block::Code {
            instructions: left_inst,
        },
        Block::Branch {
            cases: right_cases, ..
        },
        Block::Code {
            instructions: right_inst,
        },
        Block::Branch {
            cases: op_cases, ..
        },
        Block::Code {
            instructions: op_inst,
        },
        Block::Merge {
            listeners,
            selector_node: merge_selector,
            incoming,
        },
        ..,
    ] = blocks.blocks
    {
        debug_assert_eq!(op_inst.start, *merge_selector, "This is a bug");
        /* If lhs gets pruned: (consecutive branching is already done).

        bleft | cleft | bright | cright | bop | cop | merge
          │                       ↑  │    ↑ │          ↑
          └───────────────────────┘  └────┘ └──────────┘
        */
        link_jump(
            bleft,
            left_cases,
            cright,
            listeners,
            PruningType::Left,
            None,
        );
        link_jump(bop, op_cases, merge, listeners, PruningType::Left, None);
        incoming.push(Incoming {
            block: bop,
            output: Some(right_inst.end - 1),
        });
        /* If rhs gets pruned: (consecutive branching is already done).

        bleft | cleft | bright | cright | bop | cop | merge
          │      ↑ │     ↑  │                           ↑
          └──────┘ └─────┘  └───────────────────────────┘
         */
        link_jump(
            bright,
            right_cases,
            merge,
            listeners,
            PruningType::Right,
            None,
        );
        incoming.push(Incoming {
            block: bright,
            output: Some(left_inst.end - 1),
        });
    } else {
        unreachable!("Wrong types of block. This is a bug");
    }
}

fn link_bin_op_left_prunable(blocks: BlockGroup<'_, 6>) {
    let [bleft, _, cright, bop, _, merge] = blocks.indices;
    if let [
        Block::Branch {
            cases: left_cases, ..
        },
        Block::Code { instructions: _ },
        Block::Code {
            instructions: right_inst,
        },
        Block::Branch {
            cases: op_cases, ..
        },
        Block::Code {
            instructions: op_inst,
        },
        Block::Merge {
            listeners,
            selector_node: merge_selector,
            incoming,
        },
        ..,
    ] = blocks.blocks
    {
        debug_assert_eq!(op_inst.start, *merge_selector, "This is a bug");
        /* Only lhs is prunable and it gets pruned.

        bleft | cleft | cright | bop | cop | merge
          │              ↑  │    ↑ │          ↑
          └──────────────┘  └────┘ └──────────┘
         */
        link_jump(
            bleft,
            left_cases,
            cright,
            listeners,
            PruningType::Left,
            None,
        );
        link_jump(bop, op_cases, merge, listeners, PruningType::Left, None);
        incoming.push(Incoming {
            block: bop,
            output: Some(right_inst.end - 1),
        });
    } else {
        unreachable!("Wrong types of block. This is a bug");
    }
}

fn link_bin_op_right_prunable(blocks: BlockGroup<'_, 6>) {
    let [_, bright, _, _, _, merge] = blocks.indices;
    if let [
        Block::Code { instructions: _ },
        Block::Branch {
            cases: right_cases, ..
        },
        Block::Code {
            instructions: right_inst,
        },
        Block::Branch { cases: _, .. },
        Block::Code {
            instructions: op_inst,
        },
        Block::Merge {
            listeners,
            selector_node: merge_selector,
            incoming,
        },
        ..,
    ] = blocks.blocks
    {
        debug_assert_eq!(op_inst.start, *merge_selector, "This is a bug");
        /* Only rhs is purnable and it gets pruned.

        cleft | bright | cright | bop | cop | merge
          │      ↑ │                            ↑
          └──────┘ └────────────────────────────┘
         */
        link_jump(
            bright,
            right_cases,
            merge,
            listeners,
            PruningType::Right,
            None,
        );
        incoming.push(Incoming {
            block: bright,
            output: Some(right_inst.end - 1),
        });
    } else {
        unreachable!("Wrong types of block. This is a bug");
    }
}

fn link_jump(
    branch: usize,
    cases: &mut Vec<Case>,
    target: usize,
    listeners: &mut Vec<Listener>,
    prune: PruningType,
    output: Option<Value>,
) {
    let case = cases.len();
    cases.push(Case {
        target_block: target,
        output,
    });
    listeners.push(Listener {
        branch,
        case: case as u32,
        prune,
    });
}

struct BlockGroup<'a, const N: usize> {
    blocks: [&'a mut Block; N],
    indices: [usize; N],
    phantom: PhantomData<&'a mut [Block]>,
}

impl<'a, const N: usize> BlockGroup<'a, N> {
    fn new(slice: &'a mut [Block], indices: [usize; N]) -> Self {
        assert!(
            indices.windows(2).all(|window| window[0] < window[1]),
            "The indices must be increasing: {:?}",
            indices
        );
        assert!(
            indices.iter().all(|i| *i < slice.len()),
            "All indices must be within bounds: {:?}",
            indices
        );
        let ptr = slice.as_mut_ptr();
        Self {
            // # SAFETY: The two asserts above ensure the indices are
            // non-overlapping and within bounds. So this is safe.
            blocks: unsafe { indices.map(|i| &mut *ptr.add(i)) },
            phantom: PhantomData,
            indices,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Split {
    Branch(usize),
    Merge(usize),
    Direct(usize),
}

fn make_layout(tree: &Tree, threshold: usize, ndom: &[usize]) -> (Box<[Split]>, Box<[bool]>) {
    let mut splits: Vec<Split> = Vec::with_capacity(tree.len() / 2);
    let mut is_selector: Box<[bool]> = vec![false; tree.len()].into_boxed_slice();
    for (i, node) in tree.nodes().iter().enumerate() {
        match node {
            Constant(_) | Symbol(_) | Unary(_, _) => continue,
            Binary(op, lhs, rhs) => match op {
                Add | Subtract | Multiply | Divide | Pow | Remainder | Equal | NotEqual | And
                | Or => continue,
                Min | Max => {
                    let ldom = ndom[*lhs];
                    let rdom = ndom[*rhs];
                    match (ldom > threshold, rdom > threshold) {
                        (true, true) => {
                            // branch | ldom, lhs | branch | rdom, rhs | branch | op | merge
                            splits.extend_from_slice(&[
                                Split::Branch(*lhs - ldom),
                                Split::Branch(*rhs - rdom),
                                Split::Branch(i),
                                Split::Merge(i + 1),
                            ]);
                            is_selector[i] = true;
                        }
                        (true, false) => {
                            // branch | ldom, lhs | rdom, rhs | branch | op | merge
                            splits.extend_from_slice(&[
                                Split::Branch(*lhs - ldom),
                                Split::Direct(*rhs - rdom),
                                Split::Branch(i),
                                Split::Merge(i + 1),
                            ]);
                            is_selector[i] = true;
                        }
                        (false, true) => {
                            // | ldom, lhs | branch | rdom, rhs | branch | op | merge
                            splits.extend_from_slice(&[
                                Split::Direct(*lhs - ldom),
                                Split::Branch(*rhs - rdom),
                                Split::Branch(i),
                                Split::Merge(i + 1),
                            ]);
                            is_selector[i] = true;
                        }
                        (false, false) => continue,
                    }
                }
                Less | LessOrEqual | Greater | GreaterOrEqual => {
                    // branch | cond-dom, cond | merge
                    let n = ndom[i];
                    if n > threshold {
                        splits.extend_from_slice(&[Split::Branch(i - n), Split::Merge(i + 1)]);
                        is_selector[i] = true;
                    }
                }
            },
            Ternary(op, cond, tt, ff) => match op {
                Choose => {
                    if is_selector[*cond] {
                        let ttdom = ndom[*tt];
                        let ffdom = ndom[*ff];
                        match (ttdom > threshold, ffdom > threshold) {
                            (true, true) => {
                                // branch | ttdom, tt | branch | ffdom, ff | branch | choose | merge.
                                splits.extend_from_slice(&[
                                    Split::Branch(*tt - ttdom),
                                    Split::Branch(*ff - ffdom),
                                    Split::Branch(i),
                                    Split::Merge(i + 1),
                                ]);
                                is_selector[i] = true;
                            }
                            (true, false) => {
                                // branch | ttdom, tt | ffdom, ff | branch | choose | merge.
                                splits.extend_from_slice(&[
                                    Split::Branch(*tt - ttdom),
                                    Split::Direct(*ff - ffdom),
                                    Split::Branch(i),
                                    Split::Merge(i + 1),
                                ]);
                                is_selector[i] = true;
                            }
                            (false, true) => {
                                // | ttdom, tt | branch | ffdom, ff | branch | choose | merge.
                                splits.extend_from_slice(&[
                                    Split::Branch(*tt - ttdom),
                                    Split::Branch(*ff - ffdom),
                                    Split::Direct(i),
                                    Split::Merge(i + 1),
                                ]);
                                is_selector[i] = true;
                            }
                            (false, false) => continue,
                        }
                    }
                }
            },
        }
    }
    splits.sort_by(|a, b| match (a, b) {
        (Split::Branch(a), Split::Branch(b))
        | (Split::Merge(a), Split::Merge(b))
        | (Split::Direct(a), Split::Direct(b)) => a.cmp(b),
        (Split::Branch(a), Split::Merge(b))
        | (Split::Direct(a), Split::Branch(b))
        | (Split::Direct(a), Split::Merge(b)) => (*a, 1).cmp(&(*b, 0)), // Prefer b.
        (Split::Branch(a), Split::Direct(b))
        | (Split::Merge(a), Split::Branch(b))
        | (Split::Merge(a), Split::Direct(b)) => (*a, 0).cmp(&(*b, 1)), // Prefer a.
    });
    splits.dedup_by(|a, b| match (a, b) {
        (Split::Branch(a), Split::Branch(b))
        | (Split::Merge(a), Split::Merge(b))
        | (Split::Direct(a), Split::Direct(b)) => a == b,
        (Split::Branch(_), Split::Merge(_)) | (Split::Merge(_), Split::Branch(_)) => false,
        (Split::Branch(a), Split::Direct(b))
        | (Split::Direct(a), Split::Branch(b))
        | (Split::Merge(a), Split::Direct(b))
        | (Split::Direct(a), Split::Merge(b))
            if a == b =>
        {
            true
        }
        _ => false,
    });
    (splits.into_boxed_slice(), is_selector)
}

type NativePruningFunc = unsafe extern "C" fn(
    *const c_void, // Inputs
    *mut c_void,   // Outputs
    *mut u32,      // Signals - tells future pruned evaluations how to skip and jump.
);

type NativeSingleFunc = unsafe extern "C" fn(
    *const c_void, // Inputs,
    *const u32,    // Signals,
    *mut c_void,
);

pub type NativeSimdFunc = unsafe extern "C" fn(
    *const c_void, // Inputs
    *const u32,    // Signals,
    *mut c_void,   // Outputs
    u64,           // Number of evals.
);

pub type NativeIntervalFunc = unsafe extern "C" fn(
    *const c_void, // Inputs
    *const u32,    // Signals,
    *mut c_void,   // Outputs
);

fn compile_pruning_func<'ctx, T: NumberType>(
    tree: &Tree,
    threshold: usize,
    context: &'ctx JitContext,
    params: &str,
) -> Result<(), Error> {
    if !tree.is_scalar() {
        // Only support scalar output trees.
        return Err(Error::TypeMismatch);
    }
    let ranges = interval::compute_ranges(tree)?;
    let func_name = context.new_func_name::<T>(Some("interval"));
    let context = &context.inner;
    let compiler = JitCompiler::new(context)?;
    let builder = &compiler.builder;
    let interval_type = T::jit_type(context).vec_type(2);
    let iptr_type = context.ptr_type(AddressSpace::default());
    let mut constants = interval::Constants::create::<T>(context);
    let fn_type = context
        .void_type()
        .fn_type(&[iptr_type.into(), iptr_type.into()], false);
    let function = compiler.module.add_function(&func_name, fn_type, None);
    compiler.set_attributes(function, context)?;
    let entry_bb = context.append_basic_block(function, "entry");
    builder.position_at_end(entry_bb);
    let mut regs = Vec::<BasicValueEnum>::with_capacity(tree.len());
    let blocks = make_blocks(tree, threshold)?;
    let mut bbs: Box<[BasicBlock<'ctx>]> = blocks
        .iter()
        .enumerate()
        .map(|(bi, block)| {
            context.append_basic_block(
                function,
                &format!(
                    "{}_block_{bi}",
                    match block {
                        Block::Branch { .. } => "branch",
                        Block::Code { .. } => "code",
                        Block::Merge { .. } => "merge",
                    }
                ),
            )
        })
        .collect();
    let branch_signal_map: Box<[usize]> = {
        blocks
            .iter()
            .scan(0usize, |idx, block| match block {
                Block::Branch { .. } => Some(std::mem::replace(idx, *idx + 1)),
                Block::Code { .. } | Block::Merge { .. } => Some(*idx),
            })
            .collect()
    };
    let signal_ptrs = {
        let signals_arg = function
            .get_nth_param(2)
            .ok_or(Error::JitCompilationError(
                "Cannot read output address".to_string(),
            ))?
            .into_pointer_value();
        blocks
            .iter()
            .filter(|block| match block {
                Block::Branch { .. } => true,
                Block::Code { .. } | Block::Merge { .. } => false,
            })
            .enumerate()
            .try_fold(
                Vec::with_capacity(blocks.len()),
                |mut signals, (bi, _block)| -> Result<Vec<PointerValue>, Error> {
                    signals.push(unsafe {
                        builder.build_gep(
                            context.i32_type(),
                            signals_arg,
                            &[constants.int_32(bi as u32, false)],
                            &format!("signal_ptr_{bi}"),
                        )?
                    });
                    Ok(signals)
                },
            )?
            .into_boxed_slice()
    };
    let signals = {
        signal_ptrs
            .iter()
            .copied()
            .enumerate()
            .try_fold(
                Vec::with_capacity(blocks.len()),
                |mut signals, (i, ptr)| -> Result<Vec<IntValue>, Error> {
                    signals.push(
                        builder
                            .build_load(context.i32_type(), ptr, &format!("load_signal_{i}"))?
                            .into_int_value(),
                    );
                    Ok(signals)
                },
            )?
            .into_boxed_slice()
    };
    let mut branch_outputs = HashMap::<(usize, usize), BasicValueEnum>::new();
    for (bi, block) in blocks.into_iter().enumerate() {
        match block {
            Block::Branch { cases } => {
                let signal = signals[branch_signal_map[bi]];
                // Some of these cases may be trying to forward a constant value
                // to their target branch. So we create those constant
                // valuesnow, in the appropriate basic block.
                build_branch_forwarding_outputs(
                    &mut constants,
                    &cases,
                    &mut branch_outputs,
                    builder,
                    bi,
                    signal,
                    &bbs,
                )?;
                // Now switch to the target branches.
                builder.position_at_end(bbs[bi]);
                let cases: Box<[(IntValue, BasicBlock)]> = cases
                    .iter()
                    .enumerate()
                    .skip(1)
                    .map(|(case, target)| {
                        (
                            constants.int_32(case as u32, false),
                            bbs[target.target_block],
                        )
                    })
                    .collect();
                builder.build_switch(signal, bbs[bi + 1], cases.as_ref())?;
            }
            Block::Code { instructions } => {
                builder.position_at_end(bbs[bi]);
                for (index, node) in tree.nodes()[instructions].iter().copied().enumerate() {
                    let reg = interval::build_op::<T>(
                        BuildArgs {
                            nodes: tree.nodes(),
                            params,
                            ranges: &ranges,
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
                    regs.push(reg);
                }
                // It is possible while building the instruction we created
                // another basic block. So we ask the builder and overwrite.
                if let Some(newbb) = builder.get_insert_block() {
                    bbs[bi] = newbb;
                }
                // Every code blocks unconditionally branches to the block that
                // comes right after it, except for the last code block, because
                // it has nowhere to go to.
                if let Some(next_bb) = bbs.get(bi + 1).copied() {
                    builder.build_unconditional_branch(next_bb)?;
                }
            }
            Block::Merge {
                listeners,
                incoming,
                selector_node,
            } => {
                builder.position_at_end(bbs[bi]);
                // Merge incoming values into a phi and overwrite the output of `selector_node`.
                let phi = builder.build_phi(
                    match is_node_scalar(tree.nodes(), selector_node) {
                        true => interval_type,
                        false => context.bool_type().vec_type(2),
                    },
                    &format!("merge_block_phi_{selector_node}"),
                )?;
                let incoming: Box<[(&dyn BasicValue, BasicBlock)]> = incoming
                    .iter()
                    .map(|Incoming { block, output }| match output {
                        Some(output) => (&regs[*output] as &dyn BasicValue, bbs[*block]),
                        None => match branch_outputs.get(&(*block, bi)) {
                            Some(output) => (output as &dyn BasicValue, bbs[*block]),
                            None => unreachable!(
                                "This CFG edge has neither a forwarding constant, nor a previously computed value. This is a bug."
                            ),
                        },
                    }).collect();
                phi.add_incoming(&incoming);
                regs[selector_node] = phi.as_basic_value();
                // Notify the listeners about the current interval.
                match tree.node(selector_node) {
                    Constant(_) | Symbol(_) | Unary(_, _) => {}
                    Binary(op, lhs, rhs) => match op {
                        Add | Subtract | Multiply | Divide | Pow | Remainder | Equal | NotEqual
                        | And | Or => {}
                        Min | Max | Less | LessOrEqual | Greater | GreaterOrEqual => {
                            let interval::InequalityFlags {
                                either_empty,
                                strictly_before,
                                strictly_after,
                                touching,
                            } = interval::build_interval_inequality_flags(
                                regs[*lhs].into_vector_value(),
                                regs[*rhs].into_vector_value(),
                                builder,
                                &compiler.module,
                                &mut constants,
                                selector_node,
                            )?;
                            match op {
                                Min => todo!(),
                                Max => todo!(),
                                Less => todo!(),
                                LessOrEqual => todo!(),
                                Greater => todo!(),
                                GreaterOrEqual => todo!(),
                                _ => unreachable!(
                                    "We already checked, this should never happen, this is a bug."
                                ),
                            }
                        }
                    },
                    Ternary(op, _, _, _) => todo!(),
                }
                todo!("Notify listeners about the current interval");
            }
        }
    }
    todo!();
}

fn build_branch_forwarding_outputs<'ctx>(
    constants: &mut interval::Constants<'ctx>,
    cases: &[Case],
    dst: &mut HashMap<(usize, usize), BasicValueEnum<'ctx>>,
    builder: &'ctx Builder,
    branch_index: usize,
    signal: IntValue<'ctx>,
    bbs: &[BasicBlock<'ctx>],
) -> Result<(), Error> {
    let mut prev: Option<(usize, usize, BasicValueEnum)> = None;
    for (case, target, value) in cases.iter().enumerate().filter_map(
        |(
            i,
            Case {
                target_block,
                output,
            },
        )| { output.map(|output| (i, *target_block, output)) },
    ) {
        match prev {
            Some((_, prev_target, prev_bv)) => {
                if prev_target == target {
                    let case_eq = builder.build_int_compare(
                        IntPredicate::EQ,
                        signal,
                        constants.int_32(case as u32, false),
                        &format!("branch_{branch_index}_case_{case}_forwarding_comparison"),
                    )?;
                    let next_bv = match value {
                        Value::Bool(flag) => builder.build_select(
                            case_eq,
                            constants.boolean(flag),
                            prev_bv.into_int_value(),
                            &format!("branch_{branch_index}_case_{case}_forwarding_comparison"),
                        )?,
                        Value::Scalar(val) => builder.build_select(
                            case_eq,
                            constants.float(val),
                            prev_bv.into_float_value(),
                            &format!("branch_{branch_index}_case_{case}_forwarding_comparison"),
                        )?,
                    };
                    prev = Some((case, target, next_bv));
                } else {
                    let conflict = dst.insert((branch_index, prev_target), prev_bv);
                    assert!(
                        conflict.is_none(),
                        "We tried to insert two different forwarding values for the same CFG edge"
                    );
                    builder.position_at_end(bbs[target]);
                    prev = Some((case, target, interval::build_const(constants, value)));
                }
            }
            None => {
                builder.position_at_end(bbs[target]);
                prev = Some((case, target, interval::build_const(constants, value)));
            }
        }
    }
    if let Some((_, target, bv)) = prev {
        let conflict = dst.insert((branch_index, target), bv);
        assert!(
            conflict.is_none(),
            "We tried to insert two different forwarding values for the same CFG edge"
        );
    }
    Ok(())
}

impl Tree {
    pub fn jit_compile_pruning<'ctx, T: NumberType>(
        &'ctx self,
        threshold: usize,
        context: &'ctx JitContext,
        params: &str,
    ) -> Result<(), Error> {
        todo!();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Node, deftree, llvm_jit::pruning::make_layout};

    #[test]
    fn t_min_sphere_layout() {
        let tree = deftree!(min
                 (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5)
                 (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5))
        .unwrap();
        let (tree, ndom) = tree
            .control_dependence_sorted()
            .expect("Dominator sorting failed");
        let (splits, is_selector) = make_layout(&tree, 10, &ndom);
        assert_eq!(is_selector.len(), tree.len());
        assert!(!is_selector.iter().take(tree.len() - 1).any(|b| *b));
        assert!(is_selector.last().unwrap());
        assert_eq!(
            splits.as_ref(),
            &[
                Split::Branch(0,),
                Split::Branch(12,),
                Split::Branch(24,),
                Split::Merge(25,)
            ]
        );
    }

    #[test]
    fn t_min_3_spheres_layout() {
        let tree = deftree!(min (min
                                  (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5)
                                  (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5))
                            (- (sqrt (+ (pow 'x 2) (pow (- 'y 1) 2))) 1.5))
        .unwrap();
        let (tree, ndom) = tree
            .control_dependence_sorted()
            .expect("Dominator sorting failed");
        let (splits, is_selector) = make_layout(&tree, 10, &ndom);
        println!("{tree}");
        assert_eq!(is_selector.len(), tree.len());
        assert_eq!(is_selector.iter().filter(|b| **b).count(), 2);
        for (i, _) in is_selector.iter().enumerate().filter(|(_i, b)| **b) {
            assert!(matches!(tree.node(i), Node::Binary(Min, _, _)));
        }
        assert_eq!(
            splits.as_ref(),
            &[
                Split::Branch(0,),
                Split::Branch(12,),
                Split::Branch(24,),
                Split::Merge(25,),
                Split::Branch(25,),
                Split::Branch(37,),
                Split::Merge(38,)
            ]
        );
    }

    #[test]
    fn t_min_sphere_blocks() {
        let tree = deftree!(min
                 (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5)
                 (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5))
        .unwrap();
        let blocks = make_blocks(&tree, 10).expect("Unable to make blocks");
        dbg!(blocks);
        assert!(false);
    }

    #[test]
    fn t_min_3_spheres_blocks() {
        let tree = deftree!(min (min
                                  (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5)
                                  (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5))
                            (- (sqrt (+ (pow 'x 2) (pow (- 'y 1) 2))) 1.5))
        .unwrap();
        let blocks = make_blocks(&tree, 10).expect("Unable to make blocks");
        dbg!(blocks);
        assert!(false);
    }

    #[test]
    fn t_simple_cond() {
        let tree = deftree!(if (< 'x 0) (+ 'x 1) (- 'x 1))
            .unwrap()
            .compacted()
            .unwrap();
        let blocks = make_blocks(&tree, 0).unwrap();
        println!("{tree}");
        dbg!(blocks);
        assert!(false);
    }

    #[test]
    fn t_simple_choose() {
        let tree = deftree!(if (< 'x 0) (- 'a 1) (- 'b 2))
            .unwrap()
            .compacted()
            .unwrap();
        let blocks = make_blocks(&tree, 0).unwrap();
        println!("{tree}");
        dbg!(blocks);
        assert!(false);
    }
}
