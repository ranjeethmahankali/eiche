use crate::{BinaryOp::*, Error, Node::*, TernaryOp::*, Tree};
use std::{collections::HashMap, marker::PhantomData, ops::Range};

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
 */

#[derive(Debug)]
pub enum PruningType {
    None,
    LeftOrTrue,
    RightOrFalse,
}

#[derive(Debug)]
pub struct Listener {
    branch: usize,
    case: usize,
    prune: PruningType,
}

#[derive(Debug)]
pub struct Incoming {
    block: usize,
    output: usize,
}

#[derive(Debug)]
pub enum Block {
    Branch {
        cases: Vec<usize>,
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
    let (blocks, _) = splits.iter().fold(
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
                output: instructions.end - 1,
            }),
            // The default case i.e. '0' in every branch just goes to the next block.
            (Block::Branch { cases }, Block::Code { .. }) => {
                cases.push(bi + 1);
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
                    // branch | ldom, lhs, rdom, rhs | branch | cond | merge
                }
            },
            Ternary(op, _, _, _) => match op {
                Choose => todo!(),
            },
        }
    }
    Ok(blocks)
}

fn link_bin_op_both_prunable(blocks: BlockGroup<'_, 7>) {
    let [bleft, _, bright, cright, bop, _, merge] = blocks.indices;
    if let [
        Block::Branch { cases: left_cases },
        Block::Code {
            instructions: left_inst,
        },
        Block::Branch { cases: right_cases },
        Block::Code {
            instructions: right_inst,
        },
        Block::Branch { cases: op_cases },
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
            PruningType::LeftOrTrue,
        );
        link_jump(bop, op_cases, merge, listeners, PruningType::LeftOrTrue);
        incoming.push(Incoming {
            block: bop,
            output: right_inst.end - 1,
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
            PruningType::RightOrFalse,
        );
        incoming.push(Incoming {
            block: bright,
            output: left_inst.end - 1,
        });
    } else {
        unreachable!("Wrong types of block. This is a bug");
    }
}

fn link_bin_op_left_prunable(blocks: BlockGroup<'_, 6>) {
    let [bleft, _, cright, bop, _, merge] = blocks.indices;
    if let [
        Block::Branch { cases: left_cases },
        Block::Code { instructions: _ },
        Block::Code {
            instructions: right_inst,
        },
        Block::Branch { cases: op_cases },
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
            PruningType::LeftOrTrue,
        );
        link_jump(bop, op_cases, merge, listeners, PruningType::LeftOrTrue);
        incoming.push(Incoming {
            block: bop,
            output: right_inst.end - 1,
        });
    } else {
        unreachable!("Wrong types of block. This is a bug");
    }
}

fn link_bin_op_right_prunable(blocks: BlockGroup<'_, 6>) {
    let [_, bright, _, _, _, merge] = blocks.indices;
    if let [
        Block::Code { instructions: _ },
        Block::Branch { cases: right_cases },
        Block::Code {
            instructions: right_inst,
        },
        Block::Branch { cases: _ },
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
            PruningType::RightOrFalse,
        );
        incoming.push(Incoming {
            block: bright,
            output: right_inst.end - 1,
        });
    } else {
        unreachable!("Wrong types of block. This is a bug");
    }
}

fn link_jump(
    branch: usize,
    cases: &mut Vec<usize>,
    target: usize,
    listeners: &mut Vec<Listener>,
    prune: PruningType,
) {
    let case = cases.len();
    cases.push(target);
    listeners.push(Listener {
        branch,
        case,
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
                    // branch | ldom, lhs, rdom, rhs | branch | cond | merge
                    let n = ndom[i];
                    if n > threshold {
                        splits.extend_from_slice(&[
                            Split::Branch(i - n),
                            Split::Branch(i),
                            Split::Merge(i + 1),
                        ]);
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
        dbg!(&splits, &is_selector);
        println!("{tree}");
        assert_eq!(is_selector.len(), tree.len());
        assert_eq!(is_selector.iter().filter(|b| **b).count(), 2);
        for (i, _) in is_selector.iter().enumerate().filter(|(i, b)| **b) {
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
}
