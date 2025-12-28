use crate::{BinaryOp::*, Error, Node::*, TernaryOp::*, Tree};
use std::ops::Range;

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

enum Block {
    Branch {
        cases: Vec<usize>,
    },
    Code {
        instructions: Range<usize>,
    },
    Merge {
        incoming: Vec<usize>,
        selector_node: usize,
        token: usize,
    },
}

#[derive(Copy, Clone)]
enum Split {
    Branch(usize),
    Merge(usize),
    Direct(usize),
}

fn make_blocks(tree: &Tree, threshold: usize) -> Result<Box<[Block]>, Error> {
    let (tree, ndom) = tree.control_dependence_sorted()?;
    assert_eq!(
        tree.len(),
        ndom.len(),
        "This should never happen, it is a bug in control dependence sorting"
    );
    let (splits, is_selector) = make_layout(&tree, threshold, &ndom);
    todo!();
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
