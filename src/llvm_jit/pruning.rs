use crate::{BinaryOp::*, Error, Node::*, TernaryOp::*, Tree, Value};
use std::{ffi::c_void, ops::Range};

#[derive(Clone, Debug, PartialEq, Eq)]
enum Alternate {
    None,
    Node(usize),
    Constant(Value),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Interrupt {
    Jump {
        before_node: usize,
        target: usize,
        alternate: Alternate,
    },
    Land {
        after_node: usize,
    },
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
enum PruneKind {
    Left,
    Right,
    AlwaysTrue,
    AlwaysFalse,
}

#[derive(Debug, PartialEq, Eq)]
struct Notification {
    receiver: usize,
    sender_node: usize,
    message: PruneKind,
}

#[derive(Debug, PartialEq, Eq)]
struct ControlFlow {
    interrupts: Box<[Interrupt]>,
    criteria: Box<[Notification]>,
}

impl ControlFlow {
    fn from_tree(tree: &Tree, threshold: usize) -> Result<ControlFlow, Error> {
        let (tree, ndom) = tree.control_dependence_sorted()?;
        let mut interrupts = Vec::<Interrupt>::with_capacity(tree.len() / 2);
        let mut criteria = Vec::<Notification>::new();
        let mut land_map: Vec<Option<usize>> = vec![None; tree.len()];
        for (ni, node) in tree.nodes().iter().enumerate() {
            match node {
                Binary(Min | Max, lhs, rhs) => {
                    let ldom = ndom[*lhs];
                    let rdom = ndom[*rhs];
                    let lskip = ldom > threshold;
                    let rskip = rdom > threshold;
                    if lskip {
                        let l = push_land(&mut interrupts, *lhs, &mut land_map);
                        let j = push_interrupt(
                            &mut interrupts,
                            Interrupt::Jump {
                                before_node: *lhs - ldom,
                                target: l,
                                alternate: Alternate::None,
                            },
                        );
                        criteria.push(Notification {
                            receiver: j,
                            sender_node: ni,
                            message: PruneKind::Left,
                        });
                    }
                    if rskip {
                        let l = push_land(&mut interrupts, *rhs, &mut land_map);
                        let j = push_interrupt(
                            &mut interrupts,
                            Interrupt::Jump {
                                before_node: *rhs - rdom,
                                target: l,
                                alternate: Alternate::None,
                            },
                        );
                        criteria.push(Notification {
                            receiver: j,
                            sender_node: ni,
                            message: PruneKind::Right,
                        });
                    }
                    if lskip || rskip {
                        let merge = push_land(&mut interrupts, ni, &mut land_map);
                        if lskip {
                            let j = push_interrupt(
                                &mut interrupts,
                                Interrupt::Jump {
                                    before_node: ni,
                                    target: merge,
                                    alternate: Alternate::Node(*rhs),
                                },
                            );
                            criteria.push(Notification {
                                receiver: j,
                                sender_node: ni,
                                message: PruneKind::Left,
                            });
                        }
                        if rskip {
                            let j = push_interrupt(
                                &mut interrupts,
                                Interrupt::Jump {
                                    before_node: ni,
                                    target: merge,
                                    alternate: Alternate::Node(*lhs),
                                },
                            );
                            criteria.push(Notification {
                                receiver: j,
                                sender_node: ni,
                                message: PruneKind::Right,
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
        let mut numbered: Vec<(usize, Interrupt)> =
            interrupts.iter().cloned().enumerate().collect();
        numbered.sort_by(|(_, a), (_, b)| -> std::cmp::Ordering {
            match (a, b) {
                (
                    Interrupt::Jump {
                        before_node: lbn,
                        target: lt,
                        alternate: _,
                    },
                    Interrupt::Jump {
                        before_node: rbn,
                        target: rt,
                        alternate: _,
                    },
                ) => {
                    let (lt, rt) = match (&interrupts[*lt], &interrupts[*rt]) {
                        (
                            Interrupt::Land { after_node: la },
                            Interrupt::Land { after_node: ra },
                        ) => (la, ra),
                        _ => unreachable!("This is a bug"),
                    };
                    (lbn, std::cmp::Reverse(lt)).cmp(&(rbn, std::cmp::Reverse(rt)))
                }
                (Interrupt::Jump { before_node, .. }, Interrupt::Land { after_node }) => {
                    (before_node, 0).cmp(&(after_node, 1))
                }
                (Interrupt::Land { after_node }, Interrupt::Jump { before_node, .. }) => {
                    (after_node, 1).cmp(&(before_node, 0))
                }
                (Interrupt::Land { after_node: la }, Interrupt::Land { after_node: ra }) => {
                    la.cmp(&ra)
                }
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
            } => Interrupt::Jump {
                before_node,
                target: idxmap[target],
                alternate,
            },
            Interrupt::Land { after_node } => Interrupt::Land { after_node },
        }));
        for c in criteria.iter_mut() {
            c.receiver = idxmap[c.receiver];
        }
        criteria.sort_by(|a, b| {
            (a.sender_node, a.message, a.receiver).cmp(&(b.sender_node, b.message, b.receiver))
        });
        Ok(ControlFlow {
            interrupts: interrupts.into_boxed_slice(),
            criteria: criteria.into_boxed_slice(),
        })
    }
}

enum Block {
    Code(Range<usize>),
    Branch(Range<usize>),
    Merge(usize),
}

fn make_blocks(interrupts: Box<[Interrupt]>, n_nodes: usize) -> Result<Box<[Block]>, Error> {
    enum PartialBlock {
        Code(usize),
        Branch(usize),
    }
    let (mut blocks, last) = interrupts.into_iter().enumerate().fold(
        (Vec::<Block>::new(), PartialBlock::Code(0)),
        |(mut blocks, current), (i, interrupt)| match (current, interrupt) {
            (PartialBlock::Code(start), Interrupt::Jump { before_node, .. }) => {
                if start < before_node {
                    blocks.push(Block::Code(start..before_node));
                }
                (blocks, PartialBlock::Branch(i))
            }
            (PartialBlock::Code(start), Interrupt::Land { after_node }) => {
                if start < after_node + 1 {
                    blocks.push(Block::Code(start..(after_node + 1)));
                }
                blocks.push(Block::Merge(after_node));
                (blocks, PartialBlock::Code(after_node + 1))
            }
            (PartialBlock::Branch(index), Interrupt::Jump { .. }) => {
                debug_assert!(index < i);
                (blocks, PartialBlock::Branch(index))
            }
            (PartialBlock::Branch(_), Interrupt::Land { .. }) => {
                unreachable!("Branching and landing without any code in between. This is a bug")
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

fn push_interrupt(dst: &mut Vec<Interrupt>, interrupt: Interrupt) -> usize {
    let idx = dst.len();
    dst.push(interrupt);
    return idx;
}

fn push_land(dst: &mut Vec<Interrupt>, after_node: usize, land_map: &mut [Option<usize>]) -> usize {
    let mapped = &mut land_map[after_node];
    match mapped {
        Some(found) => *found,
        None => {
            let out = push_interrupt(dst, Interrupt::Land { after_node });
            *mapped = Some(out);
            out
        }
    }
}

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

#[cfg(test)]
mod test {
    use super::*;
    use crate::{deftree, llvm_jit::pruning::ControlFlow};

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
        let cfg = ControlFlow::from_tree(&tree, 0).expect("Cannot compute control flow");
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
        let cfg = ControlFlow::from_tree(&tree, 0).expect("Unable to build control flow");
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
        let cfg = ControlFlow::from_tree(&tree, 0).expect("Unable to build control flow");
        dbg!(tree.nodes(), cfg);
        assert!(false);
    }
}
