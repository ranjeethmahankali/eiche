use crate::{BinaryOp::*, Error, Node::*, TernaryOp::*, Tree, Value};
use std::ffi::c_void;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Choice {
    Node(usize),
    Constant(Value),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Interrupt {
    Jump { before: usize, target: usize },
    Land { after: usize, source: usize },
    Diverge { before: usize, choice: Choice },
    Converge { after: usize },
}

#[derive(Debug, PartialEq, Eq)]
enum PruneKind {
    Left,
    Right,
    AlwaysTrue,
    AlwaysFalse,
}

#[derive(Debug, PartialEq, Eq)]
struct Criteria {
    start: usize,
    end: usize,
    owner: usize,
    kind: PruneKind,
}

#[derive(Debug, PartialEq, Eq)]
struct ControlFlow {
    interrupts: Box<[Interrupt]>,
    criteria: Box<[Criteria]>,
}

impl ControlFlow {
    fn from_tree(tree: &Tree, threshold: usize) -> Result<ControlFlow, Error> {
        let (tree, ndom) = tree.control_dependence_sorted()?;
        let mut interrupts = Vec::<Interrupt>::with_capacity(tree.len() / 2);
        let mut skippable = vec![false; tree.len()].into_boxed_slice();
        let mut criteria = Vec::<Criteria>::new();
        for (ni, node) in tree.nodes().iter().enumerate() {
            match node {
                Binary(Min | Max, lhs, rhs) => {
                    let ldom = ndom[*lhs];
                    let rdom = ndom[*rhs];
                    let lskip = ldom > threshold;
                    let rskip = rdom > threshold;
                    if lskip {
                        make_skip(
                            *lhs - ldom,
                            *lhs,
                            ni,
                            PruneKind::Left,
                            &mut skippable[*lhs],
                            &mut interrupts,
                            &mut criteria,
                        );
                        interrupts.push(Interrupt::Diverge {
                            before: ni,
                            choice: Choice::Node(*rhs),
                        });
                    }
                    if rskip {
                        make_skip(
                            *rhs - rdom,
                            *rhs,
                            ni,
                            PruneKind::Right,
                            &mut skippable[*rhs],
                            &mut interrupts,
                            &mut criteria,
                        );
                        interrupts.push(Interrupt::Diverge {
                            before: ni,
                            choice: Choice::Node(*lhs),
                        });
                    }
                    if lskip || rskip {
                        interrupts.push(Interrupt::Converge { after: ni });
                    }
                }
                Binary(Less | LessOrEqual | Greater | GreaterOrEqual, _lhs, _rhs) => {
                    let dom = ndom[ni];
                    let start = ni - dom;
                    if dom > threshold {
                        interrupts.push(Interrupt::Diverge {
                            before: start,
                            choice: Choice::Constant(Value::Bool(true)),
                        });
                        interrupts.push(Interrupt::Diverge {
                            before: start,
                            choice: Choice::Constant(Value::Bool(false)),
                        });
                        interrupts.push(Interrupt::Converge { after: ni });
                    }
                }
                Ternary(Choose, _cond, tt, ff) => {
                    let ttdom = ndom[*tt];
                    let ffdom = ndom[*ff];
                    let tskip = ttdom > threshold;
                    let fskip = ffdom > threshold;
                    if tskip {
                        make_skip(
                            *tt - ttdom,
                            *tt,
                            ni,
                            PruneKind::Left,
                            &mut skippable[*tt],
                            &mut interrupts,
                            &mut criteria,
                        );
                        interrupts.push(Interrupt::Diverge {
                            before: ni,
                            choice: Choice::Node(*ff),
                        });
                    }
                    if fskip {
                        make_skip(
                            *ff - ffdom,
                            *ff,
                            ni,
                            PruneKind::Left,
                            &mut skippable[*ff],
                            &mut interrupts,
                            &mut criteria,
                        );
                        interrupts.push(Interrupt::Diverge {
                            before: ni,
                            choice: Choice::Node(*tt),
                        });
                    }
                    if tskip || fskip {
                        interrupts.push(Interrupt::Converge { after: ni });
                    }
                }
                _ => continue,
            }
        }
        interrupts.sort_by(|a, b| match (a, b) {
            (
                Interrupt::Jump {
                    before: lb,
                    target: lt,
                },
                Interrupt::Jump {
                    before: rb,
                    target: rt,
                },
            ) => (*lb, std::cmp::Reverse(*lt)).cmp(&(*rb, std::cmp::Reverse(*rt))),
            (Interrupt::Jump { before, .. }, Interrupt::Land { after, .. }) => {
                (*before, 0).cmp(&(*after, 1)) // If the positions are the same, jump should come first.
            }
            (Interrupt::Jump { before: lb, .. }, Interrupt::Diverge { before: rb, .. })
            | (Interrupt::Diverge { before: lb, .. }, Interrupt::Jump { before: rb, .. })
            | (Interrupt::Diverge { before: lb, .. }, Interrupt::Diverge { before: rb, .. }) => {
                lb.cmp(rb)
            }
            (Interrupt::Jump { before, .. }, Interrupt::Converge { after }) => before.cmp(after),
            (Interrupt::Land { after, .. }, Interrupt::Jump { before, .. }) => after.cmp(before),
            (
                Interrupt::Land {
                    after: la,
                    source: ls,
                },
                Interrupt::Land {
                    after: ra,
                    source: rs,
                },
            ) => (*la, *ls).cmp(&(*ra, *rs)),
            (Interrupt::Land { after, .. }, Interrupt::Diverge { before, .. })
            | (Interrupt::Converge { after }, Interrupt::Jump { before, .. }) => after.cmp(before),
            (Interrupt::Land { after: la, .. }, Interrupt::Converge { after: ra })
            | (Interrupt::Converge { after: la }, Interrupt::Land { after: ra, .. })
            | (Interrupt::Converge { after: la }, Interrupt::Converge { after: ra }) => la.cmp(ra),
            (Interrupt::Diverge { before, .. }, Interrupt::Land { after, .. }) => before.cmp(after),
            (Interrupt::Diverge { before, .. }, Interrupt::Converge { after }) => {
                (*before, 0).cmp(&(*after, 1))
            }
            (Interrupt::Converge { after }, Interrupt::Diverge { before, .. }) => {
                (*after, 1).cmp(&(*before, 0))
            }
        });
        criteria.sort_by(|a, b| (a.start, a.end, a.owner).cmp(&(b.start, b.end, b.owner)));
        Ok(ControlFlow {
            interrupts: interrupts.into_boxed_slice(),
            criteria: criteria.into_boxed_slice(),
        })
    }
}

fn make_skip(
    start: usize,
    end: usize,
    owner: usize,
    kind: PruneKind,
    skippable: &mut bool,
    interrupts: &mut Vec<Interrupt>,
    criteria: &mut Vec<Criteria>,
) {
    if !std::mem::replace(skippable, true) {
        interrupts.extend_from_slice(&[
            Interrupt::Jump {
                before: start,
                target: end,
            },
            Interrupt::Land {
                after: end,
                source: start,
            },
        ]);
    }
    criteria.push(Criteria {
        start,
        end,
        owner,
        kind,
    });
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
        assert_eq!(
            cfg,
            ControlFlow {
                interrupts: vec![
                    Interrupt::Jump {
                        before: 0,
                        target: 2,
                    },
                    Interrupt::Land {
                        after: 2,
                        source: 0,
                    },
                    Interrupt::Jump {
                        before: 3,
                        target: 5,
                    },
                    Interrupt::Land {
                        after: 5,
                        source: 3,
                    },
                    Interrupt::Diverge {
                        before: 6,
                        choice: Choice::Node(5,),
                    },
                    Interrupt::Diverge {
                        before: 6,
                        choice: Choice::Node(2,),
                    },
                    Interrupt::Converge { after: 6 },
                ]
                .into_boxed_slice(),
                criteria: vec![
                    Criteria {
                        start: 0,
                        end: 2,
                        owner: 6,
                        kind: PruneKind::Left,
                    },
                    Criteria {
                        start: 3,
                        end: 5,
                        owner: 6,
                        kind: PruneKind::Right,
                    },
                ]
                .into_boxed_slice(),
            }
        );
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
        assert_eq!(
            cfg,
            ControlFlow {
                interrupts: vec![
                    Interrupt::Jump {
                        before: 0,
                        target: 6,
                    },
                    Interrupt::Jump {
                        before: 0,
                        target: 2,
                    },
                    Interrupt::Land {
                        after: 2,
                        source: 0,
                    },
                    Interrupt::Jump {
                        before: 3,
                        target: 5,
                    },
                    Interrupt::Land {
                        after: 5,
                        source: 3,
                    },
                    Interrupt::Diverge {
                        before: 6,
                        choice: Choice::Node(5,),
                    },
                    Interrupt::Diverge {
                        before: 6,
                        choice: Choice::Node(2,),
                    },
                    Interrupt::Converge { after: 6 },
                    Interrupt::Land {
                        after: 6,
                        source: 0,
                    },
                    Interrupt::Jump {
                        before: 7,
                        target: 9,
                    },
                    Interrupt::Land {
                        after: 9,
                        source: 7,
                    },
                    Interrupt::Diverge {
                        before: 10,
                        choice: Choice::Node(9,),
                    },
                    Interrupt::Diverge {
                        before: 10,
                        choice: Choice::Node(6,),
                    },
                    Interrupt::Converge { after: 10 },
                ]
                .into_boxed_slice(),
                criteria: vec![
                    Criteria {
                        start: 0,
                        end: 2,
                        owner: 6,
                        kind: PruneKind::Left,
                    },
                    Criteria {
                        start: 0,
                        end: 6,
                        owner: 10,
                        kind: PruneKind::Left,
                    },
                    Criteria {
                        start: 3,
                        end: 5,
                        owner: 6,
                        kind: PruneKind::Right,
                    },
                    Criteria {
                        start: 7,
                        end: 9,
                        owner: 10,
                        kind: PruneKind::Right,
                    },
                ]
                .into_boxed_slice(),
            }
        );
    }
}
