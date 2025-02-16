use std::collections::BTreeMap;

use crate::{
    error::Error,
    eval::ValueType,
    inari_interval::Interval,
    tree::{
        BinaryOp::{self, *},
        Node::{self, *},
        TernaryOp::{self, *},
        Tree,
        UnaryOp::{self, *},
        Value::{self, *},
    },
};

fn fold_unary_op(op: UnaryOp, input: usize, nodes: &[Node]) -> Result<Option<Node>, Error> {
    Ok(if let Constant(value) = nodes[input] {
        Some(Constant(Value::unary_op(op, value)?))
    } else {
        None
    })
}

fn fold_binary_op(
    op: BinaryOp,
    li: usize,
    ri: usize,
    nodes: &[Node],
) -> Result<Option<Node>, Error> {
    Ok(match (op, &nodes[li], &nodes[ri]) {
        // Constant folding.
        (op, Constant(a), Constant(b)) => Some(Constant(Value::binary_op(op, *a, *b)?)),
        // Identity ops.
        (Add, lhs, Constant(val)) if *val == 0. => Some(*lhs),
        (Add, Constant(val), rhs) if *val == 0. => Some(*rhs),
        (Subtract, lhs, Constant(val)) if *val == 0. => Some(*lhs),
        (Multiply, lhs, Constant(val)) if *val == 1. => Some(*lhs),
        (Multiply, Constant(val), rhs) if *val == 1. => Some(*rhs),
        (Pow, base, Constant(val)) if *val == 1. => Some(*base),
        (Divide, numerator, Constant(val)) if *val == 1. => Some(*numerator),
        (Or, lhs, Constant(rhs)) if *rhs == false => Some(*lhs),
        (Or, Constant(lhs), rhs) if *lhs == false => Some(*rhs),
        (And, lhs, Constant(rhs)) if *rhs == true => Some(*lhs),
        (And, Constant(lhs), rhs) if *lhs == true => Some(*rhs),
        // Other ops.
        (Subtract, Constant(val), _rhs) if *val == 0. => Some(Unary(Negate, ri)),
        (Pow, _base, Constant(val)) if *val == 0. => Some(Constant(Scalar(1.))),
        (Multiply, _lhs, Constant(val)) if *val == 0. => Some(Constant(Scalar(0.))),
        (Multiply, Constant(val), _rhs) if *val == 0. => Some(Constant(Scalar(0.))),
        (Divide, Constant(val), _rhs) if *val == 0. => Some(Constant(Scalar(0.))),
        (Or, _lhs, Constant(rhs)) if *rhs == true => Some(Constant(Bool(true))),
        (Or, Constant(lhs), _rhs) if *lhs == true => Some(Constant(Bool(true))),
        (And, _lhs, Constant(rhs)) if *rhs == false => Some(Constant(Bool(false))),
        (And, Constant(lhs), _rhs) if *lhs == false => Some(Constant(Bool(false))),
        _ => None,
    })
}

fn fold_ternary_op(
    op: TernaryOp,
    a: usize,
    b: usize,
    c: usize,
    nodes: &[Node],
) -> Result<Option<Node>, Error> {
    Ok(match (op, &nodes[a], &nodes[b], &nodes[c]) {
        (op, Constant(a), Constant(b), Constant(c)) => {
            Some(Constant(Value::ternary_op(op, *a, *b, *c)?))
        }
        (Choose, Constant(flag), left, right) => {
            if flag.boolean()? {
                Some(*left)
            } else {
                Some(*right)
            }
        }
        _ => None,
    })
}

/**
Compute the results of operations on constants and fold those into
constant nodes. The unused nodes after folding are not
pruned. Use a pruner for that.
*/
pub(crate) fn fold(nodes: &mut [Node]) -> Result<(), Error> {
    for index in 0..nodes.len() {
        if let Some(node) = match nodes[index] {
            Constant(_) => None,
            Symbol(_) => None,
            Unary(op, input) => fold_unary_op(op, input, nodes)?,
            Binary(op, li, ri) => fold_binary_op(op, li, ri, nodes)?,
            Ternary(op, a, b, c) => fold_ternary_op(op, a, b, c, nodes)?,
        } {
            nodes[index] = node;
        }
    }
    Ok(())
}

pub(crate) fn fold_with_interval(
    nodes: &mut [Node],
    vars: &BTreeMap<char, Interval>,
) -> Result<(), Error> {
    // This is similar to the folding above, except we use the results of the
    // interval evaluation instead of hardcoding the rules.
    let mut values: Vec<Interval> = Vec::with_capacity(nodes.len());
    for index in 0..nodes.len() {
        let (folded, value) = match nodes[index] {
            Constant(val) => (None, Interval::from_value(val)?),
            Symbol(label) => (
                None,
                vars.get(&label)
                    .copied()
                    .ok_or(Error::VariableNotFound(label))?,
            ),
            Unary(op, input) => match fold_unary_op(op, input, nodes)? {
                Some(folded) => (
                    Some(folded),
                    match &folded {
                        Constant(value) => Interval::from_value(*value)?,
                        Symbol(label) => vars
                            .get(&label)
                            .copied()
                            .ok_or(Error::VariableNotFound(*label))?,
                        Unary(op, input) => Interval::unary_op(*op, values[*input])?,
                        Binary(op, lhs, rhs) => {
                            Interval::binary_op(*op, values[*lhs], values[*rhs])?
                        }
                        Ternary(op, a, b, c) => {
                            Interval::ternary_op(*op, values[*a], values[*b], values[*c])?
                        }
                    },
                ),
                None => {
                    let tofold = match (op, values[input]) {
                        (Negate, Interval::Scalar(ii)) => ii.is_singleton() && ii.inf() == 0.,
                        (Sqrt, Interval::Scalar(ii)) => {
                            ii.is_singleton() && (ii.inf() == 0. || ii.inf() == 1.)
                        }
                        (Abs, Interval::Scalar(ii)) => ii.inf() >= 0.,
                        (Sin, Interval::Scalar(ii)) => ii.is_singleton() && ii.inf() == 0.,
                        (Cos, Interval::Scalar(_))
                        | (Tan, Interval::Scalar(_))
                        | (Log, Interval::Scalar(_))
                        | (Exp, Interval::Scalar(_)) => false,
                        (Floor, Interval::Scalar(ii)) => {
                            ii.is_singleton() && ii.inf().fract() == 0.
                        }
                        (Not, Interval::Boolean(_, _)) => false, // This should be taken care of by folding.
                        (Negate, Interval::Boolean(_, _))
                        | (Sqrt, Interval::Boolean(_, _))
                        | (Abs, Interval::Boolean(_, _))
                        | (Sin, Interval::Boolean(_, _))
                        | (Cos, Interval::Boolean(_, _))
                        | (Tan, Interval::Boolean(_, _))
                        | (Log, Interval::Boolean(_, _))
                        | (Exp, Interval::Boolean(_, _))
                        | (Floor, Interval::Boolean(_, _))
                        | (Not, Interval::Scalar(_)) => return Err(Error::TypeMismatch),
                    };
                    if tofold {
                        (Some(nodes[input]), values[input])
                    } else {
                        (None, Interval::unary_op(op, values[input])?)
                    }
                }
            },
            Binary(op, li, ri) => match fold_binary_op(op, li, ri, nodes)? {
                Some(folded) => (
                    Some(folded),
                    match &folded {
                        Constant(value) => Interval::from_value(*value)?,
                        Symbol(label) => vars
                            .get(&label)
                            .copied()
                            .ok_or(Error::VariableNotFound(*label))?,
                        Unary(op, input) => Interval::unary_op(*op, values[*input])?,
                        Binary(op, lhs, rhs) => {
                            Interval::binary_op(*op, values[*lhs], values[*rhs])?
                        }
                        Ternary(op, a, b, c) => {
                            Interval::ternary_op(*op, values[*a], values[*b], values[*c])?
                        }
                    },
                ),
                None => {
                    enum Choice {
                        Left,
                        Right,
                        Custom(Node, Interval),
                        None,
                    }
                    let choice = match (op, values[li], values[ri]) {
                        (Add, Interval::Scalar(ileft), Interval::Scalar(iright)) => {
                            if ileft.is_singleton() && ileft.inf() == 0. {
                                Choice::Right
                            } else if iright.is_singleton() && iright.inf() == 0. {
                                Choice::Left
                            } else {
                                Choice::None
                            }
                        }
                        (Subtract, Interval::Scalar(ileft), Interval::Scalar(iright)) => {
                            if iright.is_singleton() && iright.inf() == 0. {
                                Choice::Left
                            } else {
                                Choice::None
                            }
                        }
                        (Multiply, Interval::Scalar(ileft), Interval::Scalar(iright)) => {
                            if ileft.is_singleton() && ileft.inf() == 1. {
                                Choice::Right
                            } else if iright.is_singleton() && iright.inf() == 1. {
                                Choice::Left
                            } else {
                                Choice::None
                            }
                        }
                        (Divide, Interval::Scalar(ileft), Interval::Scalar(iright)) => todo!(),
                        (Pow, Interval::Scalar(ileft), Interval::Scalar(iright)) => todo!(),
                        (Min, Interval::Scalar(ileft), Interval::Scalar(iright)) => todo!(),
                        (Max, Interval::Scalar(ileft), Interval::Scalar(iright)) => todo!(),
                        (Remainder, Interval::Scalar(ileft), Interval::Scalar(iright)) => todo!(),
                        (Less, Interval::Scalar(ileft), Interval::Scalar(iright)) => todo!(),
                        (LessOrEqual, Interval::Scalar(ileft), Interval::Scalar(iright)) => {
                            todo!()
                        }
                        (Equal, Interval::Scalar(ileft), Interval::Scalar(iright)) => todo!(),
                        (NotEqual, Interval::Scalar(ileft), Interval::Scalar(iright)) => todo!(),
                        (Greater, Interval::Scalar(ileft), Interval::Scalar(iright)) => todo!(),
                        (GreaterOrEqual, Interval::Scalar(ileft), Interval::Scalar(iright)) => {
                            todo!()
                        }
                        (And, Interval::Boolean(_, _), Interval::Boolean(_, _)) => todo!(),
                        (Or, Interval::Boolean(_, _), Interval::Boolean(_, _)) => todo!(),
                        _ => return Err(Error::TypeMismatch),
                    };
                    match choice {
                        Choice::Left => (Some(nodes[li]), values[li]),
                        Choice::Right => (Some(nodes[ri]), values[ri]),
                        Choice::Custom(node, value) => (Some(node), value),
                        Choice::None => (None, Interval::binary_op(op, values[li], values[ri])?),
                    }
                }
            },
            Ternary(op, a, b, c) => match fold_ternary_op(op, a, b, c, nodes)? {
                Some(folded) => (
                    Some(folded),
                    match &folded {
                        Constant(value) => Interval::from_value(*value)?,
                        Symbol(label) => vars
                            .get(&label)
                            .copied()
                            .ok_or(Error::VariableNotFound(*label))?,
                        Unary(op, input) => Interval::unary_op(*op, values[*input])?,
                        Binary(op, lhs, rhs) => {
                            Interval::binary_op(*op, values[*lhs], values[*rhs])?
                        }
                        Ternary(op, a, b, c) => {
                            Interval::ternary_op(*op, values[*a], values[*b], values[*c])?
                        }
                    },
                ),
                None => (
                    None,
                    Interval::ternary_op(op, values[a], values[b], values[c])?,
                ),
            },
        };
        if let Some(node) = folded {
            nodes[index] = node;
        }
        values.push(value);
    }
    Ok(())
}

impl Tree {
    /// Computes the results of constant operations, and folds them
    /// into the tree. Identity operations and other expressions whose
    /// values can be inferred without evaluating the tree are also
    /// folded. The resulting tree is pruned and checked for validity
    /// befoore it is returned. If the resulting tree is not valid,
    /// the appropriate `Error` is returned.
    pub fn fold(self) -> Result<Tree, Error> {
        let (mut nodes, dims) = self.take();
        fold(&mut nodes)?;
        Tree::from_nodes(nodes, dims)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{dedup::Deduplicater, deftree, prune::Pruner, test::compare_trees, tree::min};
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn t_sconstant_folding_0() {
        let mut pruner = Pruner::new();
        let tree = deftree!(* 2. 3.)
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner)
            .unwrap();
        assert_eq!(tree.len(), 1usize);
        assert_eq!(tree.roots(), &[Constant(Scalar(2. * 3.))]);
    }

    #[test]
    fn t_constant_folding_1() {
        let tree = deftree!(
            (/
             (+ x (* 2. 3.))
             (log (+ x (/ 2. (min 5. (max 3. (- 9. 5.)))))))
        )
        .unwrap();
        let expected = deftree!(/ (+ x 6.) (log (+ x 0.5))).unwrap();
        assert!(tree.len() > expected.len());
        let mut pruner = Pruner::new();
        let tree = tree.fold().unwrap().prune(&mut pruner).unwrap();
        assert!(tree.equivalent(&expected));
        compare_trees(&tree, &expected, &[('x', 0.1, 10.)], 100, 0.);
    }

    #[test]
    fn t_constant_folding_concat() {
        let tree = deftree!(
            concat
                (/
                 (+ x (* 2. 3.))
                 (log (+ x (/ 2. (min 5. (max 3. (- 9. 5.)))))))
                (/
                 (+ x (* 3. 3.))
                 (log (+ x (/ 8. (min 5. (max 3. (- 9. 5.)))))))
        )
        .unwrap();
        let expected = deftree!(
            concat
                (/ (+ x 6.) (log (+ x 0.5)))
                (/ (+ x 9.) (log (+ x 2.)))
        )
        .unwrap();
        assert!(tree.len() > expected.len());
        let mut pruner = Pruner::new();
        let tree = tree.fold().unwrap().prune(&mut pruner).unwrap();
        assert!(tree.equivalent(&expected));
        compare_trees(&tree, &expected, &[('x', 0.1, 10.)], 100, 0.);
    }

    #[test]
    fn t_add_zero() {
        let mut pruner = Pruner::new();
        assert!(deftree!(+ (pow x (+ y 0)) 0)
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner)
            .unwrap()
            .equivalent(&deftree!(pow x y).unwrap()));
        assert!(
            deftree!(pow (+ (+ (cos (+ x 0)) (/ 1 (+ (sin y) 0))) 0) (* 2 (+ (+ x 0) y)))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y))).unwrap())
        );
    }

    #[test]
    fn t_sub_zero() {
        let mut pruner = Pruner::new();
        assert!(deftree!(- (pow (- x 0) (+ y 0)) 0)
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner)
            .unwrap()
            .equivalent(&deftree!(pow x y).unwrap()),);
        assert!(
            deftree!(pow (+ (cos (+ (- x 0) 0)) (/ 1 (- (sin (- y 0)) 0))) (* 2 (+ x (- y 0))))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y))).unwrap()),
        );
    }

    #[test]
    fn t_mul_1() {
        let mut pruner = Pruner::new();
        assert!(deftree!(+ (pow (* 1 x) (* y 1)) 0)
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner)
            .unwrap()
            .equivalent(&deftree!(pow x y).unwrap()),);
        assert!(deftree!(pow (+ (cos (* x (* 1 (+ 1 (* 0 x)))))
                          (/ 1 (* (sin (- y 0)) 1))) (* (* (+ 2 0) (+ x y)) 1))
        .unwrap()
        .fold()
        .unwrap()
        .prune(&mut pruner)
        .unwrap()
        .equivalent(&deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y))).unwrap()),);
    }

    #[test]
    fn t_pow_1() {
        let mut pruner = Pruner::new();
        assert!(deftree!(pow (pow (pow x 1) (pow y 1)) (pow 1 1))
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner)
            .unwrap()
            .equivalent(&deftree!(pow x y).unwrap()));
        assert!(
            deftree!(pow (+ (cos (pow x (pow x (* 0 x)))) (/ 1 (sin y))) (* 2 (+ x (pow y 1))))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y))).unwrap()),
        );
    }

    #[test]
    fn t_div_1() {
        let mut pruner = Pruner::new();
        assert!(deftree!(pow (/ x 1) (/ y (pow x (* t 0))))
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner)
            .unwrap()
            .equivalent(&deftree!(pow x y).unwrap()),);

        assert!(
            deftree!(pow (+ (cos (/ x 1)) (/ 1 (sin (/ y (pow t (* 0 p)))))) (* 2 (+ x y)))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y))).unwrap()),
        );
    }

    #[test]
    fn t_mul_0() {
        let mut pruner = Pruner::new();
        assert!(deftree!(pow (+ x (* t 0)) (+ y (* t 0)))
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner)
            .unwrap()
            .equivalent(&deftree!(pow x y).unwrap()),);
        assert!(
            deftree!(pow (+ (cos (+ x (* 0 t))) (/ 1 (sin (- y (* t 0)))))
                     (* 2 (* (+ x y) (pow t (* 0 t)))))
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner)
            .unwrap()
            .equivalent(&deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y))).unwrap()),
        );
    }

    #[test]
    fn t_pow_0() {
        let mut pruner = Pruner::new();
        assert!(deftree!(* (pow x (* t 0)) (pow (* x (pow t 0)) y))
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner)
            .unwrap()
            .equivalent(&deftree!(pow x y).unwrap()),);
        assert!(
            deftree!(pow (+ (cos (* x (pow t 0))) (/ 1 (sin (* y (pow t (* x 0))))))
                     (* 2 (+ x (* y (pow x 0)))))
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner)
            .unwrap()
            .equivalent(&deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y))).unwrap()),
        );
    }

    #[test]
    fn t_boolean() {
        let mut pruner = Pruner::new();
        assert!(deftree!(if (> 2 0) x (- x))
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner)
            .unwrap()
            .equivalent(&deftree!(x).unwrap()),);
        assert!(deftree!(if (not (> 2 0)) x (- x))
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner)
            .unwrap()
            .equivalent(&deftree!(-x).unwrap()),);
        assert!(deftree!(if (and (> x 0) (> 2 0)) x (-x))
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner)
            .unwrap()
            .equivalent(&deftree!(if (> x 0) x (-x)).unwrap()),);
        assert!(deftree!(if (and (> x 0) (> 0 2)) x (-x))
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner)
            .unwrap()
            .equivalent(&deftree!(-x).unwrap()),);
        assert!(deftree!(if (or (> x 0) (> 2 0)) x (-x))
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner)
            .unwrap()
            .equivalent(&deftree!(x).unwrap()),);
        assert!(deftree!(if (or (> x 0) (> 0 2)) x (-x))
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner)
            .unwrap()
            .equivalent(&deftree!(if (> x 0) x (-x)).unwrap()),);
    }

    const RADIUS_RANGE: (f64, f64) = (0.2, 2.);
    const X_RANGE: (f64, f64) = (0., 100.);
    const Y_RANGE: (f64, f64) = (0., 100.);
    const Z_RANGE: (f64, f64) = (0., 100.);
    const N_SPHERES: usize = 5000;
    const N_QUERIES: usize = 5000;

    fn sample_range(range: (f64, f64), rng: &mut StdRng) -> f64 {
        use rand::Rng;
        range.0 + rng.random::<f64>() * (range.1 - range.0)
    }

    fn sphere_union() -> Tree {
        let mut rng = StdRng::seed_from_u64(42);
        let mut make_sphere = || -> Result<Tree, Error> {
            deftree!(- (sqrt (+ (+
                                 (pow (- x (const sample_range(X_RANGE, &mut rng))) 2)
                                 (pow (- y (const sample_range(Y_RANGE, &mut rng))) 2))
                              (pow (- z (const sample_range(Z_RANGE, &mut rng))) 2)))
                     (const sample_range(RADIUS_RANGE, &mut rng)))
        };
        let mut tree = make_sphere();
        for _ in 1..N_SPHERES {
            tree = min(tree, make_sphere());
        }
        let tree = tree.unwrap();
        assert_eq!(tree.dims(), (1, 1));
        tree
    }

    fn _t_prune() {
        let mut rng = StdRng::seed_from_u64(234);
        let queries: Vec<[f64; 3]> = (0..N_QUERIES)
            .map(|_| {
                [
                    sample_range(X_RANGE, &mut rng),
                    sample_range(Y_RANGE, &mut rng),
                    sample_range(Z_RANGE, &mut rng),
                ]
            })
            .collect();
        let tree = {
            let mut dedup = Deduplicater::new();
            let mut pruner = Pruner::new();
            sphere_union()
                .fold()
                .unwrap()
                .deduplicate(&mut dedup)
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
        };
    }
}
