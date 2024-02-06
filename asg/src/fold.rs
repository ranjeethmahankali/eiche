use crate::{
    error::Error,
    tree::{BinaryOp::*, MaybeTree, Node, Node::*, TernaryOp::*, Tree, Value::*},
};

/// Compute the results of operations on constants and fold those into
/// constant nodes. The unused nodes after folding are not
/// pruned. Use a pruner for that.
pub fn fold_nodes(nodes: &mut Vec<Node>) -> Result<(), Error> {
    for index in 0..nodes.len() {
        let folded = match nodes[index] {
            Constant(_) => None,
            Symbol(_) => None,
            Unary(op, input) => {
                if let Constant(value) = nodes[input] {
                    Some(Constant(op.apply(value)?))
                } else {
                    None
                }
            }
            Binary(op, lhs, rhs) => match (op, &nodes[lhs], &nodes[rhs]) {
                // Constant folding.
                (op, Constant(a), Constant(b)) => Some(Constant(op.apply(*a, *b)?)),
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
                (Pow, _base, Constant(val)) if *val == 0. => Some(Constant(Scalar(1.))),
                (Multiply, _lhs, Constant(val)) if *val == 0. => Some(Constant(Scalar(0.))),
                (Multiply, Constant(val), _rhs) if *val == 0. => Some(Constant(Scalar(0.))),
                (Divide, Constant(val), _rhs) if *val == 0. => Some(Constant(Scalar(0.))),
                (Or, _lhs, Constant(rhs)) if *rhs == true => Some(Constant(Bool(true))),
                (Or, Constant(lhs), _rhs) if *lhs == true => Some(Constant(Bool(true))),
                (And, _lhs, Constant(rhs)) if *rhs == false => Some(Constant(Bool(false))),
                (And, Constant(lhs), _rhs) if *lhs == false => Some(Constant(Bool(false))),
                _ => None,
            },
            Ternary(op, a, b, c) => match (op, &nodes[a], &nodes[b], &nodes[c]) {
                (Choose, Constant(flag), left, right) => {
                    if flag.boolean()? {
                        Some(*left)
                    } else {
                        Some(*right)
                    }
                }
                _ => None,
            },
        };
        if let Some(node) = folded {
            nodes[index] = node;
        }
    }
    return Ok(());
}

impl Tree {
    /// Computes the results of constant operations, and folds them
    /// into the tree. Identity operations and other expressions whose
    /// values can be inferred without evaluating the tree are also
    /// folded. The resulting tree is pruned and checked for validity
    /// befoore it is returned. If the resulting tree is not valid,
    /// the appropriate `Error` is returned.
    pub fn fold(mut self) -> MaybeTree {
        fold_nodes(self.nodes_mut())?;
        return self.validated();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{deftree, prune::Pruner, test::util::compare_trees};

    #[test]
    fn t_sconstant_folding_0() {
        let mut pruner = Pruner::new();
        let tree = deftree!(* 2. 3.)
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner);
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
        let tree = tree.fold().unwrap().prune(&mut pruner);
        assert_eq!(tree, expected);
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
        let tree = tree.fold().unwrap().prune(&mut pruner);
        assert_eq!(tree, expected);
        compare_trees(&tree, &expected, &[('x', 0.1, 10.)], 100, 0.);
    }

    #[test]
    fn t_add_zero() {
        let mut pruner = Pruner::new();
        assert_eq!(
            deftree!(+ (pow x (+ y 0)) 0)
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow x y).unwrap()
        );
        assert_eq!(
            deftree!(pow (+ (+ (cos (+ x 0)) (/ 1 (+ (sin y) 0))) 0) (* 2 (+ (+ x 0) y)))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y))).unwrap()
        );
    }

    #[test]
    fn t_sub_zero() {
        let mut pruner = Pruner::new();
        assert_eq!(
            deftree!(- (pow (- x 0) (+ y 0)) 0)
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow x y).unwrap()
        );
        assert_eq!(
            deftree!(pow (+ (cos (+ (- x 0) 0)) (/ 1 (- (sin (- y 0)) 0))) (* 2 (+ x (- y 0))))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y))).unwrap()
        );
    }

    #[test]
    fn t_mul_1() {
        let mut pruner = Pruner::new();
        assert_eq!(
            deftree!(+ (pow (* 1 x) (* y 1)) 0)
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow x y).unwrap()
        );
        assert_eq!(
            deftree!(pow (+ (cos (* x (* 1 (+ 1 (* 0 x))))) (/ 1 (* (sin (- y 0)) 1))) (* (* (+ 2 0) (+ x y)) 1)).unwrap()
                .fold()
                .unwrap().prune(&mut pruner),
            deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y))).unwrap()
        );
    }

    #[test]
    fn t_pow_1() {
        let mut pruner = Pruner::new();
        assert_eq!(
            deftree!(pow (pow (pow x 1) (pow y 1)) (pow 1 1))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow x y).unwrap()
        );
        assert_eq!(
            deftree!(pow (+ (cos (pow x (pow x (* 0 x)))) (/ 1 (sin y))) (* 2 (+ x (pow y 1))))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y))).unwrap()
        );
    }

    #[test]
    fn t_div_1() {
        let mut pruner = Pruner::new();
        assert_eq!(
            deftree!(pow (/ x 1) (/ y (pow x (* t 0))))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow x y).unwrap()
        );
        assert_eq!(
            deftree!(pow (+ (cos (/ x 1)) (/ 1 (sin (/ y (pow t (* 0 p)))))) (* 2 (+ x y)))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y))).unwrap()
        );
    }

    #[test]
    fn t_mul_0() {
        let mut pruner = Pruner::new();
        assert_eq!(
            deftree!(pow (+ x (* t 0)) (+ y (* t 0)))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow x y).unwrap()
        );
        assert_eq!(
            deftree!(pow (+ (cos (+ x (* 0 t))) (/ 1 (sin (- y (* t 0)))))
                     (* 2 (* (+ x y) (pow t (* 0 t)))))
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner),
            deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y))).unwrap()
        );
    }

    #[test]
    fn t_pow_0() {
        let mut pruner = Pruner::new();
        assert_eq!(
            deftree!(* (pow x (* t 0)) (pow (* x (pow t 0)) y))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow x y).unwrap()
        );
        assert_eq!(
            deftree!(pow (+ (cos (* x (pow t 0))) (/ 1 (sin (* y (pow t (* x 0))))))
                     (* 2 (+ x (* y (pow x 0)))))
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner),
            deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y))).unwrap()
        );
    }

    #[test]
    fn t_boolean() {
        let mut pruner = Pruner::new();
        assert_eq!(
            deftree!(if (> 2 0) x (- x))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(x).unwrap()
        );
        assert_eq!(
            deftree!(if (not (> 2 0)) x (- x))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(-x).unwrap()
        );
        assert_eq!(
            deftree!(if (and (> x 0) (> 2 0)) x (-x))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(if (> x 0) x (-x)).unwrap()
        );
        assert_eq!(
            deftree!(if (and (> x 0) (> 0 2)) x (-x))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(-x).unwrap()
        );
        assert_eq!(
            deftree!(if (or (> x 0) (> 2 0)) x (-x))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(x).unwrap()
        );
        assert_eq!(
            deftree!(if (or (> x 0) (> 0 2)) x (-x))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(if (> x 0) x (-x)).unwrap()
        );
    }
}
