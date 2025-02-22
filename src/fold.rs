use crate::{
    error::Error,
    eval::ValueType,
    tree::{
        BinaryOp::*,
        Node::{self, *},
        TernaryOp::*,
        Tree,
        UnaryOp::*,
        Value::{self, *},
    },
};

enum FoldResult {
    NoFolding,
    Folded(Node),
    Failure(Error),
}

fn fold_node(node: Node, nodes: &[Node]) -> FoldResult {
    match node {
        Constant(_) | Symbol(_) => FoldResult::NoFolding,
        Unary(op, input) => match nodes[input] {
            Constant(value) => match Value::unary_op(op, value) {
                Ok(result) => FoldResult::Folded(Constant(result)),
                Err(e) => FoldResult::Failure(e),
            },
            _ => FoldResult::NoFolding,
        },
        Binary(op, li, ri) => match (op, &nodes[li], &nodes[ri]) {
            // Constant folding.
            (op, Constant(a), Constant(b)) => match Value::binary_op(op, *a, *b) {
                Ok(result) => FoldResult::Folded(Constant(result)),
                Err(e) => FoldResult::Failure(e),
            },
            // Identity ops.
            (Add, lhs, Constant(val)) if *val == 0. => FoldResult::Folded(*lhs),
            (Add, Constant(val), rhs) if *val == 0. => FoldResult::Folded(*rhs),
            (Subtract, lhs, Constant(val)) if *val == 0. => FoldResult::Folded(*lhs),
            (Multiply, lhs, Constant(val)) if *val == 1. => FoldResult::Folded(*lhs),
            (Multiply, Constant(val), rhs) if *val == 1. => FoldResult::Folded(*rhs),
            (Pow, base, Constant(val)) if *val == 1. => FoldResult::Folded(*base),
            (Divide, numerator, Constant(val)) if *val == 1. => FoldResult::Folded(*numerator),
            (Or, lhs, Constant(rhs)) if *rhs == false => FoldResult::Folded(*lhs),
            (Or, Constant(lhs), rhs) if *lhs == false => FoldResult::Folded(*rhs),
            (And, lhs, Constant(rhs)) if *rhs == true => FoldResult::Folded(*lhs),
            (And, Constant(lhs), rhs) if *lhs == true => FoldResult::Folded(*rhs),
            // Other ops.
            (Subtract, Constant(val), _rhs) if *val == 0. => FoldResult::Folded(Unary(Negate, ri)),
            (Pow, _base, Constant(val)) if *val == 0. => FoldResult::Folded(Constant(Scalar(1.))),
            (Multiply, _lhs, Constant(val)) if *val == 0. => {
                FoldResult::Folded(Constant(Scalar(0.)))
            }
            (Multiply, Constant(val), _rhs) if *val == 0. => {
                FoldResult::Folded(Constant(Scalar(0.)))
            }
            (Divide, Constant(val), _rhs) if *val == 0. => FoldResult::Folded(Constant(Scalar(0.))),
            (Or, _lhs, Constant(rhs)) if *rhs == true => FoldResult::Folded(Constant(Bool(true))),
            (Or, Constant(lhs), _rhs) if *lhs == true => FoldResult::Folded(Constant(Bool(true))),
            (And, _lhs, Constant(rhs)) if *rhs == false => {
                FoldResult::Folded(Constant(Bool(false)))
            }
            (And, Constant(lhs), _rhs) if *lhs == false => {
                FoldResult::Folded(Constant(Bool(false)))
            }
            _ => FoldResult::NoFolding,
        },
        Ternary(op, a, b, c) => match (op, &nodes[a], &nodes[b], &nodes[c]) {
            (op, Constant(a), Constant(b), Constant(c)) => {
                match Value::ternary_op(op, *a, *b, *c) {
                    Ok(result) => FoldResult::Folded(Constant(result)),
                    Err(e) => FoldResult::Failure(e),
                }
            }
            (Choose, Constant(flag), left, right) => match flag.boolean() {
                Ok(true) => FoldResult::Folded(*left),
                Ok(false) => FoldResult::Folded(*right),
                Err(e) => FoldResult::Failure(e),
            },
            _ => FoldResult::NoFolding,
        },
    }
}

/**
Compute the results of operations on constants and fold those into
constant nodes. The unused nodes after folding are not
pruned. Use a pruner for that.
*/
pub fn fold(nodes: &mut [Node]) -> Result<(), Error> {
    for index in 0..nodes.len() {
        match fold_node(nodes[index], nodes) {
            FoldResult::NoFolding => {}
            FoldResult::Folded(node) => nodes[index] = node,
            FoldResult::Failure(error) => return Err(error),
        }
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
    use crate::{deftree, prune::Pruner, test::compare_trees};

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
}
