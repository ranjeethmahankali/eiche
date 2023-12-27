use crate::tree::{BinaryOp::*, Node, Node::*, Tree, TreeError};

/// Compute the results of operations on constants and fold those into
/// constant nodes. The unused nodes after folding are not
/// pruned. Use a pruner for that.
pub fn fold_nodes(nodes: &mut Vec<Node>) {
    for index in 0..nodes.len() {
        let folded = match nodes[index] {
            ConstScalar(_) => None,
            Symbol(_) => None,
            Unary(op, input) => {
                if let ConstScalar(value) = nodes[input] {
                    Some(ConstScalar(op.apply(value)))
                } else {
                    None
                }
            }
            Binary(op, lhs, rhs) => match (op, &nodes[lhs], &nodes[rhs]) {
                // Constant folding.
                (op, ConstScalar(a), ConstScalar(b)) => Some(ConstScalar(op.apply(*a, *b))),
                // Identity ops.
                (Add, lhs, ConstScalar(val)) if *val == 0. => Some(*lhs),
                (Add, ConstScalar(val), rhs) if *val == 0. => Some(*rhs),
                (Subtract, lhs, ConstScalar(val)) if *val == 0. => Some(*lhs),
                (Multiply, lhs, ConstScalar(val)) if *val == 1. => Some(*lhs),
                (Multiply, ConstScalar(val), rhs) if *val == 1. => Some(*rhs),
                (Pow, base, ConstScalar(val)) if *val == 1. => Some(*base),
                (Divide, numerator, ConstScalar(val)) if *val == 1. => Some(*numerator),
                // Other ops.
                (Pow, _base, ConstScalar(val)) if *val == 0. => Some(ConstScalar(1.)),
                (Multiply, _lhs, ConstScalar(val)) if *val == 0. => Some(ConstScalar(0.)),
                (Multiply, ConstScalar(val), _rhs) if *val == 0. => Some(ConstScalar(0.)),
                _ => None,
            },
        };
        if let Some(node) = folded {
            nodes[index] = node;
        }
    }
}

impl Tree {
    /// Computes the results of constant operations, and folds them
    /// into the tree. Identity operations and other expressions whose
    /// values can be inferred without evaluating the tree are also
    /// folded. The resulting tree is pruned and checked for validity
    /// befoore it is returned. If the resulting tree is not valid,
    /// the appropriate `TreeError` is returned.
    pub fn fold(mut self) -> Result<Tree, TreeError> {
        fold_nodes(self.nodes_mut());
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
        let tree = deftree!(* 2. 3.).fold().unwrap().prune(&mut pruner);
        assert_eq!(tree.len(), 1usize);
        assert_eq!(tree.root(), &ConstScalar(2. * 3.));
    }

    #[test]
    fn t_constant_folding_1() {
        let tree = deftree!(
            (/
             (+ x (* 2. 3.))
             (log (+ x (/ 2. (min 5. (max 3. (- 9. 5.)))))))
        );
        let expected = deftree!(/ (+ x 6.) (log (+ x 0.5)));
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
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow x y)
        );
        assert_eq!(
            deftree!(pow (+ (+ (cos (+ x 0)) (/ 1 (+ (sin y) 0))) 0) (* 2 (+ (+ x 0) y)))
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y)))
        );
    }

    #[test]
    fn t_sub_zero() {
        let mut pruner = Pruner::new();
        assert_eq!(
            deftree!(- (pow (- x 0) (+ y 0)) 0)
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow x y)
        );
        assert_eq!(
            deftree!(pow (+ (cos (+ (- x 0) 0)) (/ 1 (- (sin (- y 0)) 0))) (* 2 (+ x (- y 0))))
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y)))
        );
    }

    #[test]
    fn t_mul_1() {
        let mut pruner = Pruner::new();
        assert_eq!(
            deftree!(+ (pow (* 1 x) (* y 1)) 0)
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow x y)
        );
        assert_eq!(
            deftree!(pow (+ (cos (* x (* 1 (+ 1 (* 0 x))))) (/ 1 (* (sin (- y 0)) 1))) (* (* (+ 2 0) (+ x y)) 1))
                .fold()
                .unwrap().prune(&mut pruner),
            deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y)))
        );
    }

    #[test]
    fn t_pow_1() {
        let mut pruner = Pruner::new();
        assert_eq!(
            deftree!(pow (pow (pow x 1) (pow y 1)) (pow 1 1))
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow x y)
        );
        assert_eq!(
            deftree!(pow (+ (cos (pow x (pow x (* 0 x)))) (/ 1 (sin y))) (* 2 (+ x (pow y 1))))
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y)))
        );
    }

    #[test]
    fn t_div_1() {
        let mut pruner = Pruner::new();
        assert_eq!(
            deftree!(pow (/ x 1) (/ y (pow x (* t 0))))
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow x y)
        );
        assert_eq!(
            deftree!(pow (+ (cos (/ x 1)) (/ 1 (sin (/ y (pow t (* 0 p)))))) (* 2 (+ x y)))
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y)))
        );
    }

    #[test]
    fn t_mul_0() {
        let mut pruner = Pruner::new();
        assert_eq!(
            deftree!(pow (+ x (* t 0)) (+ y (* t 0)))
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow x y)
        );
        assert_eq!(
            deftree!(pow (+ (cos (+ x (* 0 t))) (/ 1 (sin (- y (* t 0)))))
                     (* 2 (* (+ x y) (pow t (* 0 t)))))
            .fold()
            .unwrap()
            .prune(&mut pruner),
            deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y)))
        );
    }

    #[test]
    fn t_pow_0() {
        let mut pruner = Pruner::new();
        assert_eq!(
            deftree!(* (pow x (* t 0)) (pow (* x (pow t 0)) y))
                .fold()
                .unwrap()
                .prune(&mut pruner),
            deftree!(pow x y)
        );
        assert_eq!(
            deftree!(pow (+ (cos (* x (pow t 0))) (/ 1 (sin (* y (pow t (* x 0))))))
                     (* 2 (+ x (* y (pow x 0)))))
            .fold()
            .unwrap()
            .prune(&mut pruner),
            deftree!(pow (+ (cos x) (/ 1 (sin y))) (* 2 (+ x y)))
        );
    }
}
