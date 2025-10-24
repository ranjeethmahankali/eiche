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

/**
Compute the results of operations on constants and fold those into constant
nodes. The unused nodes after folding are not pruned. Use a pruner for
that. Returns true if the nodes were modified, false otherwise.
*/
pub fn fold(nodes: &mut [Node]) -> Result<bool, Error> {
    let mut modified = false;
    for index in 0..nodes.len() {
        let folded = match nodes[index] {
            Constant(_) => None,
            Symbol(_) => None,
            Unary(op, input) => match (op, &nodes[input]) {
                (_, Constant(value)) => Some(Constant(Value::unary_op(op, *value)?)),
                (Negate, Binary(Subtract, li, ri)) => Some(Binary(Subtract, *ri, *li)),
                (Sqrt, Binary(Multiply, li, ri)) if li == ri => Some(Unary(Abs, *li)),
                (Negate, Unary(Negate, inner)) // Chains of ops that cancel out.
                    | (Log, Unary(Exp, inner))
                    | (Exp, Unary(Log, inner))
                    | (Not, Unary(Not, inner)) => Some(nodes[*inner]),
                (Abs, Unary(Abs, _)) => Some(nodes[input]),
                _ => None,
            },
            Binary(op, li, ri) => match (op, &nodes[li], &nodes[ri]) {
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
                (Max | Min, lhs, _) if li == ri => Some(*lhs),
                (Max, _, Unary(Negate, inner)) if *inner == li => Some(Unary(Abs, li)),
                (Max, Unary(Negate, inner), _) if *inner == ri => Some(Unary(Abs, ri)),
                _ => None,
            },
            Ternary(op, a, b, c) => match (op, &nodes[a], &nodes[b], &nodes[c]) {
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
                (Choose, _, _, _) if b == c => Some(nodes[b]),
                _ => None,
            },
        };
        if let Some(node) = folded {
            nodes[index] = node;
            modified = true;
        }
    }
    Ok(modified)
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
    use crate::{Deduplicater, deftree, prune::Pruner, test::compare_trees};

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
             (+ 'x (* 2. 3.))
             (log (+ 'x (/ 2. (min 5. (max 3. (- 9. 5.)))))))
        )
        .unwrap();
        let expected = deftree!(/ (+ 'x 6.) (log (+ 'x 0.5))).unwrap();
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
                 (+ 'x (* 2. 3.))
                 (log (+ 'x (/ 2. (min 5. (max 3. (- 9. 5.)))))))
                (/
                 (+ 'x (* 3. 3.))
                 (log (+ 'x (/ 8. (min 5. (max 3. (- 9. 5.)))))))
        )
        .unwrap();
        let expected = deftree!(
            concat
                (/ (+ 'x 6.) (log (+ 'x 0.5)))
                (/ (+ 'x 9.) (log (+ 'x 2.)))
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
        assert!(
            deftree!(+ (pow 'x (+ 'y 0)) 0)
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(pow 'x 'y).unwrap())
        );
        assert!(
            deftree!(pow (+ (+ (cos (+ 'x 0)) (/ 1 (+ (sin 'y) 0))) 0) (* 2 (+ (+ 'x 0) 'y)))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(pow (+ (cos 'x) (/ 1 (sin 'y))) (* 2 (+ 'x 'y))).unwrap())
        );
    }

    #[test]
    fn t_sub_zero() {
        let mut pruner = Pruner::new();
        assert!(
            deftree!(- (pow (- 'x 0) (+ 'y 0)) 0)
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(pow 'x 'y).unwrap()),
        );
        assert!(
            deftree!(pow (+ (cos (+ (- 'x 0) 0)) (/ 1 (- (sin (- 'y 0)) 0))) (* 2 (+ 'x (- 'y 0))))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(pow (+ (cos 'x) (/ 1 (sin 'y))) (* 2 (+ 'x 'y))).unwrap()),
        );
    }

    #[test]
    fn t_mul_1() {
        let mut pruner = Pruner::new();
        assert!(
            deftree!(+ (pow (* 1 'x) (* 'y 1)) 0)
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(pow 'x 'y).unwrap()),
        );
        assert!(
            deftree!(pow (+ (cos (* 'x (* 1 (+ 1 (* 0 'x)))))
                          (/ 1 (* (sin (- 'y 0)) 1))) (* (* (+ 2 0) (+ 'x 'y)) 1))
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner)
            .unwrap()
            .equivalent(&deftree!(pow (+ (cos 'x) (/ 1 (sin 'y))) (* 2 (+ 'x 'y))).unwrap()),
        );
    }

    #[test]
    fn t_pow_1() {
        let mut pruner = Pruner::new();
        assert!(
            deftree!(pow (pow (pow 'x 1) (pow 'y 1)) (pow 1 1))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(pow 'x 'y).unwrap())
        );
        assert!(
            deftree!(pow (+ (cos (pow 'x (pow 'x (* 0 'x)))) (/ 1 (sin 'y))) (* 2 (+ 'x (pow 'y 1))))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(pow (+ (cos 'x) (/ 1 (sin 'y))) (* 2 (+ 'x 'y))).unwrap()),
        );
    }

    #[test]
    fn t_div_1() {
        let mut pruner = Pruner::new();
        assert!(
            deftree!(pow (/ 'x 1) (/ 'y (pow 'x (* 't 0))))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(pow 'x 'y).unwrap()),
        );

        assert!(
            deftree!(pow (+ (cos (/ 'x 1)) (/ 1 (sin (/ 'y (pow 't (* 0 'p)))))) (* 2 (+ 'x 'y)))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(pow (+ (cos 'x) (/ 1 (sin 'y))) (* 2 (+ 'x 'y))).unwrap()),
        );
    }

    #[test]
    fn t_mul_0() {
        let mut pruner = Pruner::new();
        assert!(
            deftree!(pow (+ 'x (* 't 0)) (+ 'y (* 't 0)))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(pow 'x 'y).unwrap()),
        );
        assert!(
            deftree!(pow (+ (cos (+ 'x (* 0 't))) (/ 1 (sin (- 'y (* 't 0)))))
                     (* 2 (* (+ 'x 'y) (pow 't (* 0 't)))))
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner)
            .unwrap()
            .equivalent(&deftree!(pow (+ (cos 'x) (/ 1 (sin 'y))) (* 2 (+ 'x 'y))).unwrap()),
        );
    }

    #[test]
    fn t_pow_0() {
        let mut pruner = Pruner::new();
        assert!(
            deftree!(* (pow 'x (* 't 0)) (pow (* 'x (pow 't 0)) 'y))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(pow 'x 'y).unwrap()),
        );
        assert!(
            deftree!(pow (+ (cos (* 'x (pow 't 0))) (/ 1 (sin (* 'y (pow 't (* 'x 0))))))
                     (* 2 (+ 'x (* 'y (pow 'x 0)))))
            .unwrap()
            .fold()
            .unwrap()
            .prune(&mut pruner)
            .unwrap()
            .equivalent(&deftree!(pow (+ (cos 'x) (/ 1 (sin 'y))) (* 2 (+ 'x 'y))).unwrap()),
        );
    }

    #[test]
    fn t_boolean() {
        let mut pruner = Pruner::new();
        assert!(
            deftree!(if (> 2 0) 'x (- 'x))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!('x).unwrap()),
        );
        assert!(
            deftree!(if (not (> 2 0)) 'x (- 'x))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(- 'x).unwrap()),
        );
        assert!(
            deftree!(if (and (> 'x 0) (> 2 0)) 'x (- 'x))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(if (> 'x 0) 'x (- 'x)).unwrap()),
        );
        assert!(
            deftree!(if (and (> 'x 0) (> 0 2)) 'x (- 'x))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(- 'x).unwrap()),
        );
        assert!(
            deftree!(if (or (> 'x 0) (> 2 0)) 'x (- 'x))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!('x).unwrap()),
        );
        assert!(
            deftree!(if (or (> 'x 0) (> 0 2)) 'x (- 'x))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(if (> 'x 0) 'x (- 'x)).unwrap()),
        );
    }

    #[test]
    fn t_double_negation() {
        let mut pruner = Pruner::new();
        // Simple case: -(-x) = x
        assert!(
            deftree!(- (- 'x))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!('x).unwrap()),
        );
        // Nested case: -(-(x + y)) = x + y
        assert!(
            deftree!(- (- (+ 'x 'y)))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(+ 'x 'y).unwrap()),
        );
        // Multiple double negations: -(-(x)) + -(-(y)) = x + y
        assert!(
            deftree!(+ (- (- 'x)) (- (- 'y)))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(+ 'x 'y).unwrap()),
        );
        // Complex expression with double negation
        assert!(
            deftree!(pow (+ (- (- (cos 'x))) (/ 1 (sin (- (- 'y))))) (* 2 (+ 'x 'y)))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(pow (+ (cos 'x) (/ 1 (sin 'y))) (* 2 (+ 'x 'y))).unwrap()),
        );
    }

    #[test]
    fn t_negate_subtract() {
        let mut pruner = Pruner::new();
        // Simple case: -(a - b) = b - a
        assert!(
            deftree!(- (- 'a 'b))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(- 'b 'a).unwrap()),
        );
        // With expressions: -(x + y - z) = z - (x + y)
        assert!(
            deftree!(- (- (+ 'x 'y) 'z))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(- 'z (+ 'x 'y)).unwrap()),
        );
        // Verify numerical equivalence with compare_trees
        let tree = deftree!(- (- 'x 'y)).unwrap();
        let expected = deftree!(- 'y 'x).unwrap();
        let folded = tree.fold().unwrap().prune(&mut pruner).unwrap();
        assert!(folded.equivalent(&expected));
        compare_trees(
            &folded,
            &expected,
            &[('x', -10., 10.), ('y', -10., 10.)],
            100,
            0.,
        );
    }

    #[test]
    fn t_combined_negate_rules() {
        let mut pruner = Pruner::new();
        assert!(
            deftree!(- (- (- (- 'a 'b))))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(- 'b 'a).unwrap()),
        );
        // Complex expression: -(-(x - y)) + -(z - w) = (x - y) + (w - z)
        let tree = deftree!(+ (- (- (- 'x 'y))) (- (- 'z 'w))).unwrap();
        let expected = deftree!(+ (- 'x 'y) (- 'w 'z))
            .unwrap()
            .prune(&mut pruner)
            .unwrap();
        let folded = tree.fold().unwrap().prune(&mut pruner).unwrap();
        assert!(folded.equivalent(&expected));
        compare_trees(
            &folded,
            &expected,
            &[
                ('x', -5., 5.),
                ('y', -5., 5.),
                ('z', -5., 5.),
                ('w', -5., 5.),
            ],
            10,
            0.,
        );
    }

    #[test]
    fn t_sqrt_square() {
        let mut deduper = Deduplicater::new();
        // Simple case: sqrt(x * x) = x
        assert!(
            deftree!(sqrt (* 'x 'x))
                .unwrap()
                .deduplicate(&mut deduper)
                .unwrap()
                .fold()
                .unwrap()
                .equivalent(&deftree!(abs 'x).unwrap()),
        );
        // Nested expression: sqrt((x + y) * (x + y)) = x + y
        assert!(
            deftree!(sqrt (* (+ 'x 'y) (+ 'x 'y)))
                .unwrap()
                .deduplicate(&mut deduper)
                .unwrap()
                .fold()
                .unwrap()
                .equivalent(&deftree!(abs (+ 'x 'y)).unwrap()),
        );
        // Complex nested case: sqrt(cos(x) * cos(x)) = cos(x)
        assert!(
            deftree!(sqrt (* (cos 'x) (cos 'x)))
                .unwrap()
                .deduplicate(&mut deduper)
                .unwrap()
                .fold()
                .unwrap()
                .equivalent(&deftree!(abs (cos 'x)).unwrap()),
        );
        // Combined with other folding: sqrt((x * 1) * (x * 1)) = x
        assert!(
            deftree!(sqrt (* (* 'x 1) (* 'x 1)))
                .unwrap()
                .deduplicate(&mut deduper)
                .unwrap()
                .fold()
                .unwrap()
                .equivalent(&deftree!(abs 'x).unwrap()),
        );
        // Verify numerically
        let tree = deftree!(+ (sqrt (* 'x 'x)) (sqrt (* 'y 'y)))
            .unwrap()
            .deduplicate(&mut deduper)
            .unwrap();
        let expected = deftree!(+ (abs 'x) (abs 'y)).unwrap();
        let folded = tree.fold().unwrap();
        assert!(folded.equivalent(&expected));
        compare_trees(
            &folded,
            &expected,
            &[('x', 0.1, 10.), ('y', 0.1, 10.)],
            10,
            0.,
        );
    }

    #[test]
    fn t_log_exp_cancellation() {
        let mut deduper = Deduplicater::new();
        // Simple case: log(exp(x)) = x
        assert!(
            deftree!(log (exp 'x))
                .unwrap()
                .deduplicate(&mut deduper)
                .unwrap()
                .fold()
                .unwrap()
                .equivalent(&deftree!('x).unwrap()),
        );
        // Nested case: log(exp(x + y)) = x + y
        assert!(
            deftree!(log (exp (+ 'x 'y)))
                .unwrap()
                .deduplicate(&mut deduper)
                .unwrap()
                .fold()
                .unwrap()
                .equivalent(&deftree!(+ 'x 'y).unwrap()),
        );
        // Multiple cancellations: log(exp(x)) + log(exp(y)) = x + y
        assert!(
            deftree!(+ (log (exp 'x)) (log (exp 'y)))
                .unwrap()
                .deduplicate(&mut deduper)
                .unwrap()
                .fold()
                .unwrap()
                .equivalent(&deftree!(+ 'x 'y).unwrap()),
        );
        // Complex expression with log(exp(...))
        assert!(
            deftree!(pow (+ (log (exp (cos 'x))) (/ 1 (sin (log (exp 'y))))) (* 2 (+ 'x 'y)))
                .unwrap()
                .deduplicate(&mut deduper)
                .unwrap()
                .fold()
                .unwrap()
                .equivalent(&deftree!(pow (+ (cos 'x) (/ 1 (sin 'y))) (* 2 (+ 'x 'y))).unwrap()),
        );
    }

    #[test]
    fn t_exp_log_cancellation() {
        let mut deduper = Deduplicater::new();
        // Simple case: exp(log(x)) = x
        assert!(
            deftree!(exp (log 'x))
                .unwrap()
                .deduplicate(&mut deduper)
                .unwrap()
                .fold()
                .unwrap()
                .equivalent(&deftree!('x).unwrap()),
        );
        // Nested case: exp(log(x + y)) = x + y
        assert!(
            deftree!(exp (log (+ 'x 'y)))
                .unwrap()
                .deduplicate(&mut deduper)
                .unwrap()
                .fold()
                .unwrap()
                .equivalent(&deftree!(+ 'x 'y).unwrap()),
        );
        // Multiple cancellations: exp(log(x)) + exp(log(y)) = x + y
        assert!(
            deftree!(+ (exp (log 'x)) (exp (log 'y)))
                .unwrap()
                .deduplicate(&mut deduper)
                .unwrap()
                .fold()
                .unwrap()
                .equivalent(&deftree!(+ 'x 'y).unwrap()),
        );
        // Verify numerical equivalence with compare_trees
        let tree = deftree!(* (exp (log 'x)) (exp (log 'y)))
            .unwrap()
            .deduplicate(&mut deduper)
            .unwrap();
        let expected = deftree!(* 'x 'y).unwrap();
        let folded = tree.fold().unwrap();
        assert!(folded.equivalent(&expected));
        compare_trees(
            &folded,
            &expected,
            &[('x', 0.1, 10.), ('y', 0.1, 10.)],
            100,
            0.,
        );
    }

    #[test]
    fn t_not_not_cancellation() {
        let mut deduper = Deduplicater::new();
        // Simple case: not(not(x)) = x
        assert!(
            deftree!(not (not (> 'x 0)))
                .unwrap()
                .deduplicate(&mut deduper)
                .unwrap()
                .fold()
                .unwrap()
                .equivalent(&deftree!(> 'x 0).unwrap()),
        );
        // Multiple cancellations: not(not(x > 0)) and not(not(y > 0)) = (x > 0) and (y > 0)
        assert!(
            deftree!(and (not (not (> 'x 0))) (not (not (> 'y 0))))
                .unwrap()
                .deduplicate(&mut deduper)
                .unwrap()
                .fold()
                .unwrap()
                .equivalent(&deftree!(and (> 'x 0) (> 'y 0)).unwrap()),
        );
        // Inside conditional: if not(not(x > y)) then x else y
        assert!(
            deftree!(if (not (not (> 'x 'y))) 'x 'y)
                .unwrap()
                .deduplicate(&mut deduper)
                .unwrap()
                .fold()
                .unwrap()
                .equivalent(&deftree!(if (> 'x 'y) 'x 'y).unwrap()),
        );
        // Complex expression with not(not(...))
        assert!(
            deftree!(if (and (not (not (> 'x 0))) (> 'y 0)) (+ 'x 'y) (- 'x 'y))
                .unwrap()
                .deduplicate(&mut deduper)
                .unwrap()
                .fold()
                .unwrap()
                .equivalent(&deftree!(if (and (> 'x 0) (> 'y 0)) (+ 'x 'y) (- 'x 'y)).unwrap()),
        );
    }

    #[test]
    fn t_combined_cancellations() {
        let mut deduper = Deduplicater::new();
        // Mix log/exp and not/not: log(exp(x)) + (if not(not(x > 0)) then y else 0)
        assert!(
            deftree!(+ (log (exp 'x)) (if (not (not (> 'x 0))) 'y 0))
                .unwrap()
                .deduplicate(&mut deduper)
                .unwrap()
                .fold()
                .unwrap()
                .equivalent(&deftree!(+ 'x (if (> 'x 0) 'y 0)).unwrap()),
        );
        // Mix exp/log with negation: -(-exp(log(x))) = exp(log(x)) = x
        assert!(
            deftree!(- (- (exp (log 'x))))
                .unwrap()
                .deduplicate(&mut deduper)
                .unwrap()
                .fold()
                .unwrap()
                .equivalent(&deftree!('x).unwrap()),
        );
        // All three types: log(exp(x)) + -(-y) + (if not(not(z > 0)) then z else 0)
        let tree = deftree!(+ (+ (log (exp 'x)) (- (- 'y))) (if (not (not (> 'z 0))) 'z 0))
            .unwrap()
            .deduplicate(&mut deduper)
            .unwrap();
        let expected = deftree!(+ (+ 'x 'y) (if (> 'z 0) 'z 0)).unwrap();
        let folded = tree.fold().unwrap();
        assert!(folded.equivalent(&expected));
        compare_trees(
            &folded,
            &expected,
            &[('x', 0.1, 10.), ('y', 0.1, 10.), ('z', -5., 5.)],
            100,
            0.,
        );
    }

    #[test]
    fn t_abs_abs_simplification() {
        let mut pruner = Pruner::new();
        // Simple case: abs(abs(x)) = abs(x), NOT x
        assert!(
            deftree!(abs (abs 'x))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(abs 'x).unwrap()),
        );
        // Verify this is NOT equivalent to just x
        assert!(
            !deftree!(abs (abs 'x))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!('x).unwrap()),
        );
        // Numerical verification: abs(abs(x)) should equal abs(x) for negative values
        let tree = deftree!(abs (abs 'x)).unwrap();
        let expected = deftree!(abs 'x).unwrap();
        let folded = tree.fold().unwrap().prune(&mut pruner).unwrap();
        assert!(folded.equivalent(&expected));
        compare_trees(&folded, &expected, &[('x', -10., 10.)], 100, 0.);
        // More complex: abs(abs(x + y)) = abs(x + y)
        assert!(
            deftree!(abs (abs (+ 'x 'y)))
                .unwrap()
                .fold()
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
                .equivalent(&deftree!(abs (+ 'x 'y)).unwrap()),
        );
        // Multiple: abs(abs(x)) + abs(abs(y)) = abs(x) + abs(y)
        let tree = deftree!(+ (abs (abs 'x)) (abs (abs 'y))).unwrap();
        let expected = deftree!(+ (abs 'x) (abs 'y)).unwrap();
        let folded = tree.fold().unwrap().prune(&mut pruner).unwrap();
        assert!(folded.equivalent(&expected));
        compare_trees(
            &folded,
            &expected,
            &[('x', -10., 10.), ('y', -10., 10.)],
            100,
            0.,
        );
    }

    #[test]
    fn t_max_min_identical_args() {
        let mut pruner = Pruner::new();
        let mut deduper = Deduplicater::new();
        // Simple case: max(x, x) = x
        let tree = deftree!(max 'x 'x)
            .unwrap()
            .deduplicate(&mut deduper)
            .unwrap();
        let expected = deftree!('x).unwrap();
        let folded = tree.fold().unwrap().prune(&mut pruner).unwrap();
        assert!(folded.equivalent(&expected));
        // Simple case: min(x, x) = x
        let tree = deftree!(min 'x 'x)
            .unwrap()
            .deduplicate(&mut deduper)
            .unwrap();
        let expected = deftree!('x).unwrap();
        let folded = tree.fold().unwrap().prune(&mut pruner).unwrap();
        assert!(folded.equivalent(&expected));
        // Expression: max(x + y, x + y) = x + y
        let tree = deftree!(max (+ 'x 'y) (+ 'x 'y))
            .unwrap()
            .deduplicate(&mut deduper)
            .unwrap();
        let expected = deftree!(+ 'x 'y).unwrap();
        let folded = tree.fold().unwrap().prune(&mut pruner).unwrap();
        assert!(folded.equivalent(&expected));
        // Expression: min(cos(x), cos(x)) = cos(x)
        let tree = deftree!(min (cos 'x) (cos 'x))
            .unwrap()
            .deduplicate(&mut deduper)
            .unwrap();
        let expected = deftree!(cos 'x).unwrap();
        let folded = tree.fold().unwrap().prune(&mut pruner).unwrap();
        assert!(folded.equivalent(&expected));
    }

    #[test]
    fn t_max_negate_to_abs() {
        let mut pruner = Pruner::new();
        let mut deduper = Deduplicater::new();
        // Simple case: max(x, -x) = abs(x)
        let tree = deftree!(max 'x (- 'x))
            .unwrap()
            .deduplicate(&mut deduper)
            .unwrap();
        let expected = deftree!(abs 'x).unwrap();
        let folded = tree.fold().unwrap().prune(&mut pruner).unwrap();
        assert!(folded.equivalent(&expected));
        // Reverse order: max(-x, x) = abs(x)
        let tree = deftree!(max (- 'x) 'x)
            .unwrap()
            .deduplicate(&mut deduper)
            .unwrap();
        let expected = deftree!(abs 'x).unwrap();
        let folded = tree.fold().unwrap().prune(&mut pruner).unwrap();
        assert!(folded.equivalent(&expected));
        // Expression: max(x + y, -(x + y)) = abs(x + y)
        let tree = deftree!(max (+ 'x 'y) (- (+ 'x 'y)))
            .unwrap()
            .deduplicate(&mut deduper)
            .unwrap();
        let expected = deftree!(abs (+ 'x 'y)).unwrap();
        let folded = tree.fold().unwrap().prune(&mut pruner).unwrap();
        assert!(folded.equivalent(&expected));
        // Complex: max(cos(x), -cos(x)) = abs(cos(x))
        let tree = deftree!(max (cos 'x) (- (cos 'x)))
            .unwrap()
            .deduplicate(&mut deduper)
            .unwrap();
        let expected = deftree!(abs (cos 'x)).unwrap();
        let folded = tree.fold().unwrap().prune(&mut pruner).unwrap();
        assert!(folded.equivalent(&expected));
        // Multiple: max(x, -x) + max(-y, y) = abs(x) + abs(y)
        let tree = deftree!(+ (max 'x (- 'x)) (max (- 'y) 'y))
            .unwrap()
            .deduplicate(&mut deduper)
            .unwrap();
        let expected = deftree!(+ (abs 'x) (abs 'y)).unwrap();
        let folded = tree.fold().unwrap().prune(&mut pruner).unwrap();
        assert!(folded.equivalent(&expected));
    }

    #[test]
    fn t_choose_identical_branches() {
        let mut pruner = Pruner::new();
        let mut deduper = Deduplicater::new();
        // Simple case: choose(cond, x, x) = x
        let tree = deftree!(if (> 'y 0) 'x 'x)
            .unwrap()
            .deduplicate(&mut deduper)
            .unwrap();
        let expected = deftree!('x).unwrap();
        let folded = tree.fold().unwrap().prune(&mut pruner).unwrap();
        assert!(folded.equivalent(&expected));
        // Expression: choose(x > 0, x + y, x + y) = x + y
        let tree = deftree!(if (> 'x 0) (+ 'x 'y) (+ 'x 'y))
            .unwrap()
            .deduplicate(&mut deduper)
            .unwrap();
        let expected = deftree!(+ 'x 'y).unwrap();
        let folded = tree.fold().unwrap().prune(&mut pruner).unwrap();
        assert!(folded.equivalent(&expected));
        // Complex: choose(x > y, cos(z), cos(z)) = cos(z)
        let tree = deftree!(if (> 'x 'y) (cos 'z) (cos 'z))
            .unwrap()
            .deduplicate(&mut deduper)
            .unwrap();
        let expected = deftree!(cos 'z).unwrap();
        let folded = tree.fold().unwrap().prune(&mut pruner).unwrap();
        assert!(folded.equivalent(&expected));
        // Nested: choose(a > 0, choose(b > 0, x, x), x) folds inner then outer
        let tree = deftree!(if (> 'a 0) (if (> 'b 0) 'x 'x) 'x)
            .unwrap()
            .compacted()
            .unwrap();
        let expected = deftree!('x).unwrap();
        let folded = tree.fold().unwrap().deduplicate(&mut deduper).unwrap();
        assert!(folded.equivalent(&expected));
    }
}
