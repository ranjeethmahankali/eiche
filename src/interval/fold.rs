use super::Interval;
use crate::{
    error::Error,
    eval::ValueType,
    fold::fold,
    tree::{
        BinaryOp::*,
        Node::{self, *},
        TernaryOp::*,
        Tree,
        UnaryOp::*,
        Value,
    },
};

/// Fold the given nodes by pruning for the given interval. The intervals are
/// assumed to be sorted by the variable labels, and each variable label must
/// appear at most once.
pub(crate) fn fold_for_interval(
    nodes: &[Node],
    interval: &[(char, Interval)],
    dst: &mut Vec<Node>,
    values: &mut Vec<Interval>,
) -> Result<(), Error> {
    values.clear();
    values.reserve(nodes.len());
    dst.clear();
    dst.reserve(nodes.len());
    for node in nodes {
        let (folded, value) = match node {
            Constant(value) => (None, Interval::from_value(*value)?),
            Symbol(label) => match interval.binary_search_by(|(var, _)| var.cmp(label)) {
                Ok(index) => (None, interval[index].1),
                Err(_) => (None, Interval::Scalar(inari::Interval::ENTIRE)),
            },
            Unary(op, input) => match (op, &values[*input], &nodes[*input]) {
                // Singleton intervals that don't come from constants can be replaced with constants.
                (_, Interval::Scalar(ii), Unary(_, _) | Binary(_, _, _) | Ternary(_, _, _, _))
                    if ii.is_singleton() =>
                {
                    let ni = dst.len();
                    dst.push(Constant(Value::Scalar(ii.inf())));
                    (
                        Some(Unary(*op, ni)),
                        Interval::unary_op(*op, Interval::Scalar(*ii))?,
                    )
                }
                (Not, Interval::Bool(true, true), _) => (
                    Some(Constant(Value::Bool(false))),
                    Interval::Bool(false, false),
                ),
                (Not, Interval::Bool(false, false), _) => (
                    Some(Constant(Value::Bool(true))),
                    Interval::Bool(true, true),
                ),
                // No folding, no optimization, default interval.
                _ => (None, Interval::unary_op(*op, values[*input])?),
            },
            Binary(op, li, ri) => match (op, &values[*li], &values[*ri]) {
                (And | Or, Interval::Scalar(_), Interval::Scalar(_)) => {
                    return Err(Error::TypeMismatch);
                }
                (op, Interval::Scalar(ileft), Interval::Scalar(iright)) => {
                    match (op, ileft.overlap(*iright)) {
                        // Choose left
                        (Min, inari::Overlap::Before | inari::Overlap::Meets)
                        | (Max, inari::Overlap::MetBy | inari::Overlap::After) => {
                            (Some(nodes[*li]), Interval::Scalar(*ileft))
                        }
                        // Choose right
                        (Min, inari::Overlap::MetBy | inari::Overlap::After)
                        | (Max, inari::Overlap::Before | inari::Overlap::Meets) => {
                            (Some(nodes[*ri]), Interval::Scalar(*iright))
                        }
                        // Always true
                        (Less, inari::Overlap::Before)
                        | (LessOrEqual, inari::Overlap::Before | inari::Overlap::Meets)
                        | (
                            NotEqual,
                            inari::Overlap::FirstEmpty
                            | inari::Overlap::SecondEmpty
                            | inari::Overlap::Before
                            | inari::Overlap::After,
                        )
                        | (Greater, inari::Overlap::After)
                        | (GreaterOrEqual, inari::Overlap::MetBy | inari::Overlap::After) => (
                            Some(Constant(Value::Bool(true))),
                            Interval::Bool(true, true),
                        ),
                        // Always false
                        (Less, inari::Overlap::After)
                        | (LessOrEqual, inari::Overlap::After)
                        | (Greater, inari::Overlap::Before)
                        | (GreaterOrEqual, inari::Overlap::Before) => (
                            Some(Constant(Value::Bool(false))),
                            Interval::Bool(false, false),
                        ),
                        // Finally handle singleton intervals.
                        _ => match (
                            ileft.is_singleton(),
                            iright.is_singleton(),
                            &nodes[*li],
                            &nodes[*ri],
                        ) {
                            // Fold constants.
                            (true, true, ..) => {
                                let val = Value::binary_op(
                                    *op,
                                    Value::Scalar(ileft.inf()),
                                    Value::Scalar(iright.inf()),
                                )?;
                                (Some(Constant(val)), Interval::from_value(val)?)
                            }
                            // Fold constant on left.
                            (true, false, Unary(..) | Binary(..) | Ternary(..), _) => {
                                let ni = dst.len();
                                dst.push(Constant(Value::Scalar(ileft.inf())));
                                (
                                    Some(Binary(*op, ni, *ri)),
                                    Interval::binary_op(
                                        *op,
                                        Interval::Scalar(*ileft),
                                        Interval::Scalar(*iright),
                                    )?,
                                )
                            }
                            // Fold constant on right.
                            (false, true, _, Unary(..) | Binary(..) | Ternary(..)) => {
                                let ni = dst.len();
                                dst.push(Constant(Value::Scalar(iright.inf())));
                                (
                                    Some(Binary(*op, *li, ni)),
                                    Interval::binary_op(
                                        *op,
                                        Interval::Scalar(*ileft),
                                        Interval::Scalar(*iright),
                                    )?,
                                )
                            }
                            // No folding, no optimization, default interval.
                            _ => (
                                None,
                                Interval::binary_op(
                                    *op,
                                    Interval::Scalar(*ileft),
                                    Interval::Scalar(*iright),
                                )?,
                            ),
                        },
                    }
                }
                (
                    Add | Subtract | Multiply | Divide | Pow | Min | Max | Remainder | Less
                    | LessOrEqual | Equal | NotEqual | Greater | GreaterOrEqual,
                    Interval::Bool(..),
                    Interval::Bool(..),
                )
                | (_, Interval::Scalar(_), Interval::Bool(_, _))
                | (_, Interval::Bool(_, _), Interval::Scalar(_)) => {
                    return Err(Error::TypeMismatch);
                }
                (And, Interval::Bool(true, true), Interval::Bool(true, true))
                | (Or, Interval::Bool(true, true), _)
                | (Or, _, Interval::Bool(true, true)) => (
                    Some(Constant(Value::Bool(true))),
                    Interval::Bool(true, true),
                ),
                (And, Interval::Bool(false, false), _)
                | (And, _, Interval::Bool(false, false))
                | (Or, Interval::Bool(false, false), Interval::Bool(false, false)) => (
                    Some(Constant(Value::Bool(false))),
                    Interval::Bool(false, false),
                ),
                // No folding, no optimization, default interval.
                (op, lhs, rhs) => (None, Interval::binary_op(*op, *lhs, *rhs)?),
            },
            Ternary(op, a, b, c) => match (op, &values[*a]) {
                (Choose, Interval::Bool(true, true)) => (Some(nodes[*b]), values[*b]),
                (Choose, Interval::Bool(false, false)) => (Some(nodes[*c]), values[*c]),
                (Choose, Interval::Bool(_, _)) => (
                    None,
                    Interval::ternary_op(*op, values[*a], values[*b], values[*c])?,
                ),
                (Choose, Interval::Scalar(_)) => return Err(Error::TypeMismatch),
            },
        };
        dst.push(folded.unwrap_or(*node));
        values.push(value);
    }
    fold(dst)?;
    Ok(())
}

impl Tree {
    /// Fold the tree by pruning for the given interval.The intervals are
    /// assumed to be sorted by the variable labels, and each variable label
    /// must appear at most once.
    pub fn folded_for_interval(&self, vars: &[(char, Interval)]) -> Result<Tree, Error> {
        let mut out = Vec::with_capacity(self.len());
        let mut values = Vec::new();
        fold_for_interval(self.nodes(), vars, &mut out, &mut values)?;
        Tree::from_nodes(out, self.dims())
    }
}

#[cfg(test)]
mod test {
    use crate::{deftree, error::Error, interval::Interval, tree::Tree};

    fn sphere(cx: f64, cy: f64, cz: f64, r: f64) -> Result<Tree, Error> {
        deftree!(- (sqrt (+
                          (pow (- x (const cx)) 2)
                          (+
                           (pow (- y (const cy)) 2)
                           (pow (- z (const cz)) 2))))
                 (const r))
    }

    fn circle(cx: f64, cy: f64, r: f64) -> Result<Tree, Error> {
        deftree!(- (sqrt (+ (pow (- x (const cx)) 2) (pow (- y (const cy)) 2))) (const r))
    }

    #[test]
    fn t_two_planes_union() {
        let tree = deftree!(min (- x 1) (- 6 x)).unwrap(); // Union of two planes.
        assert!(
            tree.folded_for_interval(&[('x', Interval::from_scalar(0., 1.).unwrap(),)])
                .unwrap()
                .equivalent(&deftree!(- x 1).unwrap())
        ); // Should get back one plane after pruning.
        assert!(
            tree.folded_for_interval(&[('x', Interval::from_scalar(6., 7.).unwrap(),)])
                .unwrap()
                .equivalent(&deftree!(- 6 x).unwrap())
        );
        assert!(
            tree.folded_for_interval(&[('x', Interval::from_scalar(2., 5.).unwrap(),)])
                .unwrap()
                .equivalent(&tree)
        )
    }

    #[test]
    fn t_two_circles_union() {
        let c1 = circle(0., 0., 1.).unwrap();
        let c2 = circle(4., 0., 1.).unwrap();
        let tree = deftree!(min {Ok(c1.clone())} {Ok(c2.clone())}).unwrap();
        assert!(
            tree.folded_for_interval(&[
                ('x', Interval::from_scalar(0., 1.).unwrap()),
                ('y', Interval::from_scalar(0., 1.).unwrap()),
            ])
            .unwrap()
            .equivalent(&c1.fold().unwrap())
        );
        assert!(
            tree.folded_for_interval(&[
                ('x', Interval::from_scalar(3., 4.).unwrap()),
                ('y', Interval::from_scalar(0., 1.).unwrap()),
            ])
            .unwrap()
            .equivalent(&c2.fold().unwrap())
        );
        assert!(
            tree.folded_for_interval(&[
                ('x', Interval::from_scalar(1., 3.).unwrap()),
                ('y', Interval::from_scalar(0., 1.).unwrap()),
            ])
            .unwrap()
            .equivalent(&tree.fold().unwrap())
        );
    }

    #[test]
    fn t_two_spheres_union() {
        let s1 = sphere(0., 0., 0., 1.).unwrap();
        let s2 = sphere(4., 0., 0., 1.).unwrap();
        let tree = deftree!(min {Ok(s1.clone())} {Ok(s2.clone())}).unwrap();
        assert!(
            tree.folded_for_interval(&[
                ('x', Interval::from_scalar(0., 1.).unwrap()),
                ('y', Interval::from_scalar(0., 1.).unwrap()),
                ('z', Interval::from_scalar(0., 1.).unwrap())
            ])
            .unwrap()
            .equivalent(&s1.fold().unwrap())
        );
        assert!(
            tree.folded_for_interval(&[
                ('x', Interval::from_scalar(3., 4.).unwrap()),
                ('y', Interval::from_scalar(0., 1.).unwrap()),
                ('z', Interval::from_scalar(0., 1.).unwrap())
            ])
            .unwrap()
            .equivalent(&s2.fold().unwrap())
        );
        assert!(
            tree.folded_for_interval(&[
                ('x', Interval::from_scalar(1., 3.).unwrap()),
                ('y', Interval::from_scalar(0., 1.).unwrap()),
                ('z', Interval::from_scalar(0., 1.).unwrap())
            ])
            .unwrap()
            .equivalent(&tree.fold().unwrap())
        )
    }

    #[test]
    fn t_two_planes_intersection() {
        let tree = deftree!(max (- x 1) (- y 1)).unwrap();
        assert!(
            tree.folded_for_interval(&[
                ('x', Interval::from_scalar(0., 1.).unwrap()),
                ('y', Interval::from_scalar(2., 3.).unwrap())
            ])
            .unwrap()
            .equivalent(&deftree!(- y 1).unwrap())
        );
        assert!(
            tree.folded_for_interval(&[
                ('x', Interval::from_scalar(2., 3.).unwrap()),
                ('y', Interval::from_scalar(0., 1.).unwrap())
            ])
            .unwrap()
            .equivalent(&deftree!(- x 1).unwrap())
        );
        assert!(
            tree.folded_for_interval(&[
                ('x', Interval::from_scalar(2., 3.).unwrap()),
                ('y', Interval::from_scalar(2., 3.).unwrap())
            ])
            .unwrap()
            .equivalent(&tree)
        );
    }

    #[test]
    fn t_two_circles_intersection() {
        let c1 = circle(0., 0., 1.).unwrap();
        let c2 = circle(1.5, 0., 1.).unwrap();
        let tree = deftree!(max {Ok(c1.clone())} {Ok(c2.clone())}).unwrap();
        assert!(
            tree.folded_for_interval(&[
                ('x', Interval::from_scalar(-2., -1.).unwrap()),
                ('y', Interval::from_scalar(0., 1.).unwrap()),
            ],)
                .unwrap()
                .equivalent(&c2.fold().unwrap())
        );
        assert!(
            tree.folded_for_interval(&[
                ('x', Interval::from_scalar(2.5, 3.5).unwrap()),
                ('y', Interval::from_scalar(0., 1.).unwrap()),
            ],)
                .unwrap()
                .equivalent(&c1.fold().unwrap())
        );
        assert!(
            tree.folded_for_interval(&[
                ('x', Interval::from_scalar(0., 1.5).unwrap()),
                ('y', Interval::from_scalar(0., 1.).unwrap()),
            ],)
                .unwrap()
                .equivalent(&tree.fold().unwrap())
        );
    }

    #[test]
    fn t_two_spheres_intersection() {
        let c1 = sphere(0., 0., 0., 1.).unwrap();
        let c2 = sphere(1.5, 0., 0., 1.).unwrap();
        let tree = deftree!(max {Ok(c1.clone())} {Ok(c2.clone())}).unwrap();
        assert!(
            tree.folded_for_interval(&[
                ('x', Interval::from_scalar(-2., -1.).unwrap()),
                ('y', Interval::from_scalar(0., 1.).unwrap()),
                ('z', Interval::from_scalar(0., 1.).unwrap()),
            ],)
                .unwrap()
                .equivalent(&c2.fold().unwrap())
        );
        assert!(
            tree.folded_for_interval(&[
                ('x', Interval::from_scalar(2.5, 3.5).unwrap()),
                ('y', Interval::from_scalar(0., 1.).unwrap()),
                ('z', Interval::from_scalar(0., 1.).unwrap()),
            ],)
                .unwrap()
                .equivalent(&c1.fold().unwrap())
        );
        assert!(
            tree.folded_for_interval(&[
                ('x', Interval::from_scalar(0., 1.5).unwrap()),
                ('y', Interval::from_scalar(0., 1.).unwrap()),
                ('z', Interval::from_scalar(0., 1.).unwrap()),
            ],)
                .unwrap()
                .equivalent(&tree.fold().unwrap())
        );
    }

    #[test]
    fn t_choose_lt() {
        let tree = deftree!(if (< x 1) true false).unwrap();
        assert!(
            tree.folded_for_interval(&[('x', Interval::from_scalar(0., 0.5).unwrap())])
                .unwrap()
                .equivalent(&deftree!(true).unwrap())
        );
        assert!(
            tree.folded_for_interval(&[('x', Interval::from_scalar(1.5, 2.).unwrap())])
                .unwrap()
                .equivalent(&deftree!(false).unwrap())
        );
        assert!(
            tree.folded_for_interval(&[('x', Interval::from_scalar(1.0, 2.).unwrap())])
                .unwrap()
                .equivalent(&tree)
        )
    }

    #[test]
    fn t_choose_lte() {
        let tree = deftree!(if (<= x 1) true false).unwrap();
        assert!(
            tree.folded_for_interval(&[('x', Interval::from_scalar(0., 0.5).unwrap())])
                .unwrap()
                .equivalent(&deftree!(true).unwrap())
        );
        assert!(
            tree.folded_for_interval(&[('x', Interval::from_scalar(1.5, 2.).unwrap())])
                .unwrap()
                .equivalent(&deftree!(false).unwrap())
        );
        assert!(
            tree.folded_for_interval(&[('x', Interval::from_scalar(0., 1.).unwrap())])
                .unwrap()
                .equivalent(&deftree!(true).unwrap())
        );
    }

    #[test]
    fn t_choose_gt() {
        let tree = deftree!(if (> x 1) true false).unwrap();
        assert!(
            tree.folded_for_interval(&[('x', Interval::from_scalar(0., 0.5).unwrap())])
                .unwrap()
                .equivalent(&deftree!(false).unwrap())
        );
        assert!(
            tree.folded_for_interval(&[('x', Interval::from_scalar(1.5, 2.).unwrap())])
                .unwrap()
                .equivalent(&deftree!(true).unwrap())
        );
        assert!(
            tree.folded_for_interval(&[('x', Interval::from_scalar(1.0, 2.).unwrap())])
                .unwrap()
                .equivalent(&tree)
        )
    }

    #[test]
    fn t_choose_gte() {
        let tree = deftree!(if (>= x 1) true false).unwrap();
        assert!(
            tree.folded_for_interval(&[('x', Interval::from_scalar(0., 0.5).unwrap())])
                .unwrap()
                .equivalent(&deftree!(false).unwrap())
        );
        assert!(
            tree.folded_for_interval(&[('x', Interval::from_scalar(1.5, 2.).unwrap())])
                .unwrap()
                .equivalent(&deftree!(true).unwrap())
        );
        assert!(
            tree.folded_for_interval(&[('x', Interval::from_scalar(1., 2.).unwrap())])
                .unwrap()
                .equivalent(&deftree!(true).unwrap())
        );
    }
}
