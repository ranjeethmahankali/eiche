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
use std::collections::BTreeMap;

pub(crate) fn fold_for_interval(
    nodes: &[Node],
    vars: &BTreeMap<char, Interval>,
) -> Result<Vec<Node>, Error> {
    let mut values: Vec<Interval> = Vec::with_capacity(nodes.len());
    let mut out = Vec::with_capacity(nodes.len());
    for node in nodes {
        let (folded, value) = match node {
            Constant(value) => (None, Interval::from_value(*value)?),
            Symbol(label) => match vars.get(label) {
                Some(value) => (None, *value),
                None => (None, Interval::Scalar(inari::Interval::ENTIRE)),
            },
            Unary(op, input) => match (op, &values[*input], &nodes[*input]) {
                // Singleton intervals that don't come from constants can be replaced with constants.
                (_, Interval::Scalar(ii), Unary(_, _) | Binary(_, _, _) | Ternary(_, _, _, _))
                    if ii.is_singleton() =>
                {
                    let ni = out.len();
                    out.push(Constant(Value::Scalar(ii.inf())));
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
                    return Err(Error::TypeMismatch)
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
                                let ni = out.len();
                                out.push(Constant(Value::Scalar(ileft.inf())));
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
                                let ni = out.len();
                                out.push(Constant(Value::Scalar(iright.inf())));
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
                    return Err(Error::TypeMismatch)
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
        out.push(folded.unwrap_or(*node));
        values.push(value);
    }
    fold(&mut out)?;
    Ok(out)
}

impl Tree {
    pub fn fold_for_interval(self, vars: &BTreeMap<char, Interval>) -> Result<Tree, Error> {
        let (nodes, dims) = self.take();
        Tree::from_nodes(fold_for_interval(&nodes, vars)?, dims)
    }
}

#[cfg(test)]
mod test {
    use std::collections::BTreeMap;

    use super::fold_for_interval;
    use crate::{deftree, error::Error, interval::Interval, tree::Tree};

    fn sphere(cx: f64, cy: f64, cz: f64, r: f64) -> Result<Tree, Error> {
        deftree!(- (sqrt (+
                          (pow (- x (const cx)) 2)
                          (+
                           (pow (- y (const cy)) 2)
                           (pow (- z (const cz)) 2))))
                 (const r))
    }

    #[test]
    fn t_two_planes() {
        assert!(deftree!(min (- x 1) (- 6 x)) // Union of two planes.
            .unwrap()
            .fold_for_interval(&BTreeMap::from([(
                'x',
                Interval::from_scalar(0., 1.).unwrap(),
            )]))
            .unwrap()
            .equivalent(&deftree!(- x 1).unwrap())); // Should get back one plane after pruning.
    }

    #[test]
    fn t_two_spheres() {
        let union = deftree!(min {sphere(0., 0., 0., 1.)} {sphere(4., 0., 0., 1.)}).unwrap();
        let smaller = Tree::from_nodes(
            fold_for_interval(
                union.nodes(),
                &BTreeMap::from([('x', Interval::from_scalar(0., 1.).unwrap())]),
            )
            .unwrap(),
            union.dims(),
        )
        .unwrap();
        println!("\n${}$\n\n", union.to_latex());
        println!("\n${}$\n\n", smaller.to_latex());
    }
}
