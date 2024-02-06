use crate::{
    error::Error,
    tree::{BinaryOp::*, MaybeTree, Node, Node::*, TernaryOp::*, Tree, UnaryOp::*, Value::*},
};
use std::ops::Range;

impl Tree {
    pub fn symbolic_derivative(mut self, params: &[char]) -> MaybeTree {
        let num_nodes = self.len();
        let (root_start, root_end) = {
            let root_indices = self.root_indices();
            (root_indices.start, root_indices.end)
        };
        let mut derivs = Vec::<Node>::new();
        let mut derivmap = Vec::<Option<usize>>::new();
        let mut rootnodes = Vec::<usize>::new();
        for param in params {
            compute_symbolic_deriv(
                self.nodes(),
                0..num_nodes,
                *param,
                &mut derivs,
                &mut derivmap,
            );
            self.nodes_mut().extend(derivs.drain(..));
            for ri in root_start..root_end {
                rootnodes.push(match derivmap[ri] {
                    Some(deriv) => deriv,
                    None => return Err(Error::CannotComputeSymbolicDerivative),
                });
            }
        }
        todo!()
    }
}

fn compute_symbolic_deriv(
    nodes: &[Node],
    input_range: Range<usize>,
    param: char,
    dst: &mut Vec<Node>,
    derivmap: &mut Vec<Option<usize>>,
) {
    let offset = nodes.len();
    dst.clear();
    derivmap.clear();
    derivmap.resize(nodes.len(), None);
    for ni in input_range {
        let deriv = match &nodes[ni] {
            Constant(_val) => Constant(Scalar(0.)),
            Symbol(label) => Constant(Scalar(if *label == param { 1. } else { 0. })),
            Unary(op, input) => {
                let inputderiv = match derivmap[*input] {
                    Some(index) => index,
                    // A unary op whose input is not differentiable is not differentiable.
                    None => continue,
                };
                match op {
                    Negate => Unary(Negate, inputderiv),
                    Sqrt => {
                        let sf = push_node(Unary(Sqrt, *input), dst) + offset;
                        let c2 = push_node(Constant(Scalar(2.)), dst) + offset;
                        let sf2 = push_node(Binary(Multiply, sf, c2), dst) + offset;
                        Binary(Divide, inputderiv, sf2)
                    }
                    Abs => {
                        // Technically the gradient should not be defined at zero
                        // exactly. But this might be a pragmatic
                        // compromise. Reconsider this decision later if it becomes
                        // a problem.
                        let zero = push_node(Constant(Scalar(0.)), dst) + offset;
                        let cond = push_node(Binary(Less, *input, zero), dst) + offset;
                        let one = push_node(Constant(Scalar(1.)), dst) + offset;
                        let neg = push_node(Constant(Scalar(-1.)), dst) + offset;
                        let df = push_node(Ternary(Choose, cond, neg, one), dst) + offset;
                        Binary(Multiply, df, inputderiv) // Chain rule.
                    }
                    Sin => Unary(Cos, *input),
                    Cos => {
                        let sin = push_node(Unary(Sin, *input), dst) + offset;
                        let negsin = push_node(Unary(Negate, sin), dst) + offset;
                        Binary(Multiply, negsin, inputderiv) // Chain rule.
                    }
                    Tan => {
                        let cos = push_node(Unary(Cos, *input), dst) + offset;
                        let one = push_node(Constant(Scalar(1.)), dst) + offset;
                        let sec = push_node(Binary(Divide, one, cos), dst) + offset;
                        let two = push_node(Constant(Scalar(2.)), dst) + offset;
                        let sec2 = push_node(Binary(Pow, sec, two), dst) + offset;
                        Binary(Multiply, sec2, inputderiv) // chain rule.
                    }
                    Log => {
                        let xlogx = push_node(Binary(Multiply, *input, ni), dst) + offset;
                        Binary(Divide, inputderiv, xlogx)
                    }
                    Exp => nodes[ni],
                    Not => continue,
                }
            }
            Binary(op, lhs, rhs) => {
                // Both inputs need to be differentiable, otherwise this node is not differentiable.
                let lderiv = match derivmap[*lhs] {
                    Some(val) => val,
                    None => continue,
                };
                let rderiv = match derivmap[*rhs] {
                    Some(val) => val,
                    None => continue,
                };
                match op {
                    Add => Binary(Add, lderiv, rderiv),
                    Subtract => Binary(Subtract, lderiv, rderiv),
                    Multiply => {
                        let lr = push_node(Binary(Multiply, *lhs, rderiv), dst) + offset;
                        let rl = push_node(Binary(Multiply, *rhs, lderiv), dst) + offset;
                        Binary(Add, lr, rl)
                    }
                    Divide => {
                        let lr = push_node(Binary(Multiply, lderiv, *rhs), dst) + offset;
                        let rl = push_node(Binary(Multiply, rderiv, *lhs), dst) + offset;
                        let sub = push_node(Binary(Subtract, lr, rl), dst) + offset;
                        let two = push_node(Constant(Scalar(2.)), dst) + offset;
                        let r2 = push_node(Binary(Pow, *rhs, two), dst) + offset;
                        Binary(Divide, sub, r2)
                    }
                    Pow => {
                        // https://www.physicsforums.com/threads/derivative-of-f-x-to-the-power-of-g-x-and-algebra-problem.273333/
                        let logf = push_node(Unary(Log, *lhs), dst) + offset;
                        let gderiv_logf = push_node(Binary(Multiply, rderiv, logf), dst) + offset;
                        let fderiv_over_f = push_node(Binary(Divide, lderiv, *lhs), dst) + offset;
                        let second_term =
                            push_node(Binary(Multiply, *rhs, fderiv_over_f), dst) + offset;
                        let sum = push_node(Binary(Add, gderiv_logf, second_term), dst) + offset;
                        Binary(Multiply, ni, sum)
                    }
                    Min => {
                        let cond = push_node(Binary(Less, *lhs, *rhs), dst) + offset;
                        Ternary(Choose, cond, lderiv, rderiv)
                    }
                    Max => {
                        let cond = push_node(Binary(Greater, *lhs, *rhs), dst) + offset;
                        Ternary(Choose, cond, lderiv, rderiv)
                    }
                    Less | LessOrEqual | Equal | NotEqual | Greater | GreaterOrEqual | And | Or => {
                        continue
                    }
                }
            }
            Ternary(op, a, b, c) => match op {
                Choose => {
                    let bderiv = match derivmap[*b] {
                        Some(val) => val,
                        None => continue,
                    };
                    let cderiv = match derivmap[*c] {
                        Some(val) => val,
                        None => continue,
                    };
                    Ternary(Choose, *a, bderiv, cderiv)
                }
            },
        };
        derivmap[ni] = Some(offset + dst.len());
        dst.push(deriv);
    }
}

fn push_node(node: Node, dst: &mut Vec<Node>) -> usize {
    let idx = dst.len();
    dst.push(node);
    return idx;
}
