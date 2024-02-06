use crate::tree::{BinaryOp::*, Node, Node::*, Tree, UnaryOp::*, Value::*};
use std::ops::Range;

impl Tree {}

fn deriv_helper(
    nodes: &[Node],
    input_range: Range<usize>,
    param: char,
    dst: &mut Vec<Node>,
    derivmap: &mut Vec<usize>,
) {
    let offset = nodes.len();
    let nodes = &nodes[input_range];
    derivmap.clear();
    derivmap.reserve(nodes.len());
    for ni in 0..nodes.len() {
        let deriv = match &nodes[ni] {
            Constant(_val) => Constant(Scalar(0.)),
            Symbol(label) => Constant(Scalar(if *label == param { 1. } else { 0. })),
            Unary(op, input) => match op {
                Negate => Unary(Negate, derivmap[*input]),
                Sqrt => {
                    let sf = push_node(Unary(Sqrt, *input), dst) + offset;
                    let c2 = push_node(Constant(Scalar(2.)), dst) + offset;
                    let sf2 = push_node(Binary(Multiply, sf, c2), dst) + offset;
                    Binary(Divide, derivmap[*input], sf2)
                }
                Abs => todo!(), // Needs piecewise functions. Will stop here and desing those first.
                Sin => todo!(),
                Cos => todo!(),
                Tan => todo!(),
                Log => todo!(),
                Exp => todo!(),
                Not => todo!(),
            },
            Binary(_, _, _) => todo!(),
            Ternary(..) => todo!(),
        };
    }
    todo!();
}

fn push_node(node: Node, dst: &mut Vec<Node>) -> usize {
    let idx = dst.len();
    dst.push(node);
    return idx;
}
