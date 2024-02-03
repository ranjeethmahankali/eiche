use crate::tree::{BinaryOp::*, Node, Node::*, Tree, UnaryOp::*};

impl Tree {}

fn deriv_wrt_one(nodes: &[Node], param: char) -> Vec<Node> {
    let derivmap = Vec::<usize>::with_capacity(nodes.len());
    let mut out = Vec::<Node>::with_capacity(nodes.len());
    for ni in 0..nodes.len() {
        let deriv = match &nodes[ni] {
            Constant(_val) => Constant(0.),
            Symbol(label) => Constant(if *label == param { 1. } else { 0. }),
            Unary(op, input) => match op {
                Negate => Unary(Negate, derivmap[*input]),
                Sqrt => {
                    let sf = push_node(Unary(Sqrt, *input), &mut out);
                    let c2 = push_node(Constant(2.), &mut out);
                    let sf2 = push_node(Binary(Multiply, sf, c2), &mut out);
                    Binary(Divide, derivmap[*input], sf2)
                }
                Abs => todo!(), // Needs piecewise functions. Will stop here and desing those first.
                Sin => todo!(),
                Cos => todo!(),
                Tan => todo!(),
                Log => todo!(),
                Exp => todo!(),
            },
            Binary(_, _, _) => todo!(),
        };
    }
    todo!();
}

fn push_node(node: Node, dst: &mut Vec<Node>) -> usize {
    let idx = dst.len();
    dst.push(node);
    return idx;
}
