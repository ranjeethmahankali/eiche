use crate::tree::{BinaryOp, UnaryOp};

pub enum TNode {
    Constant(f32),
    Symbol(char),
    Placeholder(char),
    Unary(UnaryOp, usize, usize),
    Binary(BinaryOp, usize, usize),
}

pub struct Template {
    nodes: Vec<TNode>,
}
