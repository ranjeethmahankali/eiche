#[derive(Debug, PartialEq)]
pub enum Node {
    // Leaf types
    Constant(f64),
    Symbol(char),
    // Binary operations
    Add(usize, usize),
    Subtract(usize, usize),
    Multiply(usize, usize),
    Divide(usize, usize),
    Pow(usize, usize),
    Min(usize, usize),
    Max(usize, usize),
    // Unary operations
    Negate(usize),
    Sqrt(usize),
    Abs(usize),
    Sin(usize),
    Cos(usize),
    Tan(usize),
    Log(usize),
    Exp(usize),
}

impl Into<Tree> for Node {
    fn into(self) -> Tree {
        Tree::new(self)
    }
}

use std::collections::HashMap;

use Node::*;

pub struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    pub fn new(node: Node) -> Tree {
        Tree { nodes: vec![node] }
    }

    pub fn root(&self) -> &Node {
        self.nodes
            .last()
            .expect("This Tree is empty! This should never have happened!")
    }

    fn merge(mut self, other: Tree, op: Node) -> Tree {
        let offset: usize = other.nodes.len();
        self.nodes
            .reserve(self.nodes.len() + other.nodes.len() + 1usize);
        self.nodes.extend(other.nodes.iter().map(|node| match node {
            Constant(value) => Constant(*value),
            Symbol(label) => Symbol(label.clone()),
            Add(l, r) => Add(*l + offset, *r + offset),
            Subtract(l, r) => Subtract(*l + offset, *r + offset),
            Multiply(l, r) => Multiply(*l + offset, *r + offset),
            Divide(l, r) => Divide(*l + offset, *r + offset),
            Pow(l, r) => Pow(*l + offset, *r + offset),
            Min(l, r) => Min(*l + offset, *r + offset),
            Max(l, r) => Max(*l + offset, *r + offset),
            Negate(x) => Negate(*x + offset),
            Sqrt(x) => Sqrt(*x + offset),
            Abs(x) => Abs(*x + offset),
            Sin(x) => Sin(*x + offset),
            Cos(x) => Cos(*x + offset),
            Tan(x) => Tan(*x + offset),
            Log(x) => Log(*x + offset),
            Exp(x) => Exp(*x + offset),
        }));
        self.nodes.push(op);
        return self;
    }

    fn merge_offsets(left: &Tree, right: &Tree) -> (usize, usize) {
        let li = left.nodes.len() - 1;
        let ri = li + right.nodes.len();
        (li, ri)
    }
}

macro_rules! binary_op {
    ($lhs: ident, $rhs: ident, $op: ident) => {{
        let (left, right) = Tree::merge_offsets(&$lhs, &$rhs);
        $lhs.merge($rhs, $op(left, right))
    }};
}

macro_rules! unary_op {
    ($input: ident, $op: ident) => {{
        let idx = $input.nodes.len() - 1;
        $input.nodes.push($op(idx));
        return $input;
    }};
}

impl core::ops::Add<Tree> for Tree {
    type Output = Tree;

    fn add(self, rhs: Tree) -> Tree {
        binary_op!(self, rhs, Add)
    }
}

impl core::ops::Sub<Tree> for Tree {
    type Output = Tree;

    fn sub(self, rhs: Tree) -> Self::Output {
        binary_op!(self, rhs, Subtract)
    }
}

impl core::ops::Mul<Tree> for Tree {
    type Output = Tree;

    fn mul(self, rhs: Tree) -> Tree {
        binary_op!(self, rhs, Multiply)
    }
}

impl core::ops::Div<Tree> for Tree {
    type Output = Tree;

    fn div(self, rhs: Tree) -> Self::Output {
        binary_op!(self, rhs, Divide)
    }
}

pub fn pow(base: Tree, exponent: Tree) -> Tree {
    binary_op!(base, exponent, Pow)
}

pub fn min(lhs: Tree, rhs: Tree) -> Tree {
    binary_op!(lhs, rhs, Min)
}

pub fn max(lhs: Tree, rhs: Tree) -> Tree {
    binary_op!(lhs, rhs, Max)
}

impl core::ops::Neg for Tree {
    type Output = Tree;

    fn neg(mut self) -> Self::Output {
        unary_op!(self, Negate)
    }
}

pub fn sqrt(mut x: Tree) -> Tree {
    unary_op!(x, Sqrt)
}

pub fn abs(mut x: Tree) -> Tree {
    unary_op!(x, Abs)
}

pub fn sin(mut x: Tree) -> Tree {
    unary_op!(x, Sin)
}

pub fn cos(mut x: Tree) -> Tree {
    unary_op!(x, Cos)
}

pub fn tan(mut x: Tree) -> Tree {
    unary_op!(x, Tan)
}

pub fn log(mut x: Tree) -> Tree {
    unary_op!(x, Log)
}

pub fn exp(mut x: Tree) -> Tree {
    unary_op!(x, Exp)
}

pub enum EvaluationError {
    VariableNotFound,
    Unknown,
}

#[derive(Clone)]
struct NodeState {
    value: f64,
    visited: usize,
    finished: bool,
}

pub struct Evaluator<'a> {
    tree: &'a Tree,
    state: Vec<NodeState>,
    stack: Vec<usize>,
}

impl<'a> Evaluator<'a> {
    pub fn new(tree: &'a Tree) -> Evaluator {
        Evaluator {
            tree,
            state: vec![
                NodeState {
                    value: 0.,
                    visited: 0,
                    finished: false
                };
                tree.nodes.len()
            ],
            stack: Vec::with_capacity(tree.nodes.len()),
        }
    }

    pub fn run(&mut self, vars: &HashMap<char, f64>) -> Result<f64, EvaluationError> {
        self.stack.push(self.tree.nodes.len() - 1);
        while let Some(index) = self.stack.pop() {
            let prev: usize = *(self.stack.last().unwrap_or(&index));
            let mut state = self.state[index].clone();
            if state.finished {
                continue;
            }
            match &self.tree.nodes[index] {
                Constant(value) => {
                    state.value = *value;
                    state.visited = 1;
                    state.finished = true;
                }
                Symbol(_) => todo!(),
                Add(lhs, rhs) => {
                    if state.visited < 2 {
                        state.visited += 1;
                    } else {
                        state.value = self.state[*lhs].value + self.state[*rhs].value;
                        state.finished = true;
                    }
                }
                Subtract(lhs, rhs) => {
                    if state.visited < 2 {
                        state.visited += 1;
                    } else {
                        state.value = self.state[*lhs].value - self.state[*rhs].value;
                        state.finished = true;
                    }
                }
                Multiply(_, _) => todo!(),
                Divide(_, _) => todo!(),
                Pow(_, _) => todo!(),
                Min(_, _) => todo!(),
                Max(_, _) => todo!(),
                Negate(_) => todo!(),
                Sqrt(_) => todo!(),
                Abs(_) => todo!(),
                Sin(_) => todo!(),
                Cos(_) => todo!(),
                Tan(_) => todo!(),
                Log(_) => todo!(),
                Exp(_) => todo!(),
            }
            self.state[index] = state;
        }
        match self.state.last() {
            Some(state) => Ok(state.value),
            None => Err(EvaluationError::Unknown),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Node::*, Tree};

    fn symbol(label: char) -> Tree {
        return Symbol(label).into();
    }

    #[test]
    fn add() {
        let sum = symbol('x') + symbol('y');
        assert_eq!(sum.nodes, vec![Symbol('x'), Symbol('y'), Add(0, 1)]);
    }

    #[test]
    fn multiply() {
        let sum = symbol('x') * symbol('y');
        assert_eq!(sum.nodes, vec![Symbol('x'), Symbol('y'), Multiply(0, 1)]);
    }

    #[test]
    fn subtract() {
        let sum = symbol('x') - symbol('y');
        assert_eq!(sum.nodes, vec![Symbol('x'), Symbol('y'), Subtract(0, 1)]);
    }

    #[test]
    fn divide() {
        let sum = symbol('x') / symbol('y');
        assert_eq!(sum.nodes, vec![Symbol('x'), Symbol('y'), Divide(0, 1)]);
    }

    #[test]
    fn negate() {
        let neg = -symbol('x');
        assert_eq!(neg.nodes, vec![Symbol('x'), Negate(0)]);
    }
}
