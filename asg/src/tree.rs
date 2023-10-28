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

impl From<f64> for Tree {
    fn from(value: f64) -> Self {
        return Constant(value).into();
    }
}

impl From<f64> for Node {
    fn from(value: f64) -> Self {
        return Constant(value);
    }
}

impl From<char> for Node {
    fn from(value: char) -> Self {
        return Symbol(value);
    }
}

impl From<char> for Tree {
    fn from(c: char) -> Self {
        return Symbol(c).into();
    }
}

pub struct Tree {
    nodes: Vec<Node>,
}

use Node::*;

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

use std::collections::HashMap;

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
            let mut state = self.state[index].clone();
            if state.finished {
                continue;
            }
            match &self.tree.nodes[index] {
                Constant(..) | Symbol(..) => {
                    state.visited += 1;
                    state.finished = true;
                }
                Add(lhs, rhs)
                | Subtract(lhs, rhs)
                | Multiply(lhs, rhs)
                | Divide(lhs, rhs)
                | Pow(lhs, rhs)
                | Min(lhs, rhs)
                | Max(lhs, rhs) => {
                    if state.visited < 2 {
                        state.visited += 1;
                        self.stack.push(index);
                        self.stack.push(*lhs);
                        self.stack.push(*rhs);
                    } else {
                        state.finished = true;
                    }
                }
                Negate(x) | Sqrt(x) | Abs(x) | Sin(x) | Cos(x) | Tan(x) | Log(x) | Exp(x) => {
                    if state.visited < 1 {
                        state.visited += 1;
                        self.stack.push(index);
                        self.stack.push(*x);
                    } else {
                        state.finished = true;
                    }
                }
            }
            if state.finished {
                state.value = match &self.tree.nodes[index] {
                    Constant(value) => *value,
                    Symbol(label) => {
                        if let Some(value) = vars.get(label) {
                            *value
                        } else {
                            return Err(EvaluationError::VariableNotFound);
                        }
                    }
                    Add(lhs, rhs) => self.state[*lhs].value + self.state[*rhs].value,
                    Subtract(lhs, rhs) => self.state[*lhs].value - self.state[*rhs].value,
                    Multiply(lhs, rhs) => self.state[*lhs].value * self.state[*rhs].value,
                    Divide(lhs, rhs) => self.state[*lhs].value / self.state[*rhs].value,
                    Pow(lhs, rhs) => self.state[*lhs].value.powf(self.state[*rhs].value),
                    Min(lhs, rhs) => f64::min(self.state[*lhs].value, self.state[*rhs].value),
                    Max(lhs, rhs) => f64::max(self.state[*lhs].value, self.state[*rhs].value),
                    Negate(x) => -self.state[*x].value,
                    Sqrt(x) => f64::sqrt(self.state[*x].value),
                    Abs(x) => f64::abs(self.state[*x].value),
                    Sin(x) => f64::sin(self.state[*x].value),
                    Cos(x) => f64::cos(self.state[*x].value),
                    Tan(x) => f64::tan(self.state[*x].value),
                    Log(x) => f64::log(self.state[*x].value, std::f64::consts::E),
                    Exp(x) => f64::exp(self.state[*x].value),
                }
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
    #[test]
    fn add() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let sum = x + y;
        assert_eq!(sum.nodes, vec![Symbol('x'), Symbol('y'), Add(0, 1)]);
    }

    #[test]
    fn multiply() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let sum = x * y;
        assert_eq!(sum.nodes, vec![Symbol('x'), Symbol('y'), Multiply(0, 1)]);
    }

    #[test]
    fn subtract() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let sum = x - y;
        assert_eq!(sum.nodes, vec![Symbol('x'), Symbol('y'), Subtract(0, 1)]);
    }

    #[test]
    fn divide() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let sum = x / y;
        assert_eq!(sum.nodes, vec![Symbol('x'), Symbol('y'), Divide(0, 1)]);
    }

    #[test]
    fn negate() {
        let x: Tree = 'x'.into();
        let neg = -x;
        assert_eq!(neg.nodes, vec![Symbol('x'), Negate(0)]);
    }
}
