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
        let offset: usize = self.nodes.len();
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

#[derive(Debug)]
pub enum EvaluationError {
    VariableNotFound(char),
    UninitializedValueRead,
}

pub struct Evaluator<'a> {
    tree: &'a Tree,
    regs: Box<[Option<f64>]>,
}

impl<'a> Evaluator<'a> {
    pub fn new(tree: &'a Tree) -> Evaluator {
        Evaluator {
            tree,
            regs: vec![None; tree.nodes.len()].into_boxed_slice(),
        }
    }

    pub fn set_var(&mut self, label: char, val: f64) {
        for (node, reg) in self.tree.nodes.iter().zip(self.regs.iter_mut()) {
            match node {
                Symbol(l) if *l == label => {
                    *reg = Some(val);
                }
                _ => {}
            }
        }
    }

    fn read(&self, index: usize) -> Result<f64, EvaluationError> {
        match self.regs[index] {
            Some(val) => Ok(val),
            None => Err(EvaluationError::UninitializedValueRead),
        }
    }

    fn write(&mut self, index: usize, value: f64) {
        self.regs[index] = Some(value);
    }

    pub fn run(&mut self) -> Result<f64, EvaluationError> {
        for idx in 0..self.tree.nodes.len() {
            self.write(
                idx,
                match self.tree.nodes[idx] {
                    Constant(val) => val,
                    Symbol(label) => match &self.regs[idx] {
                        None => return Err(EvaluationError::VariableNotFound(label)),
                        Some(val) => *val,
                    },
                    Add(lhs, rhs) => self.read(lhs)? + self.read(rhs)?,
                    Subtract(lhs, rhs) => self.read(lhs)? - self.read(rhs)?,
                    Multiply(lhs, rhs) => self.read(lhs)? * self.read(rhs)?,
                    Divide(lhs, rhs) => self.read(lhs)? / self.read(rhs)?,
                    Pow(lhs, rhs) => f64::powf(self.read(lhs)?, self.read(rhs)?),
                    Min(lhs, rhs) => f64::min(self.read(lhs)?, self.read(rhs)?),
                    Max(lhs, rhs) => f64::max(self.read(lhs)?, self.read(rhs)?),
                    Negate(i) => -self.read(i)?,
                    Sqrt(i) => f64::sqrt(self.read(i)?),
                    Abs(i) => f64::abs(self.read(i)?),
                    Sin(i) => f64::sin(self.read(i)?),
                    Cos(i) => f64::cos(self.read(i)?),
                    Tan(i) => f64::tan(self.read(i)?),
                    Log(i) => f64::log(self.read(i)?, std::f64::consts::E),
                    Exp(i) => f64::exp(self.read(i)?),
                },
            );
        }
        return self.read(self.regs.len() - 1);
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
