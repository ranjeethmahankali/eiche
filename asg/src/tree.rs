#[derive(Debug, PartialEq)]
/// Node types that an abstract syntax tree can be composed of.
pub enum Node {
    /// Constant node containing it's value.
    Constant(f64),
    /// Symbol with a `char` label.
    Symbol(char),
    // Binary operations
    /// Represents addition. Contains indices of the operand nodes.
    Add(usize, usize),
    /// Represents subtraction. Contains indices of the operand nodes.
    Subtract(usize, usize),
    /// Represents multiplication. Contains indices of the operand nodes.
    Multiply(usize, usize),
    /// Represents Division. Contains indices of the operand nodes.
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

/// Represents an abstract syntax tree.
pub struct Tree {
    nodes: Vec<Node>,
}

use Node::*;

impl Tree {
    pub fn new(node: Node) -> Tree {
        Tree { nodes: vec![node] }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn root(&self) -> &Node {
        self.nodes
            .last()
            .expect("This Tree is empty! This should never have happened!")
    }

    pub fn depth_first_traverse<F, T, E>(&self, mut visitor: F) -> Result<T, E>
    where
        F: FnMut(usize, Option<usize>) -> Result<T, E>,
        T: Default,
    {
        let mut stack: Vec<(usize, Option<usize>)> = vec![(self.len() - 1, None)];
        while !stack.is_empty() {
            let (index, parent) = stack
                .pop()
                .expect("Something went wrong in the depth first traversal!");
            match self.node(index) {
                Constant(_) => {} // Do nothing.
                Symbol(_) => {}   // Do nothing.
                Add(lhs, rhs)
                | Subtract(lhs, rhs)
                | Multiply(lhs, rhs)
                | Divide(lhs, rhs)
                | Pow(lhs, rhs)
                | Min(lhs, rhs)
                | Max(lhs, rhs) => {
                    stack.push((*rhs, Some(index)));
                    stack.push((*lhs, Some(index)));
                }
                Negate(input) | Sqrt(input) | Abs(input) | Sin(input) | Cos(input) | Tan(input)
                | Log(input) | Exp(input) => stack.push((*input, Some(index))),
            }
            visitor(index, parent)?;
        }
        return Result::<T, E>::Ok(T::default());
    }

    pub fn node(&self, index: usize) -> &Node {
        &self.nodes[index]
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
}

macro_rules! binary_op {
    ($lhs: ident, $rhs: ident, $op: ident) => {{
        let left = $lhs.len() - 1;
        let right = left + $rhs.len();
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

    /// Set all symbols in the evaluator matching `label` to
    /// `value`. This `value` will be used for all future evaluations,
    /// unless this function is called again with a different `value`.
    pub fn set_var(&mut self, label: char, value: f64) {
        for (node, reg) in self.tree.nodes.iter().zip(self.regs.iter_mut()) {
            match node {
                Symbol(l) if *l == label => {
                    *reg = Some(value);
                }
                _ => {}
            }
        }
    }

    /// Read the value from the `index`-th register. Returns an error
    /// if the register doesn't contain a value.
    fn read(&self, index: usize) -> Result<f64, EvaluationError> {
        match self.regs[index] {
            Some(val) => Ok(val),
            None => Err(EvaluationError::UninitializedValueRead),
        }
    }

    /// Write the `value` into the `index`-th register. The existing
    /// value is overwritten.
    fn write(&mut self, index: usize, value: f64) {
        self.regs[index] = Some(value);
    }

    /// Run the evaluator and return the result. The result may
    /// contain the output value, or an
    /// error. `Variablenotfound(label)` error means the variable
    /// matching `label` hasn't been assigned a value using `set_var`.
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
