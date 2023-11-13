#[derive(Debug, Copy, Clone, PartialEq, Hash)]
pub enum UnaryOp {
    Negate,
    Sqrt,
    Abs,
    Sin,
    Cos,
    Tan,
    Log,
    Exp,
}

#[derive(Debug, Copy, Clone, PartialEq, Hash)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Pow,
    Min,
    Max,
}

impl UnaryOp {
    /// Compute the result of the operation on `value`.
    pub fn apply(&self, value: f64) -> f64 {
        match self {
            Negate => -value,
            Sqrt => f64::sqrt(value),
            Abs => f64::abs(value),
            Sin => f64::sin(value),
            Cos => f64::cos(value),
            Tan => f64::tan(value),
            Log => f64::log(value, std::f64::consts::E),
            Exp => f64::exp(value),
        }
    }
}

impl BinaryOp {
    /// Compute the result of the operation on `lhs` and `rhs`.
    pub fn apply(&self, lhs: f64, rhs: f64) -> f64 {
        match self {
            Add => lhs + rhs,
            Subtract => lhs - rhs,
            Multiply => lhs * rhs,
            Divide => lhs / rhs,
            Pow => f64::powf(lhs, rhs),
            Min => f64::min(lhs, rhs),
            Max => f64::max(lhs, rhs),
        }
    }
}

use crate::{
    helper::{fold_constants, Deduplicater, Pruner},
    parser::{parse_tree, LispParseError},
};
use BinaryOp::*;
use UnaryOp::*;

#[derive(Debug)]
pub enum TreeError {
    WrongNodeOrder,
    ConstantFoldingFailed,
    EmptyTree,
    PruningFailed,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Node {
    Constant(f64),
    Symbol(char),
    Unary(UnaryOp, usize),
    Binary(BinaryOp, usize, usize),
}

use Node::*;

/// Represents an abstract syntax tree.
#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    /// Create a tree representing a constant value.
    pub fn constant(val: f64) -> Tree {
        Tree {
            nodes: vec![Constant(val)],
        }
    }

    /// Create a tree representing a symbol with the given `label`.
    pub fn symbol(label: char) -> Tree {
        Tree {
            nodes: vec![Symbol(label)],
        }
    }

    /// Validate the list of nodes. The topology of the nodes must be
    /// valid and no constant nodes can contain NaNs.
    fn validate(nodes: Vec<Node>) -> Result<Vec<Node>, TreeError> {
        if !(0..nodes.len()).all(|i| match &nodes[i] {
            Constant(val) => !f64::is_nan(*val),
            Symbol(_) => true,
            Unary(_op, input) => input < &i,
            Binary(_op, lhs, rhs) => lhs < &i && rhs < &i,
        }) {
            return Err(TreeError::WrongNodeOrder);
        }
        // Maybe add more checks later.
        return Ok(nodes);
    }

    /// Create a new tree with `nodes`. If the `nodes` don't meet the
    /// requirements for represeting a standart tree, an appropriate
    /// `TreeError` is returned.
    pub fn from_nodes(nodes: Vec<Node>) -> Result<Tree, TreeError> {
        Ok(Tree {
            nodes: Self::validate(nodes)?,
        })
    }

    pub fn from_lisp(lisp: &str) -> Result<Tree, LispParseError> {
        parse_tree(lisp)
    }

    /// The number of nodes in this tree.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Get a reference to root of the tree. This is the last node of
    /// the tree.
    pub fn root(&self) -> Result<&Node, TreeError> {
        self.nodes.last().ok_or(TreeError::EmptyTree)
    }

    pub fn root_index(&self) -> usize {
        self.len() - 1
    }

    pub fn node(&self, index: usize) -> &Node {
        &self.nodes[index]
    }

    pub fn nodes(&self) -> &Vec<Node> {
        &self.nodes
    }

    pub fn symbols(&self) -> Vec<char> {
        let mut chars: Vec<_> = self
            .nodes
            .iter()
            .filter_map(|n| {
                if let Symbol(label) = n {
                    Some(*label)
                } else {
                    None
                }
            })
            .collect();
        chars.sort();
        chars.dedup();
        return chars;
    }

    pub fn to_lisp(&self) -> Result<String, TreeError> {
        Ok(crate::helper::to_lisp(self.root()?, self.nodes()))
    }

    pub fn fold_constants(mut self) -> Result<Tree, TreeError> {
        let mut pruner = Pruner::new();
        let root_index = self.root_index();
        self.nodes = Self::validate(pruner.run(fold_constants(self.nodes), root_index))?;
        return Ok(self);
    }

    pub fn deduplicate(mut self) -> Result<Tree, TreeError> {
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        let root_index = self.root_index();
        self.nodes = Self::validate(pruner.run(dedup.run(self.nodes), root_index))?;
        return Ok(self);
    }

    fn binary_op(mut self, other: Tree, op: BinaryOp) -> Tree {
        let offset: usize = self.nodes.len();
        self.nodes
            .reserve(self.nodes.len() + other.nodes.len() + 1usize);
        self.nodes.extend(other.nodes.iter().map(|node| match node {
            Constant(value) => Constant(*value),
            Symbol(label) => Symbol(label.clone()),
            Unary(op, input) => Unary(*op, *input + offset),
            Binary(op, lhs, rhs) => Binary(*op, *lhs + offset, *rhs + offset),
        }));
        self.nodes
            .push(Binary(op, offset - 1, self.nodes.len() - 1));
        return self;
    }

    fn unary_op(mut self, op: UnaryOp) -> Tree {
        self.nodes.push(Unary(op, self.root_index()));
        return self;
    }
}

impl core::ops::Add<Tree> for Tree {
    type Output = Tree;

    fn add(self, rhs: Tree) -> Tree {
        self.binary_op(rhs, Add)
    }
}

impl core::ops::Sub<Tree> for Tree {
    type Output = Tree;

    fn sub(self, rhs: Tree) -> Self::Output {
        self.binary_op(rhs, Subtract)
    }
}

impl core::ops::Mul<Tree> for Tree {
    type Output = Tree;

    fn mul(self, rhs: Tree) -> Tree {
        self.binary_op(rhs, Multiply)
    }
}

impl core::ops::Div<Tree> for Tree {
    type Output = Tree;

    fn div(self, rhs: Tree) -> Self::Output {
        self.binary_op(rhs, Divide)
    }
}

pub fn pow(base: Tree, exponent: Tree) -> Tree {
    base.binary_op(exponent, Pow)
}

pub fn min(lhs: Tree, rhs: Tree) -> Tree {
    lhs.binary_op(rhs, Min)
}

pub fn max(lhs: Tree, rhs: Tree) -> Tree {
    lhs.binary_op(rhs, Max)
}

impl core::ops::Neg for Tree {
    type Output = Tree;

    fn neg(self) -> Self::Output {
        self.unary_op(Negate)
    }
}

pub fn sqrt(x: Tree) -> Tree {
    x.unary_op(Sqrt)
}

pub fn abs(x: Tree) -> Tree {
    x.unary_op(Abs)
}

pub fn sin(x: Tree) -> Tree {
    x.unary_op(Sin)
}

pub fn cos(x: Tree) -> Tree {
    x.unary_op(Cos)
}

pub fn tan(x: Tree) -> Tree {
    x.unary_op(Tan)
}

pub fn log(x: Tree) -> Tree {
    x.unary_op(Log)
}

pub fn exp(x: Tree) -> Tree {
    x.unary_op(Exp)
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
        for idx in 0..self.tree.len() {
            self.write(
                idx,
                match &self.tree.nodes[idx] {
                    Constant(val) => *val,
                    Symbol(label) => match &self.regs[idx] {
                        None => return Err(EvaluationError::VariableNotFound(*label)),
                        Some(val) => *val,
                    },
                    Binary(op, lhs, rhs) => op.apply(self.read(*lhs)?, self.read(*rhs)?),
                    Unary(op, input) => op.apply(self.read(*input)?),
                },
            );
        }
        return self.read(self.tree.root_index());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deftree;

    #[test]
    fn add() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let sum = x + y;
        assert_eq!(sum.nodes, vec![Symbol('x'), Symbol('y'), Binary(Add, 0, 1)]);
    }

    #[test]
    fn multiply() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let sum = x * y;
        assert_eq!(
            sum.nodes,
            vec![Symbol('x'), Symbol('y'), Binary(Multiply, 0, 1)]
        );
    }

    #[test]
    fn subtract() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let sum = x - y;
        assert_eq!(
            sum.nodes,
            vec![Symbol('x'), Symbol('y'), Binary(Subtract, 0, 1)]
        );
    }

    #[test]
    fn divide() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let sum = x / y;
        assert_eq!(
            sum.nodes,
            vec![Symbol('x'), Symbol('y'), Binary(Divide, 0, 1)]
        );
    }

    #[test]
    fn negate() {
        let x: Tree = 'x'.into();
        let neg = -x;
        assert_eq!(neg.nodes, vec![Symbol('x'), Unary(Negate, 0)]);
    }

    #[test]
    fn sqrt_test() {
        let x: Tree = 'x'.into();
        let y = sqrt(x);
        assert_eq!(y.nodes, vec![Symbol('x'), Unary(Sqrt, 0)]);
    }

    #[test]
    fn abs_test() {
        let x: Tree = 'x'.into();
        let y = abs(x);
        assert_eq!(y.nodes, vec![Symbol('x'), Unary(Abs, 0)]);
    }

    #[test]
    fn symbol_deftree() {
        let tree = deftree!(x);
        assert_eq!(tree.len(), 1);
        assert!(matches!(tree.root(), Ok(Symbol(label)) if *label == 'x'));
    }

    #[test]
    fn constant_deftree() {
        let tree = deftree!(2.);
        assert_eq!(tree.len(), 1);
        assert!(matches!(tree.root(), Ok(Constant(val)) if *val == 2.));
    }

    #[test]
    fn negate_deftree() {
        let tree = deftree!(-x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &vec![Symbol('x'), Unary(Negate, 0)]);
    }

    #[test]
    fn sqrt_deftree() {
        let tree = deftree!(sqrt x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &vec![Symbol('x'), Unary(Sqrt, 0)]);
    }

    #[test]
    fn abs_deftree() {
        let tree = deftree!(abs x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &vec![Symbol('x'), Unary(Abs, 0)]);
    }

    #[test]
    fn sin_deftree() {
        let tree = deftree!(sin x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &vec![Symbol('x'), Unary(Sin, 0)]);
    }

    #[test]
    fn cos_deftree() {
        let tree = deftree!(cos x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &vec![Symbol('x'), Unary(Cos, 0)]);
    }

    #[test]
    fn tan_deftree() {
        let tree = deftree!(tan x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &vec![Symbol('x'), Unary(Tan, 0)]);
    }

    #[test]
    fn log_deftree() {
        let tree = deftree!(log x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &vec![Symbol('x'), Unary(Log, 0)]);
    }

    #[test]
    fn exp_deftree() {
        let tree = deftree!(exp x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &vec![Symbol('x'), Unary(Exp, 0)]);
    }

    #[test]
    fn add_deftree() {
        let tree = deftree!(+ x y);
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &vec![Symbol('x'), Symbol('y'), Binary(Add, 0, 1)]
        );
        let tree = deftree!(+ 2. (-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &vec![
                Constant(2.),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Add, 0, 2)
            ]
        );
    }

    #[test]
    fn subtract_deftree() {
        let tree = deftree!(- x y);
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &vec![Symbol('x'), Symbol('y'), Binary(Subtract, 0, 1)]
        );
        let tree = deftree!(-2.(-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &vec![
                Constant(2.),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Subtract, 0, 2)
            ]
        );
    }

    #[test]
    fn multiply_deftree() {
        let tree = deftree!(* x y);
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &vec![Symbol('x'), Symbol('y'), Binary(Multiply, 0, 1)]
        );
        let tree = deftree!(*(2.)(-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &vec![
                Constant(2.),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Multiply, 0, 2)
            ]
        );
    }

    #[test]
    fn divide_deftree() {
        let tree = deftree!(/ x y);
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &vec![Symbol('x'), Symbol('y'), Binary(Divide, 0, 1)]
        );
        let tree = deftree!(/ 2. (-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &vec![
                Constant(2.),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Divide, 0, 2)
            ]
        );
    }

    #[test]
    fn pow_deftree() {
        let tree = deftree!(pow x y);
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &vec![Symbol('x'), Symbol('y'), Binary(Pow, 0, 1)]
        );
        let tree = deftree!(pow 2. (-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &vec![
                Constant(2.),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Pow, 0, 2)
            ]
        );
    }

    #[test]
    fn min_deftree() {
        let tree = deftree!(min x y);
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &vec![Symbol('x'), Symbol('y'), Binary(Min, 0, 1)]
        );
        let tree = deftree!(min 2. (-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &vec![
                Constant(2.),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Min, 0, 2)
            ]
        );
    }

    #[test]
    fn max_deftree() {
        let tree = deftree!(max x y);
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &vec![Symbol('x'), Symbol('y'), Binary(Max, 0, 1)]
        );
        let tree = deftree!(max 2. (-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &vec![
                Constant(2.),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Max, 0, 2)
            ]
        );
    }
}
