/// Represents an operation with one input.
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

/// Represents an operation with two inputs.
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
    /// The index of the variant for comparison and sorting.
    pub fn index(&self) -> u8 {
        use UnaryOp::*;
        match self {
            Negate => 0,
            Sqrt => 1,
            Abs => 2,
            Sin => 3,
            Cos => 4,
            Tan => 5,
            Log => 6,
            Exp => 7,
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

    /// The index of the variant for comparison and sorting.
    pub fn index(&self) -> u8 {
        use BinaryOp::*;
        match self {
            Add => 0,
            Subtract => 1,
            Multiply => 2,
            Divide => 3,
            Pow => 4,
            Min => 5,
            Max => 6,
        }
    }

    /// Check if the binary op is commutative.
    pub fn is_commutative(&self) -> bool {
        use BinaryOp::*;
        match self {
            Add => true,
            Subtract => false,
            Multiply => true,
            Divide => false,
            Pow => false,
            Min => true,
            Max => true,
        }
    }
}

use crate::{
    dedup::Deduplicater,
    fold::fold_constants,
    parser::{parse_tree, LispParseError},
    prune::Pruner,
    walk::{DepthWalker, NodeOrdering},
};
use BinaryOp::*;
use UnaryOp::*;

/// Errors that can occur when constructing a tree.
#[derive(Debug)]
pub enum TreeError {
    /// Nodes are not in a valid topological order.
    WrongNodeOrder,
    /// A constant node contains NaN.
    ContainsNaN,
    /// Tree conains no nodes.
    EmptyTree,
}

/// Represents a node in an abstract syntax `Tree`.
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

    /// Create a new tree with `nodes`. If the `nodes` don't meet the
    /// requirements for represeting a standart tree, an appropriate
    /// `TreeError` is returned.
    pub fn from_nodes(nodes: Vec<Node>) -> Result<Tree, TreeError> {
        Ok(Tree {
            nodes: Self::validate_nodes(nodes)?,
        })
    }

    /// Parse the `lisp` expression into a new tree. If the parsing
    /// fails, an appropriate `LispParseError` is returned.
    pub fn from_lisp(lisp: &str) -> Result<Tree, LispParseError> {
        parse_tree(lisp)
    }

    /// The number of nodes in this tree.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Get a reference to root of the tree. This is the last node of
    /// the tree.
    pub fn root(&self) -> &Node {
        // We can confidently unwrap because we should never create an
        // invalid tree in the first place.
        self.nodes.last().unwrap()
    }

    /// Index of the root node. This will be the index of the last
    /// node of the tree.
    pub fn root_index(&self) -> usize {
        self.len() - 1
    }

    /// Get a reference to the node at `index`.
    pub fn node(&self, index: usize) -> &Node {
        &self.nodes[index]
    }

    /// Reference to the nodes of this tree.
    pub fn nodes(&self) -> &Vec<Node> {
        &self.nodes
    }

    /// Drop this tree and take ownership of the nodes. The nodes are
    /// returned.
    pub fn take_nodes(self) -> Vec<Node> {
        self.nodes
    }

    /// Get a unique list of all symbols in this tree.
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

    /// Fold constants in this tree.
    pub fn fold_constants(mut self) -> Result<Tree, TreeError> {
        let mut pruner = Pruner::new();
        let root_index = self.root_index();
        self.nodes = Self::validate_nodes(pruner.run(fold_constants(self.nodes), root_index))?;
        return Ok(self);
    }

    /// Deduplicate the common subtrees in this tree.
    pub fn deduplicate(mut self) -> Result<Tree, TreeError> {
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        let root_index = self.root_index();
        self.nodes = Self::validate_nodes(pruner.run(dedup.run(self.nodes), root_index))?;
        return Ok(self);
    }

    /// Check if `nodes` can represent a valid tree.
    fn validate_nodes(nodes: Vec<Node>) -> Result<Vec<Node>, TreeError> {
        if nodes.is_empty() {
            return Err(TreeError::EmptyTree);
        }
        for i in 0..nodes.len() {
            match &nodes[i] {
                Constant(val) if f64::is_nan(*val) => return Err(TreeError::ContainsNaN),
                Unary(_, input) if *input >= i => return Err(TreeError::WrongNodeOrder),
                Binary(_, l, r) if *l >= i || *r >= i => return Err(TreeError::WrongNodeOrder),
                Symbol(_) | _ => {} // Do nothing.
            }
        }
        // Maybe add more checks later.
        return Ok(nodes);
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

/// Construct a tree that represents raising `base` to the power of
/// `exponent`.
pub fn pow(base: Tree, exponent: Tree) -> Tree {
    base.binary_op(exponent, Pow)
}

/// Construct a tree that represents the smaller of `lhs` and `rhs`.
pub fn min(lhs: Tree, rhs: Tree) -> Tree {
    lhs.binary_op(rhs, Min)
}

/// Construct a tree that represents the larger of `lhs` and `rhs`.
pub fn max(lhs: Tree, rhs: Tree) -> Tree {
    lhs.binary_op(rhs, Max)
}

impl core::ops::Neg for Tree {
    type Output = Tree;

    fn neg(self) -> Self::Output {
        self.unary_op(Negate)
    }
}

/// Construct a tree representing the square root of `x`.
pub fn sqrt(x: Tree) -> Tree {
    x.unary_op(Sqrt)
}

/// Construct a tree representing the absolute value of `x`.
pub fn abs(x: Tree) -> Tree {
    x.unary_op(Abs)
}

/// Construct a tree representing the sine of `x`.
pub fn sin(x: Tree) -> Tree {
    x.unary_op(Sin)
}

/// Construct a tree representing the cosine of `x`.
pub fn cos(x: Tree) -> Tree {
    x.unary_op(Cos)
}

/// Construct a tree representing the tangent of 'x'.
pub fn tan(x: Tree) -> Tree {
    x.unary_op(Tan)
}

/// Construct a tree representing the natural logarithm of `x`.
pub fn log(x: Tree) -> Tree {
    x.unary_op(Log)
}

/// Construct a tree representing `e` raised to the power of `x`.
pub fn exp(x: Tree) -> Tree {
    x.unary_op(Exp)
}

impl From<f64> for Tree {
    fn from(value: f64) -> Self {
        return Self::constant(value);
    }
}

impl From<char> for Tree {
    fn from(c: char) -> Self {
        return Self::symbol(c);
    }
}

impl std::fmt::Display for Tree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        enum Token {
            Branch,
            Pass,
            Turn,
            Gap,
            Newline,
            NodeIndex(usize),
        }
        use Token::*;
        // Walk the tree and collect tokens.
        let tokens = {
            // First pass of collecting tokens with no branching.
            let mut tokens = {
                let mut tokens: Vec<Token> = Vec::with_capacity(self.len()); // Likely need more memory.
                let mut walker = DepthWalker::new();
                let mut node_depths: Box<[usize]> = vec![0; self.len()].into_boxed_slice();
                for (index, parent) in walker.walk_tree(self, false, NodeOrdering::Original) {
                    if let Some(pi) = parent {
                        node_depths[index] = node_depths[pi] + 1;
                    }
                    let depth = node_depths[index];
                    if depth > 0 {
                        for _ in 0..(depth - 1) {
                            tokens.push(Gap);
                        }
                        tokens.push(Turn);
                    }
                    tokens.push(NodeIndex(index));
                    tokens.push(Newline);
                }
                tokens
            };
            // Insert branching tokens where necessary.
            let mut line_start: usize = 0;
            for i in 0..tokens.len() {
                match tokens[i] {
                    Branch | Pass | Gap | NodeIndex(_) => {} // Do nothing.
                    Newline => line_start = i,
                    Turn => {
                        let offset = i - line_start;
                        for li in (0..line_start).rev() {
                            if let Newline = tokens[li] {
                                let ti = li + offset;
                                tokens[ti] = match &tokens[ti] {
                                    Branch | Pass | NodeIndex(_) => break,
                                    Turn => Branch,
                                    Gap => Pass,
                                    Newline => panic!("FATAL: Failed to convert tree to a string"),
                                }
                            }
                        }
                    }
                }
            }
            tokens
        };
        // Write all the tokens out.
        write!(f, "\n")?;
        for token in tokens.iter() {
            match token {
                Branch => write!(f, " ├── ")?,
                Pass => write!(f, " │   ")?,
                Turn => write!(f, " └── ")?,
                Gap => write!(f, "     ")?,
                Newline => write!(f, "\n")?,
                NodeIndex(index) => write!(f, "[{}] {}", *index, &self.node(*index))?,
            };
        }
        write!(f, "\n")
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Constant(value) => write!(f, "Constant({})", value),
            Symbol(label) => write!(f, "Symbol({})", label),
            Unary(op, input) => write!(f, "{:?}({})", op, input),
            Binary(op, lhs, rhs) => write!(f, "{:?}({}, {})", op, lhs, rhs),
        }
    }
}

impl PartialOrd for Node {
    /// This implementation only accounts for the node, its type and
    /// the data held inside the node. It DOES NOT take into account
    /// the children of the node when comparing two nodes.
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering::*;
        match (self, other) {
            // Constant
            (Constant(a), Constant(b)) => a.partial_cmp(b),
            (Constant(_), Symbol(_)) => Some(Less),
            (Constant(_), Unary(_, _)) => Some(Less),
            (Constant(_), Binary(_, _, _)) => Some(Less),
            // Symbol
            (Symbol(_), Constant(_)) => Some(Greater),
            (Symbol(a), Symbol(b)) => Some(a.cmp(b)),
            (Symbol(_), Unary(_, _)) => Some(Less),
            (Symbol(_), Binary(_, _, _)) => Some(Less),
            // Unary
            (Unary(_, _), Constant(_)) => Some(Greater),
            (Unary(_, _), Symbol(_)) => Some(Greater),
            (Unary(op1, _), Unary(op2, _)) => Some(op1.index().cmp(&op2.index())),
            (Unary(_, _), Binary(_, _, _)) => Some(Less),
            // Binary
            (Binary(_, _, _), Constant(_)) => Some(Greater),
            (Binary(_, _, _), Symbol(_)) => Some(Greater),
            (Binary(_, _, _), Unary(_, _)) => Some(Greater),
            (Binary(op1, _, _), Binary(op2, _, _)) => Some(op1.index().cmp(&op2.index())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
