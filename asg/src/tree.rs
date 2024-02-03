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

use {BinaryOp::*, UnaryOp::*};

/// Errors that can occur when constructing a tree.
#[derive(Debug)]
pub enum TreeError {
    /// Nodes are not in a valid topological order.
    WrongNodeOrder,
    /// A constant node contains NaN.
    ContainsNaN,
    /// Tree conains no nodes.
    EmptyTree,
    /// A mismatch between two dimensions, for example, during a reshape operation.
    DimensionMismatch((usize, usize), (usize, usize)),
}

/// Represents a node in an abstract syntax `Tree`.
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Node {
    Constant(f64),
    Symbol(char),
    Unary(UnaryOp, usize),
    Binary(BinaryOp, usize, usize),
}

use std::ops::Range;

use Node::*;

/// Represents an abstract syntax tree.
#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    nodes: Vec<Node>,
    dims: (usize, usize),
}

const fn matsize(dims: (usize, usize)) -> usize {
    dims.0 * dims.1
}

impl Tree {
    /// Create a tree representing a constant value.
    pub fn constant(val: f64) -> Tree {
        Tree {
            nodes: vec![Constant(val)],
            dims: (1, 1),
        }
    }

    /// Create a tree representing a symbol with the given `label`.
    pub fn symbol(label: char) -> Tree {
        Tree {
            nodes: vec![Symbol(label)],
            dims: (1, 1),
        }
    }

    pub fn concat(mut lhs: Tree, mut rhs: Tree) -> Tree {
        let (llen, lsize) = (lhs.len(), lhs.num_roots());
        let (rlen, rsize) = (rhs.len(), rhs.num_roots());
        {
            // Copy nodes. We're ignoring the root nodes of `lhs` when
            // computing the offset. That is fine because these root nodes will
            // later be rotated to the end of the buffer.
            let offset = llen - lsize;
            lhs.nodes.extend(rhs.nodes.drain(..).map(|n| match n {
                Constant(_) => n,
                Symbol(_) => n,
                Unary(op, input) => Unary(op, input + offset),
                Binary(op, lhs, rhs) => Binary(op, lhs + offset, rhs + offset),
            }));
        }
        // After we just concatenated the nodes as is. This rotation after the
        // concatenations makes sure all the root nodes are at the end.
        lhs.nodes[(llen - lsize)..(llen + rlen - rsize)].rotate_left(lsize);
        lhs.dims = (lsize + rsize, 1);
        return lhs;
    }

    pub fn transposed(mut self) -> Tree {
        self.dims = (self.dims.1, self.dims.0);
        return self;
    }

    /// The number of nodes in this tree.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_roots(&self) -> usize {
        matsize(self.dims)
    }

    pub fn dims(&self) -> (usize, usize) {
        self.dims
    }

    pub fn reshape(self, rows: usize, cols: usize) -> Result<Tree, TreeError> {
        if matsize((rows, cols)) == self.num_roots() {
            Ok(Tree {
                nodes: self.nodes,
                dims: (rows, cols),
            })
        } else {
            Err(TreeError::DimensionMismatch(self.dims, (rows, cols)))
        }
    }

    /// Get a reference to root of the tree. This is the last node of
    /// the tree.
    pub fn roots(&self) -> &[Node] {
        // We can confidently unwrap because we should never create an
        // invalid tree in the first place.
        &self.nodes[(self.nodes.len() - self.num_roots())..]
    }

    /// Indices of the root nodes of the tree. These nodes will be at the end of
    /// the tree.
    pub fn root_indices(&self) -> Range<usize> {
        (self.len() - self.num_roots())..self.len()
    }

    /// Get a reference to the node at `index`.
    pub fn node(&self, index: usize) -> &Node {
        &self.nodes[index]
    }

    /// Reference to the nodes of this tree.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn nodes_mut(&mut self) -> &mut Vec<Node> {
        &mut self.nodes
    }

    /// Get a unique list of all symbols in this tree. The list of
    /// chars is expected to be sorted.
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

    /// Check the tree for errors and return a Result that contains the tree if
    /// no errors were found, or the first error encountered with the tree.
    pub fn validated(self) -> Result<Tree, TreeError> {
        if self.nodes.is_empty() {
            return Err(TreeError::EmptyTree);
        }
        for i in 0..self.nodes.len() {
            match &self.nodes[i] {
                Constant(val) if f64::is_nan(*val) => return Err(TreeError::ContainsNaN),
                Unary(_, input) if *input >= i => return Err(TreeError::WrongNodeOrder),
                Binary(_, l, r) if *l >= i || *r >= i => return Err(TreeError::WrongNodeOrder),
                Symbol(_) | _ => {} // Do nothing.
            }
        }
        // Maybe add more checks later.
        return Ok(self);
    }

    fn binary_op(mut self, other: Tree, op: BinaryOp) -> Tree {
        if self.num_roots() > other.num_roots() {
            return other.binary_op(self, op);
        }
        let offset: usize = self.nodes.len();
        self.nodes.reserve(self.nodes.len() + other.nodes.len() + 1);
        self.nodes.extend(other.nodes.iter().map(|node| match node {
            Constant(value) => Constant(*value),
            Symbol(label) => Symbol(label.clone()),
            Unary(op, input) => Unary(*op, *input + offset),
            Binary(op, lhs, rhs) => Binary(*op, *lhs + offset, *rhs + offset),
        }));
        if self.num_roots() == 1 {
            for r in other.root_indices() {
                self.nodes.push(Binary(op, offset - 1, r + offset));
            }
        } else {
            for (l, r) in ((offset - self.num_roots())..offset).zip(other.root_indices()) {
                self.nodes.push(Binary(op, l, r + offset));
            }
        }
        return self;
    }

    fn unary_op(mut self, op: UnaryOp) -> Tree {
        for root in self.root_indices() {
            self.nodes.push(Unary(op, root));
        }
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
            (Constant(_), Unary(..)) => Some(Less),
            (Constant(_), Binary(..)) => Some(Less),
            // Symbol
            (Symbol(_), Constant(_)) => Some(Greater),
            (Symbol(a), Symbol(b)) => Some(a.cmp(b)),
            (Symbol(_), Unary(..)) => Some(Less),
            (Symbol(_), Binary(..)) => Some(Less),
            // Unary
            (Unary(..), Constant(_)) => Some(Greater),
            (Unary(..), Symbol(_)) => Some(Greater),
            (Unary(op1, _), Unary(op2, _)) => Some(op1.index().cmp(&op2.index())),
            (Unary(..), Binary(..)) => Some(Less),
            // Binary
            (Binary(..), Constant(_)) => Some(Greater),
            (Binary(..), Symbol(_)) => Some(Greater),
            (Binary(..), Unary(..)) => Some(Greater),
            (Binary(op1, ..), Binary(op2, ..)) => Some(op1.index().cmp(&op2.index())),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::deftree;

    use super::*;

    #[test]
    fn t_add() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let sum = x + y;
        assert_eq!(sum.nodes, vec![Symbol('x'), Symbol('y'), Binary(Add, 0, 1)]);
    }

    #[test]
    fn t_multiply() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let sum = x * y;
        assert_eq!(
            sum.nodes,
            vec![Symbol('x'), Symbol('y'), Binary(Multiply, 0, 1)]
        );
    }

    #[test]
    fn t_subtract() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let sum = x - y;
        assert_eq!(
            sum.nodes,
            vec![Symbol('x'), Symbol('y'), Binary(Subtract, 0, 1)]
        );
    }

    #[test]
    fn t_divide() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let sum = x / y;
        assert_eq!(
            sum.nodes,
            vec![Symbol('x'), Symbol('y'), Binary(Divide, 0, 1)]
        );
    }

    #[test]
    fn t_pow() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let p = pow(x, y);
        assert_eq!(p.nodes, vec![Symbol('x'), Symbol('y'), Binary(Pow, 0, 1)]);
    }

    #[test]
    fn t_min() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let m = min(x, y);
        assert_eq!(m.nodes, vec![Symbol('x'), Symbol('y'), Binary(Min, 0, 1)]);
    }

    #[test]
    fn t_max() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let m = max(x, y);
        assert_eq!(m.nodes, vec![Symbol('x'), Symbol('y'), Binary(Max, 0, 1)]);
    }

    #[test]
    fn t_negate() {
        let x: Tree = 'x'.into();
        let neg = -x;
        assert_eq!(neg.nodes, vec![Symbol('x'), Unary(Negate, 0)]);
    }

    #[test]
    fn t_sqrt_test() {
        let x: Tree = 'x'.into();
        let y = sqrt(x);
        assert_eq!(y.nodes, vec![Symbol('x'), Unary(Sqrt, 0)]);
    }

    #[test]
    fn t_abs_test() {
        let x: Tree = 'x'.into();
        let y = abs(x);
        assert_eq!(y.nodes, vec![Symbol('x'), Unary(Abs, 0)]);
    }

    #[test]
    fn t_sin_test() {
        let x: Tree = 'x'.into();
        let y = sin(x);
        assert_eq!(y.nodes, vec![Symbol('x'), Unary(Sin, 0)]);
    }

    #[test]
    fn t_cos_test() {
        let x: Tree = 'x'.into();
        let y = cos(x);
        assert_eq!(y.nodes, vec![Symbol('x'), Unary(Cos, 0)]);
    }

    #[test]
    fn t_tan_test() {
        let x: Tree = 'x'.into();
        let y = tan(x);
        assert_eq!(y.nodes, vec![Symbol('x'), Unary(Tan, 0)]);
    }

    #[test]
    fn t_log_test() {
        let x: Tree = 'x'.into();
        let y = log(x);
        assert_eq!(y.nodes, vec![Symbol('x'), Unary(Log, 0)]);
    }

    #[test]
    fn t_exp_test() {
        let x: Tree = 'x'.into();
        let y = exp(x);
        assert_eq!(y.nodes, vec![Symbol('x'), Unary(Exp, 0)]);
    }

    #[test]
    fn t_element_wise_unary_op() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let p = Tree::concat(x, y);
        assert_eq!(p.dims, (2, 1));
        assert_eq!(p.nodes, vec![Symbol('x'), Symbol('y')]);
        let p = p * 2.0.into();
        assert_eq!(
            p.nodes,
            vec![
                Constant(2.),
                Symbol('x'),
                Symbol('y'),
                Binary(Multiply, 0, 1),
                Binary(Multiply, 0, 2)
            ]
        );
    }

    #[test]
    fn t_element_wise_binary_op() {
        // Matrix and a scalar.
        let tree = Tree::concat(Tree::concat('x'.into(), 'y'.into()), 'z'.into()) * 2.0.into();
        let expected = vec![
            Constant(2.),
            Symbol('x'),
            Symbol('y'),
            Symbol('z'),
            Binary(Multiply, 0, 1),
            Binary(Multiply, 0, 2),
            Binary(Multiply, 0, 3),
        ];
        assert_eq!(tree.nodes, expected);
        // Scalar and a matrix
        let tree =
            Tree::constant(2.) * Tree::concat(Tree::concat('x'.into(), 'y'.into()), 'z'.into());
        assert_eq!(tree.nodes, expected);
        // Matrix and a matrix - multiply
        let tree = Tree::concat(Tree::concat('x'.into(), 'y'.into()), 'z'.into())
            * Tree::concat(Tree::concat('a'.into(), 'b'.into()), 'c'.into());
        assert_eq!(
            tree.nodes,
            vec![
                Symbol('x'),
                Symbol('y'),
                Symbol('z'),
                Symbol('a'),
                Symbol('b'),
                Symbol('c'),
                Binary(Multiply, 0, 3),
                Binary(Multiply, 1, 4),
                Binary(Multiply, 2, 5),
            ]
        );
        // Matrix and a matrix - add.
        let tree = Tree::concat(Tree::concat('x'.into(), 'y'.into()), 'z'.into())
            + Tree::concat(Tree::concat('a'.into(), 'b'.into()), 'c'.into());
        assert_eq!(
            tree.nodes,
            vec![
                Symbol('x'),
                Symbol('y'),
                Symbol('z'),
                Symbol('a'),
                Symbol('b'),
                Symbol('c'),
                Binary(Add, 0, 3),
                Binary(Add, 1, 4),
                Binary(Add, 2, 5),
            ]
        );
        // Matrices of different sizes.
        let tree = Tree::concat(Tree::concat('x'.into(), 'y'.into()), 'z'.into())
            * Tree::concat('a'.into(), 'b'.into());
        let expected = vec![
            Symbol('a'),
            Symbol('b'),
            Symbol('x'),
            Symbol('y'),
            Symbol('z'),
            Binary(Multiply, 0, 2),
            Binary(Multiply, 1, 3),
        ];
        assert_eq!(tree.nodes, expected);
        let tree = Tree::concat('a'.into(), 'b'.into())
            * Tree::concat('x'.into(), Tree::concat('y'.into(), 'z'.into()));
        assert_eq!(tree.nodes, expected);
    }

    #[test]
    fn t_reshape() {
        let mat = deftree!(concat a b c p q r x y z).reshape(3, 3).unwrap();
        assert_eq!(mat.dims(), (3, 3));
        let mat = mat.reshape(1, 9).unwrap();
        assert_eq!(mat.dims(), (1, 9));
        let mat = mat.reshape(9, 1).unwrap();
        assert_eq!(mat.dims(), (9, 1));
        let result = mat.reshape(7, 3);
        matches!(result, Err(TreeError::DimensionMismatch((9, 1), (7, 3))));
    }
}
