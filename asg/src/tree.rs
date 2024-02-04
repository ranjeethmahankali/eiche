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
#[derive(Debug, PartialEq, Copy, Clone)]
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
    Scalar(f64),
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

pub type MaybeTree = Result<Tree, TreeError>;

const fn matsize(dims: (usize, usize)) -> usize {
    dims.0 * dims.1
}

impl Tree {
    /// Create a tree representing a constant value.
    pub fn constant(val: f64) -> Tree {
        Tree {
            nodes: vec![Scalar(val)],
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

    pub fn concat(lhs: MaybeTree, rhs: MaybeTree) -> MaybeTree {
        let mut lhs = lhs?;
        let mut rhs = rhs?;
        let (llen, lsize) = (lhs.len(), lhs.num_roots());
        let (rlen, rsize) = (rhs.len(), rhs.num_roots());
        {
            // Copy nodes. We're ignoring the root nodes of `lhs` when
            // computing the offset. That is fine because these root nodes will
            // later be rotated to the end of the buffer.
            let offset = llen - lsize;
            lhs.nodes.extend(rhs.nodes.drain(..).map(|n| match n {
                Scalar(_) => n,
                Symbol(_) => n,
                Unary(op, input) => Unary(op, input + offset),
                Binary(op, lhs, rhs) => Binary(op, lhs + offset, rhs + offset),
            }));
        }
        // After we just concatenated the nodes as is. This rotation after the
        // concatenations makes sure all the root nodes are at the end.
        lhs.nodes[(llen - lsize)..(llen + rlen - rsize)].rotate_left(lsize);
        lhs.dims = (lsize + rsize, 1);
        return Ok(lhs);
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

    pub fn reshape(self, rows: usize, cols: usize) -> MaybeTree {
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
    pub fn validated(self) -> MaybeTree {
        if self.nodes.is_empty() {
            return Err(TreeError::EmptyTree);
        }
        for i in 0..self.nodes.len() {
            match &self.nodes[i] {
                Scalar(val) if f64::is_nan(*val) => return Err(TreeError::ContainsNaN),
                Unary(_, input) if *input >= i => return Err(TreeError::WrongNodeOrder),
                Binary(_, l, r) if *l >= i || *r >= i => return Err(TreeError::WrongNodeOrder),
                Symbol(_) | _ => {} // Do nothing.
            }
        }
        // Maybe add more checks later.
        return Ok(self);
    }

    fn binary_op(mut self, other: Tree, op: BinaryOp) -> MaybeTree {
        let nroots = self.num_roots();
        let other_nroots = other.num_roots();
        if nroots > other_nroots {
            return other.binary_op(self, op);
        } else if nroots != 1 && nroots != other_nroots {
            return Err(TreeError::DimensionMismatch(self.dims, other.dims));
        }
        let offset: usize = self.nodes.len();
        self.nodes.reserve(self.nodes.len() + other.nodes.len() + 1);
        self.nodes.extend(other.nodes.iter().map(|node| match node {
            Scalar(value) => Scalar(*value),
            Symbol(label) => Symbol(label.clone()),
            Unary(op, input) => Unary(*op, *input + offset),
            Binary(op, lhs, rhs) => Binary(*op, *lhs + offset, *rhs + offset),
        }));
        if nroots == 1 {
            for r in other.root_indices() {
                self.nodes.push(Binary(op, offset - 1, r + offset));
            }
        } else {
            for (l, r) in ((offset - nroots)..offset).zip(other.root_indices()) {
                self.nodes.push(Binary(op, l, r + offset));
            }
        }
        return Ok(self);
    }

    fn unary_op(mut self, op: UnaryOp) -> MaybeTree {
        for root in self.root_indices() {
            self.nodes.push(Unary(op, root));
        }
        return Ok(self);
    }
}

pub fn add(lhs: MaybeTree, rhs: MaybeTree) -> MaybeTree {
    lhs?.binary_op(rhs?, Add)
}

pub fn sub(lhs: MaybeTree, rhs: MaybeTree) -> MaybeTree {
    lhs?.binary_op(rhs?, Subtract)
}

pub fn mul(lhs: MaybeTree, rhs: MaybeTree) -> MaybeTree {
    lhs?.binary_op(rhs?, Multiply)
}

pub fn div(lhs: MaybeTree, rhs: MaybeTree) -> MaybeTree {
    lhs?.binary_op(rhs?, Divide)
}

/// Construct a tree that represents raising `base` to the power of
/// `exponent`.
pub fn pow(base: MaybeTree, exponent: MaybeTree) -> MaybeTree {
    base?.binary_op(exponent?, Pow)
}

/// Construct a tree that represents the smaller of `lhs` and `rhs`.
pub fn min(lhs: MaybeTree, rhs: MaybeTree) -> MaybeTree {
    lhs?.binary_op(rhs?, Min)
}

/// Construct a tree that represents the larger of `lhs` and `rhs`.
pub fn max(lhs: MaybeTree, rhs: MaybeTree) -> MaybeTree {
    lhs?.binary_op(rhs?, Max)
}

pub fn negate(tree: MaybeTree) -> MaybeTree {
    tree?.unary_op(Negate)
}

/// Construct a tree representing the square root of `x`.
pub fn sqrt(x: MaybeTree) -> MaybeTree {
    x?.unary_op(Sqrt)
}

/// Construct a tree representing the absolute value of `x`.
pub fn abs(x: MaybeTree) -> MaybeTree {
    x?.unary_op(Abs)
}

/// Construct a tree representing the sine of `x`.
pub fn sin(x: MaybeTree) -> MaybeTree {
    x?.unary_op(Sin)
}

/// Construct a tree representing the cosine of `x`.
pub fn cos(x: MaybeTree) -> MaybeTree {
    x?.unary_op(Cos)
}

/// Construct a tree representing the tangent of 'x'.
pub fn tan(x: MaybeTree) -> MaybeTree {
    x?.unary_op(Tan)
}

/// Construct a tree representing the natural logarithm of `x`.
pub fn log(x: MaybeTree) -> MaybeTree {
    x?.unary_op(Log)
}

/// Construct a tree representing `e` raised to the power of `x`.
pub fn exp(x: MaybeTree) -> MaybeTree {
    x?.unary_op(Exp)
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
            // Scalar
            (Scalar(a), Scalar(b)) => a.partial_cmp(b),
            (Scalar(_), Symbol(_)) => Some(Less),
            (Scalar(_), Unary(..)) => Some(Less),
            (Scalar(_), Binary(..)) => Some(Less),
            // Symbol
            (Symbol(_), Scalar(_)) => Some(Greater),
            (Symbol(a), Symbol(b)) => Some(a.cmp(b)),
            (Symbol(_), Unary(..)) => Some(Less),
            (Symbol(_), Binary(..)) => Some(Less),
            // Unary
            (Unary(..), Scalar(_)) => Some(Greater),
            (Unary(..), Symbol(_)) => Some(Greater),
            (Unary(op1, _), Unary(op2, _)) => Some(op1.index().cmp(&op2.index())),
            (Unary(..), Binary(..)) => Some(Less),
            // Binary
            (Binary(..), Scalar(_)) => Some(Greater),
            (Binary(..), Symbol(_)) => Some(Greater),
            (Binary(..), Unary(..)) => Some(Greater),
            (Binary(op1, ..), Binary(op2, ..)) => Some(op1.index().cmp(&op2.index())),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::deftree;

    #[test]
    fn t_element_wise_unary_op() {
        let p = deftree!(* 2 (concat x y)).unwrap();
        assert_eq!(
            p.nodes,
            vec![
                Scalar(2.),
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
        let tree = deftree!(* 2 (concat x y z)).unwrap();
        let expected = vec![
            Scalar(2.),
            Symbol('x'),
            Symbol('y'),
            Symbol('z'),
            Binary(Multiply, 0, 1),
            Binary(Multiply, 0, 2),
            Binary(Multiply, 0, 3),
        ];
        assert_eq!(tree.nodes, expected);
        // Scalar and a matrix
        let tree = deftree!(* 2 (concat x y z)).unwrap();
        assert_eq!(tree.nodes, expected);
        // Matrix and a matrix - multiply
        let tree = deftree!(* (concat x y z) (concat a b c)).unwrap();
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
        let tree = deftree!(+ (concat x y z) (concat a b c)).unwrap();
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
        matches!(
            mul(
                Tree::concat(Tree::concat(Ok('x'.into()), Ok('y'.into())), Ok('z'.into())),
                Tree::concat(Ok('a'.into()), Ok('b'.into())),
            ),
            Err(TreeError::DimensionMismatch((2, 1), (3, 1)))
        );
        matches!(
            mul(
                Tree::concat(Ok('a'.into()), Ok('b'.into())),
                Tree::concat(Ok('x'.into()), Tree::concat(Ok('y'.into()), Ok('z'.into()))),
            ),
            Err(TreeError::DimensionMismatch((2, 1), (3, 1)))
        );
    }

    #[test]
    fn t_reshape() {
        let mat = deftree!(concat a b c p q r x y z)
            .unwrap()
            .reshape(3, 3)
            .unwrap();
        assert_eq!(mat.dims(), (3, 3));
        let mat = mat.reshape(1, 9).unwrap();
        assert_eq!(mat.dims(), (1, 9));
        let mat = mat.reshape(9, 1).unwrap();
        assert_eq!(mat.dims(), (9, 1));
        matches!(
            mat.reshape(7, 3),
            Err(TreeError::DimensionMismatch((9, 1), (7, 3)))
        );
    }
}
