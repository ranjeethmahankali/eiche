use crate::{dedup::Deduplicater, error::Error, fold::fold_nodes, prune::Pruner, sort::TopoSorter};
use std::ops::Range;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    Scalar(f64),
}
use Value::*;

/// Represents an operation with one input.
#[derive(Debug, Copy, Clone, PartialEq, Hash)]
pub enum UnaryOp {
    // Scalar
    Negate,
    Sqrt,
    Abs,
    Sin,
    Cos,
    Tan,
    Log,
    Exp,
    // Boolean
    Not,
}

/// Represents an operation with two inputs.
#[derive(Debug, Copy, Clone, PartialEq, Hash)]
pub enum BinaryOp {
    // Scalar
    Add,
    Subtract,
    Multiply,
    Divide,
    Pow,
    Min,
    Max,
    // Boolean
    Less,
    LessOrEqual,
    Equal,
    NotEqual,
    Greater,
    GreaterOrEqual,
    And,
    Or,
}

#[derive(Debug, Copy, Clone, PartialEq, Hash)]
pub enum TernaryOp {
    Choose,
}

impl UnaryOp {
    /// The index of the variant for comparison and sorting.
    pub fn index(&self) -> u8 {
        use UnaryOp::*;
        match self {
            // Scalar
            Negate => 0,
            Sqrt => 1,
            Abs => 2,
            Sin => 3,
            Cos => 4,
            Tan => 5,
            Log => 6,
            Exp => 7,
            // Boolean
            Not => 8,
        }
    }
}

impl BinaryOp {
    /// The index of the variant for comparison and sorting.
    pub fn index(&self) -> u8 {
        use BinaryOp::*;
        match self {
            // Scalar
            Add => 0,
            Subtract => 1,
            Multiply => 2,
            Divide => 3,
            Pow => 4,
            Min => 5,
            Max => 6,
            // Boolean
            Less => 7,
            LessOrEqual => 8,
            Equal => 9,
            NotEqual => 10,
            Greater => 11,
            GreaterOrEqual => 12,
            And => 13,
            Or => 14,
        }
    }

    /// Check if the binary op is commutative.
    pub fn is_commutative(&self) -> bool {
        use BinaryOp::*;
        match self {
            // Scalar
            Add => true,
            Subtract => false,
            Multiply => true,
            Divide => false,
            Pow => false,
            Min => true,
            Max => true,
            // Boolean
            Less => false,
            LessOrEqual => false,
            Equal => true,
            NotEqual => true,
            Greater => false,
            GreaterOrEqual => false,
            And => true,
            Or => true,
        }
    }
}

impl TernaryOp {
    pub fn index(&self) -> u8 {
        use TernaryOp::*;
        match self {
            Choose => 0,
        }
    }
}

use {BinaryOp::*, TernaryOp::*, UnaryOp::*};

/// Represents a node in an abstract syntax `Tree`.
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Node {
    Constant(Value),
    Symbol(char),
    Unary(UnaryOp, usize),
    Binary(BinaryOp, usize, usize),
    Ternary(TernaryOp, usize, usize, usize),
}

use Node::*;

/// Represents an abstract syntax tree.
#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    nodes: Vec<Node>,
    dims: (usize, usize),
}

pub type MaybeTree = Result<Tree, Error>;

const fn matsize(dims: (usize, usize)) -> usize {
    dims.0 * dims.1
}

impl Tree {
    pub fn from_nodes(nodes: Vec<Node>, dims: (usize, usize)) -> MaybeTree {
        let t = Tree { nodes, dims };
        return t.validated();
    }

    /// Create a tree representing a constant value.
    pub fn constant(val: Value) -> Tree {
        Tree {
            nodes: vec![Constant(val)],
            dims: (1, 1),
        }
    }

    pub fn compacted(mut self) -> MaybeTree {
        let mut topo_sorter = TopoSorter::new();
        let root_indices = self.root_indices();
        topo_sorter.run_from_range(self.nodes_mut(), root_indices)?;
        fold_nodes(self.nodes_mut())?;
        let mut deduper = Deduplicater::new();
        deduper.run(self.nodes_mut());
        let mut pruner = Pruner::new();
        let root_indices = self.root_indices();
        pruner.run_from_range(self.nodes_mut(), root_indices);
        return self.validated();
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
                Constant(_) => n,
                Symbol(_) => n,
                Unary(op, input) => Unary(op, input + offset),
                Binary(op, lhs, rhs) => Binary(op, lhs + offset, rhs + offset),
                Ternary(op, a, b, c) => Ternary(op, a + offset, b + offset, c + offset),
            }));
        }
        // After we just concatenated the nodes as is. This rotation after the
        // concatenations makes sure all the root nodes are at the end.
        lhs.nodes[(llen - lsize)..(llen + rlen - rsize)].rotate_left(lsize);
        lhs.dims = (lsize + rsize, 1);
        return Ok(lhs);
    }

    pub fn piecewise(cond: MaybeTree, iftrue: MaybeTree, iffalse: MaybeTree) -> MaybeTree {
        return cond?.ternary_op(iftrue?, iffalse?, Choose);
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

    pub fn with_dims(mut self, rows: usize, cols: usize) -> MaybeTree {
        self.dims = (rows, cols);
        return self.validated();
    }

    pub fn reshape(self, rows: usize, cols: usize) -> MaybeTree {
        if matsize((rows, cols)) == self.num_roots() {
            Ok(Tree {
                nodes: self.nodes,
                dims: (rows, cols),
            })
        } else {
            Err(Error::DimensionMismatch(self.dims, (rows, cols)))
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

    /// Get a unique list of all symbols in this tree. The symbols will appear
    /// in the same order as they first appear in the tree.
    pub fn symbols(&self) -> Vec<char> {
        let symbols: Vec<_> = self
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
        let mut indices: Vec<usize> = (0..symbols.len()).collect();
        indices.sort_by(|a, b| symbols[*a].cmp(&symbols[*b]));
        indices.dedup_by(|a, b| symbols[*a] == symbols[*b]);
        indices.sort();
        return indices.iter().map(|i| symbols[*i]).collect();
    }

    /// Check the tree for errors and return a Result that contains the tree if
    /// no errors were found, or the first error encountered with the tree.
    pub fn validated(self) -> MaybeTree {
        if self.nodes.is_empty() {
            return Err(Error::EmptyTree);
        }
        // Make sure nodes only depend on the nodes that came before them.
        let roots = self.root_indices();
        for i in 0..self.nodes.len() {
            match &self.nodes[i] {
                Constant(val) => match val {
                    Scalar(val) => {
                        if f64::is_nan(*val) {
                            return Err(Error::ContainsNaN);
                        }
                    }
                    Bool(_) => {} // Do nothing.
                },
                Symbol(_) => {} // Do nothing.
                Unary(_, input) => {
                    if *input >= i {
                        return Err(Error::WrongNodeOrder);
                    }
                    if roots.contains(input) {
                        return Err(Error::DependentRootNodes);
                    }
                }
                Binary(_, l, r) => {
                    if *l >= i || *r >= i {
                        return Err(Error::WrongNodeOrder);
                    }
                    if roots.contains(l) || roots.contains(r) {
                        return Err(Error::WrongNodeOrder);
                    }
                }
                Ternary(_, a, b, c) => {
                    if *a >= i || *b >= i || *c >= i {
                        return Err(Error::WrongNodeOrder);
                    }
                    if roots.contains(a) || roots.contains(b) || roots.contains(c) {
                        return Err(Error::DependentRootNodes);
                    }
                }
            }
        }
        // Maybe add more checks later.
        return Ok(self);
    }

    fn unary_op(mut self, op: UnaryOp) -> MaybeTree {
        for root in self.root_indices() {
            self.nodes.push(Unary(op, root));
        }
        return Ok(self);
    }

    fn binary_op(mut self, other: Tree, op: BinaryOp) -> MaybeTree {
        let nroots = self.num_roots();
        let other_nroots = other.num_roots();
        if nroots != 1 && other_nroots != 1 && nroots != other_nroots {
            return Err(Error::DimensionMismatch(self.dims, other.dims));
        }
        self.nodes
            .reserve(self.nodes.len() + other.nodes.len() + usize::max(nroots, other_nroots));
        let offset = self.push_nodes(&other);
        if nroots == 1 {
            let root = offset - 1;
            for r in other.root_indices() {
                self.nodes.push(Binary(op, root, r + offset));
            }
        } else if other_nroots == 1 {
            let root = self.len() - 1;
            for r in (offset - nroots)..offset {
                self.nodes.push(Binary(op, r, root));
            }
        } else {
            for (l, r) in ((offset - nroots)..offset).zip(other.root_indices()) {
                self.nodes.push(Binary(op, l, r + offset));
            }
        }
        return Ok(self);
    }

    fn ternary_op(mut self, a: Tree, b: Tree, op: TernaryOp) -> MaybeTree {
        let anroots = a.num_roots();
        let bnroots = b.num_roots();
        if anroots != bnroots {
            return Err(Error::DimensionMismatch(a.dims, b.dims));
        }
        let nroots = self.num_roots();
        if nroots != 1 && nroots != anroots {
            return Err(Error::DimensionMismatch(self.dims, a.dims));
        }
        self.nodes
            .reserve(self.nodes.len() + a.nodes.len() + b.nodes.len() + 1);
        let a_offset = self.push_nodes(&a);
        let b_offset = self.push_nodes(&b);
        if nroots == 1 {
            let root = a_offset - 1;
            for (ar, br) in a.root_indices().zip(b.root_indices()) {
                self.nodes
                    .push(Ternary(op, root, ar + a_offset, br + b_offset));
            }
        } else {
            for (r, (ar, br)) in self
                .root_indices()
                .zip(a.root_indices().zip(b.root_indices()))
            {
                self.nodes
                    .push(Ternary(op, r, ar + a_offset, br + b_offset));
            }
        }
        return Ok(self);
    }

    fn push_nodes(&mut self, other: &Tree) -> usize {
        let offset: usize = self.nodes.len();
        self.nodes.extend(other.nodes.iter().map(|node| match node {
            Constant(value) => Constant(*value),
            Symbol(label) => Symbol(label.clone()),
            Unary(op, input) => Unary(*op, *input + offset),
            Binary(op, lhs, rhs) => Binary(*op, *lhs + offset, *rhs + offset),
            Ternary(op, a, b, c) => Ternary(*op, *a + offset, *b + offset, *c + offset),
        }));
        return offset;
    }
}

macro_rules! unary_func {
    ($name:ident, $op:ident) => {
        pub fn $name(tree: MaybeTree) -> MaybeTree {
            tree?.unary_op($op)
        }
    };
}

unary_func!(negate, Negate);
unary_func!(sqrt, Sqrt);
unary_func!(abs, Abs);
unary_func!(sin, Sin);
unary_func!(cos, Cos);
unary_func!(tan, Tan);
unary_func!(log, Log);
unary_func!(exp, Exp);
unary_func!(not, Not);

macro_rules! binary_func {
    ($name:ident, $op:ident) => {
        pub fn $name(lhs: MaybeTree, rhs: MaybeTree) -> MaybeTree {
            lhs?.binary_op(rhs?, $op)
        }
    };
}

binary_func!(add, Add);
binary_func!(sub, Subtract);
binary_func!(mul, Multiply);
binary_func!(div, Divide);
binary_func!(pow, Pow);
binary_func!(min, Min);
binary_func!(max, Max);
binary_func!(less, Less);
binary_func!(greater, Greater);
binary_func!(equals, Equal);
binary_func!(neq, NotEqual);
binary_func!(leq, LessOrEqual);
binary_func!(geq, GreaterOrEqual);
binary_func!(and, And);
binary_func!(or, Or);

pub fn reshape(tree: MaybeTree, rows: usize, cols: usize) -> MaybeTree {
    tree?.reshape(rows, cols)
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Scalar(value)
    }
}

impl From<i32> for Value {
    fn from(value: i32) -> Self {
        Scalar(value as f64)
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Bool(value)
    }
}

impl From<f64> for Tree {
    fn from(value: f64) -> Self {
        Self::constant(Scalar(value))
    }
}

impl From<bool> for Tree {
    fn from(value: bool) -> Self {
        Self::constant(Bool(value))
    }
}

impl From<char> for Tree {
    fn from(c: char) -> Self {
        return Self::symbol(c);
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering::*;
        match (self, other) {
            (Bool(l), Bool(r)) => l.partial_cmp(r),
            (Bool(_), Scalar(_)) => Some(Less),
            (Scalar(_), Bool(_)) => Some(Greater),
            (Scalar(l), Scalar(r)) => l.partial_cmp(r),
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
            // Scalar
            (Constant(a), Constant(b)) => a.partial_cmp(b),
            (Constant(_), Symbol(_)) => Some(Less),
            (Constant(_), Unary(..)) => Some(Less),
            (Constant(_), Binary(..)) => Some(Less),
            (Constant(_), Ternary(..)) => Some(Less),
            // Symbol
            (Symbol(_), Constant(_)) => Some(Greater),
            (Symbol(a), Symbol(b)) => Some(a.cmp(b)),
            (Symbol(_), Unary(..)) => Some(Less),
            (Symbol(_), Binary(..)) => Some(Less),
            (Symbol(_), Ternary(..)) => Some(Less),
            // Unary
            (Unary(..), Constant(_)) => Some(Greater),
            (Unary(..), Symbol(_)) => Some(Greater),
            (Unary(op1, _), Unary(op2, _)) => Some(op1.index().cmp(&op2.index())),
            (Unary(..), Binary(..)) => Some(Less),
            (Unary(..), Ternary(..)) => Some(Less),
            // Binary
            (Binary(..), Constant(_)) => Some(Greater),
            (Binary(..), Symbol(_)) => Some(Greater),
            (Binary(..), Unary(..)) => Some(Greater),
            (Binary(op1, ..), Binary(op2, ..)) => Some(op1.index().cmp(&op2.index())),
            (Binary(..), Ternary(..)) => Some(Less),
            // Ternary
            (Ternary(..), Constant(_)) => Some(Greater),
            (Ternary(..), Symbol(_)) => Some(Greater),
            (Ternary(..), Unary(_, _)) => Some(Greater),
            (Ternary(..), Binary(_, _, _)) => Some(Greater),
            (Ternary(op1, ..), Ternary(op2, ..)) => Some(op1.index().cmp(&op2.index())),
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
                Constant(Scalar(2.)),
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
            Constant(Scalar(2.)),
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
            Err(Error::DimensionMismatch((2, 1), (3, 1)))
        );
        matches!(
            mul(
                Tree::concat(Ok('a'.into()), Ok('b'.into())),
                Tree::concat(Ok('x'.into()), Tree::concat(Ok('y'.into()), Ok('z'.into()))),
            ),
            Err(Error::DimensionMismatch((2, 1), (3, 1)))
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
            Err(Error::DimensionMismatch((9, 1), (7, 3)))
        );
    }

    #[test]
    fn t_choose_greater() {
        let tree = deftree!(if (> x 0) x (- x)).unwrap();
        assert_eq!(
            tree.nodes(),
            &[
                Symbol('x'),
                Constant(Scalar(0.0)),
                Binary(Greater, 0, 1),
                Symbol('x'),
                Symbol('x'),
                Unary(Negate, 4),
                Ternary(Choose, 2, 3, 5)
            ]
        );
    }

    #[test]
    fn t_choose_geq() {
        let tree = deftree!(if (>= x 0) x (- x)).unwrap();
        assert_eq!(
            tree.nodes(),
            &[
                Symbol('x'),
                Constant(Scalar(0.0)),
                Binary(GreaterOrEqual, 0, 1),
                Symbol('x'),
                Symbol('x'),
                Unary(Negate, 4),
                Ternary(Choose, 2, 3, 5)
            ]
        );
    }

    #[test]
    fn t_concat_op_inside_macro() {
        let tree = deftree!(/ (concat x y) 2.).unwrap();
        assert_eq!(
            tree.nodes(),
            &[
                Symbol('x'),
                Symbol('y'),
                Constant(Scalar(2.0)),
                Binary(Divide, 0, 2),
                Binary(Divide, 1, 2)
            ]
        );
    }

    #[test]
    fn t_symbols() {
        let tree = deftree!(* (+ x y) a).unwrap();
        assert_eq!(tree.symbols(), vec!['x', 'y', 'a']);
    }
}
