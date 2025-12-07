use crate::{dedup::Deduplicater, error::Error, fold::fold, prune::Pruner};
use std::ops::Range;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    Scalar(f64),
}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
    }
}

impl std::cmp::Eq for Value {}

use Value::*;

/// Represents an operation with one input.
#[derive(Debug, Copy, Clone, PartialEq, Hash, Eq)]
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
    Floor,
    // Boolean
    Not,
}

/// Represents an operation with two inputs.
#[derive(Debug, Copy, Clone, PartialEq, Hash, Eq)]
pub enum BinaryOp {
    // Scalar
    Add,
    Subtract,
    Multiply,
    Divide,
    Pow,
    Min,
    Max,
    Remainder, // Floating point remainder
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

#[derive(Debug, Copy, Clone, PartialEq, Hash, Eq)]
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
            Floor => 8,
            // Boolean
            Not => 9,
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
            Remainder => 7,
            // Boolean
            Less => 8,
            LessOrEqual => 9,
            Equal => 10,
            NotEqual => 11,
            Greater => 12,
            GreaterOrEqual => 13,
            And => 14,
            Or => 15,
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
            Remainder => false,
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
#[derive(Debug, PartialEq, Copy, Clone, Hash, Eq)]
pub enum Node {
    Constant(Value),
    Symbol(char),
    Unary(UnaryOp, usize),
    Binary(BinaryOp, usize, usize),
    Ternary(TernaryOp, usize, usize, usize),
}

use Node::*;

pub fn is_node_scalar(nodes: &[Node], index: usize) -> bool {
    match &nodes[index] {
        Constant(value) => match value {
            Bool(_) => false,
            Scalar(_) => true,
        },
        Symbol(_) => true,
        Unary(op, _) => match op {
            Negate => true,
            Sqrt => true,
            Abs => true,
            Sin => true,
            Cos => true,
            Tan => true,
            Log => true,
            Exp => true,
            Floor => true,
            Not => false,
        },
        Binary(op, _, _) => match op {
            Add => true,
            Subtract => true,
            Multiply => true,
            Divide => true,
            Pow => true,
            Min => true,
            Max => true,
            Remainder => true,
            Less => false,
            LessOrEqual => false,
            Equal => false,
            NotEqual => false,
            Greater => false,
            GreaterOrEqual => false,
            And => false,
            Or => false,
        },
        Ternary(op, _, a, b) => match op {
            Choose => is_node_scalar(nodes, *a) && is_node_scalar(nodes, *b),
        },
    }
}

pub(crate) fn is_topological_order(nodes: &[Node]) -> bool {
    nodes.iter().enumerate().all(|(i, node)| match node {
        Constant(_) | Symbol(_) => true,
        Unary(_, input) => *input < i,
        Binary(_, l, r) => *l < i && *r < i,
        Ternary(_, a, b, c) => *a < i && *b < i && *c < i,
    })
}

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
    pub fn from_nodes(nodes: Vec<Node>, dims: (usize, usize)) -> Result<Tree, Error> {
        let t = Tree { nodes, dims };
        t.validated()
    }

    /// Create a tree representing a constant value.
    pub fn constant(val: Value) -> Tree {
        Tree {
            nodes: vec![Constant(val)],
            dims: (1, 1),
        }
    }

    /// Fold the constants, deduplicate subtrees, prune unused subtrees and
    /// return a topologically sorted compacted equivalent to this tree.
    pub fn compacted(mut self) -> Result<Tree, Error> {
        let mut deduper = Deduplicater::new();
        loop {
            if !fold(&mut self.nodes)? && !deduper.run(&mut self.nodes)? {
                break;
            }
        }
        let mut pruner = Pruner::new();
        let roots = self.root_indices();
        let (mut nodes, dims) = self.take();
        let roots = pruner.run_from_range(&mut nodes, roots)?;
        // We don't need to check because we just ran the pruner on these nodes, which sorts them topologically.
        deduper.run(&mut nodes)?;
        let mut pruner = Pruner::new();
        pruner.run_from_range(&mut nodes, roots)?;
        Tree::from_nodes(nodes, dims)
    }

    /// Prunes the tree and topologically sorts the nodes.
    pub fn prune(self, pruner: &mut Pruner) -> Result<Tree, Error> {
        let roots = self.root_indices();
        let (mut nodes, dims) = self.take();
        pruner.run_from_range(&mut nodes, roots)?;
        Tree::from_nodes(nodes, dims)
    }

    /// Create a tree representing a symbol with the given `label`.
    pub fn symbol(label: char) -> Tree {
        Tree {
            nodes: vec![Symbol(label)],
            dims: (1, 1),
        }
    }

    /// Concatenate the two trees into a vector. If the input trees are
    /// matrices, they are flattened and then concatenated into a flat
    /// vector. If the caller doesn't want a vector, they can use the `reshape`
    /// function to reshape the tree after concatenation.
    pub fn concat(lhs: Result<Tree, Error>, rhs: Result<Tree, Error>) -> Result<Tree, Error> {
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
        Ok(lhs)
    }

    /// Get a piece wise expression from the given condition, value when true
    /// and value when false.
    pub fn piecewise(
        cond: Result<Tree, Error>,
        iftrue: Result<Tree, Error>,
        iffalse: Result<Tree, Error>,
    ) -> Result<Tree, Error> {
        cond?.ternary_op(iftrue?, iffalse?, Choose)
    }

    /// The number of nodes in this tree.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Check if the tree's output is scalar.
    pub fn is_scalar(&self) -> bool {
        self.root_indices()
            .all(|ri| is_node_scalar(self.nodes(), ri))
    }

    /// Get the number of roots in this tree.
    pub fn num_roots(&self) -> usize {
        matsize(self.dims)
    }

    /// Get the dimensions of this tree.
    pub fn dims(&self) -> (usize, usize) {
        self.dims
    }

    /// Change the shape of this tree. If the new shape doesn't correspond to
    /// the same number of elements, an error is returned.
    pub fn reshape(self, rows: usize, cols: usize) -> Result<Tree, Error> {
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

    /// The nodes and the dimensions of the tree. This drops the tree and gives
    /// the ownership of the data to the caller.
    ///
    /// This is meant to be used by algorithms that want to perform direct
    /// surgery on the nodes of the tree, rearrange them etc. Our goal is to
    /// make sure it is impossible for an invalid tree to ever exist. For this
    /// reason, these algos are expected to take full owner ship of the data in
    /// the tree, do what they want to do and construct a new tree from the
    /// modified data. When they construct the new tree, we perform checks to
    /// make sure the tree is valid. Once constructed, the tree must remain
    /// immutable, unless someone calls this function to take ownership.
    pub fn take(self) -> (Vec<Node>, (usize, usize)) {
        (self.nodes, self.dims)
    }

    /// Get a unique list of all symbols in this tree. The symbols will appear
    /// in the same order as they first appear in the tree.
    /// ```
    /// use eiche::deftree;
    /// assert_eq!(
    ///     deftree!(* (+ 'x 'y) 'a).unwrap().symbols(),
    ///     vec!['x', 'y', 'a']
    /// );
    /// assert_eq!(
    ///     deftree!(* (+ 'a 'x) 'y).unwrap().symbols(),
    ///     vec!['a', 'x', 'y']
    /// );
    /// assert_eq!(
    ///     deftree!(* (+ 'a 'y) 'x).unwrap().symbols(),
    ///     vec!['a', 'y', 'x']
    /// );
    /// ```
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
        indices.iter().map(|i| symbols[*i]).collect()
    }

    /// Check the tree for errors and return a Result that contains the tree if
    /// no errors were found, or the first error encountered with the tree.
    fn validated(self) -> Result<Tree, Error> {
        /* We make sure the inputs of every node appear before that node
         * itself. This is important when evaluating the tree, but also ensures
         * there are no cycles in the tree.
         */
        if self.nodes.is_empty() {
            return Err(Error::EmptyTree);
        }
        let roots = self.root_indices();
        if roots.start >= roots.end || roots.end != self.nodes.len() {
            return Err(Error::InvalidRoots);
        }
        // Check for NaNs.
        if self.nodes.iter().any(|node| match node {
            Constant(val) => match val {
                Bool(_) => false,
                Scalar(val) => f64::is_nan(*val),
            },
            Symbol(_) | Unary(_, _) | Binary(_, _, _) | Ternary(_, _, _, _) => false,
        }) {
            return Err(Error::ContainsNaN);
        }
        if !is_topological_order(&self.nodes) {
            return Err(Error::WrongNodeOrder);
        }
        // Check if any nodes depend on roots as their inputs. This is not allowed.
        if self.nodes.iter().any(|node| match node {
            Constant(_) | Symbol(_) => false, // Do nothing.
            Unary(_, input) => roots.contains(input),
            Binary(_, l, r) => roots.contains(l) || roots.contains(r),
            Ternary(_, a, b, c) => roots.contains(a) || roots.contains(b) || roots.contains(c),
        }) {
            return Err(Error::DependentRootNodes);
        }
        // Maybe add more checks later.
        Ok(self)
    }

    fn unary_op(mut self, op: UnaryOp) -> Result<Tree, Error> {
        for root in self.root_indices() {
            self.nodes.push(Unary(op, root));
        }
        Ok(self)
    }

    fn binary_op(mut self, other: Tree, op: BinaryOp) -> Result<Tree, Error> {
        let nroots = self.num_roots();
        let other_nroots = other.num_roots();
        if nroots != 1 && other_nroots != 1 && nroots != other_nroots {
            return Err(Error::DimensionMismatch(self.dims, other.dims));
        }
        self.nodes
            .reserve(other.nodes.len() + usize::max(nroots, other_nroots));
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
        Ok(self)
    }

    fn ternary_op(mut self, a: Tree, b: Tree, op: TernaryOp) -> Result<Tree, Error> {
        let anroots = a.num_roots();
        let bnroots = b.num_roots();
        if anroots != bnroots {
            return Err(Error::DimensionMismatch(a.dims, b.dims));
        }
        let nroots = self.num_roots();
        if nroots != 1 && nroots != anroots {
            return Err(Error::DimensionMismatch(self.dims, a.dims));
        }
        self.nodes.reserve(a.nodes.len() + b.nodes.len() + 1);
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
        Ok(self)
    }

    fn push_nodes(&mut self, other: &Tree) -> usize {
        extend_nodes_from_slice(&mut self.nodes, &other.nodes)
    }
}

pub(crate) fn extend_nodes_from_slice(dst: &mut Vec<Node>, other: &[Node]) -> usize {
    let offset = dst.len();
    dst.extend(other.iter().map(|node| match node {
        Constant(value) => Constant(*value),
        Symbol(label) => Symbol(*label),
        Unary(op, input) => Unary(*op, *input + offset),
        Binary(op, lhs, rhs) => Binary(*op, *lhs + offset, *rhs + offset),
        Ternary(op, a, b, c) => Ternary(*op, *a + offset, *b + offset, *c + offset),
    }));
    offset
}

macro_rules! unary_func {
    ($name:ident, $op:ident) => {
        pub fn $name(tree: Result<Tree, Error>) -> Result<Tree, Error> {
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
unary_func!(floor, Floor);
unary_func!(not, Not);

macro_rules! binary_func {
    ($name:ident, $op:ident) => {
        pub fn $name(lhs: Result<Tree, Error>, rhs: Result<Tree, Error>) -> Result<Tree, Error> {
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
binary_func!(rem, Remainder);
binary_func!(less, Less);
binary_func!(greater, Greater);
binary_func!(equals, Equal);
binary_func!(neq, NotEqual);
binary_func!(leq, LessOrEqual);
binary_func!(geq, GreaterOrEqual);
binary_func!(and, And);
binary_func!(or, Or);

pub fn reshape(tree: Result<Tree, Error>, rows: usize, cols: usize) -> Result<Tree, Error> {
    tree?.reshape(rows, cols)
}

pub fn dot(lhs: Result<Tree, Error>, rhs: Result<Tree, Error>) -> Result<Tree, Error> {
    lhs?.dot_product(rhs?)
}

pub fn matmul(lhs: Result<Tree, Error>, rhs: Result<Tree, Error>) -> Result<Tree, Error> {
    lhs?.matmul(rhs?)
}

pub fn transpose(mat: Result<Tree, Error>) -> Result<Tree, Error> {
    mat?.transpose()
}

pub fn l2norm(vec: Result<Tree, Error>) -> Result<Tree, Error> {
    vec?.l2norm()
}

pub fn normalize(vec: Result<Tree, Error>) -> Result<Tree, Error> {
    vec?.normalize()
}

pub fn extract(mat: Result<Tree, Error>, indices: &[(usize, usize)]) -> Result<Tree, Error> {
    mat?.extract(indices)
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
        Self::symbol(c)
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
        let p = deftree!(* 2 (concat 'x 'y)).unwrap();
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
        let tree = deftree!(* 2 (concat 'x 'y 'z)).unwrap();
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
        let tree = deftree!(* 2 (concat 'x 'y 'z)).unwrap();
        assert_eq!(tree.nodes, expected);
        // Matrix and a matrix - multiply
        let tree = deftree!(* (concat 'x 'y 'z) (concat 'a 'b 'c)).unwrap();
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
        let tree = deftree!(+ (concat 'x 'y 'z) (concat 'a 'b 'c)).unwrap();
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
        let mat = deftree!(concat 'a 'b 'c 'p 'q 'r 'x 'y 'z)
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
        let tree = deftree!(if (> 'x 0) 'x (- 'x)).unwrap();
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
        let tree = deftree!(if (>= 'x 0) 'x (- 'x)).unwrap();
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
        let tree = deftree!(/ (concat 'x 'y) 2.).unwrap();
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
        assert_eq!(
            deftree!(* (+ 'x 'y) 'a).unwrap().symbols(),
            vec!['x', 'y', 'a']
        );
        assert_eq!(
            deftree!(* (+ 'a 'x) 'y).unwrap().symbols(),
            vec!['a', 'x', 'y']
        );
        assert_eq!(
            deftree!(* (+ 'a 'y) 'x).unwrap().symbols(),
            vec!['a', 'y', 'x']
        );
    }
}
