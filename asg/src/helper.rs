use std::collections::HashMap;

use crate::tree::{BinaryOp, Node, Node::*, Tree, UnaryOp};

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

impl UnaryOp {
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
    /// the data heal inside the node. It DOES NOT take into account
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

pub struct Deduplicater {
    indices: Vec<usize>,
    hashes: Vec<u64>,
    walker1: DepthWalker,
    walker2: DepthWalker,
    hash_to_index: HashMap<u64, usize>,
}

impl Deduplicater {
    pub fn new() -> Self {
        Deduplicater {
            indices: vec![],
            hashes: vec![],
            walker1: DepthWalker::new(),
            walker2: DepthWalker::new(),
            hash_to_index: HashMap::new(),
        }
    }

    fn calc_hashes(&mut self, nodes: &Vec<Node>) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        // Using a boxed slice to avoid accidental resizing later.
        self.hashes.clear();
        self.hashes.resize(nodes.len(), 0);
        for index in 0..nodes.len() {
            let hash: u64 = match nodes[index] {
                Constant(value) => value.to_bits().into(),
                Symbol(label) => {
                    let mut s: DefaultHasher = Default::default();
                    label.hash(&mut s);
                    s.finish()
                }
                Unary(op, input) => {
                    let mut s: DefaultHasher = Default::default();
                    op.hash(&mut s);
                    self.hashes[input].hash(&mut s);
                    s.finish()
                }
                Binary(op, lhs, rhs) => {
                    let (hash1, hash2) = {
                        let mut hash1 = self.hashes[lhs];
                        let mut hash2 = self.hashes[rhs];
                        if op.is_commutative() && hash1 > hash2 {
                            std::mem::swap(&mut hash1, &mut hash2);
                        }
                        (hash1, hash2)
                    };
                    let mut s: DefaultHasher = Default::default();
                    op.hash(&mut s);
                    hash1.hash(&mut s);
                    hash2.hash(&mut s);
                    s.finish()
                }
            };
            self.hashes[index] = hash;
        }
    }

    pub fn run(&mut self, mut nodes: Vec<Node>) -> Vec<Node> {
        // Compute unique indices after deduplication.
        self.indices.clear();
        self.indices.extend(0..nodes.len());
        self.calc_hashes(&nodes);
        self.hash_to_index.clear();
        for i in 0..self.hashes.len() {
            let h = self.hashes[i];
            let entry = self.hash_to_index.entry(h).or_insert(i);
            if *entry != i && equivalent(*entry, i, &nodes, &mut self.walker1, &mut self.walker2) {
                // The i-th node should be replaced with entry-th node.
                self.indices[i] = *entry;
            }
        }
        // Update nodes.
        for node in nodes.iter_mut() {
            match node {
                Constant(_) => {}
                Symbol(_) => {}
                Unary(_, input) => {
                    *input = self.indices[*input];
                }
                Binary(_, lhs, rhs) => {
                    *lhs = self.indices[*lhs];
                    *rhs = self.indices[*rhs];
                }
            }
        }
        return nodes;
    }
}

/// Check if the nodes at indices `left` and `right` are
/// equivalent.
///
/// Two nodes need not share the same input needs to be
/// equivalent. They just need to represent the same mathematical
/// expression. For example, two distinct constant nodes with the
/// holding the same value are equivalent. Two nodes of the same type
/// with equivalent inputs are considered equivalent. For binary nodes
/// with commutative operations, checking the equivalence of the
/// inputs is done in an order agnostic way.
///
/// This implementation avoids recursion by using `walker1` and
/// `walker2` are used to traverse the tree depth wise and perform the
/// comparison.
pub fn equivalent(
    left: usize,
    right: usize,
    nodes: &[Node],
    walker1: &mut DepthWalker,
    walker2: &mut DepthWalker,
) -> bool {
    {
        use crate::helper::NodeOrdering::*;
        // Zip the depth first iterators and compare.
        let mut iter1 = walker1.walk_nodes(&nodes, left, false, Deterministic);
        let mut iter2 = walker2.walk_nodes(&nodes, right, false, Deterministic);
        loop {
            match (iter1.next(), iter2.next()) {
                (None, None) => {
                    // Both iterators ended.
                    return true;
                }
                (None, Some(_)) | (Some(_), None) => {
                    // One of the iterators ended prematurely.
                    return false;
                }
                (Some((i1, _p1)), Some((i2, _p2))) => {
                    if i1 == i2 {
                        iter1.skip_children();
                        iter2.skip_children();
                        continue;
                    }
                    if !match (nodes[i1], nodes[i2]) {
                        (Constant(v1), Constant(v2)) => v1 == v2,
                        (Symbol(c1), Symbol(c2)) => c1 == c2,
                        (Unary(op1, _input1), Unary(op2, _input2)) => op1 == op2,
                        (Binary(op1, _lhs1, _rhs1), Binary(op2, _lhs2, _rhs2)) => op1 == op2,
                        _ => false,
                    } {
                        return false;
                    }
                }
            }
        }
    }
}

pub struct DepthWalker {
    stack: Vec<(usize, Option<usize>)>,
    visited: Vec<bool>,
}

pub enum NodeOrdering {
    Original,
    Deterministic,
}

impl DepthWalker {
    pub fn new() -> DepthWalker {
        DepthWalker {
            stack: vec![],
            visited: vec![],
        }
    }

    pub fn walk_tree<'a>(
        &'a mut self,
        tree: &'a Tree,
        unique: bool,
        ordering: NodeOrdering,
    ) -> DepthIterator<'a> {
        self.walk_nodes(&tree.nodes(), tree.root_index(), unique, ordering)
    }

    pub fn walk_nodes<'a>(
        &'a mut self,
        nodes: &'a [Node],
        root_index: usize,
        unique: bool,
        ordering: NodeOrdering,
    ) -> DepthIterator<'a> {
        // Prep the stack.
        self.stack.clear();
        self.stack.reserve(nodes.len());
        self.stack.push((root_index, None));
        // Reset the visited flags.
        self.visited.clear();
        self.visited.resize(nodes.len(), false);
        // Create the iterator.
        DepthIterator {
            unique,
            ordering,
            walker: self,
            nodes: &nodes,
            last_pushed: 0,
        }
    }
}

pub struct DepthIterator<'a> {
    unique: bool,
    ordering: NodeOrdering,
    last_pushed: usize,
    walker: &'a mut DepthWalker,
    nodes: &'a [Node],
}

impl<'a> DepthIterator<'a> {
    fn sort_children(&self, parent: &Node, children: &mut [usize]) {
        use std::cmp::Ordering;
        use NodeOrdering::*;
        if children.len() < 2 {
            // Nothing to sort.
            return;
        }
        match parent {
            // Nothing to do when number children is 1 or less.
            Constant(_) | Symbol(_) | Unary(_, _) => {}
            Binary(op, _, _) => {
                match self.ordering {
                    Original => {} // Do nothing.
                    Deterministic => {
                        if op.is_commutative() {
                            children.sort_by(|a, b| {
                                match self.nodes[*a].partial_cmp(&self.nodes[*b]) {
                                    Some(ord) => ord,
                                    // This is tied to the PartialOrd
                                    // implementation for Node. Assuming the
                                    // only time we return None is with two
                                    // constant nodes with Nan's in them. This
                                    // seems like a harmless edge case for
                                    // now.
                                    None => Ordering::Equal,
                                }
                            });
                        }
                    }
                }
            }
        }
    }

    /// Skip the children of the current node. The whole subtree from
    /// the current node will be skipped, unless used as inputs by
    /// some other node.
    pub fn skip_children(&mut self) {
        for _ in 0..self.last_pushed {
            self.walker.stack.pop();
        }
    }
}

impl<'a> Iterator for DepthIterator<'a> {
    type Item = (usize, Option<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        let (index, parent) = {
            // Pop the stack until we find a node we didn't already visit.
            let (mut i, mut p) = self.walker.stack.pop()?;
            while self.unique && self.walker.visited[i] {
                (i, p) = self.walker.stack.pop()?;
            }
            (i, p)
        };
        // Push the children on to the stack.
        let node = &self.nodes[index];
        match node {
            Constant(_) | Symbol(_) => {
                self.last_pushed = 0;
            }
            Unary(_op, input) => {
                self.walker.stack.push((*input, Some(index)));
                self.last_pushed = 1;
            }
            Binary(_op, lhs, rhs) => {
                // Pushing rhs first because last in first out.
                let mut children = [*rhs, *lhs];
                // Sort according to the requested ordering.
                self.sort_children(node, &mut children);
                for child in children {
                    self.walker.stack.push((child, Some(index)));
                }
                self.last_pushed = children.len();
            }
        }
        self.walker.visited[index] = true;
        return Some((index, parent));
    }
}

pub struct Pruner {
    indices: Vec<usize>,
    pruned: Vec<Node>,
    walker: DepthWalker,
}

impl Pruner {
    pub fn new() -> Pruner {
        Pruner {
            indices: vec![],
            pruned: vec![],
            walker: DepthWalker::new(),
        }
    }

    /// The given `nodes` are walked depth-first using the `walker`
    /// and nodes that are not visited are filtered out. The filtered
    /// `nodes` are returned. You can minimize allocations by using
    /// the same pruner multiple times.
    pub fn run(&mut self, mut nodes: Vec<Node>, root_index: usize) -> Vec<Node> {
        self.indices.clear();
        self.indices.resize(nodes.len(), 0);
        // Mark used nodes.
        self.walker
            .walk_nodes(&nodes, root_index, true, NodeOrdering::Original)
            .for_each(|(index, _parent)| {
                self.indices[index] = 1;
            });
        // Reserve space for new nodes.
        self.pruned.clear();
        self.pruned.reserve(self.indices.iter().sum());
        {
            // Do inclusive scan.
            let mut sum = 0usize;
            for index in self.indices.iter_mut() {
                sum += *index;
                *index = sum;
            }
        }
        // Filter, update and copy nodes.
        for i in 0..self.indices.len() {
            let index = self.indices[i];
            if index > 0 && (i == 0 || self.indices[i - 1] < index) {
                // We subtract 1 from all indices because we did an inclusive sum.
                self.pruned.push(match nodes[i] {
                    Constant(val) => Constant(val),
                    Symbol(label) => Symbol(label),
                    Unary(op, input) => Unary(op, self.indices[input] - 1),
                    Binary(op, lhs, rhs) => {
                        Binary(op, self.indices[lhs] - 1, self.indices[rhs] - 1)
                    }
                });
            }
        }
        std::mem::swap(&mut self.pruned, &mut nodes);
        return nodes;
    }
}

/// Compute the results of operations on constants and fold those into
/// constant nodes. The unused nodes after folding are not
/// pruned. Use a pruner for that.
pub fn fold_constants(mut nodes: Vec<Node>) -> Vec<Node> {
    for index in 0..nodes.len() {
        let constval = match nodes[index] {
            Constant(_) => None,
            Symbol(_) => None,
            Unary(op, input) => {
                if let Constant(value) = nodes[input] {
                    Some(op.apply(value))
                } else {
                    None
                }
            }
            Binary(op, lhs, rhs) => {
                if let (Constant(a), Constant(b)) = (&nodes[lhs], &nodes[rhs]) {
                    Some(op.apply(*a, *b))
                } else {
                    None
                }
            }
        };
        if let Some(value) = constval {
            nodes[index] = Constant(value);
        }
    }
    return nodes;
}

pub struct TopoSorter {
    depths: Vec<usize>,
    indices: Vec<usize>,
    sorted: Vec<Node>,
    walker: DepthWalker,
}

impl TopoSorter {
    pub fn new() -> TopoSorter {
        TopoSorter {
            depths: Vec::new(),
            indices: Vec::new(),
            sorted: Vec::new(),
            walker: DepthWalker::new(),
        }
    }

    pub fn run(&mut self, mut nodes: Vec<Node>, mut root_index: usize) -> (Vec<Node>, usize) {
        // Compute depths of all nodes.
        self.depths.clear();
        self.depths.resize(nodes.len(), 0);
        for (index, maybe_parent) in
            self.walker
                .walk_nodes(&nodes, root_index, false, NodeOrdering::Original)
        {
            if let Some(parent) = maybe_parent {
                self.depths[index] = usize::max(self.depths[index], 1 + self.depths[parent]);
            }
        }
        // Sort the node indices by depth.
        self.indices.clear();
        self.indices.extend(0..nodes.len());
        self.indices
            .sort_by(|a, b| self.depths[*b].cmp(&self.depths[*a])); // Highest depth at the start.
                                                                    // Gather the sorted nodes.
        self.sorted.clear();
        self.sorted
            .extend(
                self.indices
                    .iter()
                    .zip(0..self.indices.len())
                    .map(|(index, i)| -> Node {
                        if *index == root_index {
                            root_index = i;
                        }
                        match nodes[*index] {
                            Constant(v) => Constant(v),
                            Symbol(label) => Symbol(label),
                            Unary(op, input) => Unary(op, self.indices[input]),
                            Binary(op, lhs, rhs) => {
                                Binary(op, self.indices[lhs], self.indices[rhs])
                            }
                        }
                    }),
            );
        // Swap the sorted nodes and the incoming nodes.
        std::mem::swap(&mut self.sorted, &mut nodes);
        return (nodes, root_index);
    }
}

pub fn to_lisp(node: &Node, nodes: &Vec<Node>) -> String {
    match node {
        Constant(val) => val.to_string(),
        Symbol(label) => label.to_string(),
        Unary(op, input) => format!(
            "({} {})",
            {
                match op {
                    UnaryOp::Negate => "-",
                    UnaryOp::Sqrt => "sqrt",
                    UnaryOp::Abs => "abs",
                    UnaryOp::Sin => "sin",
                    UnaryOp::Cos => "cos",
                    UnaryOp::Tan => "tan",
                    UnaryOp::Log => "log",
                    UnaryOp::Exp => "exp",
                }
            },
            to_lisp(&nodes[*input], nodes)
        ),
        Binary(op, lhs, rhs) => format!(
            "({} {} {})",
            {
                match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Subtract => "-",
                    BinaryOp::Multiply => "*",
                    BinaryOp::Divide => "/",
                    BinaryOp::Pow => "pow",
                    BinaryOp::Min => "min",
                    BinaryOp::Max => "max",
                }
            },
            to_lisp(&nodes[*lhs], nodes),
            to_lisp(&nodes[*rhs], nodes)
        ),
    }
}
