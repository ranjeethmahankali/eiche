use crate::{
    tree::{Node, Node::*, Tree},
    walk::{DepthWalker, NodeOrdering},
};

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

/// Tree pruner.
///
/// An instance of `Pruner` can be used to prune a list of nodes to
/// remove unused nodes.
pub struct Pruner {
    indices: Vec<usize>,
    pruned: Vec<Node>,
    walker: DepthWalker,
}

impl Pruner {
    /// Create a new `Pruner` instance.
    pub fn new() -> Pruner {
        Pruner {
            indices: vec![],
            pruned: vec![],
            walker: DepthWalker::new(),
        }
    }

    /// Prune `nodes` to remove unused ones.
    ///
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

#[derive(Debug)]
pub enum TopologicalError {
    CyclicGraph,
}

/// Topological sorter.
///
/// For the topology of a tree to be considered valid, the root of the
/// tree must be the last node, and every node must appear after its
/// inputs. A `TopoSorter` instance can be used to sort a vector of
/// nodes to be topologically valid.
pub struct TopoSorter {
    depths: Vec<usize>,
    sorted_indices: Vec<usize>,
    index_map: Vec<usize>,
    sorted: Vec<Node>,
    walker: DepthWalker,
}

impl TopoSorter {
    /// Create a new `TopoSorter` instance.
    pub fn new() -> TopoSorter {
        TopoSorter {
            depths: Vec::new(),
            sorted_indices: Vec::new(),
            index_map: Vec::new(),
            sorted: Vec::new(),
            walker: DepthWalker::new(),
        }
    }

    /// Sort `nodes` to be topologically valid, with `root_index` as
    /// the new root. Depending on the choice of root, the output
    /// vector may be littered with unused nodes, and may require
    /// pruning later.
    pub fn run(
        &mut self,
        mut nodes: Vec<Node>,
        mut root_index: usize,
    ) -> Result<(Vec<Node>, usize), TopologicalError> {
        // Compute depths of all nodes.
        self.depths.clear();
        self.depths.resize(nodes.len(), 0);
        for (index, maybe_parent) in
            self.walker
                .walk_nodes(&nodes, root_index, false, NodeOrdering::Original)
        {
            if let Some(parent) = maybe_parent {
                self.depths[index] = usize::max(self.depths[index], 1 + self.depths[parent]);
                if self.depths[index] >= nodes.len() {
                    // TODO: This is a terrible way to detect large
                    // cycles. As you'd have to traverse the whole
                    // cycle many times to reach this
                    // condition. Implement proper cycle detection
                    // later.
                    return Err(TopologicalError::CyclicGraph);
                }
            }
        }
        // Sort the node indices by depth.
        self.sorted_indices.clear();
        self.sorted_indices.extend(0..nodes.len());
        // Highest depth at the start.
        self.sorted_indices
            .sort_by(|a, b| self.depths[*b].cmp(&self.depths[*a]));
        // Build a map from old indices to new indices.
        self.index_map.clear();
        self.index_map.resize(nodes.len(), 0);
        for (index, i) in self.sorted_indices.iter().zip(0..self.sorted_indices.len()) {
            self.index_map[*index] = i;
            if *index == root_index {
                root_index = i;
            }
        }
        // Gather the sorted nodes.
        self.sorted.clear();
        self.sorted
            .extend(self.sorted_indices.iter().map(|index| -> Node {
                match nodes[*index] {
                    Constant(v) => Constant(v),
                    Symbol(label) => Symbol(label),
                    Unary(op, input) => Unary(op, self.index_map[input]),
                    Binary(op, lhs, rhs) => Binary(op, self.index_map[lhs], self.index_map[rhs]),
                }
            }));
        // Swap the sorted nodes and the incoming nodes.
        std::mem::swap(&mut self.sorted, &mut nodes);
        return Ok((nodes, root_index));
    }
}

#[cfg(test)]
mod tests {
    use crate::tree::TreeError;
    use {
        super::*,
        crate::tree::{BinaryOp::*, UnaryOp::*},
    };

    #[test]
    fn topological_sorting_0() {
        let mut sorter = TopoSorter::new();
        let (nodes, root) = sorter
            .run(vec![Symbol('x'), Binary(Add, 0, 2), Symbol('y')], 1)
            .unwrap();
        assert_eq!(root, 2);
        assert_eq!(nodes, vec![Symbol('x'), Symbol('y'), Binary(Add, 0, 1)]);
    }

    #[test]
    fn topological_sorting_1() {
        let nodes = vec![
            Symbol('x'),            // 0
            Binary(Add, 0, 2),      // 1
            Constant(2.245),        // 2
            Binary(Multiply, 1, 5), // 3
            Unary(Sqrt, 3),         // 4 - root
            Symbol('y'),            // 5
        ];
        assert!(matches!(
            Tree::from_nodes(nodes.clone()),
            Err(TreeError::WrongNodeOrder)
        ));
        let mut sorter = TopoSorter::new();
        let (nodes, root) = sorter.run(nodes, 4).unwrap();
        assert_eq!(root, 5);
        assert!(matches!(
            Tree::from_nodes(nodes),
            Ok(tree) if tree.len() == 6,
        ));
    }

    #[test]
    fn topological_sorting_2() {
        let nodes = vec![
            Symbol('a'),            // 0
            Binary(Add, 0, 2),      // 1
            Symbol('b'),            // 2
            Unary(Log, 5),          // 3
            Symbol('x'),            // 4
            Binary(Add, 4, 6),      // 5
            Symbol('y'),            // 6
            Symbol('p'),            // 7
            Binary(Add, 7, 9),      // 8
            Symbol('p'),            // 9
            Binary(Pow, 11, 8),     // 10 - root.
            Binary(Multiply, 3, 1), // 11
        ];
        assert!(matches!(
            Tree::from_nodes(nodes.clone()),
            Err(TreeError::WrongNodeOrder)
        ));
        let mut sorter = TopoSorter::new();
        let (nodes, root) = sorter.run(nodes, 10).unwrap();
        assert_eq!(root, 11);
        assert!(matches!(
            Tree::from_nodes(nodes),
            Ok(tree) if tree.len() == 12,
        ));
    }

    #[test]
    fn topological_sorting_3() {
        let mut sorter = TopoSorter::new();
        assert!(matches!(
            sorter.run(
                vec![
                    Binary(Pow, 8, 9),      // 0
                    Symbol('x'),            // 1
                    Binary(Multiply, 0, 1), // 2
                    Symbol('y'),            // 3
                    Binary(Multiply, 0, 3), // 4
                    Binary(Add, 2, 4),      // 5
                    Binary(Add, 1, 3),      // 6
                    Binary(Divide, 5, 6),   // 7
                    Unary(Sqrt, 0),         // 8
                    Constant(2.0),          // 9
                ],
                0,
            ),
            Err(TopologicalError::CyclicGraph)
        ));
    }
}
