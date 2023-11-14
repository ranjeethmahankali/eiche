use crate::{
    tree::{Node, Node::*},
    walk::{DepthWalker, NodeOrdering},
};

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
