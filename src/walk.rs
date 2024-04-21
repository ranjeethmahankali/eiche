use crate::tree::{Node, Node::*, Tree};

/// Helper struct for traversing the tree depth first.
///
/// Doing a non-recursive depth first traversal requires
/// allocations. Those buffers are owned by this instance. So reusing
/// the same walker many times is recommended to avoid unnecessary
/// allocations.
pub struct DepthWalker {
    stack: Vec<(usize, Option<usize>)>,
    visited: Vec<bool>,
    priorities: Vec<usize>,
}

impl DepthWalker {
    pub fn new() -> DepthWalker {
        DepthWalker {
            stack: vec![],
            visited: vec![],
            priorities: vec![],
        }
    }

    pub fn priorities_mut(&mut self) -> &mut Vec<usize> {
        return &mut self.priorities;
    }

    pub fn init_from_roots<I: Iterator<Item = usize>>(&mut self, num_nodes: usize, roots: I) {
        // Prep the stack.
        self.stack.clear();
        self.stack.reserve(num_nodes);
        self.stack.extend(roots.map(|r| (r, None)));
        // Reverse the roots to preserve their order during traversal.
        self.stack.reverse();
        // Reset the visited flags and priorities.
        self.visited.clear();
        self.visited.resize(num_nodes, false);
        self.priorities.resize(num_nodes, 0);
    }

    /// Get an iterator that walks the nodes of `tree`. If `unique` is true, no
    /// node will be visited more than once. The choice of `order` will affect
    /// the order in which the children of certain nodes are traversed. See the
    /// documentation of `NodeOrdering` for more details.
    pub fn init_from_tree<'a>(&'a mut self, tree: &'a Tree) {
        self.init_from_roots(tree.len(), tree.root_indices());
    }

    /// Get an iterator that walks the given `nodes` starting from the nodes in
    /// the range `root_indices`. If `unique` is true, no node will be visited
    /// more than once. The choice of `order` will affect the order in which the
    /// children of certain nodes are traversed. See the documentation of
    /// `NodeOrdering` for more details.
    pub fn walk<'a>(
        &'a mut self,
        nodes: &'a [Node],
        unique: bool,
        ordering: NodeOrdering,
    ) -> DepthIterator<'a> {
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

/// When traversing a tree depth first, sometimes the subtrees
/// children of a node can be visited in more than one possible
/// order. For example, this is the case with commutative binary ops.
pub enum NodeOrdering {
    /// Traverse children in the order they appear in the parent.
    Original,
    /// Traverse the children in the order reverse of how they appear in the parent.
    Reversed,
    /// Sort the children in a deterministic way, irrespective of the
    /// order they appear in the parent.
    Deterministic,
}

/// Iterator that walks the tree depth first.
///
/// The lifetime of this iterator is bound to the lifetime of the
/// nodes it's traversing. For that reason, this is a separate struct
/// from `DepthWalker`. That way, the `DepthWalker` instance won't get
/// tangled up in lifetimes and it can be used multiple traversals,
/// even on different trees.
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
            Constant(_) | Symbol(_) | Unary(..) => {}
            Binary(op, ..) => {
                match self.ordering {
                    Original => {} // Do nothing.
                    Reversed => children.reverse(),
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
            Ternary(..) => {
                // There might be ternary ops in the future that are order
                // agnostic. That is not the case right now.
            }
        }
        children.sort_by(|a, b| self.walker.priorities[*b].cmp(&self.walker.priorities[*a]));
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
            Ternary(_opp, a, b, c) => {
                // Push in reverse order because last in first out.
                let mut children = [*c, *b, *a];
                self.sort_children(node, &mut children);
                self.walker
                    .stack
                    .extend(children.iter().map(|child| (*child, Some(index))));
                self.last_pushed = children.len();
            }
        }
        self.walker.visited[index] = true;
        return Some((index, parent));
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::deftree;

    #[test]
    fn t_depth_traverse() {
        let mut walker = DepthWalker::new();
        {
            let tree = deftree!(+ (pow x 2.) (pow y 2.)).unwrap();
            // Make sure two successive traversal yield the same nodes.
            walker.init_from_tree(&tree);
            let a: Vec<_> = walker
                .walk(tree.nodes(), true, NodeOrdering::Original)
                .map(|(index, parent)| (index, parent))
                .collect();
            walker.init_from_tree(&tree);
            let b: Vec<_> = walker
                .walk(tree.nodes(), true, NodeOrdering::Original)
                .map(|(index, parent)| (index, parent))
                .collect();
            assert_eq!(a, b);
            // Same thing in reverse traversal order
            walker.init_from_tree(&tree);
            let a: Vec<_> = walker
                .walk(tree.nodes(), true, NodeOrdering::Reversed)
                .map(|(index, parent)| (index, parent))
                .collect();
            walker.init_from_tree(&tree);
            let b: Vec<_> = walker
                .walk(tree.nodes(), true, NodeOrdering::Reversed)
                .map(|(index, parent)| (index, parent))
                .collect();
            assert_eq!(a, b);
        }
        {
            // Make sure the same DepthWalker can be used on multiple trees.
            let tree = deftree!(+ (pow x 3.) (pow y 3.)).unwrap();
            walker.init_from_tree(&tree);
            let a: Vec<_> = walker
                .walk(tree.nodes(), true, NodeOrdering::Original)
                .map(|(index, parent)| (index, parent))
                .collect();
            let tree2 = tree.clone();
            walker.init_from_tree(&tree2);
            let b: Vec<_> = walker
                .walk(tree2.nodes(), true, NodeOrdering::Original)
                .map(|(index, parent)| (index, parent))
                .collect();
            assert_eq!(a, b);
        }
    }
}
