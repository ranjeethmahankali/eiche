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
}

impl DepthWalker {
    pub fn new() -> DepthWalker {
        DepthWalker {
            stack: vec![],
            visited: vec![],
        }
    }

    /// Get an iterator that walks the nodes of `tree`. If `unique` is
    /// true, no node will be visited more than once. The choice of
    /// `order` will affect the order in which the children of certain
    /// nodes are traversed. See the documentation of `NodeOrdering`
    /// for more details.
    pub fn walk_tree<'a>(
        &'a mut self,
        tree: &'a Tree,
        unique: bool,
        ordering: NodeOrdering,
    ) -> DepthIterator<'a> {
        self.walk_nodes(&tree.nodes(), tree.root_index(), unique, ordering)
    }

    /// Get an iterator that walks the given `nodes` starting from the
    /// node at `root_index`. If `unique` is true, no node will be
    /// visited more than once. The choice of `order` will affect the
    /// order in which the children of certain nodes are
    /// traversed. See the documentation of `NodeOrdering` for more
    /// details.
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

/// When traversing a tree depth first, sometimes the subtrees
/// children of a node can be visited in more than one possible
/// order. For example, this is the case with commutative binary ops.
pub enum NodeOrdering {
    /// Traverse children in the order they appear in the parent.
    Original,
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
