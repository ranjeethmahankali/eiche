use crate::{
    error::Error,
    tree::{
        Node::{self, *},
        Tree,
    },
};

#[derive(Debug)]
struct StackElement {
    index: usize,
    parent: Option<usize>,
    visited_children: bool,
}

/// Helper struct for traversing the tree depth first.
///
/// Doing a non-recursive depth first traversal requires
/// allocations. Those buffers are owned by this instance. So reusing
/// the same walker many times is recommended to avoid unnecessary
/// allocations.
pub struct DepthWalker {
    stack: Vec<StackElement>,
    visited: Vec<bool>, // Whether a node is already visited.
    on_path: Vec<bool>, // Whether a node is present on the path between the current node and the root.
}

impl DepthWalker {
    pub fn new() -> DepthWalker {
        DepthWalker {
            stack: Vec::new(),
            visited: Vec::new(),
            on_path: Vec::new(),
        }
    }

    fn init_from_roots<I: Iterator<Item = usize>>(&mut self, num_nodes: usize, roots: I) {
        // Prep the stack.
        self.stack.clear();
        self.stack.reserve(num_nodes);
        self.stack.extend(roots.map(|r| StackElement {
            index: r,
            parent: None,
            visited_children: false,
        }));
        // Reverse the roots to preserve their order during traversal.
        self.stack.reverse();
        // Reset the flags.
        self.visited.clear();
        self.visited.resize(num_nodes, false);
        self.on_path.clear();
        self.on_path.resize(num_nodes, false);
    }

    /// Get an iterator that walks the given `nodes` starting from the nodes in
    /// the range `root_indices`. If `unique` is true, no node will be visited
    /// more than once. The choice of `order` will affect the order in which the
    /// children of certain nodes are traversed. See the documentation of
    /// `NodeOrdering` for more details.
    pub fn walk_nodes<'a, I: Iterator<Item = usize>>(
        &'a mut self,
        nodes: &'a [Node],
        roots: I,
        unique: bool,
        ordering: NodeOrdering,
    ) -> DepthIterator<'a, true> {
        self.init_from_roots(nodes.len(), roots);
        // Create the iterator.
        DepthIterator {
            unique,
            ordering,
            walker: self,
            nodes: &nodes,
            last_pushed: 0,
        }
    }

    /// Get an interator that walks the given tree starting from its roots. If
    /// `unique` is true, no node will be visited more than once. The choice of
    /// `order` will affect the order in which the children of certain nodes are
    /// traversed. See the documentation of `NodeOrdering` for more details.
    pub fn walk_tree<'a>(
        &'a mut self,
        tree: &'a Tree,
        unique: bool,
        ordering: NodeOrdering,
    ) -> DepthIterator<'a, false> {
        self.init_from_roots(tree.len(), tree.root_indices());
        // Create the iterator.
        DepthIterator {
            unique,
            ordering,
            walker: self,
            nodes: tree.nodes(),
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
pub struct DepthIterator<'a, const CHECK_FOR_CYCLES: bool> {
    unique: bool,
    ordering: NodeOrdering,
    last_pushed: usize,
    walker: &'a mut DepthWalker,
    nodes: &'a [Node],
}

impl<'a, const CHECK_FOR_CYCLES: bool> DepthIterator<'a, CHECK_FOR_CYCLES> {
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
    }

    /// Skip the children of the current node. The whole subtree from
    /// the current node will be skipped, unless used as inputs by
    /// some other node.
    pub fn skip_children(&mut self) {
        for _ in 0..self.last_pushed {
            self.walker.stack.pop();
        }
    }

    /// Push the children of the node at `index` onto the stack.
    fn push_children(&mut self, index: usize) {
        let node = &self.nodes[index];
        match node {
            Constant(_) | Symbol(_) => {
                self.last_pushed = 0;
            }
            Unary(_op, input) => {
                self.walker.stack.push(StackElement {
                    index: *input,
                    parent: Some(index),
                    visited_children: false,
                });
                self.last_pushed = 1;
            }
            Binary(_op, lhs, rhs) => {
                // Pushing rhs first because last in first out.
                let mut children = [*rhs, *lhs];
                // Sort according to the requested ordering.
                self.sort_children(node, &mut children);
                for child in children {
                    self.walker.stack.push(StackElement {
                        index: child,
                        parent: Some(index),
                        visited_children: false,
                    });
                }
                self.last_pushed = children.len();
            }
            Ternary(_opp, a, b, c) => {
                // Push in reverse order because last in first out.
                let mut children = [*c, *b, *a];
                self.sort_children(node, &mut children);
                self.walker
                    .stack
                    .extend(children.iter().map(|child| StackElement {
                        index: *child,
                        parent: Some(index),
                        visited_children: false,
                    }));
                self.last_pushed = children.len();
            }
        }
    }
}

impl<'a> Iterator for DepthIterator<'a, false> {
    type Item = (usize, Option<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        let StackElement {
            index,
            parent,
            visited_children: _,
        } = {
            let mut elem = self.walker.stack.pop()?;
            debug_assert!(!elem.visited_children);
            while self.unique && self.walker.visited[elem.index] {
                elem = self.walker.stack.pop()?;
            }
            elem
        };
        self.push_children(index);
        self.walker.visited[index] = true;
        return Some((index, parent));
    }
}

impl<'a> Iterator for DepthIterator<'a, true> {
    type Item = Result<(usize, Option<usize>), Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let StackElement {
            index,
            parent,
            visited_children,
        } = {
            let mut elem = self.walker.stack.pop()?;
            loop {
                if !elem.visited_children && self.walker.on_path[elem.index] {
                    return Some(Err(Error::CyclicGraph));
                }
                if elem.visited_children {
                    self.walker.on_path[elem.index] = false;
                }
                if !(self.unique && self.walker.visited[elem.index]) && !elem.visited_children {
                    break;
                }
                elem = self.walker.stack.pop()?;
            }
            elem
        };
        println!("popped {}, {:?}, {}", index, parent, visited_children); // DEBUG
        debug_assert!(!visited_children, "Invalid depth first traversal.");
        if self.walker.on_path[index] {
            return Some(Err(Error::CyclicGraph));
        }
        self.walker.on_path[index] = true;
        self.walker.stack.push(StackElement {
            index,
            parent,
            visited_children: true,
        });
        self.push_children(index);
        self.walker.visited[index] = true;
        return Some(Ok((index, parent)));
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
            let a: Vec<_> = walker
                .walk_tree(&tree, true, NodeOrdering::Original)
                .map(|(index, parent)| (index, parent))
                .collect();
            let b: Vec<_> = walker
                .walk_tree(&tree, true, NodeOrdering::Original)
                .map(|(index, parent)| (index, parent))
                .collect();
            assert_eq!(a, b);
        }
        {
            // Make sure the same DepthWalker can be used on multiple trees.
            let tree = deftree!(+ (pow x 3.) (pow y 3.)).unwrap();
            let a: Vec<_> = walker
                .walk_tree(&tree, true, NodeOrdering::Original)
                .map(|(index, parent)| (index, parent))
                .collect();
            let tree2 = tree.clone();
            let b: Vec<_> = walker
                .walk_tree(&tree2, true, NodeOrdering::Original)
                .map(|(index, parent)| (index, parent))
                .collect();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn t_cyclic_graph() {
        use crate::tree::{BinaryOp::*, UnaryOp::*, Value::*};
        let nodes = vec![
            Binary(Pow, 8, 9),      // 0
            Symbol('x'),            // 1
            Binary(Multiply, 0, 1), // 2
            Symbol('y'),            // 3
            Binary(Multiply, 0, 3), // 4
            Binary(Add, 2, 4),      // 5
            Binary(Add, 1, 3),      // 6
            Binary(Divide, 5, 6),   // 7
            Unary(Sqrt, 0),         // 8
            Constant(Scalar(2.0)),  // 9
        ];
        let mut walker = DepthWalker::new();
        assert!(matches!(
            walker
                .walk_nodes(&nodes, 0..1, true, NodeOrdering::Deterministic)
                .fold(Ok(()), |acc, current| match (acc, current) {
                    (Ok(_), Ok(_)) => Ok(()),
                    (Ok(_), Err(e)) => Err(e),
                    (Err(e), Ok(_)) => Err(e),
                    (Err(e), Err(_)) => Err(e),
                }),
            Err(Error::CyclicGraph)
        ));
    }
}
