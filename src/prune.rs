use std::ops::Range;

use crate::{
    error::Error,
    tree::Node::{self, *},
};

struct StackElement {
    index: usize,
    visited_children: bool, // Whether we're visiting this node after visiting all it's children.
    is_root: bool,
}

/// Pruner and topological sorter.
///
/// For the topology of a tree to be considered valid, the root of the
/// tree must be the last node, and every node must appear after its
/// inputs. A `Pruner` instance can be used to sort a vector of
/// nodes to be topologically valid, and remove unused nodes.
pub struct Pruner {
    index_map: Vec<usize>,
    sorted: Vec<Node>,
    roots: Vec<Node>,
    heights: Vec<usize>,
    visited: Vec<bool>,
    stack: Vec<StackElement>,
    on_path: Vec<bool>,
}

impl Default for Pruner {
    fn default() -> Self {
        Self::new()
    }
}

impl Pruner {
    /// Create a new `Pruner` instance.
    pub fn new() -> Pruner {
        Pruner {
            index_map: Vec::new(),
            sorted: Vec::new(),
            roots: Vec::new(),
            heights: Vec::new(),
            visited: Vec::new(),
            stack: Vec::new(),
            on_path: Vec::new(),
        }
    }

    /// Run the pruning and topological sorting using root node indices provided as a range.
    pub fn run_from_range(
        &mut self,
        mut nodes: Vec<Node>,
        root_indices: Range<usize>,
    ) -> Result<(Vec<Node>, Range<usize>), Error> {
        // Heights are easier to compute on nodes that are already topologically
        // sorted. So we prune and sort them once without caring about their
        // heights, then compute the heights of all nodes, then sort them again
        // but this time we take the computed heights into account.
        self.init_traverse(nodes.len(), root_indices.clone());
        self.sort_nodes(&nodes, false)?;
        std::mem::swap(&mut self.sorted, &mut nodes);
        self.compute_heights(&nodes);
        // Second pass now using heights.
        let root_indices = (nodes.len() - root_indices.len())..nodes.len();
        // self.init_traverse(nodes.len(), root_indices.clone());
        self.init_traverse(nodes.len(), root_indices.clone());
        self.sort_nodes(&nodes, true)?;
        std::mem::swap(&mut self.sorted, &mut nodes);
        Ok((nodes, root_indices))
    }

    /// Run the pruning and topological sorting using root node indices provided as a slice.
    pub fn run_from_slice(
        &mut self,
        mut nodes: Vec<Node>,
        roots: &mut [usize],
    ) -> Result<Vec<Node>, Error> {
        // Heights are easier to compute on nodes that are already topologically
        // sorted. So we prune and sort them once without caring about their
        // heights, then compute the heights of all nodes, then sort them again
        // but this time we take the computed heights into account.
        self.init_traverse(nodes.len(), roots.iter().copied());
        self.sort_nodes(&nodes, false)?;
        std::mem::swap(&mut self.sorted, &mut nodes);
        self.compute_heights(&nodes);
        // Second pass after computing heights.
        let newroots = (nodes.len() - roots.len())..nodes.len();
        self.init_traverse(nodes.len(), newroots.clone());
        self.sort_nodes(&nodes, true)?;
        std::mem::swap(&mut self.sorted, &mut nodes);
        for (r, i) in roots.iter_mut().zip(newroots) {
            *r = i;
        }
        Ok(nodes)
    }

    /// Prune and sort the nodes. This function does a depth-first traversal of
    /// the tree, with the pre-populated stack. It only keeps the nodes that are
    /// visited, and uses the visiting order to determine the valid topological
    /// order of the nodes. If 'use_height' is true, heights are read from
    /// self.heights. That means, 'use_height' should only be true if heights
    /// are precomputed by calling self.compute_heights. This flag when set to
    /// true will cause shallower subtrees to appear before deeper
    /// subtrees. This is beneficial for optimizing register allocation, as it
    /// minimizes the liveness range of registers.
    fn sort_nodes(&mut self, nodes: &[Node], use_height: bool) -> Result<(), Error> {
        self.roots.clear();
        self.roots.reserve(self.stack.len());
        self.index_map.clear();
        self.index_map.resize(nodes.len(), usize::MAX);
        self.sorted.clear();
        self.sorted.reserve(nodes.len());
        while let Some(StackElement {
            index,
            visited_children,
            is_root,
        }) = self.stack.pop()
        {
            if self.visited[index] {
                continue;
            }
            if visited_children {
                // We're backtracking after processing the children of this node. So we remove it from the path.
                // Since we visited all children of this node, we're ready to push it into the sorted list, and mark it as
                // processed.
                self.on_path[index] = false;
                if is_root {
                    self.roots.push(nodes[index]);
                } else {
                    self.index_map[index] = self.sorted.len();
                    self.sorted.push(nodes[index]);
                }
                self.visited[index] = true;
                continue;
            } else if self.on_path[index] {
                // Haven't visited this node's children, but it's already on the path. This means we found a cycle.
                return Err(Error::CyclicGraph);
            }
            // We reached this node for the first time. We push this node to the stack again for backtracking after it's
            // children are visited. And we push its children on to the stack.
            self.on_path[index] = true;
            self.stack.push(StackElement {
                index,
                visited_children: true,
                is_root,
            });
            let num_children = match nodes[index] {
                Constant(_) | Symbol(_) => 0, // No children.
                Unary(_op, input) => {
                    self.stack.push(StackElement {
                        index: input,
                        visited_children: false,
                        is_root: false,
                    });
                    1
                }
                Binary(_op, lhs, rhs) => {
                    let children = [rhs, lhs];
                    self.stack.extend(children.iter().map(|ci| StackElement {
                        index: *ci,
                        visited_children: false,
                        is_root: false,
                    }));
                    children.len()
                }
                Ternary(_op, a, b, c) => {
                    let children = [c, b, a];
                    self.stack.extend(children.iter().map(|ci| StackElement {
                        index: *ci,
                        visited_children: false,
                        is_root: false,
                    }));
                    children.len()
                }
            };
            if use_height {
                let num = self.stack.len();
                self.stack[(num - num_children)..].sort_by(
                    |StackElement { index: a, .. }, StackElement { index: b, .. }| {
                        self.heights[*a].cmp(&self.heights[*b])
                    },
                );
            }
        }
        // Push the roots.
        self.sorted.append(&mut self.roots);
        // Update the inputs of all the nodes with our index map.
        for node in &mut self.sorted {
            match node {
                Constant(_) | Symbol(_) => {} // Nothing.
                Unary(_, input) => *input = self.index_map[*input],
                Binary(_, lhs, rhs) => {
                    *lhs = self.index_map[*lhs];
                    *rhs = self.index_map[*rhs];
                }
                Ternary(_, a, b, c) => {
                    *a = self.index_map[*a];
                    *b = self.index_map[*b];
                    *c = self.index_map[*c];
                }
            }
        }
        Ok(())
    }

    /// Clear all the temporary storage and prepare for a new depth-first
    /// traversal for the given number of nodes and root nodes.
    fn init_traverse<I: Iterator<Item = usize>>(&mut self, num_nodes: usize, roots: I) {
        self.stack.clear();
        self.stack.extend(roots.map(|r| StackElement {
            index: r,
            visited_children: false,
            is_root: true,
        }));
        self.stack.reverse();
        self.visited.clear();
        self.visited.resize(num_nodes, false);
        self.on_path.clear();
        self.on_path.resize(num_nodes, false);
    }

    /// Root node is the highest, while constants and variable have a zero height.
    fn compute_heights(&mut self, nodes: &[Node]) {
        self.heights.clear();
        self.heights.resize(nodes.len(), 0);
        for (i, node) in nodes.iter().enumerate() {
            self.heights[i] = usize::max(
                self.heights[i],
                match node {
                    Constant(_) | Symbol(_) => 0,
                    Unary(_, input) => 1 + self.heights[*input],
                    Binary(_, lhs, rhs) => 1 + usize::max(self.heights[*lhs], self.heights[*rhs]),
                    Ternary(_, a, b, c) => {
                        1 + usize::max(
                            self.heights[*a],
                            usize::max(self.heights[*b], self.heights[*c]),
                        )
                    }
                },
            );
        }
    }
}

#[cfg(test)]
mod test {
    use {
        super::*,
        crate::tree::{BinaryOp::*, UnaryOp::*, Value::*},
    };

    #[test]
    fn t_topological_sorting_0() {
        let mut sorter = Pruner::new();
        let (nodes, roots) = sorter
            .run_from_range(vec![Symbol('x'), Binary(Add, 0, 2), Symbol('y')], 1..2)
            .unwrap();
        assert_eq!(roots, 2..3);
        assert_eq!(nodes, vec![Symbol('x'), Symbol('y'), Binary(Add, 0, 1)]);
    }

    #[test]
    fn t_topological_sorting_1() {
        let mut sorter = Pruner::new();
        let (nodes, roots) = sorter
            .run_from_range(
                vec![
                    Symbol('x'),             // 0
                    Binary(Add, 0, 2),       // 1
                    Constant(Scalar(2.245)), // 2
                    Binary(Multiply, 1, 5),  // 3
                    Unary(Sqrt, 3),          // 4 - root
                    Symbol('y'),             // 5
                ],
                4..5,
            )
            .unwrap();
        assert_eq!(roots, 5..6);
        assert_eq!(
            nodes,
            vec![
                Symbol('x'),
                Constant(Scalar(2.245)),
                Binary(Add, 0, 1),
                Symbol('y'),
                Binary(Multiply, 2, 3),
                Unary(Sqrt, 4)
            ]
        );
    }

    #[test]
    fn t_topological_sorting_2() {
        let mut sorter = Pruner::new();
        let (nodes, roots) = sorter
            .run_from_range(
                vec![
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
                ],
                10..11,
            )
            .unwrap();
        assert_eq!(roots, 11..12);
        assert_eq!(
            nodes,
            vec![
                Symbol('x'),
                Symbol('y'),
                Binary(Add, 0, 1),
                Unary(Log, 2),
                Symbol('a'),
                Symbol('b'),
                Binary(Add, 4, 5),
                Binary(Multiply, 3, 6),
                Symbol('p'),
                Symbol('p'),
                Binary(Add, 8, 9),
                Binary(Pow, 7, 10)
            ]
        );
    }

    #[test]
    fn t_sort_concat() {
        let mut sorter = Pruner::new();
        let (nodes, roots) = sorter
            .run_from_range(
                vec![
                    Symbol('p'),            // 0
                    Symbol('x'),            // 1
                    Binary(Multiply, 0, 1), // 2: p * x
                    Symbol('y'),            // 3
                    Binary(Multiply, 0, 3), // 4: p * y
                    Binary(Multiply, 0, 7), // 5: p * (x + y)
                    Constant(Scalar(1.0)),  // 6
                    Binary(Add, 1, 3),      // 7: x + y
                    Binary(Add, 2, 4),      // 8: p * x + p * y
                ],
                5..7,
            )
            .unwrap();
        assert_eq!(roots, 4..6);
        assert_eq!(
            nodes,
            vec![
                Symbol('x'),
                Symbol('y'),
                Binary(Add, 0, 1),
                Symbol('p'),
                Binary(Multiply, 3, 2),
                Constant(Scalar(1.0))
            ]
        );
    }

    #[test]
    fn t_sorting_3() {
        let mut sorter = Pruner::new();
        let (nodes, roots) = sorter
            .run_from_range(
                vec![
                    Symbol('x'),             // 0
                    Constant(Scalar(2.0)),   // 1
                    Binary(Pow, 0, 1),       // 2: x^2
                    Unary(Exp, 2),           // 3: e^(x^2)
                    Constant(Scalar(1.0)),   // 4
                    Constant(Scalar(0.0)),   // 5
                    Unary(Log, 0),           // 6: log(x)
                    Binary(Multiply, 5, 6),  // 7: 0 * log(x)
                    Binary(Divide, 4, 0),    // 8: 1 / x
                    Binary(Multiply, 1, 8),  // 9: 2 * (1 / x)
                    Binary(Add, 7, 9),       // 10: 2 * (1 / x)
                    Binary(Multiply, 2, 10), // 11: x^2 * (2 * (1 / x))
                    Binary(Multiply, 3, 11), // 12: e^(x^2) * 2 * x
                ],
                12..13,
            )
            .unwrap();
        assert_eq!(roots, 12..13);
        assert_eq!(
            nodes,
            vec![
                Symbol('x'),
                Unary(Log, 0),
                Constant(Scalar(0.0)),
                Binary(Multiply, 2, 1),
                Constant(Scalar(1.0)),
                Binary(Divide, 4, 0),
                Constant(Scalar(2.0)),
                Binary(Multiply, 6, 5),
                Binary(Add, 3, 7),
                Binary(Pow, 0, 6),
                Binary(Multiply, 9, 8),
                Unary(Exp, 9),
                Binary(Multiply, 11, 10)
            ]
        );
    }

    #[test]
    fn t_prune_cyclic() {
        use crate::tree::{BinaryOp::*, UnaryOp::*, Value::*};
        let mut sorter = Pruner::new();
        assert!(matches!(
            sorter.run_from_range(
                vec![
                    Binary(Pow, 8, 9),      // 0 - root
                    Symbol('x'),            // 1
                    Binary(Multiply, 0, 1), // 2
                    Symbol('y'),            // 3
                    Binary(Multiply, 0, 3), // 4
                    Binary(Add, 2, 4),      // 5
                    Binary(Add, 1, 3),      // 6
                    Binary(Divide, 5, 6),   // 7
                    Unary(Sqrt, 0),         // 8
                    Constant(Scalar(2.0)),  // 9
                ],
                0..1,
            ),
            Err(Error::CyclicGraph)
        ));
        assert!(matches!(
            sorter.run_from_range(
                vec![
                    Symbol('x'),       // 0
                    Symbol('y'),       // 1
                    Binary(Pow, 0, 6), // 2 - root
                    Binary(Add, 0, 2), // 3
                    Unary(Sqrt, 3),    // 4
                    Unary(Log, 4),     // 5
                    Unary(Negate, 5),  // 6
                ],
                2..3
            ),
            Err(Error::CyclicGraph)
        ));
    }
}
