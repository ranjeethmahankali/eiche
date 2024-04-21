use std::ops::Range;

use crate::tree::{Node, Node::*, Tree};

/// Topological sorter.
///
/// For the topology of a tree to be considered valid, the root of the
/// tree must be the last node, and every node must appear after its
/// inputs. A `Pruner` instance can be used to sort a vector of
/// nodes to be topologically valid, and remove unused nodes.
pub struct Pruner {
    index_map: Vec<usize>,
    traverse: Vec<(Node, bool)>,
    scan: Vec<usize>,
    visited: Vec<bool>,
    sorted: Vec<Node>,
    roots: Vec<Node>,
    heights: Vec<usize>,
    stack: Vec<(usize, bool)>,
}

impl Pruner {
    /// Create a new `TopoSorter` instance.
    pub fn new() -> Pruner {
        Pruner {
            index_map: Vec::new(),
            traverse: Vec::new(),
            scan: Vec::new(),
            visited: Vec::new(),
            sorted: Vec::new(),
            roots: Vec::new(),
            heights: Vec::new(),
            stack: Vec::new(),
        }
    }

    /// Sort `nodes` to be topologically valid, with `root_indices` as the new
    /// range of root nodes. Depending on the choice of root, the output vector
    /// may be littered with unused nodes, and may require pruning later. If
    /// successful, the new root index is returned.
    pub fn run_from_range(&mut self, nodes: &mut Vec<Node>, root_indices: Range<usize>) {
        self.init_traverse(nodes.len(), root_indices.clone());
        self.sort_nodes(nodes, root_indices.len(), false);
        std::mem::swap(&mut self.sorted, nodes);
        self.compute_heights(nodes);
        self.init_traverse(nodes.len(), (nodes.len() - root_indices.len())..nodes.len());
        self.sort_nodes(nodes, root_indices.len(), true);
        std::mem::swap(&mut self.sorted, nodes);
    }

    pub fn run_from_slice(&mut self, nodes: &mut Vec<Node>, roots: &mut [usize]) {
        self.init_traverse(nodes.len(), roots.iter().map(|r| *r));
        self.sort_nodes(nodes, roots.len(), false);
        std::mem::swap(&mut self.sorted, nodes);
        self.compute_heights(nodes);
        self.init_traverse(nodes.len(), (nodes.len() - roots.len())..nodes.len());
        self.sort_nodes(nodes, roots.len(), true);
        std::mem::swap(&mut self.sorted, nodes);
        let num_roots = roots.len();
        for (r, i) in roots.iter_mut().zip((nodes.len() - num_roots)..nodes.len()) {
            *r = i;
        }
    }

    fn init_traverse<I: Iterator<Item = usize>>(&mut self, num_nodes: usize, roots: I) {
        self.stack.clear();
        self.stack.extend(roots.map(|r| (r, true)));
        self.stack.reverse();
        self.visited.clear();
        self.visited.resize(num_nodes, false);
    }

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

    fn sort_nodes(&mut self, nodes: &[Node], num_roots: usize, use_height: bool) {
        self.traverse.clear();
        self.traverse.reserve(nodes.len());
        self.roots.clear();
        self.roots.reserve(num_roots);
        self.index_map.clear();
        self.index_map.resize(nodes.len(), 0);
        self.visited.clear();
        self.visited.resize(nodes.len(), false);
        while let Some((index, is_root)) = self.stack.pop() {
            if self.visited[index] {
                self.traverse[self.index_map[index]].1 = false;
            }
            self.visited[index] = true;
            if is_root {
                self.roots.push(nodes[index]);
            } else {
                self.index_map[index] = self.traverse.len();
                self.traverse.push((nodes[index], true));
            }
            let num_children = match nodes[index] {
                Constant(_) | Symbol(_) => 0,
                Unary(_op, input) => {
                    self.stack.push((input, false));
                    1
                }
                Binary(_op, lhs, rhs) => {
                    self.stack.push((lhs, false));
                    self.stack.push((rhs, false));
                    2
                }
                Ternary(_op, a, b, c) => {
                    self.stack.push((a, false));
                    self.stack.push((b, false));
                    self.stack.push((c, false));
                    3
                }
            };
            if use_height {
                let num = self.stack.len();
                self.stack[(num - num_children)..].sort_by(|(a, _), (b, _)| {
                    return self.heights[*b].cmp(&self.heights[*a]);
                });
            }
        }
        // Only some of the nodes from the traverse will be retained. We're
        // doing an exclusive scan to get the indices of the nodes that are
        // retained.
        self.scan.clear();
        self.scan.reserve(self.traverse.len());
        self.sorted.clear();
        {
            let mut i = 0usize;
            for (node, keep) in self.traverse.iter() {
                self.scan.push(i);
                if *keep {
                    self.sorted.push(*node);
                    i += 1;
                }
            }
        }
        if !self.sorted.is_empty() {
            for i in self.index_map.iter_mut() {
                *i = self.scan[*i];
            }
        }
        if self.sorted.len() > 1 {
            // The correct topological order is the reverse of a depth first
            // traversal. So we reverse the nodes and adjust the indices.
            self.sorted.reverse();
            for i in self.index_map.iter_mut() {
                *i = self.sorted.len() - *i - 1;
            }
        }
        // Push the roots.
        self.sorted.extend(self.roots.drain(..));
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
    }
}

impl Tree {
    /// Prunes the tree and topologically sorts the nodes.
    pub fn prune(mut self, pruner: &mut Pruner) -> Tree {
        let roots = self.root_indices();
        pruner.run_from_range(self.nodes_mut(), roots);
        return self;
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
        let mut nodes = vec![Symbol('x'), Binary(Add, 0, 2), Symbol('y')];
        sorter.run_from_range(&mut nodes, 1..2);
        assert_eq!(nodes, vec![Symbol('x'), Symbol('y'), Binary(Add, 0, 1)]);
    }

    #[test]
    fn t_topological_sorting_1() {
        let mut nodes = vec![
            Symbol('x'),             // 0
            Binary(Add, 0, 2),       // 1
            Constant(Scalar(2.245)), // 2
            Binary(Multiply, 1, 5),  // 3
            Unary(Sqrt, 3),          // 4 - root
            Symbol('y'),             // 5
        ];
        let mut sorter = Pruner::new();
        sorter.run_from_range(&mut nodes, 4..5);
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
        let mut nodes = vec![
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
        let mut sorter = Pruner::new();
        sorter.run_from_range(&mut nodes, 10..11);
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
        let mut nodes = vec![
            Symbol('p'),            // 0
            Symbol('x'),            // 1
            Binary(Multiply, 0, 1), // 2: p * x
            Symbol('y'),            // 3
            Binary(Multiply, 0, 3), // 4: p * y
            Binary(Multiply, 0, 7), // 5: p * (x + y)
            Constant(Scalar(1.0)),  // 6
            Binary(Add, 1, 3),      // 7: x + y
            Binary(Add, 2, 4),      // 8: p * x + p * y
        ];
        sorter.run_from_range(&mut nodes, 5..7);
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
        let mut nodes = vec![
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
        ];
        let mut sorter = Pruner::new();
        sorter.run_from_range(&mut nodes, 12..13);
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
}
