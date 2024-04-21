use std::ops::Range;

use crate::{
    error::Error,
    tree::{Node, Node::*},
    walk::{DepthWalker, NodeOrdering},
};

/// Topological sorter.
///
/// For the topology of a tree to be considered valid, the root of the
/// tree must be the last node, and every node must appear after its
/// inputs. A `TopoSorter` instance can be used to sort a vector of
/// nodes to be topologically valid.
pub struct TopoSorter {
    index_map: Vec<usize>,
    traverse: Vec<(Node, bool)>,
    scan: Vec<usize>,
    visited: Vec<bool>,
    sorted: Vec<Node>,
    roots: Vec<Node>,
    walker: DepthWalker,
}

impl TopoSorter {
    /// Create a new `TopoSorter` instance.
    pub fn new() -> TopoSorter {
        TopoSorter {
            index_map: Vec::new(),
            traverse: Vec::new(),
            scan: Vec::new(),
            visited: Vec::new(),
            sorted: Vec::new(),
            roots: Vec::new(),
            walker: DepthWalker::new(),
        }
    }

    /// Sort `nodes` to be topologically valid, with `root_indices` as the new
    /// range of root nodes. Depending on the choice of root, the output vector
    /// may be littered with unused nodes, and may require pruning later. If
    /// successful, the new root index is returned.
    pub fn run_from_range(
        &mut self,
        nodes: &mut Vec<Node>,
        root_indices: Range<usize>,
    ) -> Result<Range<usize>, Error> {
        self.walker.priorities_mut().clear();
        self.walker
            .init_from_roots(nodes.len(), root_indices.clone());
        Self::sort_nodes(
            nodes,
            self.walker.walk(&nodes, false, NodeOrdering::Reversed),
            &mut self.index_map,
            root_indices.len(),
            &mut self.traverse,
            &mut self.scan,
            &mut self.visited,
            &mut self.sorted,
            &mut self.roots,
        );
        std::mem::swap(&mut self.sorted, nodes);
        Self::compute_heights(nodes, self.walker.priorities_mut());
        self.walker
            .init_from_roots(nodes.len(), (nodes.len() - root_indices.len())..nodes.len());
        Self::sort_nodes(
            nodes,
            self.walker.walk(&nodes, false, NodeOrdering::Reversed),
            &mut self.index_map,
            root_indices.len(),
            &mut self.traverse,
            &mut self.scan,
            &mut self.visited,
            &mut self.sorted,
            &mut self.roots,
        );
        std::mem::swap(&mut self.sorted, nodes);
        // TODO: No need to return error from this function.
        return Ok((nodes.len() - root_indices.len())..nodes.len());
    }

    pub fn run_from_slice(
        &mut self,
        nodes: &mut Vec<Node>,
        roots: &mut [usize],
    ) -> Result<(), Error> {
        self.walker.priorities_mut().clear();
        self.walker
            .init_from_roots(nodes.len(), roots.iter().map(|r| *r));
        Self::sort_nodes(
            nodes,
            self.walker.walk(nodes, false, NodeOrdering::Reversed),
            &mut self.index_map,
            roots.len(),
            &mut self.traverse,
            &mut self.scan,
            &mut self.visited,
            &mut self.sorted,
            &mut self.roots,
        );
        std::mem::swap(&mut self.sorted, nodes);
        Self::compute_heights(nodes, self.walker.priorities_mut());
        self.walker
            .init_from_roots(nodes.len(), (nodes.len() - roots.len())..nodes.len());
        Self::sort_nodes(
            nodes,
            self.walker.walk(nodes, false, NodeOrdering::Reversed),
            &mut self.index_map,
            roots.len(),
            &mut self.traverse,
            &mut self.scan,
            &mut self.visited,
            &mut self.sorted,
            &mut self.roots,
        );
        std::mem::swap(&mut self.sorted, nodes);
        let num_roots = roots.len();
        for (r, i) in roots.iter_mut().zip((nodes.len() - num_roots)..nodes.len()) {
            *r = i;
        }
        // TODO: This function need not return an error.
        return Ok(());
    }

    fn compute_heights(nodes: &[Node], heights: &mut Vec<usize>) {
        heights.clear();
        heights.resize(nodes.len(), 0);
        for (i, node) in nodes.iter().enumerate() {
            heights[i] = usize::max(
                heights[i],
                match node {
                    Constant(_) | Symbol(_) => 0,
                    Unary(_, input) => 1 + heights[*input],
                    Binary(_, lhs, rhs) => 1 + usize::max(heights[*lhs], heights[*rhs]),
                    Ternary(_, a, b, c) => {
                        1 + usize::max(heights[*a], usize::max(heights[*b], heights[*c]))
                    }
                },
            );
        }
    }

    fn sort_nodes<I: Iterator<Item = (usize, Option<usize>)>>(
        nodes: &[Node],
        depth_first_walk: I,
        indexmap: &mut Vec<usize>,
        num_roots: usize,
        traverse: &mut Vec<(Node, bool)>,
        scan: &mut Vec<usize>,
        visited: &mut Vec<bool>,
        sorted: &mut Vec<Node>,
        roots: &mut Vec<Node>,
    ) {
        traverse.clear();
        traverse.reserve(nodes.len());
        roots.clear();
        roots.reserve(num_roots);
        indexmap.clear();
        indexmap.resize(nodes.len(), 0);
        visited.clear();
        visited.resize(nodes.len(), false);
        for (index, maybe_parent) in depth_first_walk {
            if visited[index] {
                traverse[indexmap[index]].1 = false;
            }
            visited[index] = true;
            match maybe_parent {
                Some(_) => {
                    indexmap[index] = traverse.len();
                    traverse.push((nodes[index], true));
                }
                None => {
                    // This is a root node because it has no parent.
                    roots.push(nodes[index]);
                }
            }
        }
        scan.clear();
        scan.reserve(traverse.len());
        sorted.clear();
        {
            let mut i = 0usize;
            for (node, keep) in traverse.iter() {
                scan.push(i);
                if *keep {
                    sorted.push(*node);
                    i += 1;
                }
            }
        }
        if !sorted.is_empty() {
            // Remap indices after deleting nodes.
            for i in indexmap.iter_mut() {
                *i = scan[*i];
            }
        }
        if sorted.len() > 1 {
            // Reverse the nodes.
            sorted.reverse();
            for i in indexmap.iter_mut() {
                *i = sorted.len() - *i - 1;
            }
        }
        sorted.extend(roots.drain(..));
        for node in sorted {
            match node {
                Constant(_) | Symbol(_) => {} // Nothing.
                Unary(_, input) => *input = indexmap[*input],
                Binary(_, lhs, rhs) => {
                    *lhs = indexmap[*lhs];
                    *rhs = indexmap[*rhs];
                }
                Ternary(_, a, b, c) => {
                    *a = indexmap[*a];
                    *b = indexmap[*b];
                    *c = indexmap[*c];
                }
            }
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
        let mut sorter = TopoSorter::new();
        let mut nodes = vec![Symbol('x'), Binary(Add, 0, 2), Symbol('y')];
        let root = sorter.run_from_range(&mut nodes, 1..2).unwrap();
        assert_eq!(root, 2..3);
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
        let mut sorter = TopoSorter::new();
        let root = sorter.run_from_range(&mut nodes, 4..5).unwrap();
        assert_eq!(root, 5..6);
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
        let mut sorter = TopoSorter::new();
        let root = sorter.run_from_range(&mut nodes, 10..11).unwrap();
        assert_eq!(root, 11..12);
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
        let mut sorter = TopoSorter::new();
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
        let roots = sorter.run_from_range(&mut nodes, 5..7).unwrap();
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
        let mut sorter = TopoSorter::new();
        let roots = sorter.run_from_range(&mut nodes, 12..13).unwrap();
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
}
