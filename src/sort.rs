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
    sorted: Vec<Node>,
    roots: Vec<Node>,
    walker: DepthWalker,
}

impl TopoSorter {
    /// Create a new `TopoSorter` instance.
    pub fn new() -> TopoSorter {
        TopoSorter {
            index_map: Vec::new(),
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
        Self::sort_nodes(
            nodes,
            self.walker
                .walk_from_roots(&nodes, root_indices.clone(), true, NodeOrdering::Reversed),
            &mut self.index_map,
            root_indices.len(),
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
        Self::sort_nodes(
            nodes,
            self.walker.walk_from_roots(
                nodes,
                roots.iter().map(|r| *r),
                true,
                NodeOrdering::Reversed,
            ),
            &mut self.index_map,
            roots.len(),
            &mut self.sorted,
            &mut self.roots,
        );
        std::mem::swap(&mut self.sorted, nodes);
        let num_roots = roots.len();
        for (r, i) in roots.iter_mut().zip((nodes.len() - num_roots)..nodes.len()) {
            *r = i;
        }
        return Ok(());
    }

    fn sort_nodes<I: Iterator<Item = (usize, Option<usize>)>>(
        nodes: &[Node],
        depth_first_walk: I,
        indexmap: &mut Vec<usize>,
        num_roots: usize,
        sorted: &mut Vec<Node>,
        roots: &mut Vec<Node>,
    ) {
        sorted.clear();
        sorted.reserve(nodes.len());
        roots.clear();
        roots.reserve(num_roots);
        indexmap.clear();
        indexmap.resize(nodes.len(), 0);
        for (index, maybe_parent) in depth_first_walk {
            match maybe_parent {
                Some(_) => {
                    indexmap[index] = sorted.len();
                    sorted.push(nodes[index]);
                }
                None => {
                    // This is a root node because it has no parent.
                    roots.push(nodes[index]);
                }
            }
        }
        sorted.reverse();
        for i in indexmap.iter_mut() {
            *i = sorted.len() - *i - 1;
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
                Symbol('p'),
                Symbol('x'),
                Symbol('y'),
                Binary(Add, 1, 2),
                Binary(Multiply, 0, 3),
                Constant(Scalar(1.0))
            ]
        );
    }
}
