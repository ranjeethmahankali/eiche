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

    /// Sort `nodes` to be topologically valid, with `root_indices` as the new
    /// range of root nodes. Depending on the choice of root, the output vector
    /// may be littered with unused nodes, and may require pruning later. If
    /// successful, the new root index is returned.
    pub fn run_from_range(
        &mut self,
        nodes: &mut Vec<Node>,
        root_indices: Range<usize>,
    ) -> Result<Range<usize>, Error> {
        // Compute depths of all nodes.
        Self::compute_depths(
            nodes,
            &mut self.depths,
            self.walker.walk_from_range(
                &nodes,
                root_indices.clone(),
                false,
                NodeOrdering::Original,
            ),
        )?;
        self.sort_by_depth(nodes);
        // Find the new range of roots and return.
        return match self
            .sorted_indices
            .iter()
            .position(|index| *index == root_indices.start)
        {
            Some(i) => Ok(i..(i + (root_indices.end - root_indices.start))),
            None => Err(Error::InvalidTopology),
        };
    }

    pub fn run_from_slice(
        &mut self,
        nodes: &mut Vec<Node>,
        roots: &mut [usize],
    ) -> Result<(), Error> {
        Self::compute_depths(
            nodes,
            &mut self.depths,
            self.walker
                .walk_from_slice(&nodes, roots, false, NodeOrdering::Original),
        )?;
        self.sort_by_depth(nodes);
        // Update roots
        for root in roots {
            *root = match self.sorted_indices.iter().position(|index| *index == *root) {
                Some(i) => i,
                None => return Err(Error::InvalidTopology),
            };
        }
        return Ok(());
    }

    fn compute_depths<I: Iterator<Item = (usize, Option<usize>)>>(
        nodes: &[Node],
        depths: &mut Vec<usize>,
        roots: I,
    ) -> Result<(), Error> {
        depths.clear();
        depths.resize(nodes.len(), 0);
        for (index, maybe_parent) in roots {
            if let Some(parent) = maybe_parent {
                depths[index] = usize::max(depths[index], 1 + depths[parent]);
                if depths[index] >= nodes.len() {
                    // TODO: This is a terrible way to detect large
                    // cycles. As you'd have to traverse the whole
                    // cycle many times to reach this
                    // condition. Implement proper cycle detection
                    // later.
                    return Err(Error::CyclicGraph);
                }
            }
        }
        return Ok(());
    }

    fn sort_by_depth(&mut self, nodes: &mut Vec<Node>) {
        // Sort the node indices by depth.
        self.sorted_indices.clear();
        self.sorted_indices.extend(0..nodes.len());
        // Highest depth at the start.
        self.sorted_indices
            .sort_by(|a, b| self.depths[*b].cmp(&self.depths[*a]));
        // Build a map from old indices to new indices.
        self.index_map.clear();
        self.index_map.resize(nodes.len(), 0);
        for (i, index) in self.sorted_indices.iter().enumerate() {
            self.index_map[*index] = i;
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
                    Ternary(op, a, b, c) => {
                        Ternary(op, self.index_map[a], self.index_map[b], self.index_map[c])
                    }
                }
            }));
        // Swap the sorted nodes and the incoming nodes.
        std::mem::swap(&mut self.sorted, nodes);
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
                Symbol('a'),
                Symbol('b'),
                Binary(Add, 0, 1),
                Binary(Add, 2, 3),
                Unary(Log, 4),
                Symbol('p'),
                Symbol('p'),
                Binary(Add, 7, 8),
                Binary(Multiply, 6, 5),
                Binary(Pow, 10, 9)
            ]
        );
    }

    #[test]
    fn t_cyclic_graph() {
        let mut sorter = TopoSorter::new();
        let mut nodes = vec![
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
        assert!(matches!(
            sorter.run_from_range(&mut nodes, 0..1),
            Err(Error::CyclicGraph)
        ));
    }

    #[test]
    fn t_sort_concat() {
        let mut sorter = TopoSorter::new();
        let mut nodes = vec![
            Symbol('p'),
            Symbol('x'),
            Binary(Multiply, 0, 1),
            Symbol('y'),
            Binary(Multiply, 0, 3),
            Binary(Multiply, 0, 7),
            Constant(Scalar(1.0)),
            Binary(Add, 1, 3),
            Binary(Add, 2, 4),
        ];
        let roots = sorter.run_from_range(&mut nodes, 5..7).unwrap();
        assert_eq!(roots, 6..8);
        assert_eq!(
            nodes,
            vec![
                Symbol('x'),
                Symbol('y'),
                Symbol('p'),
                Binary(Add, 0, 1),
                Binary(Multiply, 2, 0),
                Binary(Multiply, 2, 1),
                Binary(Multiply, 2, 3),
                Constant(Scalar(1.0)),
                Binary(Add, 4, 5)
            ]
        );
    }
}
