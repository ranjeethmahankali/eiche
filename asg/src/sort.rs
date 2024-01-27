use crate::{
    tree::{Node, Node::*},
    walk::{DepthWalker, NodeOrdering},
};

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
    /// pruning later. If successful, the new root index is returned.
    pub fn run(
        &mut self,
        nodes: &mut Vec<Node>,
        mut root_index: usize,
    ) -> Result<usize, TopologicalError> {
        // Compute depths of all nodes.
        self.depths.clear();
        self.depths.resize(nodes.len(), 0);
        for (index, maybe_parent) in
            self.walker
                .walk_one(&nodes, root_index, false, NodeOrdering::Original)
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
        std::mem::swap(&mut self.sorted, nodes);
        return Ok(root_index);
    }
}

#[cfg(test)]
mod test {
    use {
        super::*,
        crate::{
            sort::{TopoSorter, TopologicalError},
            tree::{BinaryOp::*, UnaryOp::*},
        },
    };

    #[test]
    fn t_topological_sorting_0() {
        let mut sorter = TopoSorter::new();
        let mut nodes = vec![Symbol('x'), Binary(Add, 0, 2), Symbol('y')];
        let root = sorter.run(&mut nodes, 1).unwrap();
        assert_eq!(root, 2);
        assert_eq!(nodes, vec![Symbol('x'), Symbol('y'), Binary(Add, 0, 1)]);
    }

    #[test]
    fn t_topological_sorting_1() {
        let mut nodes = vec![
            Symbol('x'),            // 0
            Binary(Add, 0, 2),      // 1
            Constant(2.245),        // 2
            Binary(Multiply, 1, 5), // 3
            Unary(Sqrt, 3),         // 4 - root
            Symbol('y'),            // 5
        ];
        let mut sorter = TopoSorter::new();
        let root = sorter.run(&mut nodes, 4).unwrap();
        assert_eq!(root, 5);
        assert_eq!(
            nodes,
            vec![
                Symbol('x'),
                Constant(2.245),
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
        let root = sorter.run(&mut nodes, 10).unwrap();
        assert_eq!(root, 11);
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
            Constant(2.0),          // 9
        ];
        assert!(matches!(
            sorter.run(&mut nodes, 0,),
            Err(TopologicalError::CyclicGraph)
        ));
    }
}
