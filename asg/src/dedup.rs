use crate::{
    hash::hash_nodes,
    tree::{MaybeTree, Node, Node::*, Tree},
    walk::{DepthWalker, NodeOrdering},
};
use std::{collections::HashMap, ops::Range};

/// Helper struct for deduplicating common subtrees.
///
/// Deduplication requires allocations. Those buffers are owned by
/// this struct, so reusing the same instance of `Deduplicater` can
/// avoid unnecessary allocations.
pub struct Deduplicater {
    indices: Vec<usize>,
    hashes: Vec<u64>,
    walker1: DepthWalker,
    walker2: DepthWalker,
    hash_to_index: HashMap<u64, usize>,
}

impl Deduplicater {
    /// Create a new deduplicater.
    pub fn new() -> Self {
        Deduplicater {
            indices: vec![],
            hashes: vec![],
            walker1: DepthWalker::new(),
            walker2: DepthWalker::new(),
            hash_to_index: HashMap::new(),
        }
    }

    /// Deduplicate `nodes`. The `nodes` are expected to be
    /// topologically sorted. If they are not, this function might
    /// produce incorrect results. If you suspect the nodes are not
    /// topologically sorted, use the `TopoSorter` to sort them first.
    ///
    /// If a subtree appears twice, any node with the second subtree
    /// as its input will be rewired to the first subtree. That means,
    /// after deduplication, there can be `dead` nodes remaining, that
    /// are not connected to the root. Consider pruning the tree
    /// afterwards.
    pub fn run(&mut self, nodes: &mut Vec<Node>) {
        // Compute unique indices after deduplication.
        self.indices.clear();
        self.indices.extend(0..nodes.len());
        hash_nodes(&nodes, &mut self.hashes);
        self.hash_to_index.clear();
        for i in 0..self.hashes.len() {
            let h = self.hashes[i];
            let entry = self.hash_to_index.entry(h).or_insert(i);
            if *entry != i
                && equivalent(
                    *entry,
                    i,
                    &nodes,
                    &nodes,
                    &mut self.walker1,
                    &mut self.walker2,
                )
            {
                // The i-th node should be replaced with entry-th node.
                self.indices[i] = *entry;
            }
        }
        // Update nodes.
        for node in nodes.iter_mut() {
            match node {
                Constant(_) => {}
                Symbol(_) => {}
                Unary(_, input) => {
                    *input = self.indices[*input];
                }
                Binary(_, lhs, rhs) => {
                    // Copy to temporary buffer to avoid side effects if lhs and rhs are the same.
                    let mapped = [self.indices[*lhs], self.indices[*rhs]];
                    *lhs = mapped[0];
                    *rhs = mapped[1];
                }
                Ternary(_, a, b, c) => {
                    // Copy to temporary buffer to avoid side effects when a, b, c are not unique.
                    let mapped = [self.indices[*a], self.indices[*b], self.indices[*c]];
                    *a = mapped[0];
                    *b = mapped[1];
                    *c = mapped[2];
                }
            }
        }
    }
}

/// Check if the nodes at indices `left` and `right` are
/// equivalent.
///
/// Two nodes need not share the same input needs to be
/// equivalent. They just need to represent the same mathematical
/// expression. For example, two distinct constant nodes with the
/// holding the same value are equivalent. Two nodes of the same type
/// with equivalent inputs are considered equivalent. For binary nodes
/// with commutative operations, checking the equivalence of the
/// inputs is done in an order agnostic way.
///
/// This implementation avoids recursion by using `walker1` and
/// `walker2` are used to traverse the tree depth wise and perform the
/// comparison.
pub fn equivalent_many(
    left: Range<usize>,
    right: Range<usize>,
    lnodes: &[Node],
    rnodes: &[Node],
    lwalker: &mut DepthWalker,
    rwalker: &mut DepthWalker,
) -> bool {
    {
        // Zip the depth first iterators and compare.
        let mut liter = lwalker.walk_many(&lnodes, left, false, NodeOrdering::Deterministic);
        let mut riter = rwalker.walk_many(&rnodes, right, false, NodeOrdering::Deterministic);
        loop {
            match (liter.next(), riter.next()) {
                (None, None) => {
                    // Both iterators ended.
                    return true;
                }
                (None, Some(_)) | (Some(_), None) => {
                    // One of the iterators ended prematurely.
                    return false;
                }
                (Some((li, _lp)), Some((ri, _rp))) => {
                    if std::ptr::eq(lnodes, rnodes) && li == ri {
                        liter.skip_children();
                        riter.skip_children();
                        continue;
                    }
                    if !match (lnodes[li], rnodes[ri]) {
                        (Constant(v1), Constant(v2)) => v1 == v2,
                        (Symbol(c1), Symbol(c2)) => c1 == c2,
                        (Unary(op1, _input1), Unary(op2, _input2)) => op1 == op2,
                        (Binary(op1, _lhs1, _rhs1), Binary(op2, _lhs2, _rhs2)) => op1 == op2,
                        _ => false,
                    } {
                        return false;
                    }
                }
            }
        }
    }
}

pub fn equivalent(
    left: usize,
    right: usize,
    lnodes: &[Node],
    rnodes: &[Node],
    lwalker: &mut DepthWalker,
    rwalker: &mut DepthWalker,
) -> bool {
    equivalent_many(
        left..(left + 1),
        right..(right + 1),
        lnodes,
        rnodes,
        lwalker,
        rwalker,
    )
}

impl Tree {
    /// Deduplicate the common subtrees in this tree.
    pub fn deduplicate(mut self, dedup: &mut Deduplicater) -> MaybeTree {
        dedup.run(self.nodes_mut());
        return self.validated();
    }

    pub fn equivalent(&self, other: &Tree) -> bool {
        let mut lwalker = DepthWalker::new();
        let mut rwalker = DepthWalker::new();
        equivalent_many(
            self.root_indices(),
            other.root_indices(),
            self.nodes(),
            other.nodes(),
            &mut lwalker,
            &mut rwalker,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{deftree, prune::Pruner, test::util::compare_trees, tree::BinaryOp::*};

    #[test]
    fn t_recursive_compare() {
        // Check if 'Add' node with mirrored inputs is compared
        // correctly.
        let mut nodes = vec![
            Symbol('y'),            // 0
            Symbol('x'),            // 1
            Binary(Add, 0, 1),      // 2
            Symbol('x'),            // 3
            Symbol('y'),            // 4
            Binary(Add, 3, 4),      // 5
            Binary(Add, 5, 2),      // 6
            Binary(Add, 2, 2),      // 7
            Binary(Multiply, 6, 7), // 8
        ];
        let mut walker1 = DepthWalker::new();
        let mut walker2 = DepthWalker::new();
        assert!(equivalent(2, 5, &nodes, &nodes, &mut walker1, &mut walker2));
        assert!(equivalent(6, 7, &nodes, &nodes, &mut walker1, &mut walker2));
        // Try more mirroring
        nodes[6] = Binary(Add, 2, 5);
        assert!(equivalent(2, 5, &nodes, &nodes, &mut walker1, &mut walker2));
        assert!(equivalent(6, 7, &nodes, &nodes, &mut walker1, &mut walker2));
        // Multiply node with mirrored inputs.
        nodes[2] = Binary(Multiply, 0, 1);
        nodes[5] = Binary(Multiply, 3, 4);
        assert!(equivalent(2, 5, &nodes, &nodes, &mut walker1, &mut walker2));
        assert!(equivalent(6, 7, &nodes, &nodes, &mut walker1, &mut walker2));
        // Min node with mirrored inputs.
        nodes[2] = Binary(Min, 0, 1);
        nodes[5] = Binary(Min, 3, 4);
        assert!(equivalent(2, 5, &nodes, &nodes, &mut walker1, &mut walker2));
        assert!(equivalent(6, 7, &nodes, &nodes, &mut walker1, &mut walker2));
        // Max node with mirrored inputs.
        nodes[2] = Binary(Max, 0, 1);
        nodes[5] = Binary(Max, 3, 4);
        assert!(equivalent(2, 5, &nodes, &nodes, &mut walker1, &mut walker2));
        assert!(equivalent(6, 7, &nodes, &nodes, &mut walker1, &mut walker2));
        // Subtract node with mirrored inputs.
        nodes[2] = Binary(Subtract, 0, 1);
        nodes[5] = Binary(Subtract, 3, 4);
        assert!(!equivalent(
            2,
            5,
            &nodes,
            &nodes,
            &mut walker1,
            &mut walker2
        ));
        assert!(!equivalent(
            6,
            7,
            &nodes,
            &nodes,
            &mut walker1,
            &mut walker2
        ));
        // Divide node with mirrored inputs.
        nodes[2] = Binary(Divide, 0, 1);
        nodes[5] = Binary(Divide, 3, 4);
        assert!(!equivalent(
            2,
            5,
            &nodes,
            &nodes,
            &mut walker1,
            &mut walker2
        ));
        assert!(!equivalent(
            6,
            7,
            &nodes,
            &nodes,
            &mut walker1,
            &mut walker2
        ));
        // Pow node with mirrored inputs.
        nodes[2] = Binary(Pow, 0, 1);
        nodes[5] = Binary(Pow, 3, 4);
        assert!(!equivalent(
            2,
            5,
            &nodes,
            &nodes,
            &mut walker1,
            &mut walker2
        ));
        assert!(!equivalent(
            6,
            7,
            &nodes,
            &nodes,
            &mut walker1,
            &mut walker2
        ));
    }

    #[test]
    fn t_recursive_compare_2() {
        let tree1 = deftree!(/ (+ (* x y) (+ b a)) (* (+ y x) (* a b))).unwrap();
        let tree2 = deftree!(/ (+ (* y x) (+ a b)) (* (+ y x) (* b a))).unwrap();
        assert!(tree1.equivalent(&tree2));
    }

    #[test]
    fn t_recursive_compare_concat() {
        let tree1 = deftree!(concat (+ x y) (* y x)).unwrap();
        let tree2 = deftree!(concat (+ y x) (* x y)).unwrap();
        assert!(tree1.equivalent(&tree2));
        let tree1 = deftree!(concat
                             (/ 1 (log (+ x y)))
                             (/ (+ (* x y) (+ b a)) (* (+ y x) (* a b))))
        .unwrap();
        let tree2 = deftree!(concat
                             (/ 1 (log (+ x y)))
                             (/ (+ (* y x) (+ a b)) (* (+ y x) (* b a))))
        .unwrap();
        assert!(tree1.equivalent(&tree2));
        let tree2 = deftree!(concat
                             (/ 1 (log (* x y)))
                             (/ (+ (* y x) (+ a b)) (* (+ y x) (* b a))))
        .unwrap();
        assert!(!tree1.equivalent(&tree2));
    }

    #[test]
    fn t_recursive_compare_3() {
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        // Sanity check with the same tree.
        let a = deftree!(/ (* k (+ x y)) (+ x y))
            .unwrap()
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner);
        let b = deftree!(/ (* k (+ x y)) (+ x y))
            .unwrap()
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner);
        let mut walker1 = DepthWalker::new();
        let mut walker2 = DepthWalker::new();
        assert!(equivalent_many(
            a.root_indices(),
            b.root_indices(),
            a.nodes(),
            b.nodes(),
            &mut walker1,
            &mut walker2
        ));
        assert!(a.equivalent(&b));
    }

    #[test]
    fn t_deduplication_1() {
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        let tree = deftree!(
            (max (min
                  (- (sqrt (+ (+ (pow (- x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 2.75)
                  (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 4.))
             (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (+ y 3.) 2.)) (pow (- z 4.) 2.))) 5.25))
        )
        .unwrap();
        let nodup = tree
            .clone()
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner);
        assert!(tree.len() > nodup.len());
        assert_eq!(nodup.len(), 32);
        compare_trees(
            &tree,
            &nodup,
            &[('x', -10., 10.), ('y', -9., 10.), ('z', -11., 12.)],
            20,
            0.,
        );
    }

    #[test]
    fn t_deduplication_2() {
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        let tree = deftree!(/ (pow (log (+ (sin x) 2.)) 3.) (+ (cos x) 2.)).unwrap();
        let nodup = tree
            .clone()
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner);
        assert!(tree.len() > nodup.len());
        assert_eq!(nodup.len(), 10);
        compare_trees(&tree, &nodup, &[('x', -10., 10.)], 400, 0.);
    }

    #[test]
    fn t_deduplication_3() {
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        let tree = deftree!(
            (/
             (+ (pow (sin x) 2.) (+ (pow (cos x) 2.) (* 2. (* (sin x) (cos x)))))
             (+ (pow (sin y) 2.) (+ (pow (cos y) 2.) (* 2. (* (sin y) (cos y))))))
        )
        .unwrap();
        let nodup = tree
            .clone()
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner);
        assert!(tree.len() > nodup.len());
        assert_eq!(nodup.len(), 20);
        compare_trees(&tree, &nodup, &[('x', -10., 10.), ('y', -9., 10.)], 20, 0.);
    }
}
