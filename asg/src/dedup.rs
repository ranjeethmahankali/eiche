use crate::{
    tree::{Node, Node::*},
    walk::{DepthWalker, NodeOrdering},
};
use std::collections::HashMap;

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

    fn calc_hashes(&mut self, nodes: &Vec<Node>) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        // Using a boxed slice to avoid accidental resizing later.
        self.hashes.clear();
        self.hashes.resize(nodes.len(), 0);
        for index in 0..nodes.len() {
            let hash: u64 = match nodes[index] {
                Constant(value) => value.to_bits().into(),
                Symbol(label) => {
                    let mut s: DefaultHasher = Default::default();
                    label.hash(&mut s);
                    s.finish()
                }
                Unary(op, input) => {
                    let mut s: DefaultHasher = Default::default();
                    op.hash(&mut s);
                    self.hashes[input].hash(&mut s);
                    s.finish()
                }
                Binary(op, lhs, rhs) => {
                    let (hash1, hash2) = {
                        let mut hash1 = self.hashes[lhs];
                        let mut hash2 = self.hashes[rhs];
                        if op.is_commutative() && hash1 > hash2 {
                            std::mem::swap(&mut hash1, &mut hash2);
                        }
                        (hash1, hash2)
                    };
                    let mut s: DefaultHasher = Default::default();
                    op.hash(&mut s);
                    hash1.hash(&mut s);
                    hash2.hash(&mut s);
                    s.finish()
                }
            };
            self.hashes[index] = hash;
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
    pub fn run(&mut self, mut nodes: Vec<Node>) -> Vec<Node> {
        // Compute unique indices after deduplication.
        self.indices.clear();
        self.indices.extend(0..nodes.len());
        self.calc_hashes(&nodes);
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
                    *lhs = self.indices[*lhs];
                    *rhs = self.indices[*rhs];
                }
            }
        }
        return nodes;
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
pub fn equivalent(
    left: usize,
    right: usize,
    lnodes: &[Node],
    rnodes: &[Node],
    lwalker: &mut DepthWalker,
    rwalker: &mut DepthWalker,
) -> bool {
    {
        // Zip the depth first iterators and compare.
        let mut liter = lwalker.walk_nodes(&lnodes, left, false, NodeOrdering::Deterministic);
        let mut riter = rwalker.walk_nodes(&rnodes, right, false, NodeOrdering::Deterministic);
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
                        println!("Bailing out early!");
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
