use crate::{Node::*, Tree};

/// Used to manage a dominator mapping between nodes of a tree.
///
/// Because this is used to manage the dominator mapping of any node with any
/// other node, the width and the height of the table are always eual to the
/// number of nodes, i.e. `size`.
struct DomTable {
    bits: Vec<u64>,
    n_chunks: usize, // Number of bytes per row.
}

impl DomTable {
    const CHUNK_SIZE: usize = 64;

    fn set(bits: &mut [u64], i: usize) {
        let quot = i / Self::CHUNK_SIZE;
        let rem = i % Self::CHUNK_SIZE;
        bits[quot] |= 1 << rem;
    }

    pub fn new(n_nodes: usize) -> Self {
        let quot = n_nodes / Self::CHUNK_SIZE;
        let rem = n_nodes % Self::CHUNK_SIZE;
        let n_chunks = quot + (if rem == 0 { 0 } else { 1 });
        let bits = {
            // Initially each node is it's own dominator.
            let mut out = vec![0u64; n_chunks * n_chunks];
            for (ni, chunk) in out.chunks_exact_mut(n_chunks).enumerate() {
                Self::set(chunk, ni);
            }
            out
        };
        DomTable { bits, n_chunks }
    }

    pub fn dominate(&mut self, parent: usize, child: usize) {
        let (poff, coff) = (parent * self.n_chunks, child * self.n_chunks);
        let [parent_bits, child_bits] = unsafe {
            self.bits.get_disjoint_unchecked_mut([
                coff..(coff + self.n_chunks),
                poff..(poff + self.n_chunks),
            ])
        };
        for (p, c) in parent_bits.iter().zip(child_bits.iter_mut()) {
            *c &= *p;
        }
    }

    pub fn immediate_dominator(&self, child: usize) -> usize {
        let offset = child * self.n_chunks;
        // Iterate through flags in reverse and find the index of the first set flag.
        self.bits[offset..(offset + self.n_chunks)]
            .iter()
            .enumerate()
            .find_map(|(i, flags)| match flags.trailing_zeros() {
                64 => None,
                n => Some(i * Self::CHUNK_SIZE + n as usize),
            })
            .unwrap_or(child) // If no dominator found then return the node itself.
    }
}

impl Tree {
    /// Performs dominator sort on this tree. After this, each node will be
    /// precended by a contiguous range of it's dependencies are dominatoed by
    /// that node. This criteria also holds recursively for the nodes within
    /// that contiguous range. A vector containing the sizes of these dominated
    /// ranges is returned. i.e. for each node the entry in this vector
    /// indicates the number of nodes it exclusively dominates.
    pub fn dominator_sort(self) -> (Tree, Vec<usize>) {
        let sorted: Vec<_> = {
            let (indices, rev_indices) = {
                let keys: Vec<_> = {
                    let domtable = {
                        let mut table = DomTable::new(self.len());
                        for (i, node) in self.nodes().iter().enumerate().rev() {
                            match node {
                                Constant(_) | Symbol(_) => {} // Do nothing.
                                Unary(_, input) => table.dominate(i, *input),
                                Binary(_, lhs, rhs) => {
                                    table.dominate(i, *lhs);
                                    table.dominate(i, *rhs);
                                }
                                Ternary(_, a, b, c) => {
                                    table.dominate(i, *a);
                                    table.dominate(i, *b);
                                    table.dominate(i, *c);
                                }
                            }
                        }
                        table
                    };
                    let mut deps = vec![usize::MAX, self.len()];
                    for (pi, node) in self.nodes().iter().enumerate() {
                        match node {
                            Constant(_) | Symbol(_) => {} // Do nothing.
                            Unary(_, input) => {
                                deps[*input] = usize::min(deps[*input], pi);
                            }
                            Binary(_, lhs, rhs) => {
                                deps[*lhs] = usize::min(deps[*lhs], pi);
                                deps[*rhs] = usize::min(deps[*rhs], pi);
                            }
                            Ternary(_, a, b, c) => {
                                deps[*a] = usize::min(deps[*a], pi);
                                deps[*b] = usize::min(deps[*b], pi);
                                deps[*c] = usize::min(deps[*c], pi);
                            }
                        }
                    }
                    deps.iter()
                        .enumerate()
                        .map(|(ni, dep)| (domtable.immediate_dominator(ni), *dep))
                        .collect()
                };
                let mut indices: Vec<_> = (0..self.len()).collect();
                indices.sort_by_key(|i| keys[*i]);
                let rev_indices = indices.iter().enumerate().fold(
                    vec![usize::MAX; self.len()],
                    |mut r, (newi, oldi)| {
                        r[*oldi] = newi;
                        r
                    },
                );
                (indices, rev_indices)
            };
            let (nodes, dims) = self.take();
            indices
                .iter()
                .map(|ni| match nodes[*ni] {
                    Constant(value) => Constant(value),
                    Symbol(label) => Symbol(label),
                    Unary(op, input) => Unary(op, rev_indices[input]),
                    Binary(op, lhs, rhs) => Binary(op, rev_indices[lhs], rev_indices[rhs]),
                    Ternary(op, a, b, c) => {
                        Ternary(op, rev_indices[a], rev_indices[b], rev_indices[c])
                    }
                })
                .collect()
        };
        todo!("Incomplete");
    }
}
