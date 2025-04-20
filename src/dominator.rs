use crate::{Node::*, Tree};

/// Used to manage a dominator mapping between nodes of a tree.
///
/// Because this is used to manage the dominator mapping of any node with any
/// other node, the width and the height of the table are always eual to the
/// number of nodes, i.e. `size`.
struct DomTable {
    bits: Box<[u64]>,
    n_chunks: usize, // Number of bytes per row.
}

impl DomTable {
    const CHUNK_SIZE: usize = 64;

    fn set(bits: &mut [u64], i: usize) {
        let quot = i / Self::CHUNK_SIZE;
        let rem = i % Self::CHUNK_SIZE;
        bits[quot] |= 1 << rem;
    }

    fn unset(bits: &mut [u64], i: usize) {
        let quot = i / Self::CHUNK_SIZE;
        let rem = i % Self::CHUNK_SIZE;
        bits[quot] &= !(1 << rem);
    }

    pub fn from_tree(tree: &Tree) -> Self {
        // Empty tree.
        let mut table = {
            let quot = tree.len() / Self::CHUNK_SIZE;
            let rem = tree.len() % Self::CHUNK_SIZE;
            let n_chunks = quot + (if rem == 0 { 0 } else { 1 });
            DomTable {
                bits: vec![0u64; n_chunks * tree.len()].into_boxed_slice(),
                n_chunks,
            }
        };
        // Everynode dominates itself at the start.
        for (ni, chunk) in table.bits.chunks_exact_mut(table.n_chunks).enumerate() {
            Self::set(chunk, ni);
        }
        let mut visited = vec![false; tree.len()].into_boxed_slice();
        // Parents try to dominate children.
        for (i, node) in tree.nodes().iter().enumerate().rev() {
            match node {
                Constant(_) | Symbol(_) => {} // Do nothing.
                Unary(_, input) => table.dominate(i, *input, &mut visited),
                Binary(_, lhs, rhs) => {
                    table.dominate(i, *lhs, &mut visited);
                    table.dominate(i, *rhs, &mut visited);
                }
                Ternary(_, a, b, c) => {
                    table.dominate(i, *a, &mut visited);
                    table.dominate(i, *b, &mut visited);
                    table.dominate(i, *c, &mut visited);
                }
            }
        }
        // Nodes dominating themselves is only useful while making the
        // table. After that, this is implicit and makes it harder to find an
        // immediate dominator that is not the node itself. So we remove it.
        for (ni, chunk) in table.bits.chunks_exact_mut(table.n_chunks).enumerate() {
            Self::unset(chunk, ni);
        }
        table
    }

    fn dominate(&mut self, parent: usize, child: usize, visited: &mut [bool]) {
        let (poff, coff) = (parent * self.n_chunks, child * self.n_chunks);
        let [parent_bits, child_bits] = unsafe {
            self.bits.get_disjoint_unchecked_mut([
                poff..(poff + self.n_chunks),
                coff..(coff + self.n_chunks),
            ])
        };
        if std::mem::replace(&mut visited[child], true) {
            for (p, c) in parent_bits.iter().zip(child_bits.iter_mut()) {
                *c &= *p;
            }
            Self::set(child_bits, child); // Always dominates itself.
        } else {
            for (p, c) in parent_bits.iter().zip(child_bits.iter_mut()) {
                *c |= *p;
            }
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

    pub fn counts(&self) -> Vec<usize> {
        todo!("TODO: Get the number of nodes each node dominates")
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
        let (sorted, dims, domcounts): (Vec<_>, (usize, usize), Vec<_>) = {
            let (indices, rev_indices, domcounts) = {
                let (keys, domcounts): (Vec<_>, Vec<_>) = {
                    let domtable = DomTable::from_tree(&self);
                    let domcounts = domtable.counts();
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
                    let keys = deps
                        .iter()
                        .enumerate()
                        .map(|(ni, dep)| (domtable.immediate_dominator(ni), *dep))
                        .collect();
                    (keys, domcounts)
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
                (indices, rev_indices, domcounts)
            };
            let (nodes, dims) = self.take();
            let sorted = indices
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
                .collect();
            (sorted, dims, domcounts)
        };
        (
            Tree::from_nodes(sorted, dims).expect("This should never fail"),
            domcounts,
        )
    }
}

#[cfg(test)]
mod test {
    use crate::deftree;

    use super::DomTable;

    #[test]
    fn t_one_chain() {
        let tree = deftree!(sin (abs (log x))).unwrap();
        let table = DomTable::from_tree(&tree);
        assert_eq!(table.immediate_dominator(0), 1);
        assert_eq!(table.immediate_dominator(1), 2);
        assert_eq!(table.immediate_dominator(2), 3);
        assert_eq!(table.immediate_dominator(3), 3);
    }
}
