/// Used to manage a dominator mapping between nodes of a tree.
///
/// Because this is used to manage the dominator mapping of any node with any
/// other node, the width and the height of the table are always eual to the
/// number of nodes, i.e. `size`.
struct DomTable {
    bits: Vec<u64>,
    n_nodes: usize,  // Number of nodes, i.e. rows and cols.
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
        DomTable {
            bits,
            n_nodes,
            n_chunks,
        }
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
        Self::set(child_bits, child);
    }

    pub fn immediate_dominator(&self, child: usize) -> usize {
        let offset = child * self.n_chunks;
        // Iterate through flags in reverse and find the index of the first set flag.
        self.bits[offset..(offset + self.n_chunks)]
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, flags)| match flags.trailing_zeros() {
                64 => None,
                n => Some(i * Self::CHUNK_SIZE + n as usize),
            })
            .expect("The node has no immediate dominator. This should never happen.")
    }
}
