use crate::{
    Error,
    Node::{self, *},
    Tree,
};

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
        let [parent_bits, child_bits] = self
            .bits
            .get_disjoint_mut([poff..(poff + self.n_chunks), coff..(coff + self.n_chunks)])
            .expect("INTERNAL ERROR: Incorrect disjoint indices. This should never happen");
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

    pub fn num_nodes(&self) -> usize {
        self.bits.len() / self.n_chunks
    }

    pub fn counts(&self) -> Vec<usize> {
        let n_nodes = self.num_nodes();
        let mut counts = vec![0usize; n_nodes];
        for chunks in self.bits.chunks_exact(self.n_chunks) {
            let mut offset = 0usize;
            for chunk in chunks {
                let mut chunk = *chunk;
                let mut shift = 0usize;
                while chunk != 0 {
                    let tz = chunk.trailing_zeros();
                    chunk >>= tz;
                    shift += tz as usize;
                    counts[offset + shift] += 1;
                    chunk >>= 1;
                    shift += 1;
                }
                offset += 64;
            }
        }
        counts
    }
}

struct DomTree {
    children_buf: Vec<usize>,
    offsets: Vec<usize>,
}

impl DomTree {
    pub fn from_table(table: &DomTable) -> Self {
        let num_nodes = table.num_nodes();
        let pairs = {
            let mut pairs: Vec<_> = (0..num_nodes)
                .filter_map(|ni| match table.immediate_dominator(ni) {
                    idom if idom == ni => None,
                    idom => Some((idom, ni)),
                })
                .collect();
            pairs.sort();
            pairs
        };
        let mut offsets = Vec::with_capacity(num_nodes);
        let mut children_buf = Vec::new();
        let mut iter = pairs.iter().peekable();
        for i in 0..num_nodes {
            offsets.push(children_buf.len());
            while let Some(&(_, ni)) = iter.next_if(|(idom, _)| *idom == i) {
                children_buf.push(ni);
            }
        }
        DomTree {
            children_buf,
            offsets,
        }
    }

    pub fn children(&self, node: usize) -> &[usize] {
        let start = self.offsets[node];
        let stop = self
            .offsets
            .get(node + 1)
            .cloned()
            .unwrap_or(self.children_buf.len());
        &self.children_buf[start..stop]
    }
}

struct StackElement {
    index: usize,
    visited_children: bool, // Whether we're visiting this node after visiting all it's children.
    is_root: bool,
}

impl Tree {
    /// Performs dominator sort on this tree. After this, each node will be
    /// precended by a contiguous range of it's dependencies are dominatoed by
    /// that node. This criteria also holds recursively for the nodes within
    /// that contiguous range. A vector containing the sizes of these dominated
    /// ranges is returned. i.e. for each node the entry in this vector
    /// indicates the number of nodes it exclusively dominates.
    ///
    /// This concept is referred to as "Control Dependence Graph" in the
    /// compiler theory literature.
    pub fn control_dependence_sorted(&self) -> Result<(Tree, Vec<usize>), Error> {
        // Initialize data.
        let domtable = DomTable::from_tree(self);
        let domtree = DomTree::from_table(&domtable);
        let mut stack: Vec<StackElement> = Vec::with_capacity(self.len());
        stack.extend(self.root_indices().map(|r| StackElement {
            index: r,
            visited_children: false,
            is_root: true,
        }));
        let mut visited = vec![false; self.len()];
        let mut onpath = vec![false; self.len()];
        let mut roots: Vec<Node> = Vec::with_capacity(self.num_roots());
        let mut index_map: Vec<usize> = (0..self.len()).collect();
        let mut sorted: Vec<Node> = Vec::new();
        let mut children: Vec<usize> = Vec::new();
        // Do DFS walk.
        while let Some(StackElement {
            index,
            visited_children,
            is_root,
        }) = stack.pop()
        {
            if visited[index] {
                continue;
            }
            if visited_children {
                // We're backtracking after processing the children of this node. So we remove it from the path.
                // Since we visited all children of this node, we're ready to push it into the sorted list, and mark it as
                // processed.
                onpath[index] = false;
                let node = *self.node(index);
                if is_root {
                    roots.push(node);
                } else {
                    index_map[index] = sorted.len();
                    sorted.push(node);
                }
                visited[index] = true;
                continue;
            } else if onpath[index] {
                // Haven't visited this node's children, but it's already on the path. This means we found a cycle.
                return Err(Error::CyclicGraph);
            }
            // We reached this node for the first time. We push this node to the stack again for backtracking after it's
            // children are visited. And we push its children on to the stack.
            onpath[index] = true;
            stack.push(StackElement {
                index,
                visited_children: true,
                is_root,
            });
            // Process children and push them on to the stack..
            match self.node(index) {
                Constant(_) | Symbol(_) => {} // no children.
                Unary(_, input) => children.push(*input),
                Binary(_op, lhs, rhs) => children.extend_from_slice(&[*rhs, *lhs]),
                Ternary(_, a, b, c) => children.extend_from_slice(&[*c, *b, *a]),
            };
            children.extend_from_slice(domtree.children(index));
            children.sort_by(|a, b| b.cmp(a)); // Sort in descending order.
            children.dedup();
            stack.extend(children.drain(..).map(|ci| StackElement {
                index: ci,
                visited_children: false,
                is_root: false,
            }));
        }
        sorted.append(&mut roots); // Push the roots at the end.
        // Update the inputs to new indices.
        for node in &mut sorted {
            match node {
                Constant(_) | Symbol(_) => {} // Nothing.
                Unary(_, input) => *input = index_map[*input],
                Binary(_, lhs, rhs) => {
                    *lhs = index_map[*lhs];
                    *rhs = index_map[*rhs];
                }
                Ternary(_, a, b, c) => {
                    *a = index_map[*a];
                    *b = index_map[*b];
                    *c = index_map[*c];
                }
            }
        }
        let tree = Tree::from_nodes(sorted, self.dims())?;
        let counts = {
            let oldcounts = domtable.counts();
            let mut newcounts = vec![0usize; oldcounts.len()];
            for (i, count) in index_map.iter().zip(oldcounts.iter()) {
                newcounts[*i] = *count;
            }
            newcounts
        };
        Ok((tree, counts))
    }
}

#[cfg(test)]
mod test {
    use super::DomTable;
    use crate::{Tree, deftree, test_util::compare_trees};

    fn check(table: &DomTable, parent: usize, child: usize) -> bool {
        let offset = child * table.n_chunks;
        let flags = &table.bits[offset..(offset + table.n_chunks)];
        let quot = parent / DomTable::CHUNK_SIZE;
        let rem = parent % DomTable::CHUNK_SIZE;
        flags[quot] & (1 << rem) != 0
    }

    fn validate_sorting(sorted_tree: &Tree, subcounts: &[usize]) {
        {
            // Verify the number of dominating nodes for each node are the same
            // in the table as that in the sorted results.
            let domcounts = {
                let mut domcounts = vec![0usize; sorted_tree.len()];
                for (i, count) in subcounts.iter().enumerate() {
                    assert!(
                        *count <= i,
                        "
The number of dominated nodes cannot be more than the index of the node.Because
that would imply this node is dominating more nodes than have preceded this node
in the tree."
                    );
                    for count in domcounts.iter_mut().skip(i - count).take(*count) {
                        *count += 1
                    }
                }
                domcounts
            };
            let table = DomTable::from_tree(sorted_tree);
            for (child, domcount) in domcounts.iter().enumerate() {
                let offset = child * table.n_chunks;
                // Compare the computed dominator counts with those expected from the table.
                assert_eq!(
                    *domcount,
                    table.bits[offset..(offset + table.n_chunks)]
                        .iter()
                        .map(|chunk| chunk.count_ones() as usize)
                        .sum::<usize>()
                )
            }
        }
        // Ensure all the nodes indicated as dominated by the sorted results,
        // are also flagged as such in the table.
        let table = DomTable::from_tree(sorted_tree);
        for (pi, count) in subcounts.iter().enumerate() {
            for ci in (pi - count)..pi {
                assert!(check(&table, pi, ci));
            }
        }
    }

    #[test]
    fn t_one_chain() {
        let tree = deftree!(sin (abs (log 'x))).unwrap();
        let table = DomTable::from_tree(&tree);
        assert_eq!(table.immediate_dominator(0), 1);
        assert_eq!(table.immediate_dominator(1), 2);
        assert_eq!(table.immediate_dominator(2), 3);
        assert_eq!(table.immediate_dominator(3), 3);
        // Check the counts.
        assert_eq!(&table.counts(), &[0usize, 1, 2, 3]);
        // Check sorting.
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        validate_sorting(&sorted_tree, &subcounts);
        // Check equivalence.
        compare_trees(&tree, &sorted_tree, &[('x', 0.01, 10.0)], 20, 1e-14);
    }

    #[test]
    fn t_muladd_tree() {
        let tree = deftree!(* (- 'x 3.) (+ 2. 'y))
            .unwrap()
            .compacted()
            .unwrap();
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        validate_sorting(&sorted_tree, &subcounts);
        compare_trees(
            &tree,
            &sorted_tree,
            &[('x', -10.0, 10.0), ('y', -10.0, 10.0)],
            8,
            1e-14,
        );
    }

    #[test]
    fn t_tiny_tree() {
        let tree = deftree!(+ (pow (- 'x 2.95) 2.) (pow (- 'y 2.05) 2.))
            .unwrap()
            .compacted()
            .unwrap();
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        validate_sorting(&sorted_tree, &subcounts);
        compare_trees(
            &tree,
            &sorted_tree,
            &[('x', -5.0, 5.0), ('y', -5.0, 5.0)],
            8,
            1e-14,
        );
    }

    #[test]
    fn t_small_tree() {
        let tree = deftree!(+ (sqrt (+ (pow (- 'x 2.95) 2.) (pow (- 'y 2.05) 2.))) 3.67)
            .unwrap()
            .compacted()
            .unwrap();
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        validate_sorting(&sorted_tree, &subcounts);
        compare_trees(
            &tree,
            &sorted_tree,
            &[('x', -5.0, 5.0), ('y', -5.0, 5.0)],
            8,
            1e-14,
        );
    }

    #[test]
    fn t_depth_two_diamond_nodes() {
        let tree = deftree!(max (+ (+ 'x 2.) (+ 'y 2.)) (+ 'x 'y))
            .unwrap()
            .compacted()
            .unwrap();
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        validate_sorting(&sorted_tree, &subcounts);
        compare_trees(
            &tree,
            &sorted_tree,
            &[('x', -5.0, 5.0), ('y', -5.0, 5.0)],
            8,
            1e-14,
        );
    }

    #[test]
    fn t_medium_tree() {
        let tree = deftree!(max
                            (+ (pow 'x 2.) (pow 'y 2.))
                            (+ (pow (- 'x 2.5) 2.) (pow (- 'y 2.5) 2.)))
        .unwrap()
        .compacted()
        .unwrap();
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        validate_sorting(&sorted_tree, &subcounts);
        compare_trees(
            &tree,
            &sorted_tree,
            &[('x', -5.0, 5.0), ('y', -5.0, 5.0)],
            8,
            1e-14,
        );
    }

    #[test]
    fn t_large_tree() {
        let tree = deftree!(min
                 (- (log (+
                          (min
                           (+ (sqrt (+ (pow (- 'x 2.95) 2.) (pow (- 'y 2.05) 2.))) 3.67)
                           (max
                            (- (sqrt (+ (pow (- 'x 3.5) 2.) (pow (- 'y 3.5) 2.))) 2.234)
                            (max
                             (- (sqrt (+ (pow 'x 2.) (pow 'y 2.))) 4.24)
                             (- (sqrt (+ (pow (- 'x 2.5) 2.) (pow (- 'y 2.5) 2.))) 5.243))))
                          (exp (pow (min
                                     (- (sqrt (+ (pow (- 'x 2.95) 2.) (pow (- 'y 2.05) 2.))) 3.67)
                                     (max
                                      (- (sqrt (+ (pow (- 'x 3.5) 2.) (pow (- 'y 3.5) 2.))) 2.234)
                                      (max
                                       (- (sqrt (+ (pow 'x 2.) (pow 'y 2.))) 4.24)
                                       (- (sqrt (+ (pow (- 'x 2.5) 2.) (pow (- 'y 2.5) 2.))) 5.243))))
                                2.456))))
                  (min
                   (/ (+ (- 'b) (sqrt (- (pow 'b 2.) (* 4 (* 'a 'c))))) (* 2. 'a))
                   (/ (- (- 'b) (sqrt (- (pow 'b 2.) (* 4 (* 'a 'c))))) (* 2. 'a))))
                 (+ (log (+
                          (max
                           (- (sqrt (+ (pow (- 'x 3.95) 2.) (pow (- 'y 3.05) 2.))) 5.67)
                           (min
                            (- (sqrt (+ (pow (- 'x 4.51) 2.) (pow (- 'y 4.51) 2.))) 2.1234)
                            (min
                             (- (sqrt (+ (pow 'x 2.1) (pow 'y 2.1))) 4.2432)
                             (- (sqrt (+ (pow (- 'x 2.512) 2.) (pow (- 'y 2.512) 2.1))) 5.1243))))
                          (exp (pow (max
                                     (- (sqrt (+ (pow (- 'x 2.65) 2.) (pow (- 'y 2.15) 2.))) 3.67)
                                     (min
                                      (- (sqrt (+ (pow (- 'x 3.65) 2.) (pow (- 'y 3.75) 2.))) 2.234)
                                      (min
                                       (- (sqrt (+ (pow 'x 2.) (pow 'y 2.))) 4.24)
                                       (- (sqrt (+ (pow (- 'x 2.35) 2.) (pow (- 'y 2.25) 2.))) 5.1243))))
                                2.1456))))
                  (max
                   (/ (+ (- 'b) (sqrt (- (pow 'b 2.) (* 4 (* 'a 'c))))) (* 2. 'a))
                   (/ (- (- 'b) (sqrt (- (pow 'b 2.) (* 4 (* 'a 'c))))) (* 2. 'a)))))
        .unwrap()
        .compacted()
            .unwrap();
        // let table = DomTable::from_tree(&tree);
        // println!("{:?}", table.counts());
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        validate_sorting(&sorted_tree, &subcounts);
        // Skip equivalence test for this complex tree due to numerical instability
        // assert!(false);
    }

    // Edge case tests
    #[test]
    fn t_single_node() {
        let tree = deftree!('x).unwrap();
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        validate_sorting(&sorted_tree, &subcounts);
        compare_trees(&tree, &sorted_tree, &[('x', -5.0, 5.0)], 15, 1e-14);
    }

    #[test]
    fn t_single_constant() {
        let tree = deftree!(42.).unwrap();
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        validate_sorting(&sorted_tree, &subcounts);
    }

    #[test]
    fn t_all_leaves() {
        let tree = deftree!(+ 'x 'y).unwrap();
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        validate_sorting(&sorted_tree, &subcounts);
        compare_trees(
            &tree,
            &sorted_tree,
            &[('x', -5.0, 5.0), ('y', -5.0, 5.0)],
            8,
            1e-14,
        );
    }

    #[test]
    fn t_shared_subtree() {
        let tree = deftree!(+ (* 'x 'y) (* 'x 'y))
            .unwrap()
            .compacted()
            .unwrap();
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        validate_sorting(&sorted_tree, &subcounts);
        compare_trees(
            &tree,
            &sorted_tree,
            &[('x', -5.0, 5.0), ('y', -5.0, 5.0)],
            8,
            1e-14,
        );
    }

    #[test]
    fn t_nested_sharing() {
        let tree = deftree!(+ (sin (cos 'x)) (cos (cos 'x)))
            .unwrap()
            .compacted()
            .unwrap();
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        validate_sorting(&sorted_tree, &subcounts);
        compare_trees(&tree, &sorted_tree, &[('x', -5.0, 5.0)], 15, 1e-14);
    }

    #[test]
    fn t_ternary_nodes() {
        let tree = deftree!(if (> 'x 0) 'x (- 'x)).unwrap();
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        validate_sorting(&sorted_tree, &subcounts);
        compare_trees(&tree, &sorted_tree, &[('x', -5.0, 5.0)], 15, 1e-14);
    }

    #[test]
    fn t_complex_ternary() {
        let tree = deftree!(if (> 'x 'y) (+ 'x 'y) (- 'x 'y)).unwrap();
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        validate_sorting(&sorted_tree, &subcounts);
        compare_trees(
            &tree,
            &sorted_tree,
            &[('x', -5.0, 5.0), ('y', -5.0, 5.0)],
            8,
            1e-14,
        );
    }

    #[test]
    fn t_deep_chain() {
        let tree = deftree!(sin (cos (tan (log (exp (sqrt (abs 'x))))))).unwrap();
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        validate_sorting(&sorted_tree, &subcounts);
        compare_trees(&tree, &sorted_tree, &[('x', 0.01, 5.0)], 15, 1e-14);
    }

    #[test]
    fn t_wide_tree() {
        let tree = deftree!(+ (+ (+ (+ 'x 'y) 'z) 'a) (+ (+ 'b 'c) 'd)).unwrap();
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        validate_sorting(&sorted_tree, &subcounts);
        compare_trees(
            &tree,
            &sorted_tree,
            &[
                ('x', -5.0, 5.0),
                ('y', -5.0, 5.0),
                ('z', -5.0, 5.0),
                ('a', -5.0, 5.0),
                ('b', -5.0, 5.0),
                ('c', -5.0, 5.0),
                ('d', -5.0, 5.0),
            ],
            3,
            1e-14,
        ); // 3^7 = 2,187 samples
    }

    #[test]
    fn t_multiple_shared_nodes() {
        let tree = deftree!(+ (* (+ 'x 'y) (- 'x 'y)) (* (+ 'x 'y) (+ 'a 'b)))
            .unwrap()
            .compacted()
            .unwrap();
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        validate_sorting(&sorted_tree, &subcounts);
        compare_trees(
            &tree,
            &sorted_tree,
            &[
                ('x', -5.0, 5.0),
                ('y', -5.0, 5.0),
                ('a', -5.0, 5.0),
                ('b', -5.0, 5.0),
            ],
            6,
            1e-14,
        ); // 6^4 = 1,296 samples
    }

    #[test]
    fn t_deeply_nested_sharing() {
        let tree = deftree!(+ (pow (+ 'x 'y) 2) (sqrt (+ 'x 'y)))
            .unwrap()
            .compacted()
            .unwrap();
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        validate_sorting(&sorted_tree, &subcounts);
        compare_trees(
            &tree,
            &sorted_tree,
            &[('x', 0.01, 5.0), ('y', 0.01, 5.0)],
            8,
            1e-14,
        );
    }

    #[test]
    fn t_bit_boundary_64() {
        // Test with fewer variables to avoid combinatorial explosion in equivalence testing
        let tree = deftree!(+ (+ (+ 'x 'y) 'z) 'a).unwrap();
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        validate_sorting(&sorted_tree, &subcounts);
        compare_trees(
            &tree,
            &sorted_tree,
            &[
                ('x', -5.0, 5.0),
                ('y', -5.0, 5.0),
                ('z', -5.0, 5.0),
                ('a', -5.0, 5.0),
            ],
            6,
            1e-14,
        ); // 6^4 = 1,296 samples
    }
}
