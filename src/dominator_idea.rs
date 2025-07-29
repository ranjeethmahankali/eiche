use crate::{
    Error,
    Node::{self, *},
    Tree,
};

fn compute_immediate_dominators(tree: &Tree) -> Vec<Option<usize>> {
    let mut idom = vec![None; tree.len()];
    let mut visited = vec![false; tree.len()];

    // Process nodes in reverse topological order
    for (i, node) in tree.nodes().iter().enumerate().rev() {
        match node {
            Constant(_) | Symbol(_) => {} // Leaves have no dominator
            Unary(_, input) => {
                update_dominator(&mut idom, &mut visited, i, *input);
            }
            Binary(_, lhs, rhs) => {
                update_dominator(&mut idom, &mut visited, i, *lhs);
                update_dominator(&mut idom, &mut visited, i, *rhs);
            }
            Ternary(_, a, b, c) => {
                update_dominator(&mut idom, &mut visited, i, *a);
                update_dominator(&mut idom, &mut visited, i, *b);
                update_dominator(&mut idom, &mut visited, i, *c);
            }
        }
    }
    idom
}

fn update_dominator(idom: &mut [Option<usize>], visited: &mut [bool], parent: usize, child: usize) {
    if std::mem::replace(&mut visited[child], true) {
        // Child already visited - find LCA of current idom and parent
        idom[child] = lowest_common_ancestor(idom, idom[child], Some(parent));
    } else {
        // First visit - parent becomes immediate dominator
        idom[child] = Some(parent);
    }
}

fn lowest_common_ancestor(
    idom: &[Option<usize>],
    mut a: Option<usize>,
    mut b: Option<usize>,
) -> Option<usize> {
    while a != b {
        match (a, b) {
            (Some(av), Some(bv)) => {
                if av > bv {
                    a = idom[av];
                } else {
                    b = idom[bv];
                }
            }
            _ => return None,
        }
    }
    a
}

fn compute_domination_counts(idom: &[Option<usize>]) -> Vec<usize> {
    let mut counts = vec![0usize; idom.len()];

    // Count how many nodes each node dominates (transitive closure)
    for child in 0..idom.len() {
        let mut current = idom[child];
        while let Some(dominator) = current {
            counts[dominator] += 1;
            current = idom[dominator];
        }
    }

    counts
}

fn build_dominator_children(idom: &[Option<usize>]) -> (Vec<usize>, Vec<usize>) {
    // Build dominator tree structure for DFS traversal
    let mut pairs: Vec<_> = idom
        .iter()
        .enumerate()
        .filter_map(|(child, &dom)| dom.map(|d| (d, child)))
        .collect();
    pairs.sort();

    let mut offsets = Vec::with_capacity(idom.len());
    let mut children_buf = Vec::new();
    let mut iter = pairs.iter().peekable();

    for i in 0..idom.len() {
        offsets.push(children_buf.len());
        while let Some(&(_, child)) = iter.next_if(|(dom, _)| *dom == i) {
            children_buf.push(child);
        }
    }

    (children_buf, offsets)
}

fn get_dominator_children<'a>(
    children_buf: &'a [usize],
    offsets: &'a [usize],
    node: usize,
) -> &'a [usize] {
    let start = offsets[node];
    let stop = offsets.get(node + 1).cloned().unwrap_or(children_buf.len());
    &children_buf[start..stop]
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
    /// I think this concept is referred to as "Control Dependence Graph" in the
    /// compiler theory literature.
    pub fn control_dependence_sorted(&self) -> Result<(Tree, Vec<usize>), Error> {
        // Initialize data using efficient dominator computation
        let idom = compute_immediate_dominators(self);
        let (children_buf, offsets) = build_dominator_children(&idom);

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
            children.extend_from_slice(get_dominator_children(&children_buf, &offsets, index));
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
            let oldcounts = compute_domination_counts(&idom);
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
    use super::*;
    use crate::{Tree, deftree};

    fn dominates(idom: &[Option<usize>], dominator: usize, node: usize) -> bool {
        let mut current = node;
        while let Some(idom_current) = idom[current] {
            if idom_current == dominator {
                return true;
            }
            current = idom_current;
        }
        false
    }

    fn dominates_all_paths_from_roots(tree: &Tree, dominator: usize, child: usize) -> bool {
        // Try to reach child from roots without going through dominator
        // If we can reach it, domination fails
        for root in tree.root_indices() {
            if can_reach_without_dominator(tree, root, child, dominator) {
                return false;
            }
        }
        true
    }

    fn can_reach_without_dominator(tree: &Tree, start: usize, target: usize, dominator: usize) -> bool {
        let mut stack = vec![start];
        let mut visited = vec![false; tree.len()];
        
        while let Some(current) = stack.pop() {
            if current == target {
                return true; // Found path to target without hitting dominator
            }
            if current == dominator {
                continue; // Hit dominator, skip this path
            }
            if std::mem::replace(&mut visited[current], true) {
                continue; // Already visited
            }
            
            // Add children to stack (traverse toward leaves)
            match tree.node(current) {
                Constant(_) | Symbol(_) => {} // Dead end
                Unary(_, input) => stack.push(*input),
                Binary(_, lhs, rhs) => {
                    stack.push(*lhs);
                    stack.push(*rhs);
                }
                Ternary(_, a, b, c) => {
                    stack.push(*a);
                    stack.push(*b);
                    stack.push(*c);
                }
            }
        }
        false // Never reached target
    }

    fn validate_sorting(tree: Tree) {
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();

        // Test 1: Verify domination count consistency
        {
            let original_idom = compute_immediate_dominators(&tree);
            let original_counts = compute_domination_counts(&original_idom);
            let sorted_idom = compute_immediate_dominators(&sorted_tree);

            // Check that each node's domination count is preserved through sorting
            for (i, &count) in subcounts.iter().enumerate() {
                assert!(
                    count <= i,
                    "Dominated count {} cannot exceed node index {} in sorted order",
                    count,
                    i
                );

                // Verify the range [i-count..i] contains nodes dominated by i
                let dominated_range = (i.saturating_sub(count))..i;
                for dominated in dominated_range {
                    assert!(
                        dominates(&sorted_idom, i, dominated),
                        "Node {} should dominate {} based on subcounts",
                        i,
                        dominated
                    );
                }
            }

            // Total domination relationships should be preserved
            let total_original: usize = original_counts.iter().sum();
            let total_sorted: usize = subcounts.iter().sum();
            assert_eq!(
                total_original, total_sorted,
                "Total domination count mismatch"
            );
        }

        // Test 2: Verify immediate dominator relationships are valid
        {
            let idom = compute_immediate_dominators(&sorted_tree);
            for (child, &dominator) in idom.iter().enumerate() {
                if let Some(dom) = dominator {
                    assert!(
                        dom < child,
                        "Dominator {} must appear before dominated node {} in sorted order",
                        dom,
                        child
                    );
                    assert!(
                        dominates_all_paths_from_roots(&sorted_tree, dom, child),
                        "Dominator {} must block all paths from roots to {}",
                        dom,
                        child
                    );
                }
            }
        }
    }

    #[test]
    fn t_one_chain() {
        let tree = deftree!(sin (abs (log x))).unwrap();
        let idom = compute_immediate_dominators(&tree);
        println!("Original tree idom: {:?}", idom);
        assert_eq!(idom[0], Some(1));
        assert_eq!(idom[1], Some(2));
        assert_eq!(idom[2], Some(3));
        assert_eq!(idom[3], None);
        // Check the counts.
        let counts = compute_domination_counts(&idom);
        assert_eq!(&counts, &[0usize, 1, 2, 3]);
        
        // Debug the sorted tree
        let (sorted_tree, subcounts) = tree.control_dependence_sorted().unwrap();
        let sorted_idom = compute_immediate_dominators(&sorted_tree);
        println!("Sorted tree idom: {:?}", sorted_idom);
        println!("Subcounts: {:?}", subcounts);
        
        // Check sorting.
        validate_sorting(tree);
    }

    #[test]
    fn t_muladd_tree() {
        let tree = deftree!(* (- x 3.) (+ 2. y)).unwrap().compacted().unwrap();
        validate_sorting(tree);
    }

    #[test]
    fn t_tiny_tree() {
        let tree = deftree!(+ (pow (- x 2.95) 2.) (pow (- y 2.05) 2.))
            .unwrap()
            .compacted()
            .unwrap();
        validate_sorting(tree);
    }

    #[test]
    fn t_small_tree() {
        let tree = deftree!(+ (sqrt (+ (pow (- x 2.95) 2.) (pow (- y 2.05) 2.))) 3.67)
            .unwrap()
            .compacted()
            .unwrap();
        validate_sorting(tree);
    }

    #[test]
    fn t_depth_two_diamond_nodes() {
        let tree = deftree!(max (+ (+ x 2.) (+ y 2.)) (+ x y))
            .unwrap()
            .compacted()
            .unwrap();
        validate_sorting(tree);
    }

    #[test]
    fn t_medium_tree() {
        let tree = deftree!(max
                            (+ (pow x 2.) (pow y 2.))
                            (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.)))
        .unwrap()
        .compacted()
        .unwrap();
        validate_sorting(tree);
    }

    #[test]
    fn t_large_tree() {
        let tree = deftree!(min
                 (- (log (+
                          (min
                           (+ (sqrt (+ (pow (- x 2.95) 2.) (pow (- y 2.05) 2.))) 3.67)
                           (max
                            (- (sqrt (+ (pow (- x 3.5) 2.) (pow (- y 3.5) 2.))) 2.234)
                            (max
                             (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
                             (- (sqrt (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.))) 5.243))))
                          (exp (pow (min
                                     (- (sqrt (+ (pow (- x 2.95) 2.) (pow (- y 2.05) 2.))) 3.67)
                                     (max
                                      (- (sqrt (+ (pow (- x 3.5) 2.) (pow (- y 3.5) 2.))) 2.234)
                                      (max
                                       (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
                                       (- (sqrt (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.))) 5.243))))
                                2.456))))
                  (min
                   (/ (+ (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a))
                   (/ (- (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a))))
                 (+ (log (+
                          (max
                           (- (sqrt (+ (pow (- x 3.95) 2.) (pow (- y 3.05) 2.))) 5.67)
                           (min
                            (- (sqrt (+ (pow (- x 4.51) 2.) (pow (- y 4.51) 2.))) 2.1234)
                            (min
                             (- (sqrt (+ (pow x 2.1) (pow y 2.1))) 4.2432)
                             (- (sqrt (+ (pow (- x 2.512) 2.) (pow (- y 2.512) 2.1))) 5.1243))))
                          (exp (pow (max
                                     (- (sqrt (+ (pow (- x 2.65) 2.) (pow (- y 2.15) 2.))) 3.67)
                                     (min
                                      (- (sqrt (+ (pow (- x 3.65) 2.) (pow (- y 3.75) 2.))) 2.234)
                                      (min
                                       (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
                                       (- (sqrt (+ (pow (- x 2.35) 2.) (pow (- y 2.25) 2.))) 5.1243))))
                                2.1456))))
                  (max
                   (/ (+ (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a))
                   (/ (- (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a)))))
        .unwrap()
        .compacted()
            .unwrap();
        validate_sorting(tree);
    }

    // Edge case tests
    #[test]
    fn t_single_node() {
        let tree = deftree!(x).unwrap();
        validate_sorting(tree);
    }

    #[test]
    fn t_single_constant() {
        let tree = deftree!(42.).unwrap();
        validate_sorting(tree);
    }

    #[test]
    fn t_all_leaves() {
        let tree = deftree!(+ x y).unwrap();
        validate_sorting(tree);
    }

    #[test]
    fn t_shared_subtree() {
        let tree = deftree!(+ (* x y) (* x y)).unwrap().compacted().unwrap();
        validate_sorting(tree);
    }

    #[test]
    fn t_nested_sharing() {
        let tree = deftree!(+ (sin (cos x)) (cos (cos x)))
            .unwrap()
            .compacted()
            .unwrap();
        validate_sorting(tree);
    }

    #[test]
    fn t_ternary_nodes() {
        let tree = deftree!(if (> x 0) x (- x)).unwrap();
        validate_sorting(tree);
    }

    #[test]
    fn t_complex_ternary() {
        let tree = deftree!(if (> x y) (+ x y) (- x y)).unwrap();
        validate_sorting(tree);
    }

    #[test]
    fn t_deep_chain() {
        let tree = deftree!(sin (cos (tan (log (exp (sqrt (abs x))))))).unwrap();
        validate_sorting(tree);
    }

    #[test]
    fn t_wide_tree() {
        let tree = deftree!(+ (+ (+ (+ x y) z) a) (+ (+ b c) d)).unwrap();
        validate_sorting(tree);
    }

    #[test]
    fn t_multiple_shared_nodes() {
        let tree = deftree!(+ (* (+ x y) (- x y)) (* (+ x y) (+ a b)))
            .unwrap()
            .compacted()
            .unwrap();
        validate_sorting(tree);
    }

    #[test]
    fn t_deeply_nested_sharing() {
        let tree = deftree!(+ (pow (+ x y) 2) (sqrt (+ x y)))
            .unwrap()
            .compacted()
            .unwrap();
        validate_sorting(tree);
    }

    #[test]
    fn t_bit_boundary_64() {
        // Create a tree with approximately 64 nodes to test bit chunk boundaries
        // Using single character symbols as required by the macro
        let tree = deftree!(+ (+ (+ (+ (+ (+ (+ (+ x y) z) a) b) c) d) e) f).unwrap();
        validate_sorting(tree);
    }
}
