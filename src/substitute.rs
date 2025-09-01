use crate::{
    dedup::equivalent,
    error::Error,
    tree::{Node::*, Tree},
    walk::DepthWalker,
};

impl Tree {
    /// Substitute all subtrees (sub expressions) in this tree that are
    /// equivalent to `old` with `new`. `old` and `new` must have the same
    /// dimensions. Every node in the tree that is equivalent to an element in
    /// `old` will be replaced by the corresponding element in `new`. For
    /// example if `old` is `(concat 'x 'y 'z)` and `new` is `(concat (+ 1 'x)
    /// (+ 1 'y) (+ 1 'z))`, then all occurences of `x` will be replaced with `1
    /// + x`, `y` with `1 + y`, and `z` with `1 + z`.
    pub fn substitute(self, old: &Tree, new: &Tree) -> Result<Tree, Error> {
        if old.dims() != new.dims() {
            return Err(Error::InvalidDimensions);
        }
        // These are used to check equivalence.
        let mut lwalker = DepthWalker::default();
        let mut rwalker = DepthWalker::default();
        {
            // Ensure all roots of `old` are unique. We only detect direct
            // equivalence, and don't detect mathematical equivalence.
            let (nodes, roots) = (old.nodes(), old.root_indices());
            let end = roots.end;
            // Check every unique combination.
            roots
                .flat_map(|l| ((l + 1)..end).map(move |r| (l, r)))
                .try_for_each(|(l, r)| {
                    if equivalent(l, r, nodes, nodes, &mut lwalker, &mut rwalker)? {
                        Err(Error::InvalidRoots)
                    } else {
                        Ok(())
                    }
                })?;
        }
        // TODO: Checking for equivalence for all nodes like this can be
        // optimized by remembering the results from the previous checks and
        // avoiding checking the same subtree again. This requires refactoring
        // the current logic for checking equivalence to take advantage of
        // results from the past. I am skipping this optimization for now.
        let (mapping, found) = {
            let mut mapping = vec![None; self.len()];
            let mut found = false;
            let old_start = old.root_indices().start;
            let new_start = new.root_indices().start;
            for i in 0..self.len() {
                'inner: for j in old.root_indices() {
                    if equivalent(
                        // We don't need to check because the nodes are from a tree.
                        i,
                        j,
                        self.nodes(),
                        old.nodes(),
                        &mut lwalker,
                        &mut rwalker,
                    )? {
                        mapping[i] = Some(j + new_start - old_start);
                        found = true;
                        break 'inner;
                    }
                }
            }
            (mapping, found)
        };
        if !found {
            // No matches found for substitution.
            return Ok(self);
        }
        let offset = new.len();
        let map_input = move |i: usize| match mapping[i] {
            Some(mapped) => mapped,
            None => i + offset,
        };
        let (mut nodes, dims) = self.take();
        for node in &mut nodes {
            *node = match node {
                Constant(_) | Symbol(_) => *node,
                Unary(op, input) => Unary(*op, map_input(*input)),
                Binary(op, lhs, rhs) => Binary(*op, map_input(*lhs), map_input(*rhs)),
                Ternary(op, a, b, c) => Ternary(*op, map_input(*a), map_input(*b), map_input(*c)),
            };
        }
        // Instead of prepending the new tree nodes, we append them and rotate the the whole slice.
        nodes.extend_from_slice(new.nodes());
        nodes.rotate_right(new.len());
        Tree::from_nodes(nodes, dims)
    }
}

#[cfg(test)]
mod test {
    use crate::{Error, deftree};

    #[test]
    fn t_substitution_0() {
        let original = deftree!(/ (+ (pow 'x 2) (log 'x)) (exp (* 2 'x))).unwrap();
        let old = &deftree!('x).unwrap();
        let new = &deftree!(+ 1 'x).unwrap();
        let replaced = original.clone().substitute(old, new).unwrap();
        assert!(replaced.equivalent(
            &deftree!(/ (+ (pow (+ 1 'x) 2) (log (+ 1 'x))) (exp (* 2 (+ 1 'x)))).unwrap()
        ));
        let reverted = replaced.substitute(new, old).unwrap();
        assert!(original.equivalent(&reverted));
    }

    #[test]
    fn t_multiple_substitute() {
        let original = deftree!(+ (pow 'x 2) (+ (pow 'y 2) (pow 'z 2))).unwrap();
        let old = deftree!(concat 'x 'y 'z).unwrap();
        let new = deftree!(concat (+ 1 'x) (- 1 'y) (+ 1 'z)).unwrap();
        let replaced = original.clone().substitute(&old, &new).unwrap();
        assert!(replaced.equivalent(
            &deftree!(+ (pow (+ 'x 1) 2) (+ (pow (- 1 'y) 2) (pow (+ 'z 1) 2))).unwrap()
        ));
        let reverted = replaced.substitute(&new, &old).unwrap();
        assert!(original.equivalent(&reverted));
    }

    #[test]
    fn t_mismatched_root_counts() {
        let original = deftree!(+ 'x 'y).unwrap();
        let old = deftree!(concat 'x 'y).unwrap();
        let new = deftree!('z).unwrap();
        let result = original.substitute(&old, &new);
        assert!(match result {
            Err(Error::InvalidDimensions) => true,
            _ => false,
        });
    }

    #[test]
    fn t_deeply_nested_patterns() {
        let original = deftree!(
            pow (exp (sqrt (log (abs (sin (cos (tan (+ (* (pow (+ (* 'x (exp (+ 'y (sin (* 'z (cos (+ 'x 'y))))))) (sqrt (+ (pow 'y 3) (log (+ 'z 1))))) 2) (exp (/ 'x (+ 'y 'z)))) (* (+ 'x 'y) (- 'z (sqrt (* 'x 'y)))))))))))) 0.5).unwrap();
        let old = deftree!(* 'x 'y).unwrap();
        let new = deftree!(+ 'x 'y).unwrap();
        let replaced = original.clone().substitute(&old, &new).unwrap();
        assert!(replaced.equivalent(&deftree!(pow (exp (sqrt (log (abs (sin (cos (tan (+ (* (pow (+ (* 'x (exp (+ 'y (sin (* 'z (cos (+ 'x 'y))))))) (sqrt (+ (pow 'y 3) (log (+ 'z 1))))) 2) (exp (/ 'x (+ 'y 'z)))) (* (+ 'x 'y) (- 'z (sqrt (+ 'x 'y)))))))))))) 0.5).unwrap()));
        let reverted = replaced.substitute(&new, &old).unwrap();
        assert!(reverted.equivalent(&deftree!(pow (exp (sqrt (log (abs (sin (cos (tan (+ (* (pow (+ (* 'x (exp (+ 'y (sin (* 'z (cos (* 'x 'y))))))) (sqrt (+ (pow 'y 3) (log (+ 'z 1))))) 2) (exp (/ 'x (+ 'y 'z)))) (* (* 'x 'y) (- 'z (sqrt (* 'x 'y)))))))))))) 0.5).unwrap()));
    }
}
