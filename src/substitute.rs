use crate::{
    dedup::equivalent,
    error::Error,
    tree::{Node::*, Tree},
    walk::DepthWalker,
};

impl Tree {
    /// Substitute all subtrees (sub expressions) in this tree that are
    /// equivalent to `old` with `new`. Both `old` and `new` are expected to
    /// represent scalars, i.e. have dimensions (1, 1).
    pub fn substitute(self, old: &Tree, new: &Tree) -> Result<Tree, Error> {
        if old.dims() != (1, 1) || new.dims() != (1, 1) {
            return Err(Error::InvalidDimensions);
        }
        // TODO: Checking for equivalence for all nodes like this can be
        // optimized by remembering the results from the previous checks and
        // avoiding checking the same subtree again. This requires refactoring
        // the current logic for checking equivalence to take advantage of
        // results from the past. I am skipping this optimization for now.
        let oldroot = old.root_indices().start;
        let mut lwalker = DepthWalker::default();
        let mut rwalker = DepthWalker::default();
        let flags = {
            let mut flags = Vec::with_capacity(self.len());
            for ni in 0..self.len() {
                flags.push(equivalent(
                    // We don't need to check because the nodes are from a tree.
                    oldroot,
                    ni,
                    old.nodes(),
                    self.nodes(),
                    &mut lwalker,
                    &mut rwalker,
                )?);
            }
            flags
        };
        if !flags.iter().any(|f| *f) {
            // No matches found for substitution.
            return Ok(self);
        }
        let newroot = new.root_indices().start;
        let offset = newroot + 1;
        let map_input = move |i: usize| {
            if flags[i] { newroot } else { i + offset }
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
        nodes.extend(new.nodes().iter());
        nodes.rotate_right(new.len());
        Tree::from_nodes(nodes, dims)
    }
}

#[cfg(test)]
mod test {
    use crate::deftree;

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
}
