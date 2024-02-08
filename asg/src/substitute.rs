use crate::{
    dedup::equivalent,
    error::Error,
    tree::{MaybeTree, Node::*, Tree},
    walk::DepthWalker,
};

impl Tree {
    pub fn substitute(mut self, old: &Tree, new: &Tree) -> MaybeTree {
        if old.dims() != (1, 1) || new.dims() != (1, 1) {
            return Err(Error::InvalidDimensions);
        }
        // TODO: Checking for equivalence for all nodes like this can be
        // optimized by remembering the results from the previous checks and
        // avoiding checking the same subtree again. This requires refactoring
        // the current logic for checking equivalence to take advantage of
        // results from the past. I am skipping this optimization for now.
        let oldroot = old.root_indices().start;
        let mut lwalker = DepthWalker::new();
        let mut rwalker = DepthWalker::new();
        let flags: Vec<bool> = (0..self.len())
            .map(|ni| {
                equivalent(
                    oldroot,
                    ni,
                    old.nodes(),
                    self.nodes(),
                    &mut lwalker,
                    &mut rwalker,
                )
            })
            .collect();
        let newroot = new.root_indices().start;
        let offset = newroot + 1;
        let map_input = move |i: usize| {
            if flags[i] {
                newroot
            } else {
                i + offset
            }
        };
        for node in self.nodes_mut() {
            *node = match node {
                Constant(_) | Symbol(_) => *node,
                Unary(op, input) => Unary(*op, map_input(*input)),
                Binary(op, lhs, rhs) => Binary(*op, map_input(*lhs), map_input(*rhs)),
                Ternary(op, a, b, c) => Ternary(*op, map_input(*a), map_input(*b), map_input(*c)),
            };
        }
        self.nodes_mut().extend(new.nodes().iter());
        self.nodes_mut().rotate_right(new.len());
        return self.validated();
    }
}

#[cfg(test)]
mod test {
    use crate::deftree;

    #[test]
    fn t_substitution_0() {
        let original = deftree!(/ (+ (pow x 2) (log x)) (exp (* 2 x))).unwrap();
        let old = &deftree!(x).unwrap();
        let new = &deftree!(+ 1 x).unwrap();
        let replaced = original.clone().substitute(old, new).unwrap();
        assert!(replaced.equivalent(
            &deftree!(/ (+ (pow (+ 1 x) 2) (log (+ 1 x))) (exp (* 2 (+ 1 x)))).unwrap()
        ));
        let reverted = replaced.substitute(new, old).unwrap();
        assert!(original.equivalent(&reverted));
    }
}
