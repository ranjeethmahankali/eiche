use crate::tree::{Node, Node::*};

/// Compute the results of operations on constants and fold those into
/// constant nodes. The unused nodes after folding are not
/// pruned. Use a pruner for that.
pub fn fold_constants(mut nodes: Vec<Node>) -> Vec<Node> {
    for index in 0..nodes.len() {
        let constval = match nodes[index] {
            Constant(_) => None,
            Symbol(_) => None,
            Unary(op, input) => {
                if let Constant(value) = nodes[input] {
                    Some(op.apply(value))
                } else {
                    None
                }
            }
            Binary(op, lhs, rhs) => {
                if let (Constant(a), Constant(b)) = (&nodes[lhs], &nodes[rhs]) {
                    Some(op.apply(*a, *b))
                } else {
                    None
                }
            }
        };
        if let Some(value) = constval {
            nodes[index] = Constant(value);
        }
    }
    return nodes;
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{deftree, test::util::compare_trees};

    #[test]
    fn constant_folding() {
        // Basic multiplication.
        let tree = deftree!(* 2. 3.).fold_constants().unwrap();
        assert_eq!(tree.len(), 1usize);
        assert_eq!(tree.root(), &Constant(2. * 3.));
        // More complicated tree.
        let tree = deftree!(
            (/
             (+ x (* 2. 3.))
             (log (+ x (/ 2. (min 5. (max 3. (- 9. 5.)))))))
        );
        let expected = deftree!(/ (+ x 6.) (log (+ x 0.5)));
        assert!(tree.len() > expected.len());
        let tree = tree.fold_constants().unwrap();
        assert_eq!(tree, expected);
        compare_trees(&tree, &expected, &[('x', 0.1, 10.)], 100, 0.);
    }
}
