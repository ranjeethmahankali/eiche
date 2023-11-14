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
