use crate::{
    error::Error,
    tree::{MaybeTree, Node, Node::*, Tree},
};

impl Tree {
    pub fn substitute(mut self, var: char, tree: Tree) -> MaybeTree {
        if tree.dims() != (1, 1) {
            return Err(Error::InvalidDimensions);
        }
        todo!()
    }
}

fn is_variable(nodes: &[Node], index: usize, var: char) -> bool {
    if let Symbol(label) = &nodes[index] {
        *label == var
    } else {
        false
    }
}
