use crate::{mutate::Mutations, tree::Tree};

pub fn reduce(tree: Tree) {
    let mutations = Mutations::from(&tree);
    for tree in mutations {
        println!("{}", tree);
    }
    todo!();
}
