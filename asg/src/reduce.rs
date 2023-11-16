use crate::{heuristic::LoopCounter, mutate::Mutations, tree::Tree};

fn complexity(tree: &Tree, lc: &mut LoopCounter) -> usize {
    tree.len() + lc.run(tree.nodes(), tree.root_index())
}

pub fn reduce(tree: Tree) -> Result<Tree, ()> {
    let mutations = Mutations::from(&tree);
    let mut costs = Vec::<usize>::new();
    let mut lc = LoopCounter::new();
    for tree in mutations {
        match tree {
            Ok(t) => costs.push(complexity(&t, &mut lc)),
            Err(_) => todo!(),
        }
    }
    todo!();
}
