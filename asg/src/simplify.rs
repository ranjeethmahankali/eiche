use crate::{
    template::{get_templates, Capture},
    tree::Tree,
};

pub fn simplify_tree(tree: Tree) {
    let templates = get_templates();
    let mut capture = Capture::new();
    let mut candidates: Vec<Tree> = Vec::new();
    for t in templates {
        t.first_match(&tree, &mut capture);
        while capture.is_valid() {
            candidates.push(capture.apply(tree.clone()));
            t.next_match(&tree, &mut capture);
        }
    }
}

fn simplify_internal(tree: Tree) {
    todo!();
}
