use crate::{
    helper::{DepthWalker, NodeOrdering},
    template::{get_templates, Template},
    tree::{Node, Tree},
};

pub fn simplify_tree(tree: Tree) {
    let mut resources = Resources::new();
    let templates = get_templates();
    let mut capture = Capture::new();
    let mut candidates: Vec<Tree> = Vec::new();
    for t in templates {
        t.first_match(&tree, &mut capture, &mut resources);
        while capture.is_valid() {
            candidates.push(capture.apply(tree.clone()));
            t.next_match(&tree, &mut capture, &mut resources);
        }
    }
}

struct Resources {
    lwalk: DepthWalker,
    rwalk: DepthWalker,
}

impl Resources {
    pub fn new() -> Resources {
        Resources {
            lwalk: DepthWalker::new(),
            rwalk: DepthWalker::new(),
        }
    }
}

impl Template {
    fn match_node(
        &self,
        from: usize,
        tree: &Tree,
        capture: &mut Capture,
        res: &mut Resources,
    ) -> bool {
        // Clear any previous bindings to start over fresh.
        capture.bindings.clear();
        // Walkers for depth first traversal.
        let mut left =
            res.lwalk
                .walk_tree_from(self.ping(), from, true, NodeOrdering::Deterministic);
        let mut right = res
            .rwalk
            .walk_tree_from(tree, from, true, NodeOrdering::Deterministic);
        // Do simultaneous depth first walk on the template and the
        // tree and compare along the way.
        loop {
            match (left.next(), right.next()) {
                (None, None) => return true, // Both iterators ended.
                (None, Some(_)) | (Some(_), None) => return false, // One of the iterators ended prematurely.
                (Some((li, _p1)), Some((ri, _p2))) => {
                    if !match (self.ping().nodes()[li], tree.nodes()[ri]) {
                        (Node::Constant(v1), Node::Constant(v2)) => v1 == v2,
                        (Node::Constant(_), _) => false,
                        (Node::Symbol(label), _) => {
                            if capture.bind(label, ri) {
                                right.skip_children();
                                true
                            } else {
                                false
                            }
                        }
                        (Node::Unary(lop, _), Node::Unary(rop, _)) => lop == rop,
                        (Node::Unary(_, _), _) => false,
                        (Node::Binary(lop, _, _), Node::Binary(rop, _, _)) => lop == rop,
                        (Node::Binary(_, _, _), _) => false,
                    } {
                        return false;
                    }
                }
            }
        }
    }

    fn match_from(&self, index: usize, tree: &Tree, capture: &mut Capture, res: &mut Resources) {
        capture.node_index = None;
        for i in (index + 1)..tree.len() {
            if self.match_node(i, tree, capture, res) {
                capture.node_index = Some(i);
                return;
            }
        }
    }

    fn first_match(&self, tree: &Tree, capture: &mut Capture, res: &mut Resources) {
        self.match_from(0, tree, capture, res);
    }

    fn next_match(&self, tree: &Tree, capture: &mut Capture, res: &mut Resources) {
        match capture.node_index {
            Some(i) => self.match_from(i, tree, capture, res),
            None => return,
        }
    }
}

pub struct Capture {
    node_index: Option<usize>,
    bindings: Vec<(char, usize)>,
}

impl Capture {
    pub fn new() -> Capture {
        Capture {
            node_index: None,
            bindings: vec![],
        }
    }

    pub fn bind(&mut self, label: char, index: usize) -> bool {
        for (l, i) in self.bindings.iter() {
            if *l == label {
                return *i == index;
            }
        }
        self.bindings.push((label, index));
        return true;
    }

    pub fn is_valid(&self) -> bool {
        return self.node_index.is_some();
    }

    pub fn apply(&self, _tree: Tree) -> Tree {
        todo!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deftree;

    #[test]
    fn template_matching() {
        let templates = get_templates();
        let tree = deftree!(* 0.5 (+ (* x 2.5) (* x 1.5)))
            .deduplicate()
            .unwrap();
    }
}
