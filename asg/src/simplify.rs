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

// struct MatchIterator<'a> {
//     template: &'a Template,
//     resources: &'a Resources,
// }

// impl<'a> Iterator for MatchIterator<'a> {
//     type Item = Tree;

//     fn next(&mut self) -> Option<Self::Item> {
//         todo!()
//     }
// }

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
        let mut left = res
            .lwalk
            .walk_tree(self.ping(), false, NodeOrdering::Deterministic);
        let mut right = res
            .rwalk
            .walk_tree_from(tree, from, false, NodeOrdering::Deterministic);
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
        if index >= tree.len() {
            return;
        }
        for i in index..tree.len() {
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
            Some(i) => self.match_from(i + 1, tree, capture, res),
            None => return,
        }
    }
}

#[derive(Debug)]
struct Capture {
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
    use crate::{deftree, template::get_template_by_name};

    #[test]
    fn basic_template_matching() {
        let mut check_template = {
            let mut capture = Capture::new();
            let mut resources = Resources::new();
            let closure = move |name: &str, tree: Tree, node_index: usize| {
                capture.node_index = None;
                capture.bindings.clear();
                print!("Checking template {} ... ", name);
                let template = get_template_by_name(name).unwrap();
                print!("{}{}", template.ping(), tree); // DEBUG
                assert!(!capture.is_valid());
                template.first_match(&tree, &mut capture, &mut resources);
                assert!(capture.is_valid());
                assert!(matches!(capture.node_index, Some(i) if i == node_index));
                println!("✔ Passed.");
            };
            closure
        };
        check_template(
            "distribute_mul",
            deftree!(* 0.5 (+ (* x 2.5) (* x 1.5)))
                .deduplicate()
                .unwrap(),
            6,
        );
        check_template(
            "min_of_sqrt",
            deftree!(+ 2.57 (* 1.23 (min (sqrt 2) (sqrt 3)))),
            6,
        );
        check_template(
            "rearrange_frac",
            deftree!(sqrt (log (* (/ x 2) (/ 2 x))))
                .deduplicate()
                .unwrap(),
            4,
        );
        check_template(
            "divide_by_self",
            deftree!(+ 1 (/ p p)).deduplicate().unwrap(),
            2,
        );
        check_template("distribute_pow_div", deftree!(pow (pow (/ 2 3) 2) 2.5), 4);
        check_template("distribute_pow_mul", deftree!(pow (pow (* 2 3) 2) 2.5), 4);
        check_template(
            "square_sqrt",
            deftree!(log (+ 1 (exp (pow (sqrt 3.2556) 2)))),
            4,
        );
        check_template(
            "sqrt_square",
            deftree!(log (+ 1 (exp (sqrt (pow 3.2345 2.))))),
            4,
        );
        check_template("square_abs", deftree!(log (+ 1 (exp (pow (abs 2) 2.)))), 4);
        check_template(
            "mul_exponents",
            deftree!(log (+ 1 (exp (pow (pow x 3.) 2.)))),
            5,
        );
        check_template(
            "add_exponents",
            deftree!(log (+ 1 (exp (* (pow (log x) 2) (pow (log x) 3)))))
                .deduplicate()
                .unwrap(),
            7,
        );
        check_template(
            "add_frac",
            deftree!(log (+ 1 (exp (+ (/ 2 (sqrt (+ 2 x))) (/ 3 (sqrt (+ x 2)))))))
                .deduplicate()
                .unwrap(),
            8,
        );
        check_template(
            "add_zero",
            deftree!(log (+ 1 (exp (+ 0 (exp (+ 1 (log p))))))),
            7,
        );
        check_template(
            "sub_zero",
            deftree!(log (+ 1 (exp (- (exp (+ 1 (log p))) 0)))),
            7,
        );
        check_template(
            "mul_1",
            deftree!(log (+ 1 (exp (* (exp (+ 1 (log p))) 1)))),
            7,
        );
        check_template(
            "pow_1",
            deftree!(log (+ 1 (exp (pow (exp (+ 1 (log p))) 1)))),
            7,
        );
    }
}
