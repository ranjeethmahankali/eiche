use crate::{
    template::{get_templates, Template},
    tree::{Node, Tree},
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

fn symbolic_match(
    ldofs: &Box<[usize]>,
    li: usize,
    ltree: &Tree,
    ri: usize,
    rtree: &Tree,
    capture: &mut Capture,
) -> bool {
    match (ltree.node(li), rtree.node(ri)) {
        (Node::Constant(v1), Node::Constant(v2)) => v1 == v2,
        (Node::Constant(_), _) => return false,
        (Node::Symbol(label), _) => return capture.bind(*label, ri),
        (Node::Unary(lop, input1), Node::Unary(rop, input2)) => {
            if lop != rop {
                return false;
            } else {
                return symbolic_match(ldofs, *input1, ltree, *input2, rtree, capture);
            }
        }
        (Node::Unary(_, _), _) => return false,
        (Node::Binary(lop, l1, r1), Node::Binary(rop, l2, r2)) => {
            if lop != rop {
                return false;
            } else {
                let (l1, r1, l2, r2) = {
                    let mut l1 = *l1;
                    let mut r1 = *r1;
                    let mut l2 = *l2;
                    let mut r2 = *r2;
                    if !lop.is_commutative() && ldofs[l1] > ldofs[r1] {
                        std::mem::swap(&mut l1, &mut r1);
                        std::mem::swap(&mut l2, &mut r2);
                    }
                    (l1, r1, l2, r2)
                };
                let state = capture.binding_state();
                let ordered = symbolic_match(ldofs, l1, ltree, l2, rtree, capture)
                    && symbolic_match(ldofs, r1, ltree, r2, rtree, capture);
                if !lop.is_commutative() || ordered {
                    return ordered;
                }
                capture.restore_bindings(state);
                return symbolic_match(ldofs, l1, ltree, r2, rtree, capture)
                    && symbolic_match(ldofs, r1, ltree, l2, rtree, capture);
            }
        }
        (Node::Binary(_, _, _), _) => return false,
    }
}

impl Template {
    fn match_node(&self, from: usize, tree: &Tree, capture: &mut Capture) -> bool {
        // Clear any previous bindings to start over fresh.
        capture.bindings.clear();
        symbolic_match(
            &self.dof_ping(),
            self.ping().root_index(),
            self.ping(),
            from,
            tree,
            capture,
        )
    }

    fn match_from(&self, index: usize, tree: &Tree, capture: &mut Capture) {
        capture.node_index = None;
        if index >= tree.len() {
            return;
        }
        for i in index..tree.len() {
            if self.match_node(i, tree, capture) {
                capture.node_index = Some(i);
                return;
            }
        }
    }

    fn first_match(&self, tree: &Tree, capture: &mut Capture) {
        self.match_from(0, tree, capture);
    }

    fn next_match(&self, tree: &Tree, capture: &mut Capture) {
        match capture.node_index {
            Some(i) => self.match_from(i + 1, tree, capture),
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

    pub fn binding_state(&self) -> usize {
        self.bindings.len()
    }

    pub fn restore_bindings(&mut self, state: usize) {
        self.bindings.truncate(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{deftree, template::get_template_by_name};

    #[test]
    fn match_with_dofs_1() {
        let template = get_template_by_name("add_zero").unwrap();
        let two: Tree = (-2.0).into();
        let tree = deftree!(+ 0 {two});
        let mut capture = Capture::new();
        assert!(!capture.is_valid());
        template.first_match(&tree, &mut capture);
        assert!(capture.is_valid());
        assert!(matches!(capture.node_index, Some(i) if i == 2));
    }

    #[test]
    fn match_with_dofs_2() {
        let template = Template::from("test", deftree!(/ (+ a b) a), deftree!(+ 1 (/ b a)));
        let tree = deftree!(/ (+ p q) q).deduplicate().unwrap();
        let mut capture = Capture::new();
        assert!(!capture.is_valid());
        template.first_match(&tree, &mut capture);
        assert!(capture.is_valid());
        assert!(matches!(capture.node_index, Some(i) if i == 3));
    }

    #[test]
    fn match_with_dofs_3() {
        let template = Template::from(
            "test",
            deftree!(/ (+ (+ a b) (+ c d)) (+ a b)),
            deftree!(+ 1 (/ (+ c d) (+ a b))),
        );
        let tree = deftree!(/ (+ (+ p q) (+ r s)) (+ r s))
            .deduplicate()
            .unwrap();
        print!("{}{}", template.ping(), tree); // DEBUG
        let mut capture = Capture::new();
        assert!(!capture.is_valid());
        template.first_match(&tree, &mut capture);
        assert!(capture.is_valid());
        assert!(matches!(capture.node_index, Some(i) if i == 7));
    }

    #[test]
    fn basic_template_matching() {
        let mut check_template = {
            let mut capture = Capture::new();
            let closure = move |name: &str, tree: Tree, node_index: usize| {
                capture.node_index = None;
                capture.bindings.clear();
                print!("Checking template {} ... ", name);
                let template = get_template_by_name(name).unwrap();
                assert!(!capture.is_valid());
                template.first_match(&tree, &mut capture);
                if !capture.is_valid() || capture.node_index.unwrap() != node_index {
                    println!("Template:{}Tree:{}", template.ping(), tree);
                }
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
        check_template(
            "div_1",
            deftree!(log (+ 1 (exp (/ (exp (+ 1 (log p))) 1)))),
            7,
        );
        check_template(
            "mul_0",
            deftree!(log (+ 1 (exp (* (exp (+ 1 (log p))) 0)))),
            7,
        );
        check_template(
            "pow_0",
            deftree!(log (+ 1 (exp (pow (exp (+ 1 (log p))) 0)))),
            7,
        );
        check_template("min_expand", deftree!(log (+ 1 (exp (min x 2)))), 3);
        check_template("max_expand", deftree!(log (+ 1 (exp (max x 2)))), 3);
        check_template(
            "max_of_sub",
            deftree!(log (+ 1 (exp (min (- x 2) (- x 3)))))
                .deduplicate()
                .unwrap(),
            6,
        );
    }
}
