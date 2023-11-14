use crate::{
    dedup::Deduplicater,
    fold::fold_constants,
    prune::Pruner,
    sort::TopoSorter,
    template::{get_templates, Template},
    tree::{Node, Tree},
};

pub struct Mutations<'a> {
    tree: &'a Tree,
    capture: Capture,
    template_index: usize,
    reset: bool,
}

impl<'a> Mutations<'a> {
    pub fn from(tree: &'a Tree) -> Mutations {
        Mutations {
            tree,
            capture: Capture::new(),
            template_index: 0,
            reset: true,
        }
    }
}

impl<'a> Iterator for Mutations<'a> {
    type Item = Result<Tree, ()>;

    fn next(&mut self) -> Option<Self::Item> {
        let templates = get_templates();
        while self.template_index < templates.len() {
            while self.reset || self.capture.is_valid() {
                let template = &templates[self.template_index];
                if self.reset {
                    template.first_match(&self.tree, &mut self.capture);
                    self.reset = false;
                } else {
                    template.next_match(&self.tree, &mut self.capture);
                }
                if self.capture.is_valid() {
                    return Some(self.capture.apply(template, &self.tree));
                } else {
                    break;
                }
            }
            self.reset = true;
            self.template_index += 1;
        }
        return None;
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

struct Capture {
    node_index: Option<usize>,
    bindings: Vec<(char, usize)>,
    node_map: Vec<usize>,
    topo_sorter: TopoSorter,
    pruner: Pruner,
    deduper: Deduplicater,
}

impl Capture {
    fn new() -> Capture {
        Capture {
            node_index: None,
            bindings: vec![],
            node_map: vec![],
            topo_sorter: TopoSorter::new(),
            pruner: Pruner::new(),
            deduper: Deduplicater::new(),
        }
    }

    fn bind(&mut self, label: char, index: usize) -> bool {
        for (l, i) in self.bindings.iter() {
            if *l == label {
                return *i == index;
            }
        }
        self.bindings.push((label, index));
        return true;
    }

    fn is_valid(&self) -> bool {
        return self.node_index.is_some();
    }

    fn add_node(&mut self, dst: &mut Vec<Node>, src: usize, node: Node) {
        self.node_map[src] = dst.len();
        dst.push(node);
    }

    fn apply(&mut self, template: &Template, tree: &Tree) -> Result<Tree, ()> {
        use crate::tree::Node::*;
        let mut nodes = tree.nodes().clone();
        let root_index = tree.root_index();
        let pong = template.pong();
        self.node_map.clear();
        self.node_map.resize(pong.len(), 0);
        let oldroot = match self.node_index {
            Some(i) => i,
            None => return Err(()),
        };
        let mut newroot = oldroot;
        let num_nodes = nodes.len();
        for ni in 0..pong.len() {
            match pong.node(ni) {
                Constant(val) => self.add_node(&mut nodes, ni, Constant(*val)),
                Symbol(label) => match self.bindings.iter().find(|(ch, _i)| *ch == *label) {
                    Some((_ch, i)) => self.node_map[ni] = *i,
                    None => return Err(()),
                },
                Unary(op, input) => {
                    self.add_node(&mut nodes, ni, Unary(*op, self.node_map[*input]))
                }
                Binary(op, lhs, rhs) => self.add_node(
                    &mut nodes,
                    ni,
                    Binary(*op, self.node_map[*lhs], self.node_map[*rhs]),
                ),
            }
            if ni == pong.root_index() {
                newroot = self.node_map[ni];
            }
        }
        // Rewire old pattern root to the new pattern root. Only
        // iterate over the preexisting nodes, not the ones we just
        // added.
        for i in 0..num_nodes {
            match nodes.get_mut(i) {
                Some(node) => {
                    match node {
                        Constant(_) | Symbol(_) => {} // Do nothing.
                        Unary(_, input) => {
                            if *input == oldroot {
                                *input = newroot;
                            }
                        }
                        Binary(_, lhs, rhs) => {
                            if *lhs == oldroot {
                                *lhs = newroot;
                            }
                            if *rhs == oldroot {
                                *rhs = newroot;
                            }
                        }
                    }
                }
                None => {}
            }
        }
        let root_index = if oldroot == root_index {
            newroot
        } else {
            root_index
        };
        // Clean up and make a tree.
        let (nodes, root_index) = self.topo_sorter.run(nodes, root_index).map_err(|_| ())?;
        return Tree::from_nodes(
            self.pruner
                .run(self.deduper.run(fold_constants(nodes)), root_index),
        )
        .map_err(|_| ());
    }

    fn binding_state(&self) -> usize {
        self.bindings.len()
    }

    fn restore_bindings(&mut self, state: usize) {
        self.bindings.truncate(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dedup::equivalent, deftree, template::get_template_by_name, tests::tests::compare_trees,
        walk::DepthWalker,
    };

    fn check_bindings(capture: &Capture, template: &Template, tree: &Tree) {
        let left: Vec<_> = {
            let mut chars: Vec<_> = capture.bindings.iter().map(|(c, _i)| *c).collect();
            chars.sort();
            chars.dedup();
            chars
        };
        let right: Vec<_> = template.ping().symbols();
        assert_eq!(left, right);
        for (_c, i) in capture.bindings.iter() {
            assert!(*i < tree.len());
        }
    }

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
        check_bindings(&capture, &template, &tree);
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
        check_bindings(&capture, &template, &tree);
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
        check_bindings(&capture, &template, &tree);
    }

    fn check_template(name: &str, tree: Tree, node_index: usize) {
        let mut capture = Capture::new();
        capture.node_index = None;
        capture.bindings.clear();
        let template = get_template_by_name(name).unwrap();
        assert!(!capture.is_valid());
        template.first_match(&tree, &mut capture);
        if !capture.is_valid() || capture.node_index.unwrap() != node_index {
            panic!("Template:{}Tree:{}", template.ping(), tree);
        }
        assert!(capture.is_valid());
        assert!(matches!(capture.node_index, Some(i) if i == node_index));
        check_bindings(&capture, &template, &tree);
    }

    #[test]
    fn match_distribute_mul() {
        check_template(
            "distribute_mul",
            deftree!(* 0.5 (+ (* x 2.5) (* x 1.5)))
                .deduplicate()
                .unwrap(),
            6,
        );
    }

    #[test]
    fn match_min_of_sqrt() {
        check_template(
            "min_of_sqrt",
            deftree!(+ 2.57 (* 1.23 (min (sqrt 2) (sqrt 3)))),
            6,
        );
    }

    #[test]
    fn match_rearrange_frac() {
        check_template(
            "rearrange_frac",
            deftree!(sqrt (log (* (/ x 2) (/ 2 x))))
                .deduplicate()
                .unwrap(),
            4,
        );
    }

    #[test]
    fn match_divide_by_self() {
        check_template(
            "divide_by_self",
            deftree!(+ 1 (/ p p)).deduplicate().unwrap(),
            2,
        );
    }

    #[test]
    fn match_distribute_pow_div() {
        check_template("distribute_pow_div", deftree!(pow (pow (/ 2 3) 2) 2.5), 4);
    }

    #[test]
    fn match_distribute_pow_mul() {
        check_template("distribute_pow_mul", deftree!(pow (pow (* 2 3) 2) 2.5), 4);
    }

    #[test]
    fn match_square_sqrt() {
        check_template(
            "square_sqrt",
            deftree!(log (+ 1 (exp (pow (sqrt 3.2556) 2)))),
            4,
        );
    }

    #[test]
    fn match_sqrt_square() {
        check_template(
            "sqrt_square",
            deftree!(log (+ 1 (exp (sqrt (pow 3.2345 2.))))),
            4,
        );
    }

    #[test]
    fn match_square_abs() {
        check_template("square_abs", deftree!(log (+ 1 (exp (pow (abs 2) 2.)))), 4);
    }

    #[test]
    fn match_mul_exponents() {
        check_template(
            "mul_exponents",
            deftree!(log (+ 1 (exp (pow (pow x 3.) 2.)))),
            5,
        );
    }

    #[test]
    fn match_add_exponents() {
        check_template(
            "add_exponents",
            deftree!(log (+ 1 (exp (* (pow (log x) 2) (pow (log x) 3)))))
                .deduplicate()
                .unwrap(),
            7,
        );
    }

    #[test]
    fn match_add_frac() {
        check_template(
            "add_frac",
            deftree!(log (+ 1 (exp (+ (/ 2 (sqrt (+ 2 x))) (/ 3 (sqrt (+ x 2)))))))
                .deduplicate()
                .unwrap(),
            8,
        );
    }

    #[test]
    fn match_add_zero() {
        check_template(
            "add_zero",
            deftree!(log (+ 1 (exp (+ 0 (exp (+ 1 (log p))))))),
            7,
        );
    }

    #[test]
    fn match_sub_zero() {
        check_template(
            "sub_zero",
            deftree!(log (+ 1 (exp (- (exp (+ 1 (log p))) 0)))),
            7,
        );
    }

    #[test]
    fn match_mul_1() {
        check_template(
            "mul_1",
            deftree!(log (+ 1 (exp (* (exp (+ 1 (log p))) 1)))),
            7,
        );
    }

    #[test]
    fn match_pow_1() {
        check_template(
            "pow_1",
            deftree!(log (+ 1 (exp (pow (exp (+ 1 (log p))) 1)))),
            7,
        );
    }

    #[test]
    fn match_div_1() {
        check_template(
            "div_1",
            deftree!(log (+ 1 (exp (/ (exp (+ 1 (log p))) 1)))),
            7,
        );
    }

    #[test]
    fn match_mul_0() {
        check_template(
            "mul_0",
            deftree!(log (+ 1 (exp (* (exp (+ 1 (log p))) 0)))),
            7,
        );
    }

    #[test]
    fn match_pow_0() {
        check_template(
            "pow_0",
            deftree!(log (+ 1 (exp (pow (exp (+ 1 (log p))) 0)))),
            7,
        );
    }

    #[test]
    fn match_min_expand() {
        check_template("min_expand", deftree!(log (+ 1 (exp (min x 2)))), 3);
    }

    #[test]
    fn match_max_expand() {
        check_template("max_expand", deftree!(log (+ 1 (exp (max x 2)))), 3);
    }

    #[test]
    fn match_max_of_sub() {
        check_template(
            "max_of_sub",
            deftree!(log (+ 1 (exp (min (- x 2) (- x 3)))))
                .deduplicate()
                .unwrap(),
            6,
        );
    }

    #[test]
    fn basic_mutation() {
        let mut lwalker = DepthWalker::new();
        let mut rwalker = DepthWalker::new();
        let tree = deftree!(/ (+ (* p x) (* p y)) (+ x y))
            .deduplicate()
            .unwrap();
        let simpler = deftree!(/ (* p (+ x y)) (+ x y));
        let mut found: bool = false;
        for m in Mutations::from(&tree) {
            match m {
                Ok(mutated) => {
                    assert_ne!(mutated, tree);
                    found = found
                        || equivalent(
                            mutated.root_index(),
                            simpler.root_index(),
                            mutated.nodes(),
                            simpler.nodes(),
                            &mut lwalker,
                            &mut rwalker,
                        );
                    compare_trees(
                        &tree,
                        &mutated,
                        &[('p', 0.1, 10.), ('x', 0.1, 10.), ('y', 0.1, 10.)],
                        20,
                        1e-14,
                    );
                }
                Err(_) => assert!(false),
            }
        }
        assert!(found);
    }
}
