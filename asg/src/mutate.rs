use crate::{
    dedup::Deduplicater,
    fold::fold_nodes,
    prune::Pruner,
    sort::TopoSorter,
    template::{get_templates, Template},
    tree::{Node, Tree},
};

pub struct Mutations<'a> {
    tree: &'a Tree,
    capture: TemplateCapture,
    template_index: usize,
}

impl<'a> Mutations<'a> {
    pub fn from(tree: &'a Tree) -> Mutations {
        Mutations {
            tree,
            capture: TemplateCapture::new(),
            template_index: 0,
        }
    }
}

impl<'a> Iterator for Mutations<'a> {
    type Item = Result<Tree, ()>;

    fn next(&mut self) -> Option<Self::Item> {
        let templates = get_templates();
        while self.template_index < templates.len() {
            let template = &templates[self.template_index];
            while self.capture.next_match(template, self.tree) {
                return Some(self.capture.apply(template, &self.tree));
            }
            self.template_index += 1;
        }
        return None;
    }
}

pub struct TemplateCapture {
    node_index: Option<usize>,
    bindings: Vec<(char, usize)>,
    node_map: Vec<usize>,
    topo_sorter: TopoSorter,
    pruner: Pruner,
    deduper: Deduplicater,
}

impl TemplateCapture {
    pub fn new() -> TemplateCapture {
        TemplateCapture {
            node_index: None,
            bindings: vec![],
            node_map: vec![],
            topo_sorter: TopoSorter::new(),
            pruner: Pruner::new(),
            deduper: Deduplicater::new(),
        }
    }

    pub fn bindings(&self) -> &Vec<(char, usize)> {
        &self.bindings
    }

    pub fn next_match(&mut self, template: &Template, tree: &Tree) -> bool {
        // Set self.node_index to None and choose the starting index
        // based on what was in self.node_index before setting it to None.
        let start: usize = match std::mem::replace(&mut self.node_index, None) {
            Some(i) => i + 1,
            None => 0,
        };
        if start >= tree.len() {
            return false;
        }
        for i in start..tree.len() {
            // Clear any previous bindings to start over fresh.
            self.bindings.clear();
            if self.match_node(
                template.dof_ping(),
                template.ping().root_index(),
                template.ping(),
                i,
                tree,
            ) {
                self.node_index = Some(i);
                return true;
            }
        }
        return false;
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

    fn add_node(&mut self, dst: &mut Vec<Node>, src: usize, node: Node) {
        self.node_map[src] = dst.len();
        dst.push(node);
    }

    fn binding_checkpoint(&self) -> usize {
        self.bindings.len()
    }

    fn restore_bindings(&mut self, state: usize) {
        self.bindings.truncate(state);
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
                .run(self.deduper.run(fold_nodes(nodes)), root_index),
        )
        .map_err(|_| ());
    }

    fn match_node(
        &mut self,
        ldofs: &Box<[usize]>,
        li: usize,
        ltree: &Tree,
        ri: usize,
        rtree: &Tree,
    ) -> bool {
        match (ltree.node(li), rtree.node(ri)) {
            (Node::Constant(v1), Node::Constant(v2)) => v1 == v2,
            (Node::Constant(_), _) => return false,
            (Node::Symbol(label), _) => return self.bind(*label, ri),
            (Node::Unary(lop, input1), Node::Unary(rop, input2)) => {
                if lop != rop {
                    return false;
                } else {
                    return self.match_node(ldofs, *input1, ltree, *input2, rtree);
                }
            }
            (Node::Unary(..), _) => return false,
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
                            (l1, r1) = (r1, l1);
                            (l2, r2) = (r2, l2);
                        }
                        (l1, r1, l2, r2)
                    };
                    let checkpoint = self.binding_checkpoint();
                    let ordered = self.match_node(ldofs, l1, ltree, l2, rtree)
                        && self.match_node(ldofs, r1, ltree, r2, rtree);
                    if !lop.is_commutative() || ordered {
                        return ordered;
                    }
                    self.restore_bindings(checkpoint);
                    return self.match_node(ldofs, l1, ltree, r2, rtree)
                        && self.match_node(ldofs, r1, ltree, l2, rtree);
                }
            }
            (Node::Binary(..), _) => return false,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        dedup::equivalent, deftree, template::test::get_template_by_name, walk::DepthWalker,
    };

    fn t_check_bindings(capture: &TemplateCapture, template: &Template, tree: &Tree) {
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
    fn t_match_with_dofs_1() {
        let template = Template::from("add_zero", deftree!(+ x 0.), deftree!(x));
        let two: Tree = (-2.0).into();
        let tree = deftree!(+ 0 {two});
        let mut capture = TemplateCapture::new();
        assert!(capture.next_match(&template, &tree));
        assert!(matches!(capture.node_index, Some(i) if i == 2));
        t_check_bindings(&capture, &template, &tree);
    }

    #[test]
    fn t_match_with_dofs_2() {
        let template = Template::from(
            "fraction_rearrange",
            deftree!(/ (+ a b) a),
            deftree!(+ 1 (/ b a)),
        );
        let tree = deftree!(/ (+ p q) q).deduplicate().unwrap();
        let mut capture = TemplateCapture::new();
        assert!(capture.next_match(&template, &tree));
        assert!(matches!(capture.node_index, Some(i) if i == 3));
        t_check_bindings(&capture, &template, &tree);
    }

    #[test]
    fn t_match_with_dofs_3() {
        let template = Template::from(
            "test",
            deftree!(/ (+ (+ a b) (+ c d)) (+ a b)),
            deftree!(+ 1 (/ (+ c d) (+ a b))),
        );
        let tree = deftree!(/ (+ (+ p q) (+ r s)) (+ r s))
            .deduplicate()
            .unwrap();
        print!("{}{}", template.ping(), tree); // DEBUG
        let mut capture = TemplateCapture::new();
        assert!(capture.next_match(&template, &tree));
        assert!(matches!(capture.node_index, Some(i) if i == 7));
        t_check_bindings(&capture, &template, &tree);
    }

    fn t_check_template(name: &str, tree: Tree, node_index: usize) {
        let mut capture = TemplateCapture::new();
        capture.node_index = None;
        capture.bindings.clear();
        let template = get_template_by_name(name).unwrap();
        if !capture.next_match(&template, &tree) || capture.node_index.unwrap() != node_index {
            panic!("Template:{}Tree:{}", template.ping(), tree);
        }
        assert!(matches!(capture.node_index, Some(i) if i == node_index));
        t_check_bindings(&capture, &template, &tree);
    }

    #[test]
    fn t_match_distribute_mul() {
        t_check_template(
            "distribute_mul",
            deftree!(* 0.5 (+ (* x 2.5) (* x 1.5)))
                .deduplicate()
                .unwrap(),
            6,
        );
    }

    #[test]
    fn t_match_min_of_sqrt() {
        t_check_template(
            "min_of_sqrt",
            deftree!(+ 2.57 (* 1.23 (min (sqrt 2) (sqrt 3)))),
            6,
        );
    }

    #[test]
    fn t_match_rearrange_frac() {
        t_check_template(
            "rearrange_frac",
            deftree!(sqrt (log (* (/ x 2) (/ 2 x))))
                .deduplicate()
                .unwrap(),
            4,
        );
    }

    #[test]
    fn t_match_divide_by_self() {
        t_check_template(
            "divide_by_self",
            deftree!(+ 1 (/ p p)).deduplicate().unwrap(),
            2,
        );
    }

    #[test]
    fn t_match_distribute_pow_div() {
        t_check_template("distribute_pow_div", deftree!(pow (pow (/ 2 3) 2) 2.5), 4);
    }

    #[test]
    fn t_match_distribute_pow_mul() {
        t_check_template("distribute_pow_mul", deftree!(pow (pow (* 2 3) 2) 2.5), 4);
    }

    #[test]
    fn t_match_square_sqrt() {
        t_check_template(
            "square_sqrt",
            deftree!(log (+ 1 (exp (pow (sqrt 3.2556) 2)))),
            4,
        );
    }

    #[test]
    fn t_match_sqrt_square() {
        t_check_template(
            "sqrt_square",
            deftree!(log (+ 1 (exp (sqrt (pow 3.2345 2.))))),
            4,
        );
    }

    #[test]
    fn t_match_square_abs() {
        t_check_template("square_abs", deftree!(log (+ 1 (exp (pow (abs 2) 2.)))), 4);
    }

    #[test]
    fn t_match_mul_exponents() {
        t_check_template(
            "mul_exponents",
            deftree!(log (+ 1 (exp (pow (pow x 3.) 2.)))),
            5,
        );
    }

    #[test]
    fn t_match_add_exponents() {
        t_check_template(
            "add_exponents",
            deftree!(log (+ 1 (exp (* (pow (log x) 2) (pow (log x) 3)))))
                .deduplicate()
                .unwrap(),
            7,
        );
    }

    #[test]
    fn t_match_add_frac() {
        t_check_template(
            "add_frac",
            deftree!(log (+ 1 (exp (+ (/ 2 (sqrt (+ 2 x))) (/ 3 (sqrt (+ x 2)))))))
                .deduplicate()
                .unwrap(),
            8,
        );
    }

    #[test]
    fn t_match_min_expand() {
        t_check_template("min_expand", deftree!(log (+ 1 (exp (min x 2)))), 3);
    }

    #[test]
    fn t_match_max_expand() {
        t_check_template("max_expand", deftree!(log (+ 1 (exp (max x 2)))), 3);
    }

    #[test]
    fn t_match_max_of_sub() {
        t_check_template(
            "max_of_sub",
            deftree!(log (+ 1 (exp (min (- x 2) (- x 3)))))
                .deduplicate()
                .unwrap(),
            6,
        );
    }

    fn check_mutations(mut before: Tree, mut after: Tree) {
        before = before.deduplicate().unwrap();
        after = after.deduplicate().unwrap();
        let mut lwalker = DepthWalker::new();
        let mut rwalker = DepthWalker::new();
        assert_eq!(
            1,
            Mutations::from(&before)
                .filter_map(|t| match t {
                    Ok(tree) => {
                        if equivalent(
                            after.root_index(),
                            tree.root_index(),
                            after.nodes(),
                            tree.nodes(),
                            &mut lwalker,
                            &mut rwalker,
                        ) {
                            Some(1)
                        } else {
                            None
                        }
                    }
                    Err(_) => panic!("Error when generating mutations of a tree."),
                })
                .sum::<usize>()
        );
    }

    #[test]
    fn t_mul_add() {
        check_mutations(
            deftree!(/ (+ (* p x) (* p y)) (+ x y)),
            deftree!(/ (* p (+ x y)) (+ x y)),
        );
    }

    #[test]
    fn t_min_sqrt() {
        check_mutations(
            deftree!(log (+ 1 (exp (min (sqrt x) (sqrt y))))),
            deftree!(log (+ 1 (exp (sqrt (min x y))))),
        );
    }

    #[test]
    fn t_rearrange_frac() {
        check_mutations(
            deftree!(* (/ (+ a b) (pow x y)) (/ (+ x y) (pow a b))),
            deftree!(* (/ (+ a b) (pow a b)) (/ (+ x y) (pow x y))),
        );
    }

    #[test]
    fn t_rearrage_mul_div() {
        check_mutations(
            deftree!(/ (* (pow a b) (pow b c)) (pow c a)),
            deftree!(* (pow b c) (/ (pow a b) (pow c a))),
        );
        check_mutations(
            deftree!(/ (* (pow a b) (pow b c)) (pow c a)),
            deftree!(* (pow a b) (/ (pow b c) (pow c a))),
        );
    }

    #[test]
    fn t_divide_by_self() {
        check_mutations(deftree!(/ (+ x (pow y z)) (+ (pow y z) x)), deftree!(1));
    }

    #[test]
    fn t_distribute_pow_div() {
        check_mutations(
            deftree!(pow (/ (* x y) (* 2 3)) 5),
            deftree!(/ (pow (* x y) 5) (pow (* 2 3) 5)).fold().unwrap(),
        );
    }

    #[test]
    fn t_distribute_pow_mul() {
        check_mutations(
            deftree!(pow (* (* x y) (* 2 3)) 5),
            deftree!(* (pow (* x y) 5) (pow (* 2 3) 5)).fold().unwrap(),
        );
    }

    #[test]
    fn t_square_sqrt() {
        check_mutations(
            deftree!(+ 1 (log (pow (sqrt (+ 2 (exp (/ x 2)))) 2))),
            deftree!(+ 1 (log (+ 2 (exp (/ x 2))))),
        );
    }

    #[test]
    fn t_sqrt_square() {
        check_mutations(
            deftree!(+ 1 (log (sqrt (pow (+ 2 (exp (/ x 2))) 2)))),
            deftree!(+ 1 (log (abs (+ 2 (exp (/ x 2)))))),
        );
    }

    #[test]
    fn t_square_abs() {
        check_mutations(
            deftree!(exp (+ 1 (log (pow (abs (* p q)) 2)))),
            deftree!(exp (+ 1 (log (pow (* p q) 2)))),
        );
    }

    #[test]
    fn t_mul_exponents() {
        check_mutations(
            deftree!(exp (+ 1 (log (pow (pow p (+ 2 m)) (/ q r))))),
            deftree!(exp (+ 1 (log (pow p (* (+ 2 m) (/ q r)))))),
        );
    }

    #[test]
    fn t_add_exponents() {
        check_mutations(
            deftree!(exp (+ 1 (log (* (pow p (+ 2 m)) (pow p (/ q r)))))),
            deftree!(exp (+ 1 (log (pow p (+ (+ 2 m) (/ q r)))))),
        );
    }
}
