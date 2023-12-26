use crate::{
    dedup::Deduplicater,
    fold::fold_nodes,
    prune::Pruner,
    sort::{TopoSorter, TopologicalError},
    template::{get_templates, Template},
    tree::{Node, Tree, TreeError},
};

#[derive(Debug)]
pub enum MutationError {
    InvalidCapture,
    UnboundSymbol,
    InvalidTopology(TopologicalError),
    TreeCreationError(TreeError),
}

pub struct Mutations<'a> {
    tree: &'a Tree,
    capture: &'a mut TemplateCapture,
    template_index: usize,
}

impl<'a> Mutations<'a> {
    pub fn of(tree: &'a Tree, capture: &'a mut TemplateCapture) -> Mutations<'a> {
        Mutations {
            tree,
            capture,
            template_index: 0,
        }
    }
}

impl<'a> Iterator for Mutations<'a> {
    type Item = Result<Tree, MutationError>;

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
            if self.match_node(template.ping().root_index(), template.ping(), i, tree) {
                self.node_index = Some(i);
                return true;
            }
        }
        return false;
    }

    pub fn make_compact_tree(
        &mut self,
        mut tree: Tree,
        newroot: Option<usize>,
    ) -> Result<Tree, MutationError> {
        let root_index = match newroot {
            Some(root) => root,
            None => tree.root_index(),
        };
        let root_index = self
            .topo_sorter
            .run(tree.nodes_mut(), root_index)
            .map_err(|e| MutationError::InvalidTopology(e))?;
        fold_nodes(tree.nodes_mut());
        self.deduper.run(tree.nodes_mut());
        self.pruner.run(tree.nodes_mut(), root_index);
        return tree
            .validated()
            .map_err(|e| MutationError::TreeCreationError(e));
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

    fn checkpoint(&self) -> usize {
        self.bindings.len()
    }

    fn restore(&mut self, state: usize) {
        self.bindings.truncate(state);
    }

    fn apply(&mut self, template: &Template, tree: &Tree) -> Result<Tree, MutationError> {
        use crate::tree::Node::*;
        let mut tree = tree.clone();
        let root_index = tree.root_index();
        let num_nodes = tree.nodes().len();
        let pong = template.pong();
        self.node_map.clear();
        self.node_map.resize(pong.len(), 0);
        let oldroot = match self.node_index {
            Some(i) => i,
            None => return Err(MutationError::InvalidCapture),
        };
        let mut newroot = oldroot;
        for ni in 0..pong.len() {
            match pong.node(ni) {
                Constant(val) => self.add_node(tree.nodes_mut(), ni, Constant(*val)),
                Symbol(label) => match self.bindings.iter().find(|(ch, _i)| *ch == *label) {
                    Some((_ch, i)) => self.node_map[ni] = *i,
                    None => return Err(MutationError::UnboundSymbol),
                },
                Unary(op, input) => {
                    self.add_node(tree.nodes_mut(), ni, Unary(*op, self.node_map[*input]))
                }
                Binary(op, lhs, rhs) => self.add_node(
                    tree.nodes_mut(),
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
            match tree.nodes_mut().get_mut(i) {
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
        // Clean up and make a tree.
        return self.make_compact_tree(
            tree,
            Some(if oldroot == root_index {
                newroot
            } else {
                root_index
            }),
        );
    }

    fn match_node(&mut self, li: usize, ltree: &Tree, ri: usize, rtree: &Tree) -> bool {
        let cpt = self.checkpoint();
        let (found_1, commutable) = self.match_node_commute(li, ltree, ri, rtree, false);
        if found_1 {
            return true;
        }
        if commutable {
            self.restore(cpt);
            let (found, _commutable) = self.match_node_commute(li, ltree, ri, rtree, true);
            return found;
        }
        return false;
    }

    fn match_node_commute(
        &mut self,
        li: usize,
        ltree: &Tree,
        ri: usize,
        rtree: &Tree,
        commute: bool,
    ) -> (bool, bool) {
        match (ltree.node(li), rtree.node(ri)) {
            (Node::Constant(v1), Node::Constant(v2)) => (v1 == v2, false),
            (Node::Constant(_), _) => return (false, false),
            (Node::Symbol(label), _) => return (self.bind(*label, ri), false),
            (Node::Unary(lop, input1), Node::Unary(rop, input2)) => {
                if lop != rop {
                    return (false, false);
                } else {
                    return self.match_node_commute(*input1, ltree, *input2, rtree, commute);
                }
            }
            (Node::Unary(..), _) => return (false, false),
            (Node::Binary(lop, mut l1, mut r1), Node::Binary(rop, l2, r2)) => {
                if lop != rop {
                    return (false, false);
                }
                if lop.is_commutative() && commute {
                    (l1, r1) = (r1, l1);
                }
                let cpt = self.checkpoint();
                let (mut found_left, comm_left) =
                    self.match_node_commute(l1, ltree, *l2, rtree, commute);
                if !found_left && comm_left {
                    self.restore(cpt);
                    (found_left, _) = self.match_node_commute(l1, ltree, *l2, rtree, !commute);
                }
                let cpt = self.checkpoint();
                let (mut found_right, comm_right) =
                    self.match_node_commute(r1, ltree, *r2, rtree, commute);
                if !found_right && comm_right {
                    self.restore(cpt);
                    (found_right, _) = self.match_node_commute(r1, ltree, *r2, rtree, !commute);
                }
                return (
                    found_left && found_right,
                    lop.is_commutative() || comm_left || comm_right,
                );
            }
            (Node::Binary(..), _) => return (false, false),
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
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        let tree = deftree!(/ (+ p q) q)
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner);
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
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        let tree = deftree!(/ (+ (+ p q) (+ r s)) (+ r s))
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner);
        let mut capture = TemplateCapture::new();
        assert!(capture.next_match(&template, &tree));
        assert!(matches!(capture.node_index, Some(i) if i == 7));
        t_check_bindings(&capture, &template, &tree);
    }

    fn t_check_template(name: &str, tree: Tree, node_index: usize) {
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        let tree = tree.deduplicate(&mut dedup).unwrap().prune(&mut pruner);
        let mut capture = TemplateCapture::new();
        capture.node_index = None;
        capture.bindings.clear();
        let template = get_template_by_name(name).unwrap();
        assert!(
            capture.next_match(&template, &tree),
            "Template:{}Tree:{}",
            template.ping(),
            tree
        );
        assert!(matches!(capture.node_index, Some(_)));
        assert_eq!(node_index, capture.node_index.unwrap());
        t_check_bindings(&capture, &template, &tree);
    }

    #[test]
    fn t_match_distribute_mul() {
        t_check_template("distribute_mul", deftree!(* 0.5 (+ (* x 2.5) (* x 1.5))), 6);
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
            deftree!(sqrt (log (* (/ x 2) (/ 2 x)))),
            4,
        );
    }

    #[test]
    fn t_match_divide_by_self() {
        t_check_template("divide_by_self", deftree!(+ 1 (/ p p)), 2);
    }

    #[test]
    fn t_match_distribute_pow_div() {
        t_check_template("distribute_pow_div", deftree!(pow (pow (/ 2 3) 2) 2.5), 3);
    }

    #[test]
    fn t_match_distribute_pow_mul() {
        t_check_template("distribute_pow_mul", deftree!(pow (pow (* 2 3) 2) 2.5), 3);
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
        t_check_template("square_abs", deftree!(log (+ 1 (exp (pow (abs 2) 2.)))), 3);
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
            deftree!(log (+ 1 (exp (* (pow (log x) 2) (pow (log x) 3))))),
            7,
        );
    }

    #[test]
    fn t_match_add_frac() {
        t_check_template(
            "add_frac",
            deftree!(log (+ 1 (exp (+ (/ 2 (sqrt (+ 2 x))) (/ 3 (sqrt (+ x 2))))))),
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
    fn t_match_min_of_sub_1() {
        t_check_template(
            "min_of_sub_1",
            deftree!(log (+ 1 (exp (min (- x 2) (- x 3))))),
            6,
        );
    }

    #[test]
    fn t_match_min_of_add_1() {
        const NAME: &str = "min_of_add_1";
        // Make sure all permutations work.
        t_check_template(NAME, deftree!(min (+ x z) (+ x y)), 5);
        t_check_template(NAME, deftree!(min (+ z x) (+ x y)), 5);
        t_check_template(NAME, deftree!(min (+ z x) (+ y x)), 5);
        t_check_template(NAME, deftree!(min (+ x z) (+ y x)), 5);
    }

    #[test]
    fn t_mutate_multiple_trees() {
        fn assert_one_match(tree: Tree, expected: Tree, capture: &mut TemplateCapture) {
            let mut dedup = Deduplicater::new();
            let mut pruner = Pruner::new();
            let tree = tree.deduplicate(&mut dedup).unwrap().prune(&mut pruner);
            let expected = expected.deduplicate(&mut dedup).unwrap().prune(&mut pruner);
            assert_eq!(
                1,
                Mutations::of(&tree, capture)
                    .filter_map(|result| if result.unwrap().equivalent(&expected) {
                        Some(())
                    } else {
                        None
                    })
                    .count()
            );
        }
        let mut capture = TemplateCapture::new();
        // Ensure the same template capture can be used to mutate
        // multiple trees without having to reallocate.
        assert_one_match(
            deftree!(/ (+ (* p x) (* p y)) (+ x y)),
            deftree!(/ (* p (+ x y)) (+ x y)),
            &mut capture,
        );
        // Use same capture for a second set of trees.
        assert_one_match(
            deftree!(log (+ 1 (exp (min (sqrt x) (sqrt y))))),
            deftree!(log (+ 1 (exp (sqrt (min x y))))),
            &mut capture,
        );
        // Use the same capture on a third set of trees.
        assert_one_match(
            deftree!(/ (* (pow a b) (pow b c)) (pow c a)),
            deftree!(* (pow a b) (/ (pow b c) (pow c a))),
            &mut capture,
        );
    }

    fn check_mutations(mut before: Tree, mut after: Tree) {
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        before = before.deduplicate(&mut dedup).unwrap().prune(&mut pruner);
        after = after
            .fold()
            .unwrap()
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner);
        let mut lwalker = DepthWalker::new();
        let mut rwalker = DepthWalker::new();
        let mut capture = TemplateCapture::new();
        assert_eq!(
            1,
            Mutations::of(&before, &mut capture)
                .filter_map(|t| {
                    let tree = t.unwrap();
                    if equivalent(
                        after.root_index(),
                        tree.root_index(),
                        after.nodes(),
                        tree.nodes(),
                        &mut lwalker,
                        &mut rwalker,
                    ) {
                        Some(())
                    } else {
                        None
                    }
                })
                .count()
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
            deftree!(/ (pow (* x y) 5) (pow (* 2 3) 5)),
        );
    }

    #[test]
    fn t_distribute_pow_mul() {
        check_mutations(
            deftree!(pow (* (* x y) (* 2 3)) 5),
            deftree!(* (pow (* x y) 5) (pow (* 2 3) 5)),
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
