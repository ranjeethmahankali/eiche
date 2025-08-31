use crate::{
    dedup::Deduplicater,
    error::Error,
    fold::fold,
    prune::Pruner,
    template::{TEMPLATES, Template},
    tree::{Node, Node::*, Tree},
};
use std::ops::Range;

/// Iterator that produces all mutations of a tree.
pub struct Mutations<'a> {
    tree: &'a Tree,
    capture: &'a mut TemplateCapture,
    template_index: usize,
}

impl<'a> Mutations<'a> {
    /// Create mutations of a tree. The capture instance is reused to avoid
    /// allocations as the iterator is traversed.
    pub fn of(tree: &'a Tree, capture: &'a mut TemplateCapture) -> Mutations<'a> {
        Mutations {
            tree,
            capture,
            template_index: 0,
        }
    }
}

impl Iterator for Mutations<'_> {
    type Item = Result<Tree, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let templates = &TEMPLATES;
        while self.template_index < templates.len() {
            let template = &templates[self.template_index];
            if self.capture.next_match(template, self.tree) {
                return Some(self.capture.apply(template, self.tree));
            }
            self.template_index += 1;
        }
        None
    }
}

/// This struct remembers how the symbols in a template map to a tree.
pub struct TemplateCapture {
    // The node in the tree that maps to the root of the template.
    node_index: Option<usize>,
    // Bindings between symbols (labels) in the template, and node indices in the tree.
    bindings: Vec<(char, usize)>,
    // Mapping from the nodes of the 'pong' tree of the template to the nodes in
    // the tree being modified.
    node_map: Vec<usize>,
    // Pruner and deduplicater to be reused to avoid allocations.
    pruner: Pruner,
    deduper: Deduplicater,
}

impl TemplateCapture {
    pub fn new() -> TemplateCapture {
        TemplateCapture {
            node_index: None,
            bindings: vec![],
            node_map: vec![],
            pruner: Pruner::new(),
            deduper: Deduplicater::new(),
        }
    }

    /// Get the bindings in this capture instance. These map the variable labels
    /// in the captured template to the indices of the node they're bound to.
    pub fn bindings(&self) -> &Vec<(char, usize)> {
        &self.bindings
    }

    /// Find the next match of the `templte` in `tree`. The search will begin
    /// after the previously matched node.
    pub fn next_match(&mut self, template: &Template, tree: &Tree) -> bool {
        // Set self.node_index to None and choose the starting index
        // based on what was in self.node_index before setting it to None.
        let start: usize = match self.node_index.take() {
            Some(i) => i + 1,
            None => 0,
        };
        if start >= tree.len() {
            return false;
        }
        for i in start..tree.len() {
            // Clear any previous bindings to start over fresh.
            self.bindings.clear();
            if self.match_node(template.ping_root(), template.ping(), i, tree) {
                self.node_index = Some(i);
                return true;
            }
        }
        false
    }

    fn make_compact_tree(
        &mut self,
        mut nodes: Vec<Node>,
        dims: (usize, usize),
        root_indices: Range<usize>,
    ) -> Result<Tree, Error> {
        let root_indices = self.pruner.run_from_range(&mut nodes, root_indices)?;
        fold(&mut nodes)?;
        // We don't need to check for valid topological order because we just ran the pruner on these nodes, which sorts them.
        self.deduper.run(&mut nodes)?;
        let root_indices = (nodes.len() - root_indices.len())..nodes.len();
        self.pruner.run_from_range(&mut nodes, root_indices)?;
        Tree::from_nodes(nodes, dims)
    }

    /// Compact the tree using the deduper and pruner instances in this template
    /// capture. This can avoid reallocating memory by using existing instances.
    pub fn compact_tree(&mut self, tree: Tree) -> Result<Tree, Error> {
        let root_indices = tree.root_indices();
        let (nodes, dims) = tree.take();
        self.make_compact_tree(nodes, dims, root_indices)
    }

    /// Try to bind the given symbol (label) to the node index. Return whether
    /// the binding was successful. If the label was already bound to another
    /// index, false is returned. If the label was already bound to the same
    /// index, nothing is changed, and true is returned. If the label was not
    /// previously bound, a new binding is created.
    fn bind(&mut self, label: char, index: usize) -> bool {
        for (l, i) in self.bindings.iter() {
            if *l == label {
                return *i == index;
            }
        }
        self.bindings.push((label, index));
        true
    }

    /// This is meant to be used to add a node to the tree being mutated when
    /// applying a captured template to the tree.
    fn add_node(&mut self, dst: &mut Vec<Node>, src: usize, node: Node) {
        self.node_map[src] = dst.len();
        dst.push(node);
    }

    /// Return a checkpoint representing the current state of symbol bindings.
    fn checkpoint(&self) -> usize {
        self.bindings.len()
    }

    /// Restore the symbol bindings to the given state, which was previously
    /// returned by a checkpoint.
    fn restore(&mut self, state: usize) {
        self.bindings.truncate(state);
    }

    /// Apply the captured template to the tree and produce a mutated tree.
    fn apply(&mut self, template: &Template, tree: &Tree) -> Result<Tree, Error> {
        let root_indices = tree.root_indices();
        let num_nodes = tree.nodes().len();
        let (mut nodes, dims) = tree.clone().take();
        let pong = template.pong();
        self.node_map.clear();
        self.node_map.resize(pong.len(), 0);
        let oldroot = match self.node_index {
            Some(i) => i,
            None => return Err(Error::InvalidTemplateCapture),
        };
        for ni in 0..pong.len() {
            match pong.node(ni) {
                Constant(val) => self.add_node(&mut nodes, ni, Constant(*val)),
                Symbol(label) => match self.bindings.iter().find(|(ch, _i)| *ch == *label) {
                    Some((_ch, i)) => self.node_map[ni] = *i,
                    None => return Err(Error::UnboundTemplateSymbol),
                },
                Unary(op, input) => {
                    self.add_node(&mut nodes, ni, Unary(*op, self.node_map[*input]))
                }
                Binary(op, lhs, rhs) => self.add_node(
                    &mut nodes,
                    ni,
                    Binary(*op, self.node_map[*lhs], self.node_map[*rhs]),
                ),
                Ternary(op, a, b, c) => self.add_node(
                    &mut nodes,
                    ni,
                    Ternary(*op, self.node_map[*a], self.node_map[*b], self.node_map[*c]),
                ),
            }
            if pong.root_indices().contains(&ni) {
                let newroot = self.node_map[ni];
                nodes.swap(oldroot, newroot);
                // Any node that is referencing newroot as an input should be
                // rewired to oldroot to avoid broken topology. Only iterate
                // over the preexisting nodes.
                for i in 0..num_nodes {
                    if let Some(node) = nodes.get_mut(i) {
                        match node {
                            Constant(_) | Symbol(_) => {} // Do nothing.
                            Unary(_, input) => {
                                if *input == newroot {
                                    *input = oldroot;
                                }
                            }
                            Binary(_, lhs, rhs) => {
                                if *lhs == newroot {
                                    *lhs = oldroot;
                                }
                                if *rhs == newroot {
                                    *rhs = oldroot;
                                }
                            }
                            Ternary(_, a, b, c) => {
                                if *a == newroot {
                                    *a = oldroot;
                                }
                                if *b == newroot {
                                    *b = oldroot;
                                }
                                if *c == newroot {
                                    *c = oldroot;
                                }
                            }
                        }
                    }
                }
            }
        }
        // Clean up and make a tree.
        self.make_compact_tree(nodes, dims, root_indices)
    }

    /// Try to match the subtree in ltree at index li with the subtree in rtree
    /// at index ri. This is meant to be used to match the 'ping' tree of a tree
    /// meant to be mutated.
    fn match_node(&mut self, li: usize, ltree: &Tree, ri: usize, rtree: &Tree) -> bool {
        let cpt = self.checkpoint();
        let (found, commutable) = self.match_node_commute(li, ltree, ri, rtree, false);
        if found {
            return true;
        }
        if commutable {
            self.restore(cpt);
            let (found, _commutable) = self.match_node_commute(li, ltree, ri, rtree, true);
            return found;
        }
        false
    }

    /// The purpose is the same as match_node (this is called from inside
    /// match_node), but you can specify how to deal with commutative
    /// nodes. Setting the 'commute' argument to true causes the commutative
    /// nodes to be compared and matched even if the arguments are swapped
    /// between the left and the right trees.
    fn match_node_commute(
        &mut self,
        li: usize,
        ltree: &Tree,
        ri: usize,
        rtree: &Tree,
        commute: bool,
    ) -> (bool, bool) {
        match (ltree.node(li), rtree.node(ri)) {
            (Constant(v1), Constant(v2)) => (v1 == v2, false),
            (Constant(_), _) => (false, false),
            (Symbol(label), _) => (self.bind(*label, ri), false),
            (Unary(lop, input1), Unary(rop, input2)) => {
                if lop != rop {
                    (false, false)
                } else {
                    self.match_node_commute(*input1, ltree, *input2, rtree, commute)
                }
            }
            (Unary(..), _) => (false, false),
            (&Binary(lop, mut l1, mut r1), &Binary(rop, l2, r2)) => {
                if lop != rop {
                    return (false, false);
                }
                if lop.is_commutative() && commute {
                    (l1, r1) = (r1, l1);
                }
                let cpt = self.checkpoint();
                let (mut found_left, comm_left) =
                    self.match_node_commute(l1, ltree, l2, rtree, commute);
                if !found_left && comm_left {
                    self.restore(cpt);
                    (found_left, _) = self.match_node_commute(l1, ltree, l2, rtree, !commute);
                }
                let cpt = self.checkpoint();
                let (mut found_right, comm_right) =
                    self.match_node_commute(r1, ltree, r2, rtree, commute);
                if !found_right && comm_right {
                    self.restore(cpt);
                    (found_right, _) = self.match_node_commute(r1, ltree, r2, rtree, !commute);
                }
                (
                    found_left && found_right,
                    lop.is_commutative() || comm_left || comm_right,
                )
            }
            (Binary(..), _) => (false, false),
            (Ternary(lop, la, lb, lc), Ternary(rop, ra, rb, rc)) => {
                if lop != rop {
                    (false, false)
                } else {
                    let (amatch, acomm) = self.match_node_commute(*la, ltree, *ra, rtree, commute);
                    let (bmatch, bcomm) = self.match_node_commute(*lb, ltree, *rb, rtree, commute);
                    let (cmatch, ccomm) = self.match_node_commute(*lc, ltree, *rc, rtree, commute);
                    (amatch && bmatch && cmatch, acomm || bcomm || ccomm)
                }
            }
            (Ternary(..), _) => (false, false),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        dedup::equivalent_trees, deftree, template::test::get_template_by_name, tree::UnaryOp::*,
        walk::DepthWalker,
    };

    fn t_check_bindings(capture: &TemplateCapture, template: &Template, tree: &Tree) {
        let left: Vec<_> = {
            let mut chars: Vec<_> = capture.bindings.iter().map(|(c, _i)| *c).collect();
            chars.sort();
            chars.dedup();
            chars
        };
        let mut right: Vec<_> = template.ping().symbols();
        right.sort();
        assert_eq!(left, right);
        for (_c, i) in capture.bindings.iter() {
            assert!(*i < tree.len());
        }
    }

    #[test]
    fn t_match_with_dofs_1() {
        let template = Template::from(
            "add_zero",
            deftree!(+ 'x 0.).unwrap(),
            deftree!('x).unwrap(),
        );
        let tree = deftree!(+ 0 2).unwrap();
        let mut capture = TemplateCapture::new();
        assert!(capture.next_match(&template, &tree));
        assert!(matches!(capture.node_index, Some(i) if i == 2));
        t_check_bindings(&capture, &template, &tree);
    }

    #[test]
    fn t_match_with_dofs_2() {
        let template = Template::from(
            "fraction_rearrange",
            deftree!(/ (+ 'a 'b) 'a).unwrap(),
            deftree!(+ 1 (/ 'b 'a)).unwrap(),
        );
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        let tree = deftree!(/ (+ 'p 'q) 'q)
            .unwrap()
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner)
            .unwrap();
        let mut capture = TemplateCapture::new();
        assert!(capture.next_match(&template, &tree));
        assert!(matches!(capture.node_index, Some(i) if i == 3));
        t_check_bindings(&capture, &template, &tree);
    }

    #[test]
    fn t_match_with_dofs_3() {
        let template = Template::from(
            "test",
            deftree!(/ (+ (+ 'a 'b) (+ 'c 'd)) (+ 'a 'b)).unwrap(),
            deftree!(+ 1 (/ (+ 'c 'd) (+ 'a 'b))).unwrap(),
        );
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        let tree = deftree!(/ (+ (+ 'p 'q) (+ 'r 's)) (+ 'r 's))
            .unwrap()
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner)
            .unwrap();
        let mut capture = TemplateCapture::new();
        assert!(capture.next_match(&template, &tree));
        assert!(matches!(capture.node_index, Some(i) if i == 7));
        t_check_bindings(&capture, &template, &tree);
    }

    fn t_check_template(name: &str, tree: Tree) {
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        let tree = tree
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner)
            .unwrap();
        println!("{tree}");
        let mut capture = TemplateCapture::new();
        capture.node_index = None;
        capture.bindings.clear();
        let template = get_template_by_name(name).unwrap();
        assert!(
            capture.next_match(template, &tree),
            "Template:{}Tree:{}",
            template.ping(),
            tree
        );
        let node_index = capture.node_index.unwrap();
        match (*tree.node(node_index), template.ping().roots()[0]) {
            (Constant(l), Constant(r)) => assert_eq!(l, r),
            (Symbol(l), Symbol(r)) => assert_eq!(l, r),
            (Unary(l, _), Unary(r, _)) => assert_eq!(l, r),
            (Binary(l, _, _), Binary(r, _, _)) => assert_eq!(l, r),
            (Ternary(l, _, _, _), Ternary(r, _, _, _)) => assert_eq!(l, r),
            _ => panic!("Nodes must be of the same type"),
        }
        t_check_bindings(&capture, template, &tree);
    }

    #[test]
    fn t_match_distribute_mul() {
        t_check_template(
            "distribute_mul",
            deftree!(* 0.5 (+ (* 'x 2.5) (* 'x 1.5))).unwrap(),
        );
    }

    #[test]
    fn t_match_min_of_sqrt() {
        t_check_template(
            "min_of_sqrt",
            deftree!(+ 2.57 (* 1.23 (min (sqrt 2) (sqrt 3)))).unwrap(),
        );
    }

    #[test]
    fn t_match_rearrange_frac() {
        t_check_template(
            "rearrange_frac",
            deftree!(sqrt (log (* (/ 'x 2) (/ 2 'x)))).unwrap(),
        );
    }

    #[test]
    fn t_match_divide_by_self() {
        t_check_template("divide_by_self", deftree!(+ 1 (/ 'p 'p)).unwrap());
    }

    #[test]
    fn t_match_distribute_pow_div() {
        t_check_template(
            "distribute_pow_div",
            deftree!(pow (pow (/ 2 3) 2) 2.5).unwrap(),
        );
    }

    #[test]
    fn t_match_distribute_pow_mul() {
        t_check_template(
            "distribute_pow_mul",
            deftree!(pow (pow (* 2 3) 2) 2.5).unwrap(),
        );
    }

    #[test]
    fn t_match_square_sqrt() {
        t_check_template(
            "square_sqrt",
            deftree!(log (+ 1 (exp (pow (sqrt 3.2556) 2)))).unwrap(),
        );
    }

    #[test]
    fn t_match_sqrt_square() {
        t_check_template(
            "sqrt_square",
            deftree!(log (+ 1 (exp (sqrt (pow 3.2345 2.))))).unwrap(),
        );
    }

    #[test]
    fn t_match_square_abs() {
        t_check_template(
            "square_abs",
            deftree!(log (+ 1 (exp (pow (abs 2) 2.)))).unwrap(),
        );
    }

    #[test]
    fn t_match_mul_exponents() {
        t_check_template(
            "mul_exponents",
            deftree!(log (+ 1 (exp (pow (pow 'x 3.) 2.)))).unwrap(),
        );
    }

    #[test]
    fn t_match_add_exponents() {
        t_check_template(
            "add_exponents",
            deftree!(log (+ 1 (exp (* (pow (log 'x) 2) (pow (log 'x) 3))))).unwrap(),
        );
    }

    #[test]
    fn t_match_add_frac() {
        t_check_template(
            "add_frac",
            deftree!(log (+ 1 (exp (+ (/ 2 (sqrt (+ 2 'x))) (/ 3 (sqrt (+ 'x 2))))))).unwrap(),
        );
    }

    #[test]
    fn t_match_min_expand() {
        t_check_template("min_expand", deftree!(log (+ 1 (exp (min 'x 2)))).unwrap());
    }

    #[test]
    fn t_match_max_expand() {
        t_check_template("max_expand", deftree!(log (+ 1 (exp (max 'x 2)))).unwrap());
    }

    #[test]
    fn t_match_min_of_sub_1() {
        t_check_template(
            "min_of_sub_1",
            deftree!(log (+ 1 (exp (min (- 'x 2) (- 'x 3))))).unwrap(),
        );
    }

    #[test]
    fn t_match_min_of_add_1() {
        const NAME: &str = "min_of_add_1";
        // Make sure all permutations work.
        t_check_template(NAME, deftree!(min (+ 'x 'z) (+ 'x 'y)).unwrap());
        t_check_template(NAME, deftree!(min (+ 'z 'x) (+ 'x 'y)).unwrap());
        t_check_template(NAME, deftree!(min (+ 'z 'x) (+ 'y 'x)).unwrap());
        t_check_template(NAME, deftree!(min (+ 'x 'z) (+ 'y 'x)).unwrap());
    }

    #[test]
    fn t_mutate_multiple_trees() {
        fn assert_one_match(tree: Tree, expected: Tree, capture: &mut TemplateCapture) {
            let mut dedup = Deduplicater::new();
            let mut pruner = Pruner::new();
            let tree = tree
                .deduplicate(&mut dedup)
                .unwrap()
                .prune(&mut pruner)
                .unwrap();
            let expected = expected
                .deduplicate(&mut dedup)
                .unwrap()
                .prune(&mut pruner)
                .unwrap();
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
            deftree!(/ (+ (* 'p 'x) (* 'p 'y)) (+ 'x 'y)).unwrap(),
            deftree!(/ (* 'p (+ 'x 'y)) (+ 'x 'y)).unwrap(),
            &mut capture,
        );
        // Use same capture for a second set of trees.
        assert_one_match(
            deftree!(log (+ 1 (exp (min (sqrt 'x) (sqrt 'y))))).unwrap(),
            deftree!(log (+ 1 (exp (sqrt (min 'x 'y))))).unwrap(),
            &mut capture,
        );
        // Use the same capture on a third set of trees.
        assert_one_match(
            deftree!(/ (* (pow 'a 'b) (pow 'b 'c)) (pow 'c 'a)).unwrap(),
            deftree!(* (pow 'a 'b) (/ (pow 'b 'c) (pow 'c 'a))).unwrap(),
            &mut capture,
        );
    }

    fn check_mutations(mut before: Tree, mut after: Tree) {
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        before = before
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner)
            .unwrap();
        after = after
            .fold()
            .unwrap()
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner)
            .unwrap();
        let mut lwalker = DepthWalker::default();
        let mut rwalker = DepthWalker::default();
        let mut capture = TemplateCapture::new();
        assert_eq!(
            1,
            Mutations::of(&before, &mut capture)
                .filter_map(|t| {
                    let tree = t.unwrap();
                    if equivalent_trees(&after, &tree, &mut lwalker, &mut rwalker) {
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
            deftree!(/ (+ (* 'p 'x) (* 'p 'y)) (+ 'x 'y)).unwrap(),
            deftree!(/ (* 'p (+ 'x 'y)) (+ 'x 'y)).unwrap(),
        );
    }

    #[test]
    fn t_min_sqrt() {
        check_mutations(
            deftree!(log (+ 1 (exp (min (sqrt 'x) (sqrt 'y))))).unwrap(),
            deftree!(log (+ 1 (exp (sqrt (min 'x 'y))))).unwrap(),
        );
    }

    #[test]
    fn t_rearrange_frac() {
        check_mutations(
            deftree!(* (/ (+ 'a 'b) (pow 'x 'y)) (/ (+ 'x 'y) (pow 'a 'b))).unwrap(),
            deftree!(* (/ (+ 'a 'b) (pow 'a 'b)) (/ (+ 'x 'y) (pow 'x 'y))).unwrap(),
        );
    }

    #[test]
    fn t_rearrage_mul_div() {
        check_mutations(
            deftree!(/ (* (pow 'a 'b) (pow 'b 'c)) (pow 'c 'a)).unwrap(),
            deftree!(* (pow 'b 'c) (/ (pow 'a 'b) (pow 'c 'a))).unwrap(),
        );
        check_mutations(
            deftree!(/ (* (pow 'a 'b) (pow 'b 'c)) (pow 'c 'a)).unwrap(),
            deftree!(* (pow 'a 'b) (/ (pow 'b 'c) (pow 'c 'a))).unwrap(),
        );
    }

    #[test]
    fn t_divide_by_self() {
        check_mutations(
            deftree!(/ (+ 'x (pow 'y 'z)) (+ (pow 'y 'z) 'x)).unwrap(),
            deftree!(1).unwrap(),
        );
    }

    #[test]
    fn t_distribute_pow_div() {
        check_mutations(
            deftree!(pow (/ (* 'x 'y) (* 2 3)) 5).unwrap(),
            deftree!(/ (pow (* 'x 'y) 5) (pow (* 2 3) 5)).unwrap(),
        );
    }

    #[test]
    fn t_distribute_pow_mul() {
        check_mutations(
            deftree!(pow (* (* 'x 'y) (* 2 3)) 5).unwrap(),
            deftree!(* (pow (* 'x 'y) 5) (pow (* 2 3) 5)).unwrap(),
        );
    }

    #[test]
    fn t_square_sqrt() {
        check_mutations(
            deftree!(+ 1 (log (pow (sqrt (+ 2 (exp (/ 'x 2)))) 2))).unwrap(),
            deftree!(+ 1 (log (+ 2 (exp (/ 'x 2))))).unwrap(),
        );
    }

    #[test]
    fn t_sqrt_square() {
        check_mutations(
            deftree!(+ 1 (log (sqrt (pow (+ 2 (exp (/ 'x 2))) 2)))).unwrap(),
            deftree!(+ 1 (log (abs (+ 2 (exp (/ 'x 2)))))).unwrap(),
        );
    }

    #[test]
    fn t_square_abs() {
        check_mutations(
            deftree!(exp (+ 1 (log (pow (abs (* 'p 'q)) 2)))).unwrap(),
            deftree!(exp (+ 1 (log (pow (* 'p 'q) 2)))).unwrap(),
        );
    }

    #[test]
    fn t_mul_exponents() {
        check_mutations(
            deftree!(exp (+ 1 (log (pow (pow 'p (+ 2 'm)) (/ 'q 'r))))).unwrap(),
            deftree!(exp (+ 1 (log (pow 'p (* (+ 2 'm) (/ 'q 'r)))))).unwrap(),
        );
    }

    #[test]
    fn t_add_exponents() {
        check_mutations(
            deftree!(exp (+ 1 (log (* (pow 'p (+ 2 'm)) (pow 'p (/ 'q 'r)))))).unwrap(),
            deftree!(exp (+ 1 (log (pow 'p (+ (+ 2 'm) (/ 'q 'r)))))).unwrap(),
        );
    }

    #[test]
    fn t_binomial_square() {
        check_mutations(
            deftree!(pow (+ (log (+ 'a 1)) (log (+ 'b 1))) 2.).unwrap(),
            deftree!(+ (+ (pow (log (+ 'a 1)) 2.) (pow (log (+ 'b 1)) 2.))
                     (* 2. (* (log (+ 'a 1)) (log (+ 'b 1)))))
            .unwrap(),
        );
        check_mutations(
            deftree!(pow (- (log (+ 'a 1)) (log (+ 'b 1))) 2.).unwrap(),
            deftree!(- (+ (pow (log (+ 'a 1)) 2.) (pow (log (+ 'b 1)) 2.))
                     (* 2. (* (log (+ 'a 1)) (log (+ 'b 1)))))
            .unwrap(),
        );
    }

    #[test]
    fn t_mutate_concat() {
        check_mutations(
            deftree!(concat (+ (* 'p 'x) (* 'p 'y)) 1.)
                .unwrap()
                .reshape(1, 2)
                .unwrap(),
            deftree!(concat (* 'p (+ 'x 'y)) 1.)
                .unwrap()
                .reshape(1, 2)
                .unwrap(),
        );
    }

    #[test]
    fn t_dependent_roots_compact() {
        // Just making sure if the tree has root nodes depend on each other, use
        // each other as inputs, it is handled correctly and we never actually
        // produce dependent roots nodes.
        let tree = deftree!(concat (- 'x) 'x).unwrap();
        let mut capture = TemplateCapture::new();
        let tree = capture.compact_tree(tree).unwrap();
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Negate, 0), Symbol('x')]);
        assert_eq!(tree.dims(), (2, 1));
    }
}
