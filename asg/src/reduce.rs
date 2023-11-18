use crate::{
    mutate::{Mutations, TemplateCapture},
    tree::Tree,
    tree::{Node, Node::*},
};

struct Heuristic {
    stack: Vec<(usize, usize)>, // index, depth
    last_visit: Vec<Option<usize>>,
}

/// If two nodes have the same node as their one of their inputs, it
/// creates a diamond shaped loop. If this loop is small, i.e. if the
/// lowest common ancestor of the two nodes is near by, It is more
/// likely that a known template will match with the tree and be able
/// to simplify it. The euler-walk heuristic penalizes diamond shaped
/// loops based on their size, i.e. larger loops have a higher cost
/// than smaller loops. While the heuristic tries to add up the
/// lengths of all diamond shaped loops, the value might not always be
/// exact, depending on the traversal order. But the value is
/// guaranteed to have a positive correlation with the number of
/// diamond shaped loops and their sizes.
impl Heuristic {
    pub fn new() -> Heuristic {
        Heuristic {
            stack: Vec::new(),
            last_visit: Vec::new(),
        }
    }

    /// In a typical depth first traversal, you just push the children
    /// of the current node onto the stack. Instead, if you also push
    /// the node itself, before every child, it results in an euler
    /// walk. This is very useful because for any pair of nodes 'a'
    /// and 'b', euler-walk necessarily contains a subpath that starts
    /// at 'a' and ends at 'b' or starts at 'b' and ends at
    /// 'a'. Furthermore, this subpath necessarily goes through the
    /// lowest common ancestor of 'a' and 'b'. That means, if we're
    /// visiting a node for the second (or more) time from a parent,
    /// we've detected a diamond shaped loop, and the number of nodes
    /// traversed since the last visit roughly correlates to the size
    /// of the diamond shaped loop.
    fn euler_walk_cost(&mut self, nodes: &Vec<Node>, root: usize) -> usize {
        // Reset all buffers.
        self.stack.clear();
        self.stack.reserve(nodes.len());
        self.last_visit.clear();
        self.last_visit.resize(nodes.len(), None);
        // Start the Euler walk.
        self.stack.push((root, 0));
        let mut prevdepth: usize = 0;
        let mut counter: usize = 0;
        let mut sum: usize = 0;
        while let Some((i, depth)) = self.stack.pop() {
            match self.last_visit[i] {
                // Accumulate the size of the diamond shaped loop.
                Some(last) if prevdepth < depth => sum += counter - last,
                // Push children if visiting for the first time.
                None => match &nodes[i] {
                    Constant(_) | Symbol(_) => {} // No children to push.
                    Unary(_, input) => self
                        .stack
                        .extend_from_slice(&[(i, depth), (*input, depth + 1)]),
                    Binary(_, lhs, rhs) => self.stack.extend_from_slice(&[
                        (i, depth),
                        (*rhs, depth + 1),
                        (i, depth),
                        (*lhs, depth + 1),
                    ]),
                },
                _ => {} // Do nothing.
            }
            // Record visit, update counter and depth.
            self.last_visit[i] = Some(counter);
            counter += 1;
            prevdepth = depth;
        }
        return sum;
    }

    fn cost(&mut self, tree: &Tree) -> usize {
        tree.len() + self.euler_walk_cost(tree.nodes(), tree.root_index())
    }
}

pub fn reduce(tree: Tree) -> Result<Tree, ()> {
    let mut capture = TemplateCapture::new();
    let mutations = Mutations::of(&tree, &mut capture);
    let mut costs = Vec::<usize>::new();
    let mut h = Heuristic::new();
    for tree in mutations {
        match tree {
            Ok(t) => costs.push(h.cost(&t)),
            Err(_) => todo!(),
        }
    }
    todo!();
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::deftree;

    #[test]
    fn t_euler_walk_depth_1() {
        let mut h = Heuristic::new();
        let tree = deftree!(+ x x).deduplicate().unwrap();
        assert_eq!(h.euler_walk_cost(tree.nodes(), tree.root_index()), 2);
    }

    #[test]
    fn t_euler_walk_depth_2() {
        let mut h = Heuristic::new();
        let tree = deftree!(+ (* 2 x) (* 3 x)).deduplicate().unwrap();
        assert_eq!(h.euler_walk_cost(tree.nodes(), tree.root_index()), 6);
    }

    #[test]
    fn t_euler_walk_multiple() {
        let mut h = Heuristic::new();
        let tree = deftree!(+ (+ (* 2 x) (* 3 x)) (* 4 x))
            .deduplicate()
            .unwrap();
        assert_eq!(h.euler_walk_cost(tree.nodes(), tree.root_index()), 13);
        // Make sure the same heuristic instance can be reused on other trees.
        let tree = deftree!(+ (+ (* 2 x) (* 3 x)) (+ (* 4 x) 2))
            .deduplicate()
            .unwrap();
        assert_eq!(h.euler_walk_cost(tree.nodes(), tree.root_index()), 33);
    }

    #[test]
    fn t_euler_walk_non_leaf() {
        let mut h = Heuristic::new();
        let tree = deftree!(+ (* 2 (+ x y)) (* (+ x y) 3))
            .deduplicate()
            .unwrap();
        assert_eq!(h.euler_walk_cost(tree.nodes(), tree.root_index()), 4);
    }

    fn check_heuristic_and_mutations(before: Tree, after: Tree) {
        // Make sure the 'after' tree has lower cost than the 'before
        // tree. And that the 'after' tree is found exactly once
        // among the mutations of the 'before' tree.
        let before = before.deduplicate().unwrap();
        let after = after.deduplicate().unwrap();
        let mut h = Heuristic::new();
        let mut capture = TemplateCapture::new();
        assert!(h.cost(&before) > h.cost(&after));
        assert_eq!(
            1,
            Mutations::of(&before, &mut capture)
                .filter_map(|t| if t.unwrap().equivalent(&after) {
                    Some(())
                } else {
                    None
                })
                .count()
        );
    }

    #[test]
    fn t_heuristic_cost_1() {
        check_heuristic_and_mutations(
            deftree!(/ (+ (* p x) (* p y)) (+ x y)),
            deftree!(/ (* p (+ x y)) (+ x y)),
        );
    }

    #[test]
    fn t_heuristic_cost_2() {
        check_heuristic_and_mutations(
            deftree!(log (+ 1 (exp (min (sqrt x) (sqrt y))))),
            deftree!(log (+ 1 (exp (sqrt (min x y))))),
        );
    }
}
