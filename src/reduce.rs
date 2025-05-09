use crate::{
    error::Error,
    mutate::{Mutations, TemplateCapture},
    template::TEMPLATES,
    tree::{Node, Node::*, Tree},
};
use std::{
    collections::{BinaryHeap, HashMap},
    ops::Range,
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
    fn euler_walk_cost(&mut self, nodes: &[Node], roots: Range<usize>) -> usize {
        // Reset all buffers.
        self.stack.clear();
        self.stack.reserve(nodes.len());
        self.last_visit.clear();
        self.last_visit.resize(nodes.len(), None);
        // Start the Euler walk.
        self.stack.extend(roots.map(|r| (r, 0)));
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
                    Ternary(_, a, b, c) => self.stack.extend_from_slice(&[
                        (i, depth),
                        (*a, depth + 1),
                        (i, depth),
                        (*b, depth + 1),
                        (i, depth),
                        (*c, depth + 1),
                    ]),
                },
                _ => {} // Do nothing.
            }
            // Record visit, update counter and depth.
            self.last_visit[i] = Some(counter);
            counter += 1;
            prevdepth = depth;
        }
        sum
    }

    fn cost(&mut self, tree: &Tree) -> usize {
        tree.len() + self.euler_walk_cost(tree.nodes(), tree.root_indices())
    }
}

struct Candidate {
    tree: Tree,
    prev: usize,
    steps: usize,
    complexity: usize,
}

impl Candidate {
    pub fn cost(&self) -> usize {
        self.steps + self.tree.len() + self.complexity
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.tree == other.tree
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.cost().cmp(&self.cost())
    }
}
impl Eq for Candidate {}

/// Reduce the input `tree`. This will stop after `max_iter` iterations and
/// return the most concise form of the tree at that point.
pub fn reduce(tree: Tree, max_iter: usize) -> Result<Vec<Tree>, Error> {
    let mut capture = TemplateCapture::new();
    let tree = capture.compact_tree(tree)?;
    let mut hfn = Heuristic::new();
    let mut explored = Vec::<Candidate>::with_capacity(max_iter);
    let mut indexmap = HashMap::<u64, usize>::new();
    let mut hashbuf = Vec::<u64>::new();
    let mut heap = BinaryHeap::<Candidate>::with_capacity(TEMPLATES.len() * max_iter / 2); // Estimate.
    let mut min_complexity = usize::MAX;
    let mut best_candidate = 0;
    let start_complexity = hfn.cost(&tree);
    heap.push(Candidate {
        tree,
        prev: 0,
        steps: 0,
        complexity: start_complexity,
    });
    while let Some(cand) = heap.pop() {
        let hash = cand.tree.hash(&mut hashbuf);
        let index = explored.len();
        match indexmap.insert(hash, index) {
            Some(_old) => {
                continue;
            }
            None => explored.push(cand),
        }
        let cand = explored.last().unwrap();
        if cand.complexity < min_complexity {
            min_complexity = cand.complexity;
            best_candidate = index;
        }
        if explored.len() == max_iter {
            break;
        }
        for mutation in Mutations::of(&cand.tree, &mut capture) {
            let tree = mutation?;
            let complexity = hfn.cost(&tree);
            heap.push(Candidate {
                tree,
                prev: index,
                steps: cand.steps + 1,
                complexity,
            });
        }
    }
    let mut steps = Vec::<Tree>::new();
    let mut i = best_candidate;
    while explored[i].prev != i {
        let cand = &explored[i];
        steps.push(cand.tree.clone());
        i = cand.prev;
    }
    steps.reverse();
    Ok(steps)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{dedup::Deduplicater, deftree, prune::Pruner};

    #[test]
    fn t_euler_walk_depth_1() {
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        let mut h = Heuristic::new();
        let tree = deftree!(+ x x)
            .unwrap()
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner)
            .unwrap();
        assert_eq!(h.euler_walk_cost(tree.nodes(), tree.root_indices()), 2);
    }

    #[test]
    fn t_euler_walk_depth_2() {
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        let mut h = Heuristic::new();
        let tree = deftree!(+ (* 2 x) (* 3 x))
            .unwrap()
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner)
            .unwrap();
        assert_eq!(h.euler_walk_cost(tree.nodes(), tree.root_indices()), 6);
    }

    #[test]
    fn t_euler_walk_multiple() {
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        let mut h = Heuristic::new();
        let tree = deftree!(+ (+ (* 2 x) (* 3 x)) (* 4 x))
            .unwrap()
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner)
            .unwrap();
        assert_eq!(h.euler_walk_cost(tree.nodes(), tree.root_indices()), 13);
        // Make sure the same heuristic instance can be reused on other trees.
        let tree = deftree!(+ (+ (* 2 x) (* 3 x)) (+ (* 4 x) 2))
            .unwrap()
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner)
            .unwrap();
        assert_eq!(h.euler_walk_cost(tree.nodes(), tree.root_indices()), 33);
    }

    #[test]
    fn t_euler_walk_non_leaf() {
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        let mut h = Heuristic::new();
        let tree = deftree!(+ (* 2 (+ x y)) (* (+ x y) 3))
            .unwrap()
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner)
            .unwrap();
        assert_eq!(h.euler_walk_cost(tree.nodes(), tree.root_indices()), 4);
    }

    fn check_heuristic_and_mutations(before: Tree, after: Tree) {
        // Make sure the 'after' tree has lower cost than the 'before
        // tree. And that the 'after' tree is found exactly once
        // among the mutations of the 'before' tree.
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        let before = before
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner)
            .unwrap();
        let after = after
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner)
            .unwrap();
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
            deftree!(/ (+ (* p x) (* p y)) (+ x y)).unwrap(),
            deftree!(/ (* p (+ x y)) (+ x y)).unwrap(),
        );
    }

    #[test]
    fn t_heuristic_cost_2() {
        check_heuristic_and_mutations(
            deftree!(log (+ 1 (exp (min (sqrt x) (sqrt y))))).unwrap(),
            deftree!(log (+ 1 (exp (sqrt (min x y))))).unwrap(),
        );
    }

    #[test]
    fn t_factoring() {
        let tree = deftree!(/ (+ (* p x) (* p y)) (+ x y)).unwrap();
        let steps = reduce(tree, 8).unwrap();
        assert!(steps.last().unwrap().equivalent(&deftree!(p).unwrap()));
    }

    #[test]
    fn t_norm_vec_len() {
        let tree = deftree!(sqrt (+ (pow (/ x (sqrt (+ (pow x 2) (pow y 2)))) 2)
                                  (pow (/ y (sqrt (+ (pow x 2) (pow y 2)))) 2)))
        .unwrap();
        let steps = reduce(tree, 8).unwrap();
        assert!(steps.last().unwrap().equivalent(&deftree!(1).unwrap()));
    }

    #[test]
    fn t_concat_1() {
        let tree = deftree!(concat
                            (/ (+ (* p x) (* p y)) (+ x y))
                            1.
        )
        .unwrap();
        let steps = reduce(tree, 8).unwrap();
        assert!(
            steps
                .last()
                .unwrap()
                .equivalent(&deftree!(concat p 1).unwrap())
        );
    }

    #[test]
    fn t_concat_2() {
        let tree = deftree!(concat
                            (sqrt (+ (pow (/ x (sqrt (+ (pow x 2) (pow y 2)))) 2)
                                   (pow (/ y (sqrt (+ (pow x 2) (pow y 2)))) 2)))
                            42.)
        .unwrap();
        let steps = reduce(tree, 12).unwrap();
        assert!(
            steps
                .last()
                .unwrap()
                .equivalent(&deftree!(concat 1. 42.).unwrap())
        );
    }

    #[test]
    fn t_concat_3() {
        let tree = deftree!(concat
                            (/ (+ (* p x) (* p y)) (+ x y))
                            (sqrt (+ (pow (/ x (sqrt (+ (pow x 2) (pow y 2)))) 2)
                                   (pow (/ y (sqrt (+ (pow x 2) (pow y 2)))) 2))))
        .unwrap();
        let steps = reduce(tree, 12).unwrap();
        assert!(
            steps
                .last()
                .unwrap()
                .equivalent(&deftree!(concat p 1.).unwrap())
        );
    }

    #[test]
    fn t_gradient() {
        let tree = deftree!(sderiv (- (+ (pow x 2) (pow y 2)) 5) xy).unwrap();
        let steps = reduce(tree, 12).unwrap();
        assert!(
            steps.last().unwrap().equivalent(
                &deftree!(concat (* 2 x) (* 2 y))
                    .unwrap()
                    .reshape(1, 2)
                    .unwrap()
            )
        );
    }

    #[test]
    fn t_second_deriv() {
        let tree = deftree!(sderiv (sderiv (- (+ (pow x 3) (pow y 3)) 5) x) x).unwrap();
        let steps = reduce(tree, 32).unwrap();
        assert!(steps.last().unwrap().equivalent(&deftree!(* 6 x).unwrap()));
    }

    #[test]
    fn t_hessian() {
        let tree = deftree!(sderiv (sderiv (- (+ (pow x 3) (pow y 3)) 5) xy) xy).unwrap();
        let steps = reduce(tree, 256).unwrap();
        assert!(
            steps.last().unwrap().equivalent(
                &deftree!(concat (* x 6) 0 0 (* y 6))
                    .unwrap()
                    .reshape(2, 2)
                    .unwrap()
            )
        );
    }

    #[test]
    fn t_circle_gradient() {
        let tree = deftree!(sderiv (- (sqrt (+ (pow x 2) (pow y 2))) 3) xy).unwrap();
        let steps = reduce(tree, 64).unwrap();
        assert!(
            steps.last().unwrap().equivalent(
                &deftree!(concat
                      (/ x (sqrt (+ (pow x 2) (pow y 2))))
                      (/ y (sqrt (+ (pow x 2) (pow y 2)))))
                .unwrap()
                .reshape(1, 2)
                .unwrap()
            )
        );
    }

    #[test]
    fn t_min_max_self() {
        assert!(
            reduce(deftree!(min (+ (pow x 2) y) (+ y (pow x 2))).unwrap(), 2)
                .unwrap()
                .last()
                .unwrap()
                .equivalent(&deftree!(+ (pow x 2) y).unwrap())
        );
        assert!(
            reduce(deftree!(max (+ (pow x 2) y) (+ y (pow x 2))).unwrap(), 2)
                .unwrap()
                .last()
                .unwrap()
                .equivalent(&deftree!(+ (pow x 2) y).unwrap())
        );
    }
}
