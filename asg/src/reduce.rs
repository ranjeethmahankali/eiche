use crate::{
    mutate::Mutations,
    tree::Tree,
    tree::{Node, Node::*},
};

pub struct EulerWalkHeuristic {
    stack: Vec<(usize, usize)>,
    euler: Vec<(usize, bool)>, // Index, whether visiting from a parent node.
    node_data: Vec<(bool, usize, usize)>, // visited, last_visit, num_parents.
}

impl EulerWalkHeuristic {
    pub fn new() -> EulerWalkHeuristic {
        EulerWalkHeuristic {
            stack: Vec::new(),
            euler: Vec::new(),
            node_data: Vec::new(),
        }
    }

    fn reset(&mut self, size: usize) {
        self.stack.clear();
        self.stack.reserve(size);
        self.euler.clear();
        self.euler.reserve(3 * size); // Estimate.
        self.node_data.clear();
        self.node_data.resize(size, (false, 0, 0));
    }

    fn walk(&mut self, nodes: &Vec<Node>, root: usize) {
        self.stack.push((root, 0));
        let mut prevdepth = 0;
        while let Some((i, depth)) = self.stack.pop() {
            let (visited, _, _) = &mut self.node_data[i];
            if !(*visited) {
                match &nodes[i] {
                    Constant(_) | Symbol(_) => {}
                    Unary(_, input) => self
                        .stack
                        .extend_from_slice(&[(i, depth), (*input, depth + 1)]),
                    Binary(_, lhs, rhs) => self.stack.extend_from_slice(&[
                        (i, depth),
                        (*rhs, depth + 1),
                        (i, depth),
                        (*lhs, depth + 1),
                    ]),
                };
                *visited = true;
            }
            self.euler.push((i, prevdepth < depth));
            prevdepth = depth;
        }
    }

    fn count_parents(&mut self, nodes: &Vec<Node>) {
        for node in nodes {
            match node {
                Constant(_) | Symbol(_) => {}
                Unary(_, input) => {
                    self.node_data[*input].2 += 1;
                }
                Binary(_, lhs, rhs) => {
                    self.node_data[*lhs].2 += 1;
                    self.node_data[*rhs].2 += 1;
                }
            }
        }
    }

    fn total_euler_distance(&mut self) -> usize {
        self.euler
            .iter()
            .zip(0..self.euler.len())
            .map(|((node, from_parent), i)| {
                let last_visit = {
                    let (_, last_visit, num_parents) = &mut self.node_data[*node];
                    if *num_parents < 2 {
                        return 0;
                    }
                    last_visit
                };
                let last = std::mem::replace(last_visit, i);
                if *from_parent && last > 0 {
                    return i - last;
                } else {
                    return 0;
                }
            })
            .sum()
    }

    fn compute(&mut self, nodes: &Vec<Node>, root: usize) -> usize {
        if nodes.is_empty() {
            return 0;
        }
        self.reset(nodes.len());
        self.walk(nodes, root);
        self.count_parents(nodes);
        return self.total_euler_distance();
    }
}

fn complexity(tree: &Tree, euler: &mut EulerWalkHeuristic) -> usize {
    tree.len() + euler.compute(tree.nodes(), tree.root_index())
}

pub fn reduce(tree: Tree) -> Result<Tree, ()> {
    let mutations = Mutations::from(&tree);
    let mut costs = Vec::<usize>::new();
    let mut lc = EulerWalkHeuristic::new();
    for tree in mutations {
        match tree {
            Ok(t) => costs.push(complexity(&t, &mut lc)),
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
        let mut h = EulerWalkHeuristic::new();
        let tree = deftree!(+ x x).deduplicate().unwrap();
        assert_eq!(h.compute(tree.nodes(), tree.root_index()), 2);
    }

    #[test]
    fn t_euler_walk_depth_2() {
        let mut h = EulerWalkHeuristic::new();
        let tree = deftree!(+ (* 2 x) (* 3 x)).deduplicate().unwrap();
        assert_eq!(h.compute(tree.nodes(), tree.root_index()), 6);
    }

    #[test]
    fn t_euler_walk_multiple() {
        let mut h = EulerWalkHeuristic::new();
        let tree = deftree!(+ (+ (* 2 x) (* 3 x)) (* 4 x))
            .deduplicate()
            .unwrap();
        println!("{}", tree);
        assert_eq!(h.compute(tree.nodes(), tree.root_index()), 13);
        // Make sure the same heuristic instance can be reused on other trees.
        let tree = deftree!(+ (+ (* 2 x) (* 3 x)) (+ (* 4 x) 2))
            .deduplicate()
            .unwrap();
        assert_eq!(h.compute(tree.nodes(), tree.root_index()), 33);
    }

    #[test]
    fn t_euler_walk_non_leaf() {
        let mut h = EulerWalkHeuristic::new();
        let tree = deftree!(+ (* 2 (+ x y)) (* (+ x y) 3))
            .deduplicate()
            .unwrap();
        assert_eq!(h.compute(tree.nodes(), tree.root_index()), 4);
    }
}
