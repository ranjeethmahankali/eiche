use crate::{
    mutate::Mutations,
    tree::Tree,
    tree::{Node, Node::*},
};

pub struct EulerWalkHeuristic {
    visited: Vec<bool>,
    stack: Vec<usize>,
    euler: Vec<usize>,
    last_visited: Vec<usize>,
    num_parents: Vec<usize>,
}

impl EulerWalkHeuristic {
    pub fn new() -> EulerWalkHeuristic {
        EulerWalkHeuristic {
            visited: Vec::new(),
            stack: Vec::new(),
            euler: Vec::new(),
            last_visited: Vec::new(),
            num_parents: Vec::new(),
        }
    }

    pub fn compute(&mut self, nodes: &Vec<Node>, root: usize) -> usize {
        // Reset all buffers.
        self.visited.clear();
        self.visited.resize(nodes.len(), false);
        self.stack.clear();
        self.stack.reserve(nodes.len());
        self.euler.clear();
        self.euler.reserve(3 * nodes.len()); // Estimate.
        self.last_visited.clear();
        self.last_visited.resize(nodes.len(), 0);
        self.num_parents.clear();
        self.num_parents.resize(nodes.len(), 0);
        // Euler walk.
        self.stack.push(root);
        while let Some(i) = self.stack.pop() {
            if !self.visited[i] {
                match &nodes[i] {
                    Constant(_) | Symbol(_) => {}
                    Unary(_, input) => self.stack.extend_from_slice(&[i, *input]),
                    Binary(_, lhs, rhs) => self.stack.extend_from_slice(&[i, *rhs, i, *lhs]),
                };
                self.visited[i] = true;
            }
            self.euler.push(i);
        }
        // Count parents.
        for node in nodes {
            match node {
                Constant(_) | Symbol(_) => {}
                Unary(_, input) => self.num_parents[*input] += 1,
                Binary(_, lhs, rhs) => {
                    self.num_parents[*lhs] += 1;
                    self.num_parents[*rhs] += 1;
                }
            }
        }
        // Find distances between occurrences of nodes with multiple
        // parents and sum them.
        self.euler
            .iter()
            .zip(0..self.euler.len())
            .map(|(node, i)| {
                let last = std::mem::replace(&mut self.last_visited[*node], i);
                if self.num_parents[*node] > 1 && last > 0 {
                    i - last
                } else {
                    0
                }
            })
            .sum()
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
}
