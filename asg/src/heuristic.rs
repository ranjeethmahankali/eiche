use crate::tree::{Node, Node::*};

struct UnionFind {
    reps: Vec<usize>,
}

impl UnionFind {
    fn new() -> UnionFind {
        UnionFind { reps: Vec::new() }
    }

    fn reset(&mut self, len: usize) {
        self.reps.clear();
        self.reps.extend(0..len);
    }

    fn unite(&mut self, mut a: usize, b: usize) {
        while a != self.reps[a] {
            a = self.reps[a];
        }
        self.reps[b] = a;
    }

    fn find(&self, id: usize) -> usize {
        self.reps[id]
    }
}

struct Tie {
    index: usize,
    loopsize: usize,
}

pub struct LoopCounter {
    ufind: UnionFind,
    ancestors: Vec<usize>,
    visited: Vec<bool>,
    num_parents: Vec<usize>,
    ties: Vec<Tie>,
    depths: Vec<usize>,
}

impl LoopCounter {
    pub fn new() -> LoopCounter {
        LoopCounter {
            ufind: UnionFind::new(),
            ancestors: Vec::new(),
            visited: Vec::new(),
            num_parents: Vec::new(),
            ties: Vec::new(),
            depths: Vec::new(),
        }
    }

    fn reset(&mut self, len: usize) {
        self.ufind.reset(len);
        self.ancestors.clear();
        self.ancestors.reserve(len);
        self.ancestors.extend(0..len);
        self.visited.clear();
        self.visited.resize(len, false);
        self.num_parents.clear();
        self.num_parents.resize(len, 0);
        self.ties.clear();
        self.depths.clear();
        self.depths.resize(len, 0);
    }

    fn tarjan_lca(&mut self, index: usize, nodes: &Vec<Node>, depth: usize) {
        self.ancestors[index] = index;
        self.depths[index] = depth;
        if !self.visited[index] {
            match &nodes[index] {
                Constant(_) | Symbol(_) => {} // Do nothing.
                Unary(_, input) => {
                    self.tarjan_lca(*input, nodes, depth + 1);
                    self.ufind.unite(index, *input);
                    self.ancestors[self.ufind.find(index)] = index;
                }
                Binary(_, lhs, rhs) => {
                    // Left.
                    self.tarjan_lca(*lhs, nodes, depth + 1);
                    self.ufind.unite(index, *lhs);
                    self.ancestors[self.ufind.find(index)] = index;
                    // Right.
                    self.tarjan_lca(*rhs, nodes, depth + 1);
                    self.ufind.unite(index, *rhs);
                    self.ancestors[self.ufind.find(index)] = index;
                }
            }
            self.visited[index] = true;
        } else {
            for tie in self.ties.iter_mut() {
                if tie.index == index {
                    tie.loopsize += depth - self.depths[self.ancestors[index]];
                }
            }
        }
    }

    pub fn run(&mut self, nodes: &Vec<Node>, root: usize) -> usize {
        self.reset(nodes.len());
        // Find all ties.
        for node in nodes {
            match node {
                Constant(_) | Symbol(_) => {} // Do nothing.
                Unary(_, input) => self.num_parents[*input] += 1,
                Binary(_, lhs, rhs) => {
                    self.num_parents[*lhs] += 1;
                    self.num_parents[*rhs] += 1;
                }
            }
        }
        self.ties
            .extend((0..nodes.len()).filter_map(|i| -> Option<Tie> {
                if self.num_parents[i] > 1 {
                    Some(Tie {
                        index: i,
                        loopsize: 0,
                    })
                } else {
                    None
                }
            }));
        // Run Tarjan's algo and return total.
        self.tarjan_lca(root, nodes, 0);
        return self.ties.iter().map(|t| t.loopsize).sum();
    }
}
