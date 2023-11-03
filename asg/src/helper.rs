use crate::tree::{Node, Node::*, Tree};

impl Into<Tree> for Node {
    fn into(self) -> Tree {
        Tree::new(self)
    }
}

impl From<f64> for Tree {
    fn from(value: f64) -> Self {
        return Constant(value).into();
    }
}

impl From<f64> for Node {
    fn from(value: f64) -> Self {
        return Constant(value);
    }
}

impl From<char> for Node {
    fn from(value: char) -> Self {
        return Symbol(value);
    }
}

impl From<char> for Tree {
    fn from(c: char) -> Self {
        return Symbol(c).into();
    }
}

impl std::fmt::Display for Tree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const BRANCH: &str = " ├── ";
        const BYPASS: &str = " │   ";
        write!(f, "\n")?;
        let mut depths: Box<[usize]> = vec![0; self.len()].into_boxed_slice();
        let mut walker = DepthWalker::new();
        for (index, parent) in walker.walk_tree(self, false, false) {
            if let Some(pi) = parent {
                depths[index] = depths[pi] + 1;
            }
            let depth = depths[index];
            for d in 0..depth {
                write!(f, "{}", {
                    if d < depth - 1 {
                        BYPASS
                    } else {
                        BRANCH
                    }
                })?;
            }
            writeln!(f, "[{}] {}", index, self.node(index))?;
        }
        write!(f, "\n")
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Constant(value) => write!(f, "Constant({})", value),
            Symbol(label) => write!(f, "Symbol({})", label),
            Unary(op, input) => write!(f, "{:?}({})", op, input),
            Binary(op, lhs, rhs) => write!(f, "{:?}({}, {})", op, lhs, rhs),
        }
    }
}

pub fn eq_recursive(nodes: &[Node], li: usize, ri: usize) -> bool {
    let mut stack: Vec<(usize, usize)> = vec![(li, ri)];
    while !stack.is_empty() {
        let (a, b) = stack.pop().expect("This should never happen!");
        if a == b {
            continue;
        }
        if !(match (nodes[a], nodes[b]) {
            (Constant(v1), Constant(v2)) => v1 == v2,
            (Symbol(c1), Symbol(c2)) => c1 == c2,
            (Unary(op1, input1), Unary(op2, input2)) => {
                stack.push((input1, input2));
                op1 == op2
            }
            (Binary(op1, lhs1, rhs1), Binary(op2, lhs2, rhs2)) => {
                stack.push((usize::min(lhs1, rhs1), usize::min(lhs2, rhs2)));
                stack.push((usize::max(lhs1, rhs1), usize::max(lhs2, rhs2)));
                op1 == op2
            }
            _ => false,
        }) {
            return false;
        }
    }
    return true;
}

pub struct DepthWalker {
    stack: Vec<(usize, Option<usize>)>,
    visited: Vec<bool>,
}

impl DepthWalker {
    pub fn new() -> DepthWalker {
        DepthWalker {
            stack: vec![],
            visited: vec![],
        }
    }

    pub fn walk_tree<'a>(
        &'a mut self,
        tree: &'a Tree,
        unique: bool,
        mirrored: bool,
    ) -> DepthIterator<'a> {
        self.walk_nodes(&tree.nodes(), tree.root_index(), unique, mirrored)
    }

    pub fn walk_nodes<'a>(
        &'a mut self,
        nodes: &'a Vec<Node>,
        root_index: usize,
        unique: bool,
        mirrored: bool,
    ) -> DepthIterator<'a> {
        // Prep the stack.
        self.stack.clear();
        self.stack.reserve(nodes.len());
        self.stack.push((root_index, None));
        // Reset the visited flags.
        self.visited.clear();
        self.visited.resize(nodes.len(), false);
        // Create the iterator.
        DepthIterator {
            unique,
            mirrored,
            walker: self,
            nodes: &nodes,
        }
    }
}

pub struct DepthIterator<'a> {
    unique: bool,
    mirrored: bool,
    walker: &'a mut DepthWalker,
    nodes: &'a Vec<Node>,
}

impl<'a> Iterator for DepthIterator<'a> {
    type Item = (usize, Option<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        let (index, parent) = {
            // Pop the stack until we find a node we didn't already visit.
            let (mut i, mut p) = self.walker.stack.pop()?;
            while self.unique && self.walker.visited[i] {
                (i, p) = self.walker.stack.pop()?;
            }
            (i, p)
        };
        // Push the children on to the stack.
        match &self.nodes[index] {
            Constant(_) | Symbol(_) => {}
            Unary(_op, input) => {
                self.walker.stack.push((*input, Some(index)));
            }
            Binary(_op, lhs, rhs) => {
                if self.mirrored {
                    self.walker.stack.push((*lhs, Some(index)));
                    self.walker.stack.push((*rhs, Some(index)));
                } else {
                    self.walker.stack.push((*rhs, Some(index)));
                    self.walker.stack.push((*lhs, Some(index)));
                }
            }
        }
        self.walker.visited[index] = true;
        return Some((index, parent));
    }
}

pub struct Trimmer {
    indices: Vec<(bool, usize)>,
    trimmed: Vec<Node>,
}

impl Trimmer {
    pub fn new() -> Trimmer {
        Trimmer {
            indices: vec![],
            trimmed: vec![],
        }
    }

    pub fn trim(
        &mut self,
        mut nodes: Vec<Node>,
        root_index: usize,
        walker: &mut DepthWalker,
    ) -> Vec<Node> {
        self.indices.clear();
        self.indices.resize(nodes.len(), (false, 0));
        // Mark used nodes.
        walker
            .walk_nodes(&nodes, root_index, true, false)
            .for_each(|(index, _parent)| {
                self.indices[index] = (true, 1usize);
            });
        // Do exclusive scan.
        let mut sum = 0usize;
        for pair in self.indices.iter_mut() {
            let (keep, i) = *pair;
            let copy = sum;
            sum += i;
            *pair = (keep, copy);
        }
        // Filter, update and copy nodes.
        self.trimmed.reserve(nodes.len());
        self.trimmed.extend(
            (0..self.indices.len())
                .zip(nodes.iter())
                .filter(|(i, _node)| {
                    let (keep, _index) = self.indices[*i];
                    return keep;
                })
                .map(|(_i, node)| {
                    match node {
                        // Update the indices of this node's inputs.
                        Constant(val) => Constant(*val),
                        Symbol(label) => Symbol(*label),
                        Unary(op, input) => Unary(*op, self.indices[*input].1),
                        Binary(op, lhs, rhs) => {
                            Binary(*op, self.indices[*lhs].1, self.indices[*rhs].1)
                        }
                    }
                }),
        );
        std::mem::swap(&mut self.trimmed, &mut nodes);
        return nodes;
    }
}
