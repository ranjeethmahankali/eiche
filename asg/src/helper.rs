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
        write!(f, "\n")?;
        let mut depths: Box<[usize]> = vec![0; self.len()].into_boxed_slice();
        self.traverse_depth(
            |index, parent| {
                const BRANCH: &str = " ├── ";
                const BYPASS: &str = " │   ";
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
                writeln!(f, "[{}] {}", index, self.node(index))
            },
            false,
        )?;
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
