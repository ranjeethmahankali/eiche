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
        self.depth_first_traverse(|index, parent| {
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
            writeln!(f, "[{}] {:?}", index, self.node(index))
        })?;
        write!(f, "\n")
    }
}
