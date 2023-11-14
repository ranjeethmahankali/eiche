use crate::{
    tree::{Node, Node::*, Tree},
    walk::{DepthWalker, NodeOrdering},
};

impl From<f64> for Tree {
    fn from(value: f64) -> Self {
        return Self::constant(value);
    }
}

impl From<char> for Tree {
    fn from(c: char) -> Self {
        return Self::symbol(c);
    }
}

impl std::fmt::Display for Tree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        enum Token {
            Branch,
            Pass,
            Turn,
            Gap,
            Newline,
            NodeIndex(usize),
        }
        use Token::*;
        // Walk the tree and collect tokens.
        let tokens = {
            // First pass of collecting tokens with no branching.
            let mut tokens = {
                let mut tokens: Vec<Token> = Vec::with_capacity(self.len()); // Likely need more memory.
                let mut walker = DepthWalker::new();
                let mut node_depths: Box<[usize]> = vec![0; self.len()].into_boxed_slice();
                for (index, parent) in walker.walk_tree(self, false, NodeOrdering::Original) {
                    if let Some(pi) = parent {
                        node_depths[index] = node_depths[pi] + 1;
                    }
                    let depth = node_depths[index];
                    if depth > 0 {
                        for _ in 0..(depth - 1) {
                            tokens.push(Gap);
                        }
                        tokens.push(Turn);
                    }
                    tokens.push(NodeIndex(index));
                    tokens.push(Newline);
                }
                tokens
            };
            // Insert branching tokens where necessary.
            let mut line_start: usize = 0;
            for i in 0..tokens.len() {
                match tokens[i] {
                    Branch | Pass | Gap | NodeIndex(_) => {} // Do nothing.
                    Newline => line_start = i,
                    Turn => {
                        let offset = i - line_start;
                        for li in (0..line_start).rev() {
                            if let Newline = tokens[li] {
                                let ti = li + offset;
                                tokens[ti] = match &tokens[ti] {
                                    Branch | Pass | NodeIndex(_) => break,
                                    Turn => Branch,
                                    Gap => Pass,
                                    Newline => panic!("FATAL: Failed to convert tree to a string"),
                                }
                            }
                        }
                    }
                }
            }
            tokens
        };
        // Write all the tokens out.
        write!(f, "\n")?;
        for token in tokens.iter() {
            match token {
                Branch => write!(f, " ├── ")?,
                Pass => write!(f, " │   ")?,
                Turn => write!(f, " └── ")?,
                Gap => write!(f, "     ")?,
                Newline => write!(f, "\n")?,
                NodeIndex(index) => write!(f, "[{}] {}", *index, &self.node(*index))?,
            };
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

impl PartialOrd for Node {
    /// This implementation only accounts for the node, its type and
    /// the data held inside the node. It DOES NOT take into account
    /// the children of the node when comparing two nodes.
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering::*;
        match (self, other) {
            // Constant
            (Constant(a), Constant(b)) => a.partial_cmp(b),
            (Constant(_), Symbol(_)) => Some(Less),
            (Constant(_), Unary(_, _)) => Some(Less),
            (Constant(_), Binary(_, _, _)) => Some(Less),
            // Symbol
            (Symbol(_), Constant(_)) => Some(Greater),
            (Symbol(a), Symbol(b)) => Some(a.cmp(b)),
            (Symbol(_), Unary(_, _)) => Some(Less),
            (Symbol(_), Binary(_, _, _)) => Some(Less),
            // Unary
            (Unary(_, _), Constant(_)) => Some(Greater),
            (Unary(_, _), Symbol(_)) => Some(Greater),
            (Unary(op1, _), Unary(op2, _)) => Some(op1.index().cmp(&op2.index())),
            (Unary(_, _), Binary(_, _, _)) => Some(Less),
            // Binary
            (Binary(_, _, _), Constant(_)) => Some(Greater),
            (Binary(_, _, _), Symbol(_)) => Some(Greater),
            (Binary(_, _, _), Unary(_, _)) => Some(Greater),
            (Binary(op1, _, _), Binary(op2, _, _)) => Some(op1.index().cmp(&op2.index())),
        }
    }
}
