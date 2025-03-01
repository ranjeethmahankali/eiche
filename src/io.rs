use crate::{
    tree::{Node, Node::*, Tree, Value, Value::*},
    walk::{DepthWalker, NodeOrdering},
};

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scalar(val) => write!(f, "{}", val),
            Bool(val) => write!(f, "{}", val),
        }
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
                let mut walker = DepthWalker::default();
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
                                    Newline => return Err(std::fmt::Error),
                                }
                            }
                        }
                    }
                }
            }
            tokens
        };
        // Write all the tokens out.
        writeln!(f)?;
        for token in tokens.iter() {
            match token {
                Branch => write!(f, " ├── ")?,
                Pass => write!(f, " │   ")?,
                Turn => write!(f, " └── ")?,
                Gap => write!(f, "     ")?,
                Newline => writeln!(f)?,
                NodeIndex(index) => write!(f, "[{}] {}", *index, &self.node(*index))?,
            };
        }
        writeln!(f)
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Constant(value) => write!(f, "Constant({})", value),
            Symbol(label) => write!(f, "Symbol({})", label),
            Unary(op, input) => write!(f, "{:?}({})", op, input),
            Binary(op, lhs, rhs) => write!(f, "{:?}({}, {})", op, lhs, rhs),
            Ternary(op, a, b, c) => write!(f, "{:?}({}, {}, {})", op, a, b, c),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{dedup::Deduplicater, deftree, prune::Pruner};

    #[test]
    fn t_tree_string_formatting() {
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        let tree = deftree!(
            (max (min
                  (- (sqrt (+ (+ (pow (- x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 2.75)
                  (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 4.))
             (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (+ y 3.) 2.)) (pow (- z 4.) 2.))) 5.25))
        )
        .unwrap();
        assert_eq!(
            format!("{}", tree).trim(),
            "
[61] Max(40, 60)
 ├── [40] Min(19, 39)
 │    ├── [19] Subtract(17, 18)
 │    │    ├── [17] Sqrt(16)
 │    │    │    └── [16] Add(10, 15)
 │    │    │         ├── [10] Add(4, 9)
 │    │    │         │    ├── [4] Pow(2, 3)
 │    │    │         │    │    ├── [2] Subtract(0, 1)
 │    │    │         │    │    │    ├── [0] Symbol(x)
 │    │    │         │    │    │    └── [1] Constant(2)
 │    │    │         │    │    └── [3] Constant(2)
 │    │    │         │    └── [9] Pow(7, 8)
 │    │    │         │         ├── [7] Subtract(5, 6)
 │    │    │         │         │    ├── [5] Symbol(y)
 │    │    │         │         │    └── [6] Constant(3)
 │    │    │         │         └── [8] Constant(2)
 │    │    │         └── [15] Pow(13, 14)
 │    │    │              ├── [13] Subtract(11, 12)
 │    │    │              │    ├── [11] Symbol(z)
 │    │    │              │    └── [12] Constant(4)
 │    │    │              └── [14] Constant(2)
 │    │    └── [18] Constant(2.75)
 │    └── [39] Subtract(37, 38)
 │         ├── [37] Sqrt(36)
 │         │    └── [36] Add(30, 35)
 │         │         ├── [30] Add(24, 29)
 │         │         │    ├── [24] Pow(22, 23)
 │         │         │    │    ├── [22] Add(20, 21)
 │         │         │    │    │    ├── [20] Symbol(x)
 │         │         │    │    │    └── [21] Constant(2)
 │         │         │    │    └── [23] Constant(2)
 │         │         │    └── [29] Pow(27, 28)
 │         │         │         ├── [27] Subtract(25, 26)
 │         │         │         │    ├── [25] Symbol(y)
 │         │         │         │    └── [26] Constant(3)
 │         │         │         └── [28] Constant(2)
 │         │         └── [35] Pow(33, 34)
 │         │              ├── [33] Subtract(31, 32)
 │         │              │    ├── [31] Symbol(z)
 │         │              │    └── [32] Constant(4)
 │         │              └── [34] Constant(2)
 │         └── [38] Constant(4)
 └── [60] Subtract(58, 59)
      ├── [58] Sqrt(57)
      │    └── [57] Add(51, 56)
      │         ├── [51] Add(45, 50)
      │         │    ├── [45] Pow(43, 44)
      │         │    │    ├── [43] Add(41, 42)
      │         │    │    │    ├── [41] Symbol(x)
      │         │    │    │    └── [42] Constant(2)
      │         │    │    └── [44] Constant(2)
      │         │    └── [50] Pow(48, 49)
      │         │         ├── [48] Add(46, 47)
      │         │         │    ├── [46] Symbol(y)
      │         │         │    └── [47] Constant(3)
      │         │         └── [49] Constant(2)
      │         └── [56] Pow(54, 55)
      │              ├── [54] Subtract(52, 53)
      │              │    ├── [52] Symbol(z)
      │              │    └── [53] Constant(4)
      │              └── [55] Constant(2)
      └── [59] Constant(5.25)"
                .trim()
        );
        let tree = tree
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner)
            .unwrap();
        assert_eq!(
            format!("{}", tree).trim(),
            "
[31] Max(23, 30)
 ├── [23] Min(16, 22)
 │    ├── [16] Subtract(14, 15)
 │    │    ├── [14] Sqrt(13)
 │    │    │    └── [13] Add(8, 12)
 │    │    │         ├── [8] Add(3, 7)
 │    │    │         │    ├── [3] Pow(2, 1)
 │    │    │         │    │    ├── [2] Subtract(0, 1)
 │    │    │         │    │    │    ├── [0] Symbol(x)
 │    │    │         │    │    │    └── [1] Constant(2)
 │    │    │         │    │    └── [1] Constant(2)
 │    │    │         │    └── [7] Pow(6, 1)
 │    │    │         │         ├── [6] Subtract(4, 5)
 │    │    │         │         │    ├── [4] Symbol(y)
 │    │    │         │         │    └── [5] Constant(3)
 │    │    │         │         └── [1] Constant(2)
 │    │    │         └── [12] Pow(11, 1)
 │    │    │              ├── [11] Subtract(9, 10)
 │    │    │              │    ├── [9] Symbol(z)
 │    │    │              │    └── [10] Constant(4)
 │    │    │              └── [1] Constant(2)
 │    │    └── [15] Constant(2.75)
 │    └── [22] Subtract(21, 10)
 │         ├── [21] Sqrt(20)
 │         │    └── [20] Add(19, 12)
 │         │         ├── [19] Add(18, 7)
 │         │         │    ├── [18] Pow(17, 1)
 │         │         │    │    ├── [17] Add(0, 1)
 │         │         │    │    │    ├── [0] Symbol(x)
 │         │         │    │    │    └── [1] Constant(2)
 │         │         │    │    └── [1] Constant(2)
 │         │         │    └── [7] Pow(6, 1)
 │         │         │         ├── [6] Subtract(4, 5)
 │         │         │         │    ├── [4] Symbol(y)
 │         │         │         │    └── [5] Constant(3)
 │         │         │         └── [1] Constant(2)
 │         │         └── [12] Pow(11, 1)
 │         │              ├── [11] Subtract(9, 10)
 │         │              │    ├── [9] Symbol(z)
 │         │              │    └── [10] Constant(4)
 │         │              └── [1] Constant(2)
 │         └── [10] Constant(4)
 └── [30] Subtract(28, 29)
      ├── [28] Sqrt(27)
      │    └── [27] Add(26, 12)
      │         ├── [26] Add(18, 25)
      │         │    ├── [18] Pow(17, 1)
      │         │    │    ├── [17] Add(0, 1)
      │         │    │    │    ├── [0] Symbol(x)
      │         │    │    │    └── [1] Constant(2)
      │         │    │    └── [1] Constant(2)
      │         │    └── [25] Pow(24, 1)
      │         │         ├── [24] Add(4, 5)
      │         │         │    ├── [4] Symbol(y)
      │         │         │    └── [5] Constant(3)
      │         │         └── [1] Constant(2)
      │         └── [12] Pow(11, 1)
      │              ├── [11] Subtract(9, 10)
      │              │    ├── [9] Symbol(z)
      │              │    └── [10] Constant(4)
      │              └── [1] Constant(2)
      └── [29] Constant(5.25)"
                .trim()
        );
    }

    #[test]
    fn t_concat_string_formatting() {
        let v2 = deftree!(concat
                          (+ (pow x 2.) (pow y 2.))
                          (* (pow x 2.) (pow y 2.)))
        .unwrap();
        assert_eq!(
            format!("{}", v2).trim(),
            "
[12] Add(2, 5)
 ├── [2] Pow(0, 1)
 │    ├── [0] Symbol(x)
 │    └── [1] Constant(2)
 └── [5] Pow(3, 4)
      ├── [3] Symbol(y)
      └── [4] Constant(2)
[13] Multiply(8, 11)
 ├── [8] Pow(6, 7)
 │    ├── [6] Symbol(x)
 │    └── [7] Constant(2)
 └── [11] Pow(9, 10)
      ├── [9] Symbol(y)
      └── [10] Constant(2)
"
            .trim()
        );
    }
}
