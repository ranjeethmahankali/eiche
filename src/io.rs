use crate::{
    Error,
    tree::{
        BinaryOp::*,
        Node::{self, *},
        TernaryOp::*,
        Tree,
        UnaryOp::*,
        Value::{self, *},
    },
    walk::{DepthWalker, NodeOrdering},
};

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scalar(val) => write!(f, "{val}"),
            Bool(val) => write!(f, "{val}"),
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
            Constant(value) => write!(f, "Constant({value})"),
            Symbol(label) => write!(f, "Symbol({label})"),
            Unary(op, input) => write!(f, "{op:?}({input})"),
            Binary(op, lhs, rhs) => write!(f, "{op:?}({lhs}, {rhs})"),
            Ternary(op, a, b, c) => write!(f, "{op:?}({a}, {b}, {c})"),
        }
    }
}

mod alias {
    // Unary ops.
    pub const NEGATE: &str = "neg";
    pub const SQRT: &str = "sqrt";
    pub const ABS: &str = "abs";
    pub const SIN: &str = "sin";
    pub const COS: &str = "cos";
    pub const TAN: &str = "tan";
    pub const LOG: &str = "log";
    pub const EXP: &str = "exp";
    pub const FLOOR: &str = "floor";
    pub const NOT: &str = "not";
    // Binary ops.
    pub const ADD: &str = "add";
    pub const SUBTRACT: &str = "sub";
    pub const MULTIPLY: &str = "mul";
    pub const DIVIDE: &str = "div";
    pub const POW: &str = "pow";
    pub const MIN: &str = "min";
    pub const MAX: &str = "max";
    pub const REMAINDER: &str = "";
    pub const LESS: &str = "lt";
    pub const LESSOREQUAL: &str = "le";
    pub const EQUAL: &str = "eq";
    pub const NOTEQUAL: &str = "neq";
    pub const GREATER: &str = "gt";
    pub const GREATEROREQUAL: &str = "ge";
    pub const AND: &str = "and";
    pub const OR: &str = "or";
    // Ternary ops.
    pub const CHOOSE: &str = "choose";
}

impl Tree {
    pub fn write_to<W: std::io::Write>(&self, mut w: W) -> Result<(), std::io::Error> {
        fn write_node<W: std::io::Write>(
            node: &Node,
            index: usize,
            w: &mut W,
        ) -> Result<(), std::io::Error> {
            match node {
                Constant(value) => match value {
                    Bool(flag) => {
                        writeln!(w, "bool {} # {}", if *flag { "t" } else { "f" }, index)?
                    }
                    Scalar(value) => {
                        let bits = value.to_bits();
                        writeln!(w, "float {bits:x} # {index}: {value}")?
                    }
                },
                Symbol(label) => writeln!(w, "var {label}")?,
                Unary(op, input) => {
                    let opstr = match op {
                        Negate => alias::NEGATE,
                        Sqrt => alias::SQRT,
                        Abs => alias::ABS,
                        Sin => alias::SIN,
                        Cos => alias::COS,
                        Tan => alias::TAN,
                        Log => alias::LOG,
                        Exp => alias::EXP,
                        Floor => alias::FLOOR,
                        Not => alias::NOT,
                    };
                    writeln!(w, "{} {} # {}", opstr, input, index)?
                }
                Binary(op, lhs, rhs) => {
                    let opstr = match op {
                        Add => alias::ADD,
                        Subtract => alias::SUBTRACT,
                        Multiply => alias::MULTIPLY,
                        Divide => alias::DIVIDE,
                        Pow => alias::POW,
                        Min => alias::MIN,
                        Max => alias::MAX,
                        Remainder => alias::REMAINDER,
                        Less => alias::LESS,
                        LessOrEqual => alias::LESSOREQUAL,
                        Equal => alias::EQUAL,
                        NotEqual => alias::NOTEQUAL,
                        Greater => alias::GREATER,
                        GreaterOrEqual => alias::GREATEROREQUAL,
                        And => alias::AND,
                        Or => alias::OR,
                    };
                    writeln!(w, "{} {} {} # {}", opstr, lhs, rhs, index)?
                }
                Ternary(op, a, b, c) => {
                    let opstr = match op {
                        Choose => alias::CHOOSE,
                    };
                    writeln!(w, "{} {} {} {} # {}", opstr, a, b, c, index)?
                }
            }
            Ok(())
        }
        let (rows, cols) = self.dims();
        writeln!(w, "{rows} {cols} # output dims")?;
        // First write non-root nodes.
        let count = self.len() - self.num_roots();
        for (i, node) in self.nodes().iter().enumerate().take(count) {
            write_node(node, i, &mut w)?
        }
        writeln!(w, "\n# outputs\n")?;
        for (i, node) in self.nodes().iter().enumerate().skip(count) {
            write_node(node, i, &mut w)?
        }
        Ok(())
    }

    pub fn read_from<S: std::io::Read>(mut src: S) -> Result<Tree, Error> {
        fn read_int<'a>(iter: &mut impl Iterator<Item = &'a str>) -> Result<usize, Error> {
            iter.next()
                .ok_or_else(|| Error::IOError("Unable to read integer string".into()))
                .and_then(|word| {
                    word.parse::<usize>()
                        .map_err(|e| Error::IOError(e.to_string()))
                })
        }
        let src = {
            let mut out = String::new();
            src.read_to_string(&mut out)
                .map_err(|e| Error::IOError(e.to_string()))?;
            out
        };
        let src = src.as_str().trim();
        let mut lines = src.lines();
        let dims = {
            let (rows, cols) = lines
                .next()
                .map(|s| s.split_once(" "))
                .flatten()
                .ok_or_else(|| {
                    Error::IOError("Unable to read the output dimensions of tree.".into())
                })?;
            (
                rows.parse::<usize>()
                    .map_err(|e| Error::IOError(e.to_string()))?,
                cols.parse::<usize>()
                    .map_err(|e| Error::IOError(e.to_string()))?,
            )
        };
        let mut nodes = Vec::<Node>::new();
        for line in src.lines() {
            let line = line
                .trim()
                .split_once('#')
                .map(|(before, _after)| before)
                .unwrap_or(line)
                .trim();
            if line.is_empty() || line.starts_with("#") {
                continue;
            }
            let mut words = line.split_ascii_whitespace();
            let opstr = words
                .next()
                .ok_or_else(|| Error::IOError("Unable to read the op".into()))?;
            let node = match opstr {
                alias::NEGATE => Unary(Negate, read_int(&mut words)?),
                alias::SQRT => Unary(Sqrt, read_int(&mut words)?),
                alias::ABS => Unary(Abs, read_int(&mut words)?),
                alias::SIN => Unary(Sin, read_int(&mut words)?),
                alias::COS => Unary(Cos, read_int(&mut words)?),
                alias::TAN => Unary(Tan, read_int(&mut words)?),
                alias::LOG => Unary(Log, read_int(&mut words)?),
                alias::EXP => Unary(Exp, read_int(&mut words)?),
                alias::FLOOR => Unary(Floor, read_int(&mut words)?),
                alias::NOT => Unary(Not, read_int(&mut words)?),
                // Binary ops.
                alias::ADD => Binary(Add, read_int(&mut words)?, read_int(&mut words)?),
                alias::SUBTRACT => Binary(Subtract, read_int(&mut words)?, read_int(&mut words)?),
                alias::MULTIPLY => Binary(Multiply, read_int(&mut words)?, read_int(&mut words)?),
                alias::DIVIDE => Binary(Divide, read_int(&mut words)?, read_int(&mut words)?),
                alias::POW => Binary(Pow, read_int(&mut words)?, read_int(&mut words)?),
                alias::MIN => Binary(Min, read_int(&mut words)?, read_int(&mut words)?),
                alias::MAX => Binary(Max, read_int(&mut words)?, read_int(&mut words)?),
                alias::REMAINDER => Binary(Remainder, read_int(&mut words)?, read_int(&mut words)?),
                alias::LESS => Binary(Less, read_int(&mut words)?, read_int(&mut words)?),
                alias::LESSOREQUAL => {
                    Binary(LessOrEqual, read_int(&mut words)?, read_int(&mut words)?)
                }
                alias::EQUAL => Binary(Equal, read_int(&mut words)?, read_int(&mut words)?),
                alias::NOTEQUAL => Binary(NotEqual, read_int(&mut words)?, read_int(&mut words)?),
                alias::GREATER => Binary(Greater, read_int(&mut words)?, read_int(&mut words)?),
                alias::GREATEROREQUAL => {
                    Binary(GreaterOrEqual, read_int(&mut words)?, read_int(&mut words)?)
                }
                alias::AND => Binary(And, read_int(&mut words)?, read_int(&mut words)?),
                alias::OR => Binary(Or, read_int(&mut words)?, read_int(&mut words)?),
                // Ternary ops.
                alias::CHOOSE => Ternary(
                    Choose,
                    read_int(&mut words)?,
                    read_int(&mut words)?,
                    read_int(&mut words)?,
                ),
                _ => return Err(Error::IOError("Unrecognized op".into())),
            };
            nodes.push(node);
        }
        Tree::from_nodes(nodes, dims)
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
                  (- (sqrt (+ (+ (pow (- 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 2.75)
                  (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 4.))
             (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (+ 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 5.25))
        )
        .unwrap();
        assert_eq!(
            format!("{tree}").trim(),
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
            format!("{tree}").trim(),
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
                          (+ (pow 'x 2.) (pow 'y 2.))
                          (* (pow 'x 2.) (pow 'y 2.)))
        .unwrap();
        assert_eq!(
            format!("{v2}").trim(),
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
