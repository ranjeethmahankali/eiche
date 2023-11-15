use crate::tree::{
    BinaryOp::*,
    Node::{self, *},
    Tree,
    UnaryOp::*,
};
use lazy_static::lazy_static;
use regex::Regex;
use std::str::CharIndices;

#[derive(Debug, Copy, Clone)]
enum Token<'a> {
    Open,
    Atom(&'a str),
    Close,
}
use Token::*;

struct Tokenizer<'a> {
    lisp: &'a str,
    last: usize,
    curr: usize,
    iter: CharIndices<'a>,
    next: Option<Token<'a>>,
}

impl<'a> Tokenizer<'a> {
    fn from(lisp: &'a str) -> Tokenizer<'a> {
        Tokenizer {
            lisp,
            last: 0,
            curr: 0,
            iter: lisp.char_indices(),
            next: None,
        }
    }
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(token) = self.next {
            self.next = None;
            return Some(token);
        }
        while let Some((i, c)) = self.iter.next() {
            self.curr = i + 1;
            match c {
                '(' => {
                    if i > self.last {
                        self.next = Some(Open);
                        let token = Some(Atom(&self.lisp[self.last..i]));
                        self.last = i + 1;
                        return token;
                    } else {
                        self.last = i + 1;
                        return Some(Open);
                    }
                }
                ')' => {
                    if i > self.last {
                        self.next = Some(Close);
                        let token = Some(Atom(&self.lisp[self.last..i]));
                        self.last = i + 1;
                        return token;
                    } else {
                        self.last = i + 1;
                        return Some(Close);
                    }
                }
                _ => {
                    if c.is_whitespace() {
                        if i > self.last {
                            let token = Some(Atom(&self.lisp[self.last..i]));
                            self.last = i + 1;
                            return token;
                        } else {
                            self.last = i + 1;
                            continue;
                        }
                    } else {
                        continue;
                    }
                }
            };
        }
        if self.curr > self.last {
            let token = Some(Atom(&self.lisp[self.last..]));
            self.last = self.curr + 1;
            return token;
        }
        return None;
    }
}

fn count_nodes(lisp: &str) -> (usize, usize) {
    let mut nodecount: usize = 0;
    let mut maxdepth: usize = 0;
    let mut depth: usize = 0;
    let mut open: bool = false;
    for token in Tokenizer::from(&lisp) {
        match token {
            Open => {
                depth += 1;
                maxdepth = usize::max(depth, maxdepth);
                open = true;
            }
            Atom(_) => {
                if !open {
                    nodecount += 1;
                } else {
                    open = false;
                }
            }
            Close => {
                depth -= 1;
                nodecount += 1;
                open = false;
            }
        }
    }
    (nodecount, maxdepth)
}

#[derive(Debug)]
enum Parsed<'a> {
    Done(usize),
    Todo(&'a str),
}

use Parsed::*;

/// Errors that can occur when parsing a lisp expression into a
/// `Tree`.
#[derive(Debug)]
pub enum LispParseError {
    /// Error when parsing a floating point number.
    Float,
    /// Error when parsing a symbol's character.
    Symbol,
    /// Error when parsing an expression.
    MalformedExpression,
    /// Unrecognized token.
    InvalidToken(String),
    /// Less than expected tokens for the given operation / context.
    TooFewTokens,
    /// More than expected tokens for the given operation / context.
    TooManyTokens,
    /// Malformed parentheses.
    MalformedParentheses,
    /// All other problems.
    Unknown,
}

fn push_node(node: Node, nodes: &mut Vec<Node>) -> usize {
    let i = nodes.len();
    nodes.push(node);
    return i;
}

fn parse_atom<'a>(atom: &'a str, nodes: &mut Vec<Node>) -> Result<Parsed<'a>, LispParseError> {
    lazy_static! {
        // These are run in unit tests, so it is safe to unwrap these.
        static ref FLT_REGEX: Regex = Regex::new("^\\d+\\.*\\d*$").unwrap();
        static ref SYM_REGEX: Regex = Regex::new("^[a-zA-Z]$").unwrap();
    };
    if FLT_REGEX.is_match(atom) {
        return Ok(Done(push_node(
            Constant(atom.parse::<f64>().map_err(|_| LispParseError::Float)?),
            nodes,
        )));
    }
    if SYM_REGEX.is_match(atom) {
        return Ok(Done(push_node(
            Symbol(atom.chars().nth(0).ok_or(LispParseError::Symbol)?),
            nodes,
        )));
    }
    return Ok(Todo(atom));
}

fn parse_unary(op: &str, input: usize, nodes: &mut Vec<Node>) -> Result<usize, LispParseError> {
    Ok(push_node(
        Unary(
            match op {
                "-" => Negate,
                "sqrt" => Sqrt,
                "abs" => Abs,
                "sin" => Sin,
                "cos" => Cos,
                "tan" => Tan,
                "log" => Log,
                "exp" => Exp,
                _ => return Err(LispParseError::InvalidToken(op.to_string())),
            },
            input,
        ),
        nodes,
    ))
}

fn parse_binary(
    op: &str,
    lhs: usize,
    rhs: usize,
    nodes: &mut Vec<Node>,
) -> Result<usize, LispParseError> {
    Ok(push_node(
        Binary(
            match op {
                "+" => Add,
                "-" => Subtract,
                "*" => Multiply,
                "/" => Divide,
                "pow" => Pow,
                "min" => Min,
                "max" => Max,
                _ => return Err(LispParseError::InvalidToken(op.to_string())),
            },
            lhs,
            rhs,
        ),
        nodes,
    ))
}

fn parse_expression<'a>(
    expr: &[Parsed<'a>],
    nodes: &mut Vec<Node>,
) -> Result<usize, LispParseError> {
    match expr.len() {
        0 => Err(LispParseError::TooFewTokens),
        1 => match &expr[0] {
            Done(i) => Ok(*i),
            // Expression of length cannot be a todo item.
            Todo(token) => Err(LispParseError::InvalidToken(token.to_string())),
        },
        2 => match (&expr[0], &expr[1]) {
            (Todo(op), Done(input)) => Ok(parse_unary(op, *input, nodes)?),
            // Anything that's not a valid unary op.
            _ => Err(LispParseError::MalformedExpression),
        },
        3 => match (&expr[0], &expr[1], &expr[2]) {
            (Todo(op), Done(lhs), Done(rhs)) => Ok(parse_binary(op, *lhs, *rhs, nodes)?),
            // Anything that's not a valid binary op.
            _ => Err(LispParseError::MalformedExpression),
        },
        _ => Err(LispParseError::TooManyTokens),
    }
}

fn parse_nodes(lisp: &str) -> Result<Vec<Node>, LispParseError> {
    // First pass to collect statistics.
    let (nodecount, maxdepth) = count_nodes(&lisp);
    // Allocate memory according to collected statistics.
    let mut nodes: Vec<Node> = Vec::with_capacity(nodecount);
    let mut parens: Vec<usize> = Vec::with_capacity(maxdepth);
    let mut stack: Vec<Parsed> = Vec::with_capacity(maxdepth + 4);
    for token in Tokenizer::from(&lisp) {
        match token {
            Open => parens.push(stack.len()),
            Atom(token) => stack.push(parse_atom(token, &mut nodes)?),
            Close => {
                let last = parens.pop().ok_or(LispParseError::MalformedParentheses)?;
                let parsed = parse_expression(&stack[last..], &mut nodes)?;
                stack.truncate(last);
                stack.push(Done(parsed));
            }
        }
    }
    if stack.len() == 1 {
        if let Done(index) = stack.remove(0) {
            if index == nodes.len() - 1 {
                return Ok(nodes);
            }
        }
    }
    return Err(LispParseError::Unknown);
}

/// Parse the `lisp` expression into a `Tree`. If the parsing fails, an
/// appropriate `LispParseError` is returned.
pub fn parse_tree(lisp: &str) -> Result<Tree, LispParseError> {
    Ok(Tree::from_nodes(parse_nodes(&lisp)?).map_err(|_| LispParseError::Unknown)?)
}

/// Convert the list of nodes to a lisp string, by recursively
/// traversing the nodes starting at `root`.
fn to_lisp(root: &Node, nodes: &Vec<Node>) -> String {
    match root {
        Constant(val) => val.to_string(),
        Symbol(label) => label.to_string(),
        Unary(op, input) => format!(
            "({} {})",
            {
                match op {
                    Negate => "-",
                    Sqrt => "sqrt",
                    Abs => "abs",
                    Sin => "sin",
                    Cos => "cos",
                    Tan => "tan",
                    Log => "log",
                    Exp => "exp",
                }
            },
            to_lisp(&nodes[*input], nodes)
        ),
        Binary(op, lhs, rhs) => format!(
            "({} {} {})",
            {
                match op {
                    Add => "+",
                    Subtract => "-",
                    Multiply => "*",
                    Divide => "/",
                    Pow => "pow",
                    Min => "min",
                    Max => "max",
                }
            },
            to_lisp(&nodes[*lhs], nodes),
            to_lisp(&nodes[*rhs], nodes)
        ),
    }
}

impl Tree {
    /// Convert the tree to a lisp expression. If there is something
    /// wrong with this tree, and appropriate `TreeError` is returned.
    pub fn to_lisp(&self) -> String {
        to_lisp(self.root(), self.nodes())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{deftree, parsetree};

    #[test]
    fn t_node_counting() {
        let (nodes, depth) = count_nodes(stringify!(
            (- (sqrt (+ (pow x 2.) (pow y 2.))) 5.0)
        ));
        assert_eq!(nodes, 10);
        assert_eq!(depth, 4);
    }

    #[test]
    fn t_single_token() {
        // Constant.
        let tree = parse_tree("5.55").unwrap();
        assert!(matches!(tree.root(), Constant(val) if *val == 5.55));
        // Constant with spaces.
        let tree = parse_tree(" 5.55   ").unwrap();
        assert!(matches!(tree.root(), Constant(val) if *val == 5.55));
        // Symbol.
        let tree = parse_tree("x").unwrap();
        assert!(matches!(tree.root(), Symbol(label) if *label == 'x'));
        // Symbol with spaces.
        let tree = parse_tree(" x     ").unwrap();
        assert!(matches!(tree.root(), Symbol(label) if *label == 'x'));
    }

    #[test]
    fn t_tree_parsing() {
        let tree = parsetree!(
            (- (sqrt (+ (pow x 2.) (pow y 2.))) 6.0)
        )
        .unwrap();
        assert_eq!(tree.len(), 10);
        assert_eq!(
            format!("{}", tree).trim(),
            "
[9] Subtract(7, 8)
 ├── [7] Sqrt(6)
 │    └── [6] Add(2, 5)
 │         ├── [2] Pow(0, 1)
 │         │    ├── [0] Symbol(x)
 │         │    └── [1] Constant(2)
 │         └── [5] Pow(3, 4)
 │              ├── [3] Symbol(y)
 │              └── [4] Constant(2)
 └── [8] Constant(6)
"
            .trim()
        );
        // Slightly larger tree written over mutliple lines.
        let tree = parsetree!(
            (min
             (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
             (- (sqrt (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.))) 5.234)
            )
        )
        .unwrap();
        assert_eq!(tree.len(), 25);
        assert_eq!(
            format!("{}", tree).trim(),
            "
[24] Min(9, 23)
 ├── [9] Subtract(7, 8)
 │    ├── [7] Sqrt(6)
 │    │    └── [6] Add(2, 5)
 │    │         ├── [2] Pow(0, 1)
 │    │         │    ├── [0] Symbol(x)
 │    │         │    └── [1] Constant(2)
 │    │         └── [5] Pow(3, 4)
 │    │              ├── [3] Symbol(y)
 │    │              └── [4] Constant(2)
 │    └── [8] Constant(4.24)
 └── [23] Subtract(21, 22)
      ├── [21] Sqrt(20)
      │    └── [20] Add(14, 19)
      │         ├── [14] Pow(12, 13)
      │         │    ├── [12] Subtract(10, 11)
      │         │    │    ├── [10] Symbol(x)
      │         │    │    └── [11] Constant(2.5)
      │         │    └── [13] Constant(2)
      │         └── [19] Pow(17, 18)
      │              ├── [17] Subtract(15, 16)
      │              │    ├── [15] Symbol(y)
      │              │    └── [16] Constant(2.5)
      │              └── [18] Constant(2)
      └── [22] Constant(5.234)"
                .trim()
        );
    }

    #[test]
    fn t_parse_tree_with_comments() {
        let tree = parsetree!(
            (min /*IMPORTANT: Do not remove this comment.*/
             (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24) // IMPORTANT: Do not remove this comment.
             // IMPORTANT: Do not remove this comment.
             (- (sqrt (+ (pow (- x 2.5) 2.) /*IMPORTANT: Do not remove this comment.*/ (pow (- y 2.5) 2.))) 5.234)
            )
        )
        .unwrap();
        assert_eq!(tree.len(), 25);
    }

    #[test]
    fn t_parse_large_tree() {
        let tree = parsetree!(
            (min
             (- (log (+
                      (min
                       (+ (sqrt (+ (pow (- x 2.95) 2.) (pow (- y 2.05) 2.))) 3.67)
                       (max
                        (- (sqrt (+ (pow (- x 3.5) 2.) (pow (- y 3.5) 2.))) 2.234)
                        (max
                         (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
                         (- (sqrt (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.))) 5.243))))
                      (exp (pow (min
                                 (- (sqrt (+ (pow (- x 2.95) 2.) (pow (- y 2.05) 2.))) 3.67)
                                 (max
                                  (- (sqrt (+ (pow (- x 3.5) 2.) (pow (- y 3.5) 2.))) 2.234)
                                  (max
                                   (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
                                   (- (sqrt (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.))) 5.243))))
                            2.456))))
              (min
               (/ (+ (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a))
               (/ (- (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a))))
             (+ (log (+
                      (max
                       (- (sqrt (+ (pow (- x 2.95) 2.) (pow (- y 2.05) 2.))) 3.67)
                       (min
                        (- (sqrt (+ (pow (- x 3.5) 2.) (pow (- y 3.5) 2.))) 2.234)
                        (min
                         (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
                         (- (sqrt (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.))) 5.243))))
                      (exp (pow (max
                                 (- (sqrt (+ (pow (- x 2.95) 2.) (pow (- y 2.05) 2.))) 3.67)
                                 (min
                                  (- (sqrt (+ (pow (- x 3.5) 2.) (pow (- y 3.5) 2.))) 2.234)
                                  (min
                                   (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
                                   (- (sqrt (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.))) 5.243))))
                            2.456))))
              (max
               (/ (+ (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a))
               (/ (- (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a)))))
        )
        .unwrap();
        assert_eq!(
            format!("{}", tree).trim(),
            "
[302] Min(150, 301)
 ├── [150] Subtract(114, 149)
 │    ├── [114] Log(113)
 │    │    └── [113] Add(54, 112)
 │    │         ├── [54] Min(13, 53)
 │    │         │    ├── [13] Add(11, 12)
 │    │         │    │    ├── [11] Sqrt(10)
 │    │         │    │    │    └── [10] Add(4, 9)
 │    │         │    │    │         ├── [4] Pow(2, 3)
 │    │         │    │    │         │    ├── [2] Subtract(0, 1)
 │    │         │    │    │         │    │    ├── [0] Symbol(x)
 │    │         │    │    │         │    │    └── [1] Constant(2.95)
 │    │         │    │    │         │    └── [3] Constant(2)
 │    │         │    │    │         └── [9] Pow(7, 8)
 │    │         │    │    │              ├── [7] Subtract(5, 6)
 │    │         │    │    │              │    ├── [5] Symbol(y)
 │    │         │    │    │              │    └── [6] Constant(2.05)
 │    │         │    │    │              └── [8] Constant(2)
 │    │         │    │    └── [12] Constant(3.67)
 │    │         │    └── [53] Max(27, 52)
 │    │         │         ├── [27] Subtract(25, 26)
 │    │         │         │    ├── [25] Sqrt(24)
 │    │         │         │    │    └── [24] Add(18, 23)
 │    │         │         │    │         ├── [18] Pow(16, 17)
 │    │         │         │    │         │    ├── [16] Subtract(14, 15)
 │    │         │         │    │         │    │    ├── [14] Symbol(x)
 │    │         │         │    │         │    │    └── [15] Constant(3.5)
 │    │         │         │    │         │    └── [17] Constant(2)
 │    │         │         │    │         └── [23] Pow(21, 22)
 │    │         │         │    │              ├── [21] Subtract(19, 20)
 │    │         │         │    │              │    ├── [19] Symbol(y)
 │    │         │         │    │              │    └── [20] Constant(3.5)
 │    │         │         │    │              └── [22] Constant(2)
 │    │         │         │    └── [26] Constant(2.234)
 │    │         │         └── [52] Max(37, 51)
 │    │         │              ├── [37] Subtract(35, 36)
 │    │         │              │    ├── [35] Sqrt(34)
 │    │         │              │    │    └── [34] Add(30, 33)
 │    │         │              │    │         ├── [30] Pow(28, 29)
 │    │         │              │    │         │    ├── [28] Symbol(x)
 │    │         │              │    │         │    └── [29] Constant(2)
 │    │         │              │    │         └── [33] Pow(31, 32)
 │    │         │              │    │              ├── [31] Symbol(y)
 │    │         │              │    │              └── [32] Constant(2)
 │    │         │              │    └── [36] Constant(4.24)
 │    │         │              └── [51] Subtract(49, 50)
 │    │         │                   ├── [49] Sqrt(48)
 │    │         │                   │    └── [48] Add(42, 47)
 │    │         │                   │         ├── [42] Pow(40, 41)
 │    │         │                   │         │    ├── [40] Subtract(38, 39)
 │    │         │                   │         │    │    ├── [38] Symbol(x)
 │    │         │                   │         │    │    └── [39] Constant(2.5)
 │    │         │                   │         │    └── [41] Constant(2)
 │    │         │                   │         └── [47] Pow(45, 46)
 │    │         │                   │              ├── [45] Subtract(43, 44)
 │    │         │                   │              │    ├── [43] Symbol(y)
 │    │         │                   │              │    └── [44] Constant(2.5)
 │    │         │                   │              └── [46] Constant(2)
 │    │         │                   └── [50] Constant(5.243)
 │    │         └── [112] Exp(111)
 │    │              └── [111] Pow(109, 110)
 │    │                   ├── [109] Min(68, 108)
 │    │                   │    ├── [68] Subtract(66, 67)
 │    │                   │    │    ├── [66] Sqrt(65)
 │    │                   │    │    │    └── [65] Add(59, 64)
 │    │                   │    │    │         ├── [59] Pow(57, 58)
 │    │                   │    │    │         │    ├── [57] Subtract(55, 56)
 │    │                   │    │    │         │    │    ├── [55] Symbol(x)
 │    │                   │    │    │         │    │    └── [56] Constant(2.95)
 │    │                   │    │    │         │    └── [58] Constant(2)
 │    │                   │    │    │         └── [64] Pow(62, 63)
 │    │                   │    │    │              ├── [62] Subtract(60, 61)
 │    │                   │    │    │              │    ├── [60] Symbol(y)
 │    │                   │    │    │              │    └── [61] Constant(2.05)
 │    │                   │    │    │              └── [63] Constant(2)
 │    │                   │    │    └── [67] Constant(3.67)
 │    │                   │    └── [108] Max(82, 107)
 │    │                   │         ├── [82] Subtract(80, 81)
 │    │                   │         │    ├── [80] Sqrt(79)
 │    │                   │         │    │    └── [79] Add(73, 78)
 │    │                   │         │    │         ├── [73] Pow(71, 72)
 │    │                   │         │    │         │    ├── [71] Subtract(69, 70)
 │    │                   │         │    │         │    │    ├── [69] Symbol(x)
 │    │                   │         │    │         │    │    └── [70] Constant(3.5)
 │    │                   │         │    │         │    └── [72] Constant(2)
 │    │                   │         │    │         └── [78] Pow(76, 77)
 │    │                   │         │    │              ├── [76] Subtract(74, 75)
 │    │                   │         │    │              │    ├── [74] Symbol(y)
 │    │                   │         │    │              │    └── [75] Constant(3.5)
 │    │                   │         │    │              └── [77] Constant(2)
 │    │                   │         │    └── [81] Constant(2.234)
 │    │                   │         └── [107] Max(92, 106)
 │    │                   │              ├── [92] Subtract(90, 91)
 │    │                   │              │    ├── [90] Sqrt(89)
 │    │                   │              │    │    └── [89] Add(85, 88)
 │    │                   │              │    │         ├── [85] Pow(83, 84)
 │    │                   │              │    │         │    ├── [83] Symbol(x)
 │    │                   │              │    │         │    └── [84] Constant(2)
 │    │                   │              │    │         └── [88] Pow(86, 87)
 │    │                   │              │    │              ├── [86] Symbol(y)
 │    │                   │              │    │              └── [87] Constant(2)
 │    │                   │              │    └── [91] Constant(4.24)
 │    │                   │              └── [106] Subtract(104, 105)
 │    │                   │                   ├── [104] Sqrt(103)
 │    │                   │                   │    └── [103] Add(97, 102)
 │    │                   │                   │         ├── [97] Pow(95, 96)
 │    │                   │                   │         │    ├── [95] Subtract(93, 94)
 │    │                   │                   │         │    │    ├── [93] Symbol(x)
 │    │                   │                   │         │    │    └── [94] Constant(2.5)
 │    │                   │                   │         │    └── [96] Constant(2)
 │    │                   │                   │         └── [102] Pow(100, 101)
 │    │                   │                   │              ├── [100] Subtract(98, 99)
 │    │                   │                   │              │    ├── [98] Symbol(y)
 │    │                   │                   │              │    └── [99] Constant(2.5)
 │    │                   │                   │              └── [101] Constant(2)
 │    │                   │                   └── [105] Constant(5.243)
 │    │                   └── [110] Constant(2.456)
 │    └── [149] Min(131, 148)
 │         ├── [131] Divide(127, 130)
 │         │    ├── [127] Add(116, 126)
 │         │    │    ├── [116] Negate(115)
 │         │    │    │    └── [115] Symbol(b)
 │         │    │    └── [126] Sqrt(125)
 │         │    │         └── [125] Subtract(119, 124)
 │         │    │              ├── [119] Pow(117, 118)
 │         │    │              │    ├── [117] Symbol(b)
 │         │    │              │    └── [118] Constant(2)
 │         │    │              └── [124] Multiply(120, 123)
 │         │    │                   ├── [120] Constant(4)
 │         │    │                   └── [123] Multiply(121, 122)
 │         │    │                        ├── [121] Symbol(a)
 │         │    │                        └── [122] Symbol(c)
 │         │    └── [130] Multiply(128, 129)
 │         │         ├── [128] Constant(2)
 │         │         └── [129] Symbol(a)
 │         └── [148] Divide(144, 147)
 │              ├── [144] Subtract(133, 143)
 │              │    ├── [133] Negate(132)
 │              │    │    └── [132] Symbol(b)
 │              │    └── [143] Sqrt(142)
 │              │         └── [142] Subtract(136, 141)
 │              │              ├── [136] Pow(134, 135)
 │              │              │    ├── [134] Symbol(b)
 │              │              │    └── [135] Constant(2)
 │              │              └── [141] Multiply(137, 140)
 │              │                   ├── [137] Constant(4)
 │              │                   └── [140] Multiply(138, 139)
 │              │                        ├── [138] Symbol(a)
 │              │                        └── [139] Symbol(c)
 │              └── [147] Multiply(145, 146)
 │                   ├── [145] Constant(2)
 │                   └── [146] Symbol(a)
 └── [301] Add(265, 300)
      ├── [265] Log(264)
      │    └── [264] Add(205, 263)
      │         ├── [205] Max(164, 204)
      │         │    ├── [164] Subtract(162, 163)
      │         │    │    ├── [162] Sqrt(161)
      │         │    │    │    └── [161] Add(155, 160)
      │         │    │    │         ├── [155] Pow(153, 154)
      │         │    │    │         │    ├── [153] Subtract(151, 152)
      │         │    │    │         │    │    ├── [151] Symbol(x)
      │         │    │    │         │    │    └── [152] Constant(2.95)
      │         │    │    │         │    └── [154] Constant(2)
      │         │    │    │         └── [160] Pow(158, 159)
      │         │    │    │              ├── [158] Subtract(156, 157)
      │         │    │    │              │    ├── [156] Symbol(y)
      │         │    │    │              │    └── [157] Constant(2.05)
      │         │    │    │              └── [159] Constant(2)
      │         │    │    └── [163] Constant(3.67)
      │         │    └── [204] Min(178, 203)
      │         │         ├── [178] Subtract(176, 177)
      │         │         │    ├── [176] Sqrt(175)
      │         │         │    │    └── [175] Add(169, 174)
      │         │         │    │         ├── [169] Pow(167, 168)
      │         │         │    │         │    ├── [167] Subtract(165, 166)
      │         │         │    │         │    │    ├── [165] Symbol(x)
      │         │         │    │         │    │    └── [166] Constant(3.5)
      │         │         │    │         │    └── [168] Constant(2)
      │         │         │    │         └── [174] Pow(172, 173)
      │         │         │    │              ├── [172] Subtract(170, 171)
      │         │         │    │              │    ├── [170] Symbol(y)
      │         │         │    │              │    └── [171] Constant(3.5)
      │         │         │    │              └── [173] Constant(2)
      │         │         │    └── [177] Constant(2.234)
      │         │         └── [203] Min(188, 202)
      │         │              ├── [188] Subtract(186, 187)
      │         │              │    ├── [186] Sqrt(185)
      │         │              │    │    └── [185] Add(181, 184)
      │         │              │    │         ├── [181] Pow(179, 180)
      │         │              │    │         │    ├── [179] Symbol(x)
      │         │              │    │         │    └── [180] Constant(2)
      │         │              │    │         └── [184] Pow(182, 183)
      │         │              │    │              ├── [182] Symbol(y)
      │         │              │    │              └── [183] Constant(2)
      │         │              │    └── [187] Constant(4.24)
      │         │              └── [202] Subtract(200, 201)
      │         │                   ├── [200] Sqrt(199)
      │         │                   │    └── [199] Add(193, 198)
      │         │                   │         ├── [193] Pow(191, 192)
      │         │                   │         │    ├── [191] Subtract(189, 190)
      │         │                   │         │    │    ├── [189] Symbol(x)
      │         │                   │         │    │    └── [190] Constant(2.5)
      │         │                   │         │    └── [192] Constant(2)
      │         │                   │         └── [198] Pow(196, 197)
      │         │                   │              ├── [196] Subtract(194, 195)
      │         │                   │              │    ├── [194] Symbol(y)
      │         │                   │              │    └── [195] Constant(2.5)
      │         │                   │              └── [197] Constant(2)
      │         │                   └── [201] Constant(5.243)
      │         └── [263] Exp(262)
      │              └── [262] Pow(260, 261)
      │                   ├── [260] Max(219, 259)
      │                   │    ├── [219] Subtract(217, 218)
      │                   │    │    ├── [217] Sqrt(216)
      │                   │    │    │    └── [216] Add(210, 215)
      │                   │    │    │         ├── [210] Pow(208, 209)
      │                   │    │    │         │    ├── [208] Subtract(206, 207)
      │                   │    │    │         │    │    ├── [206] Symbol(x)
      │                   │    │    │         │    │    └── [207] Constant(2.95)
      │                   │    │    │         │    └── [209] Constant(2)
      │                   │    │    │         └── [215] Pow(213, 214)
      │                   │    │    │              ├── [213] Subtract(211, 212)
      │                   │    │    │              │    ├── [211] Symbol(y)
      │                   │    │    │              │    └── [212] Constant(2.05)
      │                   │    │    │              └── [214] Constant(2)
      │                   │    │    └── [218] Constant(3.67)
      │                   │    └── [259] Min(233, 258)
      │                   │         ├── [233] Subtract(231, 232)
      │                   │         │    ├── [231] Sqrt(230)
      │                   │         │    │    └── [230] Add(224, 229)
      │                   │         │    │         ├── [224] Pow(222, 223)
      │                   │         │    │         │    ├── [222] Subtract(220, 221)
      │                   │         │    │         │    │    ├── [220] Symbol(x)
      │                   │         │    │         │    │    └── [221] Constant(3.5)
      │                   │         │    │         │    └── [223] Constant(2)
      │                   │         │    │         └── [229] Pow(227, 228)
      │                   │         │    │              ├── [227] Subtract(225, 226)
      │                   │         │    │              │    ├── [225] Symbol(y)
      │                   │         │    │              │    └── [226] Constant(3.5)
      │                   │         │    │              └── [228] Constant(2)
      │                   │         │    └── [232] Constant(2.234)
      │                   │         └── [258] Min(243, 257)
      │                   │              ├── [243] Subtract(241, 242)
      │                   │              │    ├── [241] Sqrt(240)
      │                   │              │    │    └── [240] Add(236, 239)
      │                   │              │    │         ├── [236] Pow(234, 235)
      │                   │              │    │         │    ├── [234] Symbol(x)
      │                   │              │    │         │    └── [235] Constant(2)
      │                   │              │    │         └── [239] Pow(237, 238)
      │                   │              │    │              ├── [237] Symbol(y)
      │                   │              │    │              └── [238] Constant(2)
      │                   │              │    └── [242] Constant(4.24)
      │                   │              └── [257] Subtract(255, 256)
      │                   │                   ├── [255] Sqrt(254)
      │                   │                   │    └── [254] Add(248, 253)
      │                   │                   │         ├── [248] Pow(246, 247)
      │                   │                   │         │    ├── [246] Subtract(244, 245)
      │                   │                   │         │    │    ├── [244] Symbol(x)
      │                   │                   │         │    │    └── [245] Constant(2.5)
      │                   │                   │         │    └── [247] Constant(2)
      │                   │                   │         └── [253] Pow(251, 252)
      │                   │                   │              ├── [251] Subtract(249, 250)
      │                   │                   │              │    ├── [249] Symbol(y)
      │                   │                   │              │    └── [250] Constant(2.5)
      │                   │                   │              └── [252] Constant(2)
      │                   │                   └── [256] Constant(5.243)
      │                   └── [261] Constant(2.456)
      └── [300] Max(282, 299)
           ├── [282] Divide(278, 281)
           │    ├── [278] Add(267, 277)
           │    │    ├── [267] Negate(266)
           │    │    │    └── [266] Symbol(b)
           │    │    └── [277] Sqrt(276)
           │    │         └── [276] Subtract(270, 275)
           │    │              ├── [270] Pow(268, 269)
           │    │              │    ├── [268] Symbol(b)
           │    │              │    └── [269] Constant(2)
           │    │              └── [275] Multiply(271, 274)
           │    │                   ├── [271] Constant(4)
           │    │                   └── [274] Multiply(272, 273)
           │    │                        ├── [272] Symbol(a)
           │    │                        └── [273] Symbol(c)
           │    └── [281] Multiply(279, 280)
           │         ├── [279] Constant(2)
           │         └── [280] Symbol(a)
           └── [299] Divide(295, 298)
                ├── [295] Subtract(284, 294)
                │    ├── [284] Negate(283)
                │    │    └── [283] Symbol(b)
                │    └── [294] Sqrt(293)
                │         └── [293] Subtract(287, 292)
                │              ├── [287] Pow(285, 286)
                │              │    ├── [285] Symbol(b)
                │              │    └── [286] Constant(2)
                │              └── [292] Multiply(288, 291)
                │                   ├── [288] Constant(4)
                │                   └── [291] Multiply(289, 290)
                │                        ├── [289] Symbol(a)
                │                        └── [290] Symbol(c)
                └── [298] Multiply(296, 297)
                     ├── [296] Constant(2)
                     └── [297] Symbol(a)"
                .trim()
        );
    }

    #[test]
    fn t_tree_to_lisp() {
        let tree = deftree!(- (sqrt (+ (pow (- x 3.) 2.) (pow (- y 2.4) 2.))) 4.);
        assert_eq!(
            tree.to_lisp(),
            "(- (sqrt (+ (pow (- x 3) 2) (pow (- y 2.4) 2))) 4)"
        );
    }
}
