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

use crate::tree::{
    BinaryOp::*,
    Node::{self, *},
    Tree,
    UnaryOp::*,
};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_counting() {
        let (nodes, depth) = count_nodes(stringify!(
            (- (sqrt (+ (pow x 2.) (pow y 2.))) 5.0)
        ));
        assert_eq!(nodes, 10);
        assert_eq!(depth, 4);
    }

    #[test]
    fn single_token() {
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
}
