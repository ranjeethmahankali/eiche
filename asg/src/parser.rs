use crate::tree::{
    BinaryOp::*,
    Node::{self, *},
    Tree,
    UnaryOp::*,
};
use lazy_static::lazy_static;
use regex::Regex;
use std::str::CharIndices;

#[macro_export]
macro_rules! deftree {
    ($($exp:tt) *) => {
        parse_lisp(stringify!($($exp) *).to_string())
    };
}

#[derive(Debug, Copy, Clone)]
enum Token<'a> {
    Open,
    Atom(&'a str),
    Close,
}
use Token::*;

struct Tokenizer<'a> {
    lisp: &'a String,
    last: usize,
    iter: CharIndices<'a>,
    next: Option<Token<'a>>,
}

impl<'a> Tokenizer<'a> {
    pub fn from(lisp: &'a String) -> Tokenizer<'a> {
        Tokenizer {
            lisp,
            last: 0,
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
                ' ' if i > self.last => {
                    let token = Some(Atom(&self.lisp[self.last..i]));
                    self.last = i + 1;
                    return token;
                }
                ' ' => {
                    self.last = i + 1;
                    continue;
                }
                _ => continue,
            };
        }
        return None;
    }
}

#[derive(Debug)]
enum Parsed<'a> {
    Done(usize),
    Todo(&'a str),
}

use Parsed::*;

#[derive(Debug)]
pub enum LispParseError {
    Float,
    Symbol,
    Unary,
    Binary,
    InvalidToken(String),
    TooFewTokens,
    TooManyTokens,
    MalformedParentheses,
    Unknown,
}

fn push_node(node: Node, nodes: &mut Vec<Node>) -> usize {
    let i = nodes.len();
    nodes.push(node);
    return i;
}

fn parse_atom<'a>(atom: &'a str, nodes: &mut Vec<Node>) -> Result<Parsed<'a>, LispParseError> {
    lazy_static! {
        static ref FLT_REGEX: Regex =
            Regex::new("^\\d+\\.*\\d*$").expect("Failed to initialize regex for floats.");
        static ref SYM_REGEX: Regex =
            Regex::new("^[a-zA-Z]$").expect("Failed to initialize regex for numbers.");
    };
    if FLT_REGEX.is_match(atom) {
        return Ok(Done(push_node(
            Constant(atom.parse::<f64>().ok().ok_or(LispParseError::Float)?),
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
                _ => return Err(LispParseError::Unary),
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
                _ => return Err(LispParseError::Binary),
            },
            lhs,
            rhs,
        ),
        nodes,
    ))
}

fn parse_expression<'a>(
    expr: &mut Vec<Parsed<'a>>,
    nodes: &mut Vec<Node>,
) -> Result<usize, LispParseError> {
    match expr.len() {
        0 => Err(LispParseError::TooFewTokens),
        1 => match expr.remove(0) {
            Done(i) => Ok(i),
            // Expression of length cannot be a todo item.
            Todo(token) => Err(LispParseError::InvalidToken(token.to_string())),
        },
        2 => match (expr.remove(1), expr.remove(0)) {
            (Done(input), Todo(op)) => Ok(parse_unary(op, input, nodes)?),
            // Anything that's not a valid unary op.
            _ => Err(LispParseError::Unary),
        },
        3 => match (expr.remove(2), expr.remove(1), expr.remove(0)) {
            (Done(rhs), Done(lhs), Todo(op)) => Ok(parse_binary(op, lhs, rhs, nodes)?),
            // Anything that's not a valid binary op.
            _ => Err(LispParseError::Binary),
        },
        _ => Err(LispParseError::TooManyTokens),
    }
}

pub fn parse_lisp(lisp: String) -> Result<Tree, LispParseError> {
    let mut nodes: Vec<Node> = Vec::new();
    let mut parens: Vec<usize> = Vec::new();
    let mut stack: Vec<Parsed> = Vec::new();
    let mut toparse: Vec<Parsed> = Vec::with_capacity(4);
    for token in Tokenizer::from(&lisp) {
        match token {
            Open => parens.push(stack.len()),
            Atom(token) => stack.push(parse_atom(token, &mut nodes)?),
            Close => {
                toparse.clear();
                // Drain everything from the most recent Open paren
                // till the end into toparse to get parsed into a
                // tree.
                toparse.extend(
                    stack.drain(parens.pop().ok_or(LispParseError::MalformedParentheses)?..),
                );
                stack.push(Done(parse_expression(&mut toparse, &mut nodes)?));
            }
        }
    }
    if stack.len() == 1 {
        if let Done(index) = stack.remove(0) {
            if index == nodes.len() - 1 {
                return Ok(Tree::from_nodes(nodes)
                    .ok()
                    .ok_or(LispParseError::Unknown)?);
            }
        }
    }
    return Err(LispParseError::Unknown);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deftree_macro() {
        let sphere = deftree!(
            (- (sqrt (+ (* x x) (* y y))) 5.0)
        )
        .expect("Unable to parse tree from lisp");
        assert_eq!(sphere.len(), 10);
        assert_eq!(
            format!("{}", sphere).trim(),
            "
[9] Subtract(7, 8)
 ├── [7] Sqrt(6)
 │    └── [6] Add(2, 5)
 │         ├── [2] Multiply(0, 1)
 │         │    ├── [0] Symbol(x)
 │         │    └── [1] Symbol(x)
 │         └── [5] Multiply(3, 4)
 │              ├── [3] Symbol(y)
 │              └── [4] Symbol(y)
 └── [8] Constant(5)"
                .trim()
        );
    }
}
