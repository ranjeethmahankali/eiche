use crate::tree::{abs, cos, exp, log, max, min, pow, sin, sqrt, tan, Node::*, Tree};
use lazy_static::lazy_static;
use regex::Regex;
use std::str::CharIndices;

#[macro_export]
macro_rules! deftree {
    ($($exp:tt) *) => {
        crate::parser::parse_lisp(stringify!($($exp) *).to_string())
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
    Done(Tree),
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

fn parse_atom<'a>(atom: &'a str) -> Result<Parsed<'a>, LispParseError> {
    lazy_static! {
        static ref FLT_REGEX: Regex =
            Regex::new("^\\d+\\.*\\d*$").expect("Failed to initialize regex for floats.");
        static ref SYM_REGEX: Regex =
            Regex::new("^[a-zA-Z]$").expect("Failed to initialize regex for numbers.");
    };
    if FLT_REGEX.is_match(atom) {
        return Ok(Done(Tree::new(Constant(
            atom.parse::<f64>().ok().ok_or(LispParseError::Float)?,
        ))));
    }
    if SYM_REGEX.is_match(atom) {
        return Ok(Done(Tree::new(Symbol(
            atom.chars().nth(0).ok_or(LispParseError::Symbol)?,
        ))));
    }
    return Ok(Todo(atom));
}

fn parse_unary(op: &str, tree: Tree) -> Result<Tree, LispParseError> {
    match op {
        "-" => Ok(-tree),
        "sqrt" => Ok(sqrt(tree)),
        "abs" => Ok(abs(tree)),
        "sin" => Ok(sin(tree)),
        "cos" => Ok(cos(tree)),
        "tan" => Ok(tan(tree)),
        "log" => Ok(log(tree)),
        "exp" => Ok(exp(tree)),
        _ => Err(LispParseError::Unary),
    }
}

fn parse_binary(op: &str, lhs: Tree, rhs: Tree) -> Result<Tree, LispParseError> {
    match op {
        "+" => Ok(lhs + rhs),
        "-" => Ok(lhs - rhs),
        "*" => Ok(lhs * rhs),
        "/" => Ok(lhs / rhs),
        "pow" => Ok(pow(lhs, rhs)),
        "min" => Ok(min(lhs, rhs)),
        "max" => Ok(max(lhs, rhs)),
        _ => Err(LispParseError::Binary),
    }
}

fn parse_expression<'a>(expr: &mut Vec<Parsed<'a>>) -> Result<Tree, LispParseError> {
    match expr.len() {
        0 => Err(LispParseError::TooFewTokens),
        1 => match expr.remove(0) {
            Done(tree) => Ok(tree),
            // Expression of length cannot be a todo item.
            Todo(token) => Err(LispParseError::InvalidToken(token.to_string())),
        },
        2 => match (expr.remove(1), expr.remove(0)) {
            (Done(tree), Todo(op)) => Ok(parse_unary(op, tree)?),
            // Anything that's not a valid unary op.
            _ => Err(LispParseError::Unary),
        },
        3 => match (expr.remove(2), expr.remove(1), expr.remove(0)) {
            (Done(rhs), Done(lhs), Todo(op)) => Ok(parse_binary(op, lhs, rhs)?),
            // Anything that's not a valid binary op.
            _ => Err(LispParseError::Binary),
        },
        _ => Err(LispParseError::TooManyTokens),
    }
}

pub fn parse_lisp(lisp: String) -> Result<Tree, LispParseError> {
    let mut parens: Vec<usize> = Vec::new();
    let mut stack: Vec<Parsed> = Vec::new();
    let mut toparse: Vec<Parsed> = Vec::with_capacity(4);
    for token in Tokenizer::from(&lisp) {
        match token {
            Open => parens.push(stack.len()),
            Atom(token) => stack.push(parse_atom(token)?),
            Close => {
                toparse.clear();
                // Drain everything from the most recent Open paren
                // till the end into toparse to get parsed into a
                // tree.
                toparse.extend(
                    stack.drain(parens.pop().ok_or(LispParseError::MalformedParentheses)?..),
                );
                stack.push(Done(parse_expression(&mut toparse)?));
            }
        }
    }
    if stack.len() == 1 {
        if let Done(tree) = stack.remove(0) {
            return Ok(tree);
        }
    }
    return Err(LispParseError::Unknown);
}

#[cfg(test)]
mod tests {

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
