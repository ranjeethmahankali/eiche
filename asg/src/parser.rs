use crate::tree::{Node, Node::*, Tree};

#[macro_export]
macro_rules! deftree {
    ($($exp:tt) *) => {
        crate::parser::parse_lisp(stringify!($($exp) *).to_string())
    };
}

#[derive(Debug)]
enum Token<'a> {
    Open,
    Atom(&'a str),
    Close,
}
use Token::*;

fn tokenize(lisp: &String) -> Vec<Token> {
    let mut tokens: Vec<Token> = Vec::new();
    let mut last: usize = 0;
    for (i, c) in lisp.char_indices() {
        let token = match c {
            '(' => {
                if i > last {
                    tokens.push(Atom(&lisp[last..i]));
                }
                last = i + 1;
                Open
            }
            ')' => {
                if i > last {
                    tokens.push(Atom(&lisp[last..i]));
                }
                last = i + 1;
                Close
            }
            ' ' if i > last => {
                let token = Atom(&lisp[last..i]);
                last = i + 1;
                token
            }
            ' ' => {
                last = i + 1;
                continue;
            }
            _ => continue,
        };
        tokens.push(token);
    }
    return tokens;
}

pub fn parse_lisp(lisp: String) -> Tree {
    println!("Input: {}", lisp);
    let tokens = tokenize(&lisp);
    println!("{:?}", tokens);
    todo!()
}

#[cfg(test)]
mod tests {

    #[test]
    fn deftree_macro() {
        let sphere = deftree!(
            (- (sqrt (+ (* x x) (* y y))) 5.0)
        );
        println!("{}", sphere);
    }
}
