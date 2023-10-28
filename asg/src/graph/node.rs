#[derive(Debug, PartialEq)]
pub enum Node {
    // Leaf types
    Constant(f64),
    Symbol(String),
    // Binary operations
    Add(usize, usize),
    Subtract(usize, usize),
    Multiply(usize, usize),
    Divide(usize, usize),
    Pow(usize, usize),
    Min(usize, usize),
    Max(usize, usize),
    // Unary operations
    Negate(usize),
    Sqrt(usize),
    Abs(usize),
    Sin(usize),
    Cos(usize),
    Tan(usize),
    Log(usize),
    Exp(usize),
}

impl Into<Tree> for Node {
    fn into(self) -> Tree {
        Tree::new(self)
    }
}

use Node::*;

pub struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    pub fn new(node: Node) -> Tree {
        Tree { nodes: vec![node] }
    }

    pub fn root(&self) -> &Node {
        self.nodes
            .last()
            .expect("This Asg is empty! It has no root node!")
    }

    fn merge(mut self, other: Tree, op: Node) -> Tree {
        let offset: usize = other.nodes.len();
        self.nodes
            .reserve(self.nodes.len() + other.nodes.len() + 1usize);
        self.nodes.extend(other.nodes.iter().map(|node| match node {
            Constant(value) => Constant(*value),
            Symbol(label) => Symbol(label.clone()),
            Add(l, r) => Add(*l + offset, *r + offset),
            Subtract(l, r) => Subtract(*l + offset, *r + offset),
            Multiply(l, r) => Multiply(*l + offset, *r + offset),
            Divide(l, r) => Divide(*l + offset, *r + offset),
            Pow(l, r) => Pow(*l + offset, *r + offset),
            Min(l, r) => Min(*l + offset, *r + offset),
            Max(l, r) => Max(*l + offset, *r + offset),
            Negate(x) => Negate(*x + offset),
            Sqrt(x) => Sqrt(*x + offset),
            Abs(x) => Abs(*x + offset),
            Sin(x) => Sin(*x + offset),
            Cos(x) => Cos(*x + offset),
            Tan(x) => Tan(*x + offset),
            Log(x) => Log(*x + offset),
            Exp(x) => Exp(*x + offset),
        }));
        self.nodes.push(op);
        return self;
    }

    fn offset_root_indices(left: &Tree, right: &Tree) -> (usize, usize) {
        let li = left.nodes.len() - 1;
        let ri = li + right.nodes.len();
        (li, ri)
    }
}

// macro_rules! binary_op {
//     (&name, $lhs, $rhs, $op) => {

//     };
// }

impl core::ops::Add<Tree> for Tree {
    type Output = Tree;

    fn add(self, rhs: Tree) -> Tree {
        let (left, right) = Tree::offset_root_indices(&self, &rhs);
        self.merge(rhs, Add(left, right))
    }
}

impl core::ops::Sub<Tree> for Tree {
    type Output = Tree;

    fn sub(self, rhs: Tree) -> Self::Output {
        let (left, right) = Tree::offset_root_indices(&self, &rhs);
        self.merge(rhs, Subtract(left, right))
    }
}

impl core::ops::Mul<Tree> for Tree {
    type Output = Tree;

    fn mul(self, rhs: Tree) -> Tree {
        let (left, right) = Tree::offset_root_indices(&self, &rhs);
        self.merge(rhs, Multiply(left, right))
    }
}

impl core::ops::Div<Tree> for Tree {
    type Output = Tree;

    fn div(self, rhs: Tree) -> Self::Output {
        let (left, right) = Tree::offset_root_indices(&self, &rhs);
        self.merge(rhs, Divide(left, right))
    }
}

pub fn pow(base: Tree, exponent: Tree) -> Tree {
    let (left, right) = Tree::offset_root_indices(&base, &exponent);
    base.merge(exponent, Pow(left, right))
}

pub fn min(lhs: Tree, rhs: Tree) -> Tree {
    let (left, right) = Tree::offset_root_indices(&lhs, &rhs);
    lhs.merge(rhs, Min(left, right))
}

pub fn max(lhs: Tree, rhs: Tree) -> Tree {
    let (left, right) = Tree::offset_root_indices(&lhs, &rhs);
    lhs.merge(rhs, Max(left, right))
}

impl core::ops::Neg for Tree {
    type Output = Tree;

    fn neg(mut self) -> Self::Output {
        let idx = self.nodes.len() - 1;
        self.nodes.push(Negate(idx));
        return self;
    }
}

pub fn sqrt(mut op: Tree) -> Tree {
    let idx = op.nodes.len() - 1;
    op.nodes.push(Sqrt(idx));
    return op;
}

#[cfg(test)]
mod tests {
    use super::{Node::*, Tree};

    fn symbol(label: &str) -> Tree {
        return Symbol(label.into()).into();
    }

    #[test]
    fn add() {
        let sum = symbol("x") + symbol("y");
        assert_eq!(
            sum.nodes,
            vec![Symbol("x".into()), Symbol("y".into()), Add(0, 1)]
        );
    }

    #[test]
    fn multiply() {
        let sum = symbol("x") * symbol("y");
        assert_eq!(
            sum.nodes,
            vec![Symbol("x".into()), Symbol("y".into()), Multiply(0, 1)]
        );
    }

    #[test]
    fn subtract() {
        let sum = symbol("x") - symbol("y");
        assert_eq!(
            sum.nodes,
            vec![Symbol("x".into()), Symbol("y".into()), Subtract(0, 1)]
        );
    }

    #[test]
    fn divide() {
        let sum = symbol("x") / symbol("y");
        assert_eq!(
            sum.nodes,
            vec![Symbol("x".into()), Symbol("y".into()), Divide(0, 1)]
        );
    }

    #[test]
    fn negate() {
        let neg = -symbol("x");
        assert_eq!(neg.nodes, vec![Symbol("x".into()), Negate(0)]);
    }
}
