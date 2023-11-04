#[derive(Debug, Copy, Clone, PartialEq, Hash)]
pub enum UnaryOp {
    Negate,
    Sqrt,
    Abs,
    Sin,
    Cos,
    Tan,
    Log,
    Exp,
}

#[derive(Debug, Copy, Clone, PartialEq, Hash)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Pow,
    Min,
    Max,
}

impl UnaryOp {
    /// Compute the result of the operation on `value`.
    pub fn apply(&self, value: f64) -> f64 {
        match self {
            Negate => -value,
            Sqrt => f64::sqrt(value),
            Abs => f64::abs(value),
            Sin => f64::sin(value),
            Cos => f64::cos(value),
            Tan => f64::tan(value),
            Log => f64::log(value, std::f64::consts::E),
            Exp => f64::exp(value),
        }
    }
}

impl BinaryOp {
    /// Compute the result of the operation on `lhs` and `rhs`.
    pub fn apply(&self, lhs: f64, rhs: f64) -> f64 {
        match self {
            Add => lhs + rhs,
            Subtract => lhs - rhs,
            Multiply => lhs * rhs,
            Divide => lhs / rhs,
            Pow => f64::powf(lhs, rhs),
            Min => f64::min(lhs, rhs),
            Max => f64::max(lhs, rhs),
        }
    }
}

use BinaryOp::*;
use UnaryOp::*;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Node {
    Constant(f64),
    Symbol(char),
    Unary(UnaryOp, usize),
    Binary(BinaryOp, usize, usize),
}

/// Represents an abstract syntax tree.
#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    nodes: Vec<Node>,
}

use Node::*;

use crate::helper::{equivalent, fold_constants, DepthWalker, Trimmer};

#[derive(Debug)]
pub enum InvalidTree {
    WrongNodeOrder,
}

impl Tree {
    pub fn new(node: Node) -> Tree {
        Tree { nodes: vec![node] }
    }

    fn validate(nodes: &Vec<Node>) -> bool {
        (0..nodes.len()).all(|i| match &nodes[i] {
            Constant(_) | Symbol(_) => true,
            Unary(_op, input) => input < &i,
            Binary(_op, lhs, rhs) => lhs < &i && rhs < &i,
        })
        // Maybe add more checks later.
    }

    pub fn from_nodes(nodes: Vec<Node>) -> Result<Tree, InvalidTree> {
        if Self::validate(&nodes) {
            Ok(Tree { nodes })
        } else {
            Err(InvalidTree::WrongNodeOrder)
        }
    }

    /// The number of nodes in this tree.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn root(&self) -> &Node {
        self.nodes
            .last()
            .expect("This Tree is empty! This should never have happened!")
    }

    pub fn root_index(&self) -> usize {
        self.len() - 1
    }

    pub fn node(&self, index: usize) -> &Node {
        &self.nodes[index]
    }

    pub fn nodes(&self) -> &Vec<Node> {
        &self.nodes
    }

    pub fn fold_constants(mut self) -> Tree {
        fold_constants(&mut self.nodes);
        return self.prune();
    }

    pub fn prune(mut self) -> Tree {
        let mut walker = DepthWalker::new();
        let mut trimmer = Trimmer::new();
        let root_index = self.root_index();
        self.nodes = trimmer.trim(self.nodes, root_index, &mut walker);
        if !Self::validate(&self.nodes) {
            panic!("Something broke in the depth first traversal or trimming. This should never have happened!");
        }
        return self;
    }

    fn node_hashes(&self) -> Box<[u64]> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        // Using a boxed slice to avoid accidental resizing later.
        let mut hashes: Box<[u64]> = vec![0; self.len()].into_boxed_slice();
        for index in 0..self.len() {
            let hash: u64 = match self.nodes[index] {
                Constant(value) => value.to_bits(),
                Symbol(label) => {
                    let mut s: DefaultHasher = Default::default();
                    label.hash(&mut s);
                    s.finish()
                }
                Unary(op, input) => {
                    let mut s: DefaultHasher = Default::default();
                    op.hash(&mut s);
                    hashes[input].hash(&mut s);
                    s.finish()
                }
                Binary(op, lhs, rhs) => {
                    let mut s: DefaultHasher = Default::default();
                    op.hash(&mut s);
                    hashes[lhs].hash(&mut s);
                    hashes[rhs].hash(&mut s);
                    s.finish()
                }
            };
            hashes[index] = hash;
        }
        return hashes;
    }

    pub fn deduplicate(mut self) -> Tree {
        use std::collections::hash_map::HashMap;
        // Compute new indices after deduplication.
        let indices = {
            let mut indices: Box<[usize]> = (0..self.len()).collect();
            // Compute hashes to find potential duplicates.
            let hashes = self.node_hashes();
            // Map hashes to node indices.
            let mut revmap: HashMap<u64, usize> = HashMap::new();
            // These walkers are for checking the equivalence of the
            // nodes with the same hash.
            let mut walker1 = DepthWalker::new();
            let mut walker2 = DepthWalker::new();
            for i in 0..hashes.len() {
                let h = hashes[i];
                let entry = revmap.entry(h).or_insert(i);
                if *entry != i && equivalent(*entry, i, &self.nodes, &mut walker1, &mut walker2) {
                    // The i-th node should be replaced with entry-th node.
                    indices[i] = *entry;
                }
            }
            indices
        };
        for node in self.nodes.iter_mut() {
            match node {
                Constant(_) => {}
                Symbol(_) => {}
                Unary(_, input) => {
                    *input = indices[*input];
                }
                Binary(_, lhs, rhs) => {
                    *lhs = indices[*lhs];
                    *rhs = indices[*rhs];
                }
            }
        }
        return self.prune();
    }

    fn binary_op(mut self, other: Tree, op: BinaryOp) -> Tree {
        let offset: usize = self.nodes.len();
        self.nodes
            .reserve(self.nodes.len() + other.nodes.len() + 1usize);
        self.nodes.extend(other.nodes.iter().map(|node| match node {
            Constant(value) => Constant(*value),
            Symbol(label) => Symbol(label.clone()),
            Unary(op, input) => Unary(*op, *input + offset),
            Binary(op, lhs, rhs) => Binary(*op, *lhs + offset, *rhs + offset),
        }));
        self.nodes
            .push(Binary(op, offset - 1, self.nodes.len() - 1));
        return self;
    }

    fn unary_op(mut self, op: UnaryOp) -> Tree {
        self.nodes.push(Unary(op, self.root_index()));
        return self;
    }
}

impl core::ops::Add<Tree> for Tree {
    type Output = Tree;

    fn add(self, rhs: Tree) -> Tree {
        self.binary_op(rhs, Add)
    }
}

impl core::ops::Sub<Tree> for Tree {
    type Output = Tree;

    fn sub(self, rhs: Tree) -> Self::Output {
        self.binary_op(rhs, Subtract)
    }
}

impl core::ops::Mul<Tree> for Tree {
    type Output = Tree;

    fn mul(self, rhs: Tree) -> Tree {
        self.binary_op(rhs, Multiply)
    }
}

impl core::ops::Div<Tree> for Tree {
    type Output = Tree;

    fn div(self, rhs: Tree) -> Self::Output {
        self.binary_op(rhs, Divide)
    }
}

pub fn pow(base: Tree, exponent: Tree) -> Tree {
    base.binary_op(exponent, Pow)
}

pub fn min(lhs: Tree, rhs: Tree) -> Tree {
    lhs.binary_op(rhs, Min)
}

pub fn max(lhs: Tree, rhs: Tree) -> Tree {
    lhs.binary_op(rhs, Max)
}

impl core::ops::Neg for Tree {
    type Output = Tree;

    fn neg(self) -> Self::Output {
        self.unary_op(Negate)
    }
}

pub fn sqrt(x: Tree) -> Tree {
    x.unary_op(Sqrt)
}

pub fn abs(x: Tree) -> Tree {
    x.unary_op(Abs)
}

pub fn sin(x: Tree) -> Tree {
    x.unary_op(Sin)
}

pub fn cos(x: Tree) -> Tree {
    x.unary_op(Cos)
}

pub fn tan(x: Tree) -> Tree {
    x.unary_op(Tan)
}

pub fn log(x: Tree) -> Tree {
    x.unary_op(Log)
}

pub fn exp(x: Tree) -> Tree {
    x.unary_op(Exp)
}

#[derive(Debug)]
pub enum EvaluationError {
    VariableNotFound(char),
    UninitializedValueRead,
}

pub struct Evaluator<'a> {
    tree: &'a Tree,
    regs: Box<[Option<f64>]>,
}

impl<'a> Evaluator<'a> {
    pub fn new(tree: &'a Tree) -> Evaluator {
        Evaluator {
            tree,
            regs: vec![None; tree.nodes.len()].into_boxed_slice(),
        }
    }

    /// Set all symbols in the evaluator matching `label` to
    /// `value`. This `value` will be used for all future evaluations,
    /// unless this function is called again with a different `value`.
    pub fn set_var(&mut self, label: char, value: f64) {
        for (node, reg) in self.tree.nodes.iter().zip(self.regs.iter_mut()) {
            match node {
                Symbol(l) if *l == label => {
                    *reg = Some(value);
                }
                _ => {}
            }
        }
    }

    /// Read the value from the `index`-th register. Returns an error
    /// if the register doesn't contain a value.
    fn read(&self, index: usize) -> Result<f64, EvaluationError> {
        match self.regs[index] {
            Some(val) => Ok(val),
            None => Err(EvaluationError::UninitializedValueRead),
        }
    }

    /// Write the `value` into the `index`-th register. The existing
    /// value is overwritten.
    fn write(&mut self, index: usize, value: f64) {
        self.regs[index] = Some(value);
    }

    /// Run the evaluator and return the result. The result may
    /// contain the output value, or an
    /// error. `Variablenotfound(label)` error means the variable
    /// matching `label` hasn't been assigned a value using `set_var`.
    pub fn run(&mut self) -> Result<f64, EvaluationError> {
        for idx in 0..self.tree.len() {
            self.write(
                idx,
                match &self.tree.nodes[idx] {
                    Constant(val) => *val,
                    Symbol(label) => match &self.regs[idx] {
                        None => return Err(EvaluationError::VariableNotFound(*label)),
                        Some(val) => *val,
                    },
                    Binary(op, lhs, rhs) => op.apply(self.read(*lhs)?, self.read(*rhs)?),
                    Unary(op, input) => op.apply(self.read(*input)?),
                },
            );
        }
        return self.read(self.tree.root_index());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let sum = x + y;
        assert_eq!(sum.nodes, vec![Symbol('x'), Symbol('y'), Binary(Add, 0, 1)]);
    }

    #[test]
    fn multiply() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let sum = x * y;
        assert_eq!(
            sum.nodes,
            vec![Symbol('x'), Symbol('y'), Binary(Multiply, 0, 1)]
        );
    }

    #[test]
    fn subtract() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let sum = x - y;
        assert_eq!(
            sum.nodes,
            vec![Symbol('x'), Symbol('y'), Binary(Subtract, 0, 1)]
        );
    }

    #[test]
    fn divide() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let sum = x / y;
        assert_eq!(
            sum.nodes,
            vec![Symbol('x'), Symbol('y'), Binary(Divide, 0, 1)]
        );
    }

    #[test]
    fn negate() {
        let x: Tree = 'x'.into();
        let neg = -x;
        assert_eq!(neg.nodes, vec![Symbol('x'), Unary(Negate, 0)]);
    }

    #[test]
    fn sqrt_test() {
        let x: Tree = 'x'.into();
        let neg = sqrt(x);
        assert_eq!(neg.nodes, vec![Symbol('x'), Unary(Sqrt, 0)]);
    }

    #[test]
    fn abs_test() {
        let x: Tree = 'x'.into();
        let neg = abs(x);
        assert_eq!(neg.nodes, vec![Symbol('x'), Unary(Abs, 0)]);
    }
}
