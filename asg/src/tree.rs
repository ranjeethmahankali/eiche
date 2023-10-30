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
#[derive(Debug)]
pub struct Tree {
    nodes: Vec<Node>,
}

use Node::*;

use crate::helper::eq_recursive;

impl PartialEq for Tree {
    fn eq(&self, other: &Self) -> bool {
        self.nodes == other.nodes
    }
}

impl Tree {
    pub fn new(node: Node) -> Tree {
        Tree { nodes: vec![node] }
    }

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

    pub fn traverse_depth<F, T, E>(&self, mut visitor: F) -> Result<T, E>
    where
        F: FnMut(usize, Option<usize>) -> Result<T, E>,
        T: Default,
    {
        let mut stack: Vec<(usize, Option<usize>)> = vec![(self.root_index(), None)];
        while !stack.is_empty() {
            let (index, parent) = stack
                .pop()
                .expect("Something went wrong in the depth first traversal!");
            match self.node(index) {
                Constant(_) => {} // Do nothing.
                Symbol(_) => {}   // Do nothing.
                Unary(_, input) => stack.push((*input, Some(index))),
                Binary(_, lhs, rhs) => {
                    stack.push((*rhs, Some(index)));
                    stack.push((*lhs, Some(index)));
                }
            }
            visitor(index, parent)?;
        }
        return Result::<T, E>::Ok(T::default());
    }

    pub fn fold_constants(mut self) -> Tree {
        for index in 0..self.len() {
            let constval = match self.nodes[index] {
                Constant(_) => None,
                Symbol(_) => None,
                Unary(op, input) => {
                    if let Constant(value) = self.nodes[input] {
                        Some(op.apply(value))
                    } else {
                        None
                    }
                }
                Binary(op, lhs, rhs) => {
                    if let (Constant(a), Constant(b)) = (&self.nodes[lhs], &self.nodes[rhs]) {
                        Some(op.apply(*a, *b))
                    } else {
                        None
                    }
                }
            };
            if let Some(value) = constval {
                self.nodes[index] = Constant(value);
            }
        }
        return self.prune();
    }

    pub fn prune(mut self) -> Tree {
        const ERROR: &str = "Unreachable code path: Tree pruning failed.";
        // Use a boxed slice for correctness as it cannot be resized later by accident.
        let mut flags: Box<[(bool, usize)]> = vec![(false, 0); self.len()].into_boxed_slice();
        let mut count = 0usize;
        self.traverse_depth(|index, _parent| -> Result<(), ()> {
            flags[index] = (true, 1usize);
            count += 1usize;
            return Ok(());
        })
        .expect(ERROR);
        let indices = {
            // Do a prefix scan to to get the actual indices.
            let mut sum = 0usize;
            for pair in flags.iter_mut() {
                let (keep, i) = *pair;
                let copy = sum;
                sum += i;
                *pair = (keep, copy);
            }
            flags
        };
        for i in 0..indices.len() {
            let node = self.nodes.get_mut(i).expect(ERROR);
            match node {
                Constant(_) => {} // Do nothing.
                Symbol(_) => {}   // Do nothing.
                Unary(_, input) => {
                    *input = indices[*input].1;
                }
                Binary(_, lhs, rhs) => {
                    *lhs = indices[*lhs].1;
                    *rhs = indices[*rhs].1;
                }
            }
        }
        self.nodes = self
            .nodes
            .iter()
            .zip(indices.iter())
            .filter(|(_node, (keep, _index))| {
                return *keep;
            })
            .map(|(&node, (_keep, _index))| {
                return node;
            })
            .collect();
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
        let hashes = self.node_hashes();
        let mut revmap: HashMap<u64, usize> = HashMap::new();
        let mut indices: Box<[usize]> = (0..self.len()).collect();
        for i in 0..hashes.len() {
            let h = hashes[i];
            let entry = revmap.entry(h).or_insert(i);
            if *entry != i && eq_recursive(&self.nodes, *entry, i) {
                // The i-th node should be replaced with entry-th node.
                indices[i] = *entry;
            }
        }
        let indices = indices; // Disallow mutation from here.
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

    pub fn node(&self, index: usize) -> &Node {
        &self.nodes[index]
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
