use crate::tree::{Node::*, Tree};

/// Errors that can occur when evaluating a tree.
#[derive(Debug)]
pub enum EvaluationError {
    /// A symbol was not assigned a value before evaluating.
    VariableNotFound(char),
    /// A register with uninitialized value was encountered during
    /// evaluation. This could mean the topology of the tree is
    /// broken.
    UninitializedValueRead,
}

/// This can be used to compute the value(s) of the tree.
pub struct Evaluator<'a> {
    tree: &'a Tree,
    regs: Box<[Option<f64>]>,
}

impl<'a> Evaluator<'a> {
    /// Create a new evaluator for `tree`.
    pub fn new(tree: &'a Tree) -> Evaluator {
        Evaluator {
            tree,
            regs: vec![None; tree.len()].into_boxed_slice(),
        }
    }

    /// Set all symbols in the evaluator matching `label` to
    /// `value`. This `value` will be used for all future evaluations,
    /// unless this function is called again with a different `value`.
    pub fn set_var(&mut self, label: char, value: f64) {
        for (node, reg) in self.tree.nodes().iter().zip(self.regs.iter_mut()) {
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
                match &self.tree.node(idx) {
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
