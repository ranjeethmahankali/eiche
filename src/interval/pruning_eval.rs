use super::Interval;
use crate::{
    compile::{CompileCache, CompileOutput, compile},
    eval::ValueType,
    tree::{Node, Tree},
};
use std::{collections::BTreeMap, ops::Range};

pub struct PruningEvaluator<T>
where
    T: ValueType,
{
    ops: Vec<(Node, usize)>,
    regs: Vec<T>,
    vars: Vec<(char, T)>,
    num_roots: usize,
    root_regs: Vec<usize>,
    val_outputs: Vec<T>,
    interval_outputs: Vec<Interval>,
    bounds: Vec<BTreeMap<char, Interval>>, // Bounds stored like a stack.
    op_ranges: Vec<Range<usize>>,
    current_depth: usize,
    divisions: usize,
}

impl<T> PruningEvaluator<T>
where
    T: ValueType,
{
    pub fn new(
        tree: &Tree,
        divisions: usize,
        estimated_depth: usize,
        bounds: BTreeMap<char, Interval>,
    ) -> Self {
        let num_roots = tree.num_roots();
        let mut ops = Vec::with_capacity(tree.len() * estimated_depth);
        let mut root_regs = Vec::with_capacity(num_roots * estimated_depth);
        let mut cache = CompileCache::default();
        let num_regs = compile(
            tree.nodes(),
            tree.root_indices(),
            &mut cache,
            CompileOutput {
                ops: &mut ops,
                out_regs: &mut root_regs,
            },
        );
        debug_assert_eq!(root_regs.len(), num_roots);
        let num_ops = ops.len();
        PruningEvaluator {
            ops,
            regs: vec![T::from_scalar(0.).unwrap(); num_regs],
            vars: Vec::new(),
            num_roots,
            root_regs,
            val_outputs: vec![T::from_scalar(0.).unwrap(); num_roots],
            interval_outputs: vec![Interval::Scalar(inari::Interval::ENTIRE); num_roots],
            bounds: vec![bounds],
            op_ranges: vec![0..num_ops],
            current_depth: 0,
            divisions,
        }
    }
}
