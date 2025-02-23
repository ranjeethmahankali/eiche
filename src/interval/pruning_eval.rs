use super::Interval;
use crate::{
    compile::{compile, CompileCache, CompileOutput},
    error::Error,
    eval::ValueType,
    interval::fold::fold_for_interval,
    tree::{Node, Tree, Value},
};
use std::{collections::BTreeMap, ops::Range};

pub struct PruningEvaluator<T>
where
    T: ValueType,
{
    nodes: Vec<Node>,
    ops: Vec<(Node, usize)>,
    val_regs: Vec<T>,
    interval_regs: Vec<Interval>,
    vars: Vec<(char, T)>,
    num_roots: usize,
    root_regs: Vec<usize>,
    val_outputs: Vec<T>,
    interval_outputs: Vec<Interval>,
    bounds: Vec<BTreeMap<char, Interval>>, // Bounds stored like a stack.
    node_ranges: Vec<Range<usize>>,
    op_ranges: Vec<Range<usize>>,
    interval_indices: Vec<usize>,
    intervals_per_level: usize,
    divisions: f64,
    // Temp storage,
    temp_nodes: Vec<Node>,
    compile_cache: CompileCache,
}

#[derive(Debug)]
pub enum PruningError {
    UnexpectedEmptyState,
    CannotConstructInterval(inari::IntervalError),
}

impl From<PruningError> for Error {
    fn from(value: PruningError) -> Self {
        Error::Pruning(value)
    }
}

pub enum PruningState {
    None,
    Valid(usize, usize),
    Failure(Error),
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
        let num_bounds = bounds.len();
        PruningEvaluator {
            nodes: tree.nodes().to_vec(),
            ops,
            val_regs: vec![T::default(); num_regs],
            interval_regs: vec![Interval::default(); num_regs],
            vars: Vec::new(),
            num_roots,
            root_regs,
            val_outputs: vec![T::from_scalar(0.).unwrap(); num_roots],
            interval_outputs: vec![Interval::Scalar(inari::Interval::ENTIRE); num_roots],
            bounds: vec![bounds],
            node_ranges: vec![0..tree.len()],
            op_ranges: vec![0..num_ops],
            interval_indices: vec![0],
            intervals_per_level: divisions.pow(num_bounds as u32),
            divisions: divisions as f64,
            temp_nodes: Vec::with_capacity(tree.len()),
            compile_cache: cache,
        }
    }

    pub fn push(&mut self) -> PruningState {
        // Split the current intervals into 'divisions' and pick the first child.
        let bounds = match match self.bounds.last() {
            Some(last) => last
                .iter()
                .try_fold(BTreeMap::new(), |mut bounds, (&k, v)| {
                    bounds.insert(
                        k,
                        match v {
                            Interval::Scalar(ii) => inari::Interval::try_from((
                                ii.inf(),
                                ii.inf() + ii.wid() / self.divisions,
                            ))?
                            .into(),
                            Interval::Bool(true, true) => Interval::Bool(true, true),
                            Interval::Bool(false, false) => Interval::Bool(false, false),
                            Interval::Bool(_, _) => Interval::Bool(false, true),
                        },
                    );
                    Ok(bounds)
                }),
            None => return PruningState::Failure(PruningError::UnexpectedEmptyState.into()),
        } {
            Ok(bounds) => bounds,
            Err(e) => {
                return PruningState::Failure(PruningError::CannotConstructInterval(e).into())
            }
        };
        if let Err(e) = fold_for_interval(
            match self.node_ranges.last() {
                Some(range) => &self.nodes[range.clone()],
                None => return PruningState::Failure(PruningError::UnexpectedEmptyState.into()),
            },
            &bounds,
            &mut self.temp_nodes,
        ) {
            return PruningState::Failure(e);
        }
        let nnodes_before = self.nodes.len();
        self.nodes.extend(self.temp_nodes.drain(..));
        let node_range = nnodes_before..self.nodes.len();
        self.bounds.push(bounds);
        self.interval_indices.push(0);
        let nops_before = self.ops.len();
        let num_regs = compile(
            &self.nodes[node_range.clone()],
            (node_range.len() - self.num_roots)..node_range.len(),
            &mut self.compile_cache,
            CompileOutput {
                ops: &mut self.ops,
                out_regs: &mut self.root_regs,
            },
        );
        self.node_ranges.push(node_range);
        self.op_ranges.push(nops_before..self.ops.len());
        self.val_regs.resize(num_regs, T::default());
        self.interval_regs.resize(num_regs, Interval::default());
        todo!("Go one level deeper");
    }

    pub fn pop(&mut self) -> PruningState {
        todo!("Go back up one level");
    }

    pub fn next(&mut self) -> PruningState {
        todo!("Go to the next inteval at the same depth");
    }
}

pub type ValuePruningEvaluator = PruningEvaluator<Value>;
