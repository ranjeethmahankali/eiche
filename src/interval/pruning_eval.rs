use super::Interval;
use crate::{
    compile::{compile, CompileCache, CompileOutput},
    error::Error,
    eval::ValueType,
    interval::fold::fold_for_interval,
    tree::{Node, Tree, Value},
};
use std::collections::BTreeMap;

struct Stack<T>
where
    T: Clone,
{
    buf: Vec<T>,
    offsets: Vec<usize>,
}

impl<T> Stack<T>
where
    T: Clone,
{
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            offsets: Vec::new(),
        }
    }

    pub fn with_capacity(nslices: usize, ntotal: usize) -> Self {
        Self {
            buf: Vec::with_capacity(ntotal),
            offsets: Vec::with_capacity(nslices),
        }
    }

    pub fn from_iter(iter: impl Iterator<Item = T>) -> Self {
        let mut out = Self::new();
        out.start_slice();
        out.push_iter(iter);
        out
    }

    pub fn start_slice(&mut self) {
        self.offsets.push(self.buf.len());
    }

    pub fn pop_slice(&mut self) -> Option<usize> {
        self.offsets.pop().map(|last| {
            let num = self.buf.len() - last;
            self.buf.truncate(last);
            num
        })
    }

    pub fn push_slice(&mut self, vals: &[T]) {
        self.start_slice();
        self.buf.extend_from_slice(vals)
    }

    pub fn push_iter(&mut self, vals: impl Iterator<Item = T>) {
        self.start_slice();
        self.buf.extend(vals);
    }

    pub fn last_slice(&self) -> Option<&[T]> {
        self.offsets.last().map(|last| &self.buf[*last..])
    }

    pub fn borrow_mut(&mut self) -> &mut Vec<T> {
        self.start_slice();
        &mut self.buf
    }
}

impl<T> Default for Stack<T>
where
    T: Clone + Default,
{
    fn default() -> Self {
        Self {
            buf: Default::default(),
            offsets: Default::default(),
        }
    }
}

impl<T> From<&[T]> for Stack<T>
where
    T: Clone,
{
    fn from(value: &[T]) -> Self {
        let mut out = Self::with_capacity(1, value.len());
        out.start_slice();
        out.push_slice(value);
        out
    }
}

pub struct PruningEvaluator<T>
where
    T: ValueType,
{
    bounds: Stack<(char, Interval)>, // Bounds stored like a stack.
    nodes: Stack<Node>,
    num_regs: Vec<usize>,
    ops: Stack<(Node, usize)>,
    val_regs: Vec<T>,
    interval_regs: Vec<Interval>,
    vars: Vec<(char, T)>,
    num_roots: usize,
    root_regs: Stack<usize>,
    val_outputs: Vec<T>,
    interval_outputs: Vec<Interval>,
    interval_indices: Vec<usize>,
    intervals_per_level: usize,
    divisions: f64,
    // Temp storage,
    temp_bounds: Vec<(char, Interval)>,
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
        let mut ops = Stack::with_capacity(estimated_depth, tree.len() * estimated_depth);
        let mut root_regs = Stack::with_capacity(estimated_depth, num_roots * estimated_depth);
        let mut cache = CompileCache::default();
        let num_regs = compile(
            tree.nodes(),
            tree.root_indices(),
            &mut cache,
            CompileOutput {
                ops: ops.borrow_mut(),
                out_regs: root_regs.borrow_mut(),
            },
        );
        debug_assert_eq!(
            root_regs.last_slice().map(|s| s.len()).unwrap_or(0),
            num_roots
        );
        let num_bounds = bounds.len();
        PruningEvaluator {
            bounds: Stack::from_iter(bounds.into_iter()),
            nodes: tree.nodes().into(),
            num_regs: vec![num_regs],
            ops,
            val_regs: vec![T::default(); num_regs],
            interval_regs: vec![Interval::default(); num_regs],
            vars: Vec::new(),
            num_roots,
            root_regs,
            val_outputs: vec![T::from_scalar(0.).unwrap(); num_roots],
            interval_outputs: vec![Interval::Scalar(inari::Interval::ENTIRE); num_roots],
            interval_indices: vec![0],
            intervals_per_level: divisions.pow(num_bounds as u32),
            divisions: divisions as f64,
            temp_bounds: Vec::new(),
            temp_nodes: Vec::with_capacity(tree.len()),
            compile_cache: cache,
        }
    }

    pub fn current_depth(&self) -> usize {
        self.interval_indices.len()
    }

    pub fn push(&mut self) -> PruningState {
        // Split the current intervals into 'divisions' and pick the first child.
        self.temp_bounds.clear();
        match match self.bounds.last_slice() {
            Some(last) => last
                .iter()
                .try_fold(&mut self.temp_bounds, |bounds, (k, v)| {
                    bounds.push((
                        *k,
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
                    ));
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
            match self.nodes.last_slice() {
                Some(nodes) => nodes,
                None => return PruningState::Failure(PruningError::UnexpectedEmptyState.into()),
            },
            &self.temp_bounds,
            &mut self.temp_nodes,
        ) {
            return PruningState::Failure(e);
        }
        self.bounds.push_iter(self.temp_bounds.drain(..));
        self.nodes.push_iter(self.temp_nodes.drain(..));
        self.interval_indices.push(0);
        let nodes = match self.nodes.last_slice() {
            Some(nodes) => nodes,
            None => return PruningState::Failure(PruningError::UnexpectedEmptyState.into()),
        };
        let num_regs = compile(
            nodes,
            (nodes.len() - self.num_roots)..nodes.len(),
            &mut self.compile_cache,
            CompileOutput {
                ops: self.ops.borrow_mut(),
                out_regs: self.root_regs.borrow_mut(),
            },
        );
        self.num_regs.push(num_regs);
        self.val_regs.resize(num_regs, T::default());
        self.interval_regs.resize(num_regs, Interval::default());
        PruningState::Valid(
            self.current_depth(),
            match self.interval_indices.last() {
                Some(last) => *last,
                None => return PruningState::Failure(PruningError::UnexpectedEmptyState.into()),
            },
        )
    }

    pub fn pop(&mut self) -> PruningState {
        self.nodes.pop_slice();
        self.ops.pop_slice();
        self.num_regs.pop();
        if let Some(nregs) = self.num_regs.last() {
            self.val_regs.resize(*nregs, T::default());
            self.interval_regs.resize(*nregs, Interval::default());
        }
        self.root_regs.pop_slice();
        self.interval_indices.pop();
        PruningState::Valid(
            self.current_depth(),
            match self.interval_indices.last() {
                Some(last) => *last,
                None => return PruningState::Failure(PruningError::UnexpectedEmptyState.into()),
            },
        )
    }

    pub fn next(&mut self) -> PruningState {
        match self.bounds.pop_slice() {
            Some(_) => {}
            None => return PruningState::Failure(PruningError::UnexpectedEmptyState.into()),
        };
        todo!("Go to the next inteval at the same depth");
    }
}

pub type ValuePruningEvaluator = PruningEvaluator<Value>;
