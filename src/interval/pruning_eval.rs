use super::Interval;
use crate::{
    compile::{CompileCache, CompileOutput, compile},
    dedup::Deduplicater,
    error::Error,
    eval::ValueType,
    interval::fold::fold_for_interval,
    prune::Pruner,
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

impl<T> FromIterator<T> for Stack<T>
where
    T: Clone,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut out = Self::new();
        out.start_slice();
        out.buf.extend(iter);
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
    vars: BTreeMap<char, T>,
    num_roots: usize,
    root_regs: Stack<usize>,
    val_outputs: Vec<T>,
    interval_outputs: Stack<Interval>,
    interval_indices: Vec<usize>,
    intervals_per_level: usize,
    divisions: BTreeMap<char, usize>,
    pruner: Pruner,
    deduper: Deduplicater,
    // Temp storage,
    temp_bounds: Vec<(char, Interval)>,
    temp_intervals: Vec<Interval>,
    temp_nodes: Vec<Node>,
    compile_cache: CompileCache,
}

#[derive(Debug)]
pub enum PruningError {
    UnexpectedEmptyState,
    CannotConstructInterval(inari::IntervalError),
    UnknownVariable(char),
}

impl From<PruningError> for Error {
    fn from(value: PruningError) -> Self {
        Error::Pruning(value)
    }
}

impl From<inari::IntervalError> for PruningError {
    fn from(value: inari::IntervalError) -> Self {
        PruningError::CannotConstructInterval(value)
    }
}

#[derive(Debug)]
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
        estimated_depth: usize,
        intervals: BTreeMap<char, (Interval, usize)>,
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
        PruningEvaluator {
            bounds: Stack::from_iter(
                intervals
                    .iter()
                    .map(|(label, (bounds, _divisions))| (*label, *bounds)),
            ),
            nodes: tree.nodes().into(),
            num_regs: vec![num_regs],
            ops,
            val_regs: vec![T::default(); num_regs],
            interval_regs: vec![Interval::default(); num_regs],
            vars: BTreeMap::new(),
            num_roots,
            root_regs,
            val_outputs: Vec::with_capacity(num_roots),
            interval_outputs: Stack::with_capacity(estimated_depth, num_roots * estimated_depth),
            interval_indices: vec![0],
            intervals_per_level: intervals
                .iter()
                .fold(1usize, |prod, (_label, (_bounds, divisions))| {
                    prod * divisions
                }),
            divisions: intervals
                .iter()
                .map(|(label, (_bounds, divisions))| (*label, *divisions))
                .collect(),
            pruner: Pruner::default(),
            deduper: Deduplicater::default(),
            // Temporary storage.
            temp_bounds: Vec::new(),
            temp_intervals: Vec::new(),
            temp_nodes: Vec::with_capacity(tree.len()),
            compile_cache: cache,
        }
    }

    pub fn current_depth(&self) -> usize {
        self.interval_indices.len()
    }

    fn compact_temp_nodes(&mut self) -> Result<(), Error> {
        // Prune unused nodes, and deduplicate.
        let num_nodes = self.temp_nodes.len();
        let root_indices = self.pruner.run_from_range(
            &mut self.temp_nodes,
            (num_nodes - self.num_roots)..num_nodes,
        )?;
        debug_assert_eq!(root_indices.len(), self.num_roots);
        debug_assert_eq!(root_indices.start, self.temp_nodes.len() - self.num_roots);
        self.deduper.run(&mut self.temp_nodes)?;
        let num_nodes = self.temp_nodes.len();
        let root_indices = self.pruner.run_from_range(
            &mut self.temp_nodes,
            (num_nodes - self.num_roots)..num_nodes,
        )?;
        debug_assert_eq!(root_indices.len(), self.num_roots);
        debug_assert_eq!(root_indices.start, self.temp_nodes.len() - self.num_roots);
        Ok(())
    }

    fn push_impl(&mut self, index: usize) -> PruningState {
        debug_assert!(index < self.intervals_per_level);
        // Split the current intervals into 'divisions' and pick the first child.
        self.temp_bounds.clear();
        match match self.bounds.last_slice() {
            Some(last) => {
                debug_assert_eq!(last.len(), self.divisions.len()); // This is required for proper indexing.
                last.iter().zip(self.divisions.iter()).try_fold(
                    index,
                    |index, ((label, val), (divlabel, div))| {
                        debug_assert_eq!(divlabel, label); // Ensure the labels aren't somehow out of order.
                        let inext = index / div;
                        let icurr = index % div;
                        self.temp_bounds.push((
                            *label,
                            match val {
                                Interval::Scalar(ii) => {
                                    let span = ii.wid() / (*div as f64);
                                    let lo = ii.inf() + (span * icurr as f64);
                                    let hi = lo + span;
                                    match inari::Interval::try_from((lo, hi)) {
                                        Ok(iout) => Interval::Scalar(iout),
                                        Err(e) => {
                                            return Err(PruningError::CannotConstructInterval(e));
                                        }
                                    }
                                }
                                Interval::Bool(true, true) => Interval::Bool(true, true),
                                Interval::Bool(false, false) => Interval::Bool(false, false),
                                Interval::Bool(_, _) => Interval::Bool(false, true),
                            },
                        ));
                        Ok(inext)
                    },
                )
            }
            None => return PruningState::Failure(PruningError::UnexpectedEmptyState.into()),
        } {
            Ok(rem_index) => {
                debug_assert_eq!(rem_index, 0); // Ensure we consumed the
                // n-dimensional index fully, and that the index was not out of bounds.
            }
            Err(e) => return PruningState::Failure(e.into()),
        };
        if let Err(e) = fold_for_interval(
            match self.nodes.last_slice() {
                Some(nodes) => nodes,
                None => return PruningState::Failure(PruningError::UnexpectedEmptyState.into()),
            },
            &self.temp_bounds,
            &mut self.temp_nodes,
            &mut self.temp_intervals,
        ) {
            return PruningState::Failure(e);
        }
        match self.compact_temp_nodes() {
            Ok(roots) => roots,
            Err(e) => return PruningState::Failure(e),
        };
        // Copy the output intervals.
        self.interval_outputs
            .push_slice(&self.temp_intervals[(self.temp_intervals.len() - self.num_roots)..]);
        // Update the remaining members of the struct.
        self.bounds.push_iter(self.temp_bounds.drain(..));
        self.nodes.push_iter(self.temp_nodes.drain(..));
        self.interval_indices.push(index);
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

    pub fn push(&mut self) -> PruningState {
        self.push_impl(0)
    }

    pub fn push_or_pop_to(&mut self, depth: usize) -> PruningState {
        match self.current_depth() {
            d if d < depth => loop {
                match self.push() {
                    PruningState::None => break PruningState::None,
                    PruningState::Valid(dnext, i) if dnext == depth => {
                        break PruningState::Valid(dnext, i);
                    }
                    PruningState::Valid(_, _) => continue,
                    PruningState::Failure(error) => break PruningState::Failure(error),
                }
            },
            d if d > depth => loop {
                match self.pop() {
                    PruningState::None => break PruningState::None,
                    PruningState::Valid(dnext, i) if dnext == depth => {
                        break PruningState::Valid(dnext, i);
                    }
                    PruningState::Valid(_, _) => continue,
                    PruningState::Failure(error) => break PruningState::Failure(error),
                }
            },
            _ => match self.interval_indices.last() {
                // Already at the desired depth. Do nothing.
                Some(i) => PruningState::Valid(self.current_depth(), *i),
                None => PruningState::Failure(PruningError::UnexpectedEmptyState.into()),
            },
        }
    }

    fn pop_impl(&mut self) -> Option<[usize; 7]> {
        Some([
            self.nodes.pop_slice()?,
            self.ops.pop_slice()?,
            self.num_regs.pop()?,
            self.root_regs.pop_slice()?,
            self.bounds.pop_slice()?,
            self.interval_outputs.pop_slice()?,
            self.interval_indices.pop()?,
        ])
    }

    pub fn pop(&mut self) -> PruningState {
        if let None = self.pop_impl() {
            return PruningState::None;
        }
        if let Some(nregs) = self.num_regs.last() {
            self.val_regs.resize(*nregs, T::default());
            self.interval_regs.resize(*nregs, Interval::default());
        }
        match self.interval_indices.last() {
            Some(last) => PruningState::Valid(self.current_depth(), *last),
            None => PruningState::None,
        }
    }

    pub fn advance(&mut self, target_depth: Option<usize>) -> PruningState {
        let topush = loop {
            match self.pop_impl() {
                Some([.., i]) => {
                    if i + 1 < self.intervals_per_level {
                        break i + 1;
                    } else {
                        continue;
                    }
                }
                None => return PruningState::None,
            }
        };
        let (depth, index) = match self.push_impl(topush) {
            PruningState::None => return PruningState::None,
            PruningState::Valid(depth, index) => (depth, index),
            PruningState::Failure(error) => return PruningState::Failure(error),
        };
        match target_depth {
            Some(td) if depth < td => loop {
                match self.push_impl(0) {
                    PruningState::None => break PruningState::None,
                    PruningState::Valid(dnext, i) if dnext == td => {
                        break PruningState::Valid(dnext, i);
                    }
                    PruningState::Valid(_, _) => continue,
                    PruningState::Failure(error) => break PruningState::Failure(error),
                }
            },
            Some(_) | None => PruningState::Valid(depth, index),
        }
    }

    /// Set the value of a scalar variable with the given label. You'd do this
    /// for all the inputs before running the evaluator.
    pub fn set_value(&mut self, label: char, value: T) {
        self.vars.insert(label, value);
    }

    pub fn sample(&self, norm_val: &[f64], abs_val: &mut [f64]) -> Result<(), Error> {
        let bounds = self
            .bounds
            .last_slice()
            .ok_or(PruningError::UnexpectedEmptyState)?;
        if norm_val.len() != abs_val.len() {
            return Err(Error::InputSizeMismatch(norm_val.len(), abs_val.len()));
        }
        if norm_val.len() != bounds.len() {
            return Err(Error::InputSizeMismatch(norm_val.len(), bounds.len()));
        }
        for (((_label, interval), norm), out) in
            bounds.iter().zip(norm_val.iter()).zip(abs_val.iter_mut())
        {
            match interval {
                Interval::Scalar(ii) => *out = ii.inf() + norm * ii.wid(),
                Interval::Bool(_, _) => return Err(Error::TypeMismatch),
            }
        }
        Ok(())
    }

    pub fn run(&mut self) -> Result<&[T], Error> {
        for (node, out) in self
            .ops
            .last_slice()
            .ok_or(PruningError::UnexpectedEmptyState)?
        {
            self.val_regs[*out] = match node {
                Node::Constant(val) => T::from_value(*val).unwrap(),
                Node::Symbol(label) => match self.vars.get(label) {
                    Some(&v) => v,
                    None => return Err(Error::VariableNotFound(*label)),
                },
                Node::Unary(op, input) => T::unary_op(*op, self.val_regs[*input])?,
                Node::Binary(op, lhs, rhs) => {
                    T::binary_op(*op, self.val_regs[*lhs], self.val_regs[*rhs])?
                }
                Node::Ternary(op, a, b, c) => {
                    T::ternary_op(*op, self.val_regs[*a], self.val_regs[*b], self.val_regs[*c])?
                }
            };
        }
        self.val_outputs.clear();
        self.val_outputs.extend(
            self.root_regs
                .last_slice()
                .ok_or(PruningError::UnexpectedEmptyState)?
                .iter()
                .map(|r| self.val_regs[*r]),
        );
        Ok(&self.val_outputs)
    }
}

pub type ValuePruningEvaluator = PruningEvaluator<Value>;

#[cfg(test)]
mod test {
    use super::{PruningEvaluator, PruningState, ValuePruningEvaluator};
    use crate::{
        deftree,
        error::Error,
        eval::{ValueEvaluator, ValueType},
        interval::Interval,
        tree::{Tree, Value, min},
    };
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::time::Instant;

    type ImageBuffer = image::ImageBuffer<image::Luma<u8>, Vec<u8>>;

    fn circle(cx: f64, cy: f64, r: f64) -> Result<Tree, Error> {
        deftree!(- (sqrt (+ (pow (- x (const cx)) 2) (pow (- y (const cy)) 2))) (const r))
    }

    fn sphere(cx: f64, cy: f64, cz: f64, r: f64) -> Result<Tree, Error> {
        deftree!(- (sqrt (+
                          (pow (- x (const cx)) 2)
                          (+
                           (pow (- y (const cy)) 2)
                           (pow (- z (const cz)) 2))))
                 (const r))
    }

    fn num_instructions<T>(eval: &PruningEvaluator<T>) -> usize
    where
        T: ValueType,
    {
        eval.ops.last_slice().unwrap().len()
    }

    fn sample_range(range: (f64, f64), rng: &mut StdRng) -> f64 {
        range.0 + rng.random::<f64>() * (range.1 - range.0)
    }

    fn random_circles(
        xrange: (f64, f64),
        yrange: (f64, f64),
        rad_range: (f64, f64),
        num_circles: usize,
    ) -> Tree {
        let mut rng = StdRng::seed_from_u64(42);
        let mut tree = circle(
            sample_range(xrange, &mut rng),
            sample_range(yrange, &mut rng),
            sample_range(rad_range, &mut rng),
        );
        for _ in 1..num_circles {
            tree = min(
                tree,
                circle(
                    sample_range(xrange, &mut rng),
                    sample_range(yrange, &mut rng),
                    sample_range(rad_range, &mut rng),
                ),
            );
        }
        let tree = tree.unwrap();
        assert_eq!(tree.dims(), (1, 1));
        tree
    }

    #[test]
    fn t_two_circles() {
        let tree = deftree!(min {circle(0., 0., 1.)} {circle(4., 0., 1.)})
            .unwrap()
            .compacted()
            .unwrap();
        {
            let mut eval = ValuePruningEvaluator::new(
                &tree,
                4,
                [
                    ('x', (Interval::from_scalar(-1., 5.).unwrap(), 3)),
                    ('y', (Interval::from_scalar(-1., 1.).unwrap(), 3)),
                ]
                .into(),
            );
            let before = num_instructions(&eval);
            eval.push();
            let after = num_instructions(&eval);
            assert!(after < before);
        }
        {
            let mut eval = ValuePruningEvaluator::new(
                &tree,
                4,
                [
                    ('x', (Interval::from_scalar(-1., 5.).unwrap(), 3)),
                    ('y', (Interval::from_scalar(-1., 1.).unwrap(), 1)),
                ]
                .into(),
            );
            let before = num_instructions(&eval);
            eval.push();
            let after = num_instructions(&eval);
            assert!(after < before);
        }
    }

    #[test]
    fn t_two_spheres_union() {
        let tree = deftree!(min {sphere(0., 0., 0., 1.)} {sphere(4., 0., 0., 1.)})
            .unwrap()
            .compacted()
            .unwrap();
        {
            let mut eval = ValuePruningEvaluator::new(
                &tree,
                4,
                [
                    ('x', (Interval::from_scalar(-1., 5.).unwrap(), 3)),
                    ('y', (Interval::from_scalar(-1., 1.).unwrap(), 3)),
                    ('z', (Interval::from_scalar(-1., 1.).unwrap(), 3)),
                ]
                .into(),
            );
            let before = num_instructions(&eval);
            eval.push();
            let after = num_instructions(&eval);
            assert!(after < before);
        }
        {
            let mut eval = ValuePruningEvaluator::new(
                &tree,
                4,
                [
                    ('x', (Interval::from_scalar(-1., 5.).unwrap(), 3)),
                    ('y', (Interval::from_scalar(-1., 1.).unwrap(), 1)),
                    ('z', (Interval::from_scalar(-1., 1.).unwrap(), 1)),
                ]
                .into(),
            );
            let before = num_instructions(&eval);
            eval.push();
            let after = num_instructions(&eval);
            assert!(after < before);
        }
    }

    #[test]
    fn t_random_circles_push() {
        let circles = random_circles((0., 1.), (0., 1.), (0.02, 0.1), 100);
        let mut eval = ValuePruningEvaluator::new(
            &circles,
            4,
            [
                ('x', (Interval::from_scalar(0., 1.).unwrap(), 2)),
                ('y', (Interval::from_scalar(0., 1.).unwrap(), 2)),
                ('z', (Interval::from_scalar(0., 1.).unwrap(), 2)),
            ]
            .into(),
        );
        (0..7).fold(num_instructions(&eval), |prev, _| {
            eval.push();
            let current = num_instructions(&eval);
            assert!(current < prev);
            current
        });
    }

    #[test]
    fn t_perft() {
        const PRUNE_DEPTH: usize = 7;
        const DIMS: u32 = 1 << PRUNE_DEPTH;
        const DIMS_F64: f64 = DIMS as f64;
        const RAD_RANGE: (f64, f64) = (0.02 * DIMS_F64, 0.1 * DIMS_F64);
        let tree = random_circles((0., DIMS_F64), (0., DIMS_F64), RAD_RANGE, 100);
        let pruned_image = {
            let mut image = ImageBuffer::new(DIMS, DIMS);
            let mut eval = ValuePruningEvaluator::new(
                &tree,
                11,
                [
                    ('x', (Interval::from_scalar(0., DIMS_F64).unwrap(), 2)),
                    ('y', (Interval::from_scalar(0., DIMS_F64).unwrap(), 2)),
                ]
                .into(),
            );
            const NORM_SAMPLES: [[f64; 2]; 4] =
                [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]];
            let before = Instant::now();
            let mut state = eval.push_or_pop_to(PRUNE_DEPTH);
            loop {
                match state {
                    PruningState::None => break,
                    PruningState::Valid(_, _) => {} // Keep going.
                    PruningState::Failure(error) => panic!("Error during pruning: {:?}", error),
                }
                for norm in NORM_SAMPLES {
                    let mut sample = [0.; 2];
                    eval.sample(&norm, &mut sample)
                        .expect("Cannot generate sample");
                    eval.set_value('x', Value::Scalar(sample[0]));
                    eval.set_value('y', Value::Scalar(sample[1]));
                    let outputs = eval.run().expect("Failed to run the pruning evaluator");
                    assert_eq!(outputs.len(), 1);
                    let coords = sample.map(|c| c as u32);
                    *image.get_pixel_mut(coords[0], coords[1]) = match outputs[0] {
                        Value::Bool(_) => panic!("Expecting a scalar"),
                        Value::Scalar(val) => image::Luma([if val < 0. {
                            f64::min((-val / RAD_RANGE.1) * 255., 255.) as u8
                        } else {
                            f64::min(((RAD_RANGE.1 - val) / RAD_RANGE.1) * 255., 255.) as u8
                        }]),
                    };
                }
                state = eval.advance(Some(PRUNE_DEPTH));
            }
            let duration = Instant::now() - before;
            println!("Pruning evaluator took {} ms", duration.as_millis());
            image
        };
        let expected_image = {
            let mut image = ImageBuffer::new(DIMS, DIMS);
            let mut eval = ValueEvaluator::new(&tree);
            let before = Instant::now();
            for y in 0..DIMS {
                eval.set_value('y', Value::Scalar(y as f64 + 0.5));
                for x in 0..DIMS {
                    eval.set_value('x', Value::Scalar(x as f64 + 0.5));
                    let outputs = eval.run().expect("Failed to run value evaluator");
                    assert_eq!(outputs.len(), 1);
                    *image.get_pixel_mut(x, y) = match outputs[0] {
                        Value::Bool(_) => panic!("Expecting a scalar"),
                        Value::Scalar(val) => image::Luma([if val < 0. {
                            f64::min((-val / RAD_RANGE.1) * 255., 255.) as u8
                        } else {
                            f64::min(((RAD_RANGE.1 - val) / RAD_RANGE.1) * 255., 255.) as u8
                        }]),
                    };
                }
            }
            let duration = Instant::now() - before;
            println!("Value evaluator took {} ms", duration.as_millis());
            image
        };
        assert_eq!(pruned_image.width(), expected_image.width());
        assert_eq!(pruned_image.height(), expected_image.height());
        assert!(
            pruned_image
                .iter()
                .zip(expected_image.iter())
                .all(|(left, right)| left == right)
        );
    }
}
