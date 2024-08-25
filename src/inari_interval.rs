use crate::{
    error::Error,
    eval::{Evaluator, ValueType},
    tree::{BinaryOp, BinaryOp::*, TernaryOp, UnaryOp, UnaryOp::*, Value},
};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Interval {
    Scalar(inari::Interval),
    Boolean(bool, bool),
}

impl Interval {
    pub fn scalar(&self) -> Result<inari::Interval, Error> {
        match self {
            Interval::Scalar(it) => Ok(*it),
            _ => Err(Error::TypeMismatch),
        }
    }

    pub fn boolean(&self) -> Result<(bool, bool), Error> {
        match self {
            Interval::Boolean(lower, upper) => Ok((*lower, *upper)),
            _ => Err(Error::TypeMismatch),
        }
    }

    pub fn from_boolean(lower: bool, upper: bool) -> Result<Interval, Error> {
        Ok(if lower != upper && lower {
            Interval::Boolean(upper, lower)
        } else {
            Interval::Boolean(lower, upper)
        })
    }

    pub fn from_scalar(mut lower: f64, mut upper: f64) -> Result<Interval, Error> {
        if upper < lower {
            (lower, upper) = (upper, lower);
        }
        Ok(Interval::Scalar(
            inari::interval!(lower, upper).map_err(|_| Error::InvalidInterval)?,
        ))
    }
}

impl ValueType for Interval {
    fn from_scalar(val: f64) -> Result<Self, Error> {
        Interval::from_scalar(val, val)
    }

    fn from_boolean(val: bool) -> Result<Self, Error> {
        Ok(Interval::Boolean(val, val))
    }

    fn from_value(val: Value) -> Result<Self, Error> {
        match val {
            Value::Bool(val) => Ok(Interval::Boolean(val, val)),
            Value::Scalar(val) => Interval::from_scalar(val, val),
        }
    }

    fn unary_op(op: UnaryOp, val: Self) -> Result<Self, Error> {
        Ok(match val {
            Interval::Scalar(it) => Interval::Scalar(match op {
                Negate => it.neg(),
                Sqrt => it.sqrt(),
                Abs => it.abs(),
                Sin => it.sin(),
                Cos => it.cos(),
                Tan => it.tan(),
                Log => it.ln(),
                Exp => it.exp(),
                Floor => it.floor(),
                Not => return Err(Error::TypeMismatch),
            }),
            Interval::Boolean(lower, upper) => match op {
                Not => {
                    let (lower, upper) = match (lower, upper) {
                        (true, true) => (false, false),
                        (true, false) | (false, true) => (false, true),
                        (false, false) => (true, true),
                    };
                    Interval::from_boolean(lower, upper)?
                }
                Negate | Sqrt | Abs | Sin | Cos | Tan | Log | Exp | Floor => {
                    return Err(Error::TypeMismatch)
                }
            },
        })
    }

    fn binary_op(op: BinaryOp, lhs: Self, rhs: Self) -> Result<Self, Error> {
        use {inari::Overlap::*, Interval::*};
        match (lhs, rhs) {
            (Scalar(lhs), Scalar(rhs)) => match op {
                Add => Ok(Interval::Scalar(lhs.add(rhs))),
                Subtract => Ok(Interval::Scalar(lhs.sub(rhs))),
                Multiply => Ok(Interval::Scalar(lhs.mul(rhs))),
                Divide => Ok(Interval::Scalar(lhs.div(rhs))),
                Pow => Ok({
                    if rhs.is_singleton() && rhs.inf() == 2. {
                        // Special case for squaring to get tighter intervals.
                        Interval::Scalar(lhs.sqr())
                    } else {
                        Interval::Scalar(lhs.pow(rhs))
                    }
                }),
                Min => Ok(Interval::Scalar(lhs.min(rhs))),
                Max => Ok(Interval::Scalar(lhs.max(rhs))),
                Remainder => Ok(Interval::Scalar(lhs.sub(lhs.div(rhs).floor()))),
                Less => {
                    let (lo, hi) = if lhs.strict_precedes(rhs) {
                        (true, true)
                    } else if rhs.strict_precedes(lhs) {
                        (false, false)
                    } else {
                        (false, true)
                    };
                    Interval::from_boolean(lo, hi)
                }
                LessOrEqual => {
                    let (lo, hi) = if lhs.precedes(rhs) {
                        (true, true)
                    } else if rhs.strict_precedes(lhs) {
                        (false, false)
                    } else {
                        (false, true)
                    };
                    Interval::from_boolean(lo, hi)
                }
                Equal => {
                    let (lo, hi) = match lhs.overlap(rhs) {
                        BothEmpty => (true, true),
                        FirstEmpty | SecondEmpty | Before | After => (false, false),
                        Meets | Overlaps | Starts | ContainedBy | Finishes | StartedBy
                        | FinishedBy | OverlappedBy | Contains | MetBy => (false, true),
                        Equals => {
                            if lhs.is_singleton() {
                                (true, true)
                            } else {
                                (false, true)
                            }
                        }
                    };
                    Interval::from_boolean(lo, hi)
                }
                NotEqual => {
                    let (lo, hi) = match lhs.overlap(rhs) {
                        BothEmpty => (false, false),
                        FirstEmpty | SecondEmpty | Before | After => (true, true),
                        Meets | Overlaps | Starts | ContainedBy | Finishes | FinishedBy
                        | Contains | StartedBy | OverlappedBy | MetBy => (false, true),
                        Equals => {
                            if lhs.is_singleton() {
                                (false, false)
                            } else {
                                (false, true)
                            }
                        }
                    };
                    Interval::from_boolean(lo, hi)
                }
                Greater => {
                    let (lo, hi) = if rhs.strict_precedes(lhs) {
                        (true, true)
                    } else if lhs.strict_precedes(rhs) {
                        (false, false)
                    } else {
                        (false, true)
                    };
                    Interval::from_boolean(lo, hi)
                }
                GreaterOrEqual => {
                    let (lo, hi) = if rhs.precedes(lhs) {
                        (true, true)
                    } else if lhs.strict_precedes(rhs) {
                        (false, false)
                    } else {
                        (false, true)
                    };
                    Interval::from_boolean(lo, hi)
                }
                And | Or => return Err(Error::TypeMismatch),
            },
            (Boolean(llo, lhi), Boolean(rlo, rhi)) => match op {
                Add | Subtract | Multiply | Divide | Pow | Min | Max | Remainder | Less
                | LessOrEqual | Equal | NotEqual | Greater | GreaterOrEqual => {
                    return Err(Error::TypeMismatch)
                }
                And => {
                    let (lo, hi) = match (llo, lhi, rlo, rhi) {
                        (true, true, true, true) => (true, true),
                        (_, _, false, false) | (false, false, _, _) => (false, false),
                        _ => (false, true),
                    };
                    Interval::from_boolean(lo, hi)
                }
                Or => {
                    let (lo, hi) = match (llo, lhi, rlo, rhi) {
                        (false, false, false, false) => (false, false),
                        (_, _, true, true) | (true, true, _, _) => (true, true),
                        _ => (false, true),
                    };
                    Interval::from_boolean(lo, hi)
                }
            },
            _ => return Err(Error::TypeMismatch),
        }
    }

    fn ternary_op(op: TernaryOp, a: Self, b: Self, c: Self) -> Result<Self, Error> {
        use Interval::*;
        use TernaryOp::*;
        match op {
            Choose => match a.boolean()? {
                (true, true) => Ok(b),
                (true, false) | (false, true) => match (b, c) {
                    (Scalar(b), Scalar(c)) => Interval::from_scalar(
                        f64::min(b.inf(), c.inf()),
                        f64::max(b.sup(), c.sup()),
                    ),
                    (Scalar(_), Boolean(_, _)) | (Boolean(_, _), Scalar(_)) => {
                        return Err(Error::TypeMismatch)
                    }
                    (Boolean(blo, bhi), Boolean(clo, chi)) => {
                        if blo == bhi && blo == clo && blo == chi {
                            Interval::from_boolean(false, true)
                        } else {
                            Interval::from_boolean(blo, bhi)
                        }
                    }
                },
                (false, false) => Ok(c),
            },
            MulAdd => Ok(Interval::Scalar(
                a.scalar()?.mul_add(b.scalar()?, c.scalar()?),
            )),
        }
    }
}

impl From<inari::Interval> for Interval {
    fn from(value: inari::Interval) -> Self {
        Interval::Scalar(value)
    }
}

pub type IntervalEvaluator = Evaluator<Interval>;

#[cfg(test)]
mod test {
    use super::*;
    use crate::{deftree, eval::ValueEvaluator, test::assert_float_eq, test::Sampler, tree::Tree};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    /**
    Helper function to check interval evaluations by evaluating the given
    tree. `vardata` is expected to contain a list of variables and the lower and
    upper bounds defining the range in which those variables can be sampled during
    testing. In essence, `vardata` defines one large interval in which to sample the
    tree.

    This function will sample many sub intervals within this large interval and
    ensure that the output intervals of the sub-intervals are subsets of the output
    interval of the large interval. This function samples many values in this
    interval and ensure the values are contained in the output interval of the
    sub-interval that contains the sample. All this is just to ensure the accuracy
    of the interval evaluations.

    `samples_per_var` defines the number of values to be sampled per variable. So
    the tree will be evaluated a total of `pow(samples_per_var, vardata.len())`
    times. `intervals_per_var` defines the number of sub intervals to sample per
    variable. So the treee will be evaluated for a total of `pow(intervals_per_var,
    vardata.len())` number of sub intervals.
    */
    pub fn check_interval_eval(
        tree: Tree,
        vardata: &[(char, f64, f64)],
        samples_per_var: usize,
        intervals_per_var: usize,
    ) {
        let num_roots = tree.num_roots();
        let mut eval = ValueEvaluator::new(&tree);
        let mut ieval = IntervalEvaluator::new(&tree);
        // Evaluate the full interval and get the range of output values of the tree.
        let total_range: Vec<inari::Interval> = {
            for &(label, lower, upper) in vardata {
                ieval.set_value(label, inari::interval!(lower, upper).unwrap().into());
            }
            ieval
                .run()
                .unwrap()
                .iter()
                .map(|val| {
                    let iout = val.scalar().unwrap();
                    assert!(iout.is_common_interval());
                    iout
                })
                .collect()
        };
        assert_eq!(total_range.len(), num_roots);
        let mut sampler = Sampler::new(vardata, samples_per_var, 42);
        // When we compute a sub-interval, we will cache the results here.
        let mut computed_intervals =
            vec![inari::Interval::EMPTY; intervals_per_var.pow(vardata.len() as u32) * num_roots];
        // Flags for whether or not a sub-interval is already computed and cached.
        let mut computed = vec![false; intervals_per_var.pow(vardata.len() as u32)];
        // Steps that define the sub intervals on a per variable basis.
        let steps: Vec<_> = vardata
            .iter()
            .map(|(_label, lower, upper)| (upper - lower) / intervals_per_var as f64)
            .collect();
        let symbols: Vec<_> = vardata.iter().map(|(label, ..)| *label).collect();
        /*
        Sample values, evaluate them and ensure they're within the output
        interval of the sub-interval that contains the sample.
        */
        while let Some(sample) = sampler.next() {
            assert_eq!(sample.len(), vardata.len());
            /*
            Find the index of the interval that the sample belongs in, and also get
            the sub-interval that contains the sample. The index here is a flattened
            index, similar to `x + y * X_SIZE + z * X_SIZE * Y_SIZE`, but
            generalized for arbitrary dimensions. The dimensions are equal to the
            number of variables.
            */
            let (index, isample, _) = sample.iter().zip(vardata.iter()).zip(steps.iter()).fold(
                (0usize, Vec::new(), 1usize),
                |(mut idx, mut intervals, mut multiplier),
                 ((value, (_label, lower, _upper)), step)| {
                    let local_idx = f64::floor((value - lower) / step);
                    idx += (local_idx as usize) * multiplier;
                    let inf = lower + local_idx * step;
                    intervals.push(inari::interval!(inf, inf + step).unwrap());
                    multiplier *= intervals_per_var;
                    return (idx, intervals, multiplier);
                },
            );
            assert!(index < computed.len());
            // Get the interval that is expected to contain the values output by this sample.
            let expected_range = {
                let offset = index * num_roots;
                if !computed[index] {
                    // Evaluate the interval and cache it.
                    for (&label, &ivalue) in symbols.iter().zip(isample.iter()) {
                        ieval.set_value(label, ivalue.into());
                    }
                    let iresults = ieval.run().unwrap();
                    assert_eq!(iresults.len(), num_roots);
                    for i in 0..num_roots {
                        let iout = iresults[i].scalar().unwrap();
                        assert!(!iout.is_empty());
                        assert!(!iout.is_entire());
                        assert!(iout.is_common_interval());
                        assert!(iout.subset(total_range[i]));
                        computed_intervals[offset + i] = iout;
                    }
                    computed[index] = true;
                }
                &computed_intervals[offset..(offset + num_roots)]
            };
            // Evaluate the sample and ensure the output is within the interval.
            for (&label, &value) in symbols.iter().zip(sample.iter()) {
                eval.set_value(label, value.into());
            }
            let results = eval.run().unwrap();
            assert_eq!(num_roots, results.len());
            assert_eq!(results.len(), expected_range.len());
            for (range, value) in expected_range.iter().zip(results.iter()) {
                assert!(range.contains(value.scalar().unwrap()));
            }
        }
    }

    #[test]
    fn t_interval_pow() {
        let tree = deftree!(pow x 2).unwrap();
        let mut eval = IntervalEvaluator::new(&tree);
        let mut rng = StdRng::seed_from_u64(42);
        const MAX_VAL: f64 = 32.;
        for _ in 0..100 {
            let lo = MAX_VAL * rng.gen::<f64>();
            let hi = MAX_VAL * rng.gen::<f64>();
            let (outlo, outhi) = {
                let mut outlo = lo * lo;
                let mut outhi = hi * hi;
                if outhi < outlo {
                    (outlo, outhi) = (outhi, outlo);
                }
                (outlo, outhi)
            };
            eval.set_value('x', Interval::from_scalar(lo, hi).unwrap());
            let val = eval.run().unwrap();
            assert_eq!(val.len(), 1);
            let val = val[0].scalar().unwrap();
            assert_float_eq!(val.inf(), outlo, 1e-12);
            assert_float_eq!(val.sup(), outhi, 1e-12);
        }
    }

    #[test]
    fn t_interval_sum() {
        check_interval_eval(
            deftree!(+ x y).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.)],
            10,
            5,
        );
    }

    #[test]
    fn t_interval_tree_1() {
        check_interval_eval(
            deftree!(/ (pow (log (+ (sin x) 2.)) 3.) (+ (cos x) 2.)).unwrap(),
            &[('x', -2.5, 2.5)],
            100,
            10,
        );
    }

    #[test]
    fn t_interval_distance_to_point() {
        check_interval_eval(
            deftree!(sqrt (+ (pow (- x 2.) 2.) (pow (- y 3.) 2.))).unwrap(),
            &[('x', -10., 10.), ('y', -9., 10.)],
            20,
            5,
        );
    }

    #[test]
    fn t_interval_tree_2() {
        check_interval_eval(
            deftree!(
                (max (min
                      (- (sqrt (+ (+ (pow (- x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 2.75)
                      (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 4.))
                 (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (+ y 3.) 2.)) (pow (- z 4.) 2.))) 5.25))
            )
            .unwrap(),
            &[('x', -10., 10.), ('y', -9., 10.), ('z', -11., 12.)],
            20,
            5,
        );
    }

    #[test]
    fn t_interval_trees_concat_0() {
        check_interval_eval(
            deftree!(concat
                            (log x)
                            (+ x (pow y 2.)))
            .unwrap(),
            &[('x', 1., 10.), ('y', 1., 10.)],
            20,
            5,
        );
    }

    #[test]
    fn t_interval_trees_concat_1() {
        check_interval_eval(
            deftree!(concat
                     (/ (pow (log (+ (sin x) 2.)) 3.) (+ (cos x) 2.))
                     (+ x y)
                     ((max (min
                            (- (sqrt (+ (+ (pow (- x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 2.75)
                            (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 4.))
                       (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (+ y 3.) 2.)) (pow (- z 4.) 2.))) 5.25))
            )).unwrap(),
            &[('x', -10., 10.), ('y', -9., 10.), ('z', -11., 12.)],
            20,
            5
        );
    }

    #[test]
    fn t_interval_choose() {
        check_interval_eval(
            deftree!(if (> x 0) x (-x)).unwrap(),
            &[('x', -10., 10.)],
            100,
            10,
        );
        check_interval_eval(
            deftree!(if (< x 0) (- x) x).unwrap(),
            &[('x', -10., 10.)],
            100,
            10,
        );
    }

    #[test]
    fn t_remainder() {
        check_interval_eval(
            deftree!(rem (+ x y) (- x y)).unwrap(),
            &[('x', 1., 10.), ('y', 1., 10.)],
            20,
            5,
        );
    }
}
