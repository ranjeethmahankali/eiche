use crate::{
    error::Error,
    eval::{Evaluator, ValueType},
    tree::{
        BinaryOp::{self, *},
        TernaryOp::{self, *},
        UnaryOp::{self, *},
        Value,
    },
};
use std::f64::consts::{FRAC_PI_2, PI};

pub mod fold;
pub mod pruning_eval;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Interval {
    Scalar(f64, f64),
    Bool(bool, bool),
}

pub enum Overlap {
    BothEmpty,
    FirstEmpty,
    SecondEmpty,
    Before,
    Meets,
    Overlaps,
    Starts,
    ContainedBy,
    Finishes,
    Matches,
    FinishedBy,
    Contains,
    StartedBy,
    OverlappedBy,
    MetBy,
    After,
}

impl Interval {
    pub fn scalar(&self) -> Result<(f64, f64), Error> {
        match self {
            Interval::Scalar(lo, hi) => Ok((*lo, *hi)),
            _ => Err(Error::TypeMismatch),
        }
    }

    pub fn boolean(&self) -> Result<(bool, bool), Error> {
        match self {
            Interval::Bool(lower, upper) => Ok((*lower, *upper)),
            _ => Err(Error::TypeMismatch),
        }
    }

    pub fn from_boolean(lower: bool, upper: bool) -> Result<Interval, Error> {
        Ok(if lower != upper && lower {
            Interval::Bool(upper, lower)
        } else {
            Interval::Bool(lower, upper)
        })
    }

    pub fn from_scalar(lower: f64, upper: f64) -> Result<Interval, Error> {
        if upper < lower && lower != f64::INFINITY && upper != f64::NEG_INFINITY {
            Ok(Interval::Scalar(upper, lower))
        } else if lower != f64::INFINITY && upper != f64::NEG_INFINITY {
            Ok(Interval::Scalar(lower, upper))
        } else {
            Err(Error::InvalidInterval)
        }
    }
}

pub fn overlap((a, b): (f64, f64), (c, d): (f64, f64)) -> Overlap {
    use Overlap::*;
    use std::cmp::Ordering::*;
    match (b < a, d < c) {
        (true, true) => BothEmpty,
        (true, false) => FirstEmpty,
        (false, true) => SecondEmpty,
        (false, false) => {
            //     |  aRc  |  aRd  |  bRc  |  bRd
            //     | < = > | < = > | < = > | < = >
            // ----+-------+-------+-------+-------
            //   B | x     | x     | x     | x
            //   M | x     | x     |   x   | x
            //   O | x     | x     |     x | x
            //   S |   x   | x     |   ? ? | x
            //  Cb |     x | x     |     x | x
            //   F |     x | ? ?   |     x |   x
            //   E |   x   | ? ?   |   ? ? |   x
            //  Fb | x     | x     |   ? ? |   x
            //   C | x     | x     |     x |     x
            //  Sb |   x   | ? ?   |     x |     x
            //  Ob |     x | x     |     x |     x
            //  Mb |     x |   x   |     x |     x
            //   A |     x |     x |     x |     x
            match (
                b.total_cmp(&d),
                a.total_cmp(&c),
                b.total_cmp(&c),
                a.total_cmp(&d),
            ) {
                (Less, Less, Less, _) => Before,
                (Less, Less, Equal, _) => Meets,
                (Less, Less, ..) => Overlaps,
                (Less, Equal, ..) => Starts,
                (Less, ..) => ContainedBy,
                (Equal, Greater, ..) => Finishes,
                (Equal, Equal, ..) => Matches,
                (Equal, ..) => FinishedBy,
                (Greater, Less, ..) => Contains,
                (Greater, Equal, ..) => StartedBy,
                (Greater, Greater, _, Less) => OverlappedBy,
                (Greater, Greater, _, Equal) => MetBy,
                (Greater, Greater, _, Greater) => After,
            }
        }
    }
}

fn mul((llo, lhi): (f64, f64), (rlo, rhi): (f64, f64)) -> (f64, f64) {
    [llo * rlo, llo * rhi, lhi * rlo, lhi * rhi]
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), current| {
            (lo.min(*current), hi.max(*current))
        })
}

fn div((llo, lhi): (f64, f64), (rlo, rhi): (f64, f64)) -> Result<(f64, f64), Error> {
    use std::cmp::Ordering::*;
    match (
        rlo.total_cmp(&0.0),
        rhi.total_cmp(&0.0),
        llo.total_cmp(&0.0),
        lhi.total_cmp(&0.0),
    ) {
        (Less, Less, _, _) | (Greater, Greater, _, _) => {
            let (lo, hi) = [llo / rlo, llo / rhi, lhi / rlo, lhi / rhi]
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), current| {
                    (lo.min(*current), hi.max(*current))
                });
            Ok((lo, hi))
        }
        (Less, Equal, _, Less | Equal) => Ok(((llo / rlo).min(lhi / rlo), f64::INFINITY)),
        (Equal, Greater, Equal | Greater, _) => Ok(((llo / rhi).min(lhi / rhi), f64::INFINITY)),
        (Less, Equal, Equal | Greater, _) => Ok((f64::NEG_INFINITY, (llo / rlo).max(lhi / rlo))),
        (Equal, Greater, _, Less) => Ok((f64::NEG_INFINITY, (llo / rhi).max(lhi / rhi))),
        (Equal, Equal, _, _) => Err(Error::InvalidInterval),
        _ => Ok((f64::NEG_INFINITY, f64::INFINITY)), // everything.
    }
}

fn intersection((llo, lhi): (f64, f64), (rlo, rhi): (f64, f64)) -> (f64, f64) {
    (llo.max(rlo), lhi.min(rhi))
}

fn precedes((llo, lhi): (f64, f64), (rlo, rhi): (f64, f64)) -> bool {
    lhi < llo || rhi < rlo || lhi <= rlo
}

fn strict_precedes((llo, lhi): (f64, f64), (rlo, rhi): (f64, f64)) -> bool {
    lhi < llo || rhi < rlo || lhi < rlo
}

impl Default for Interval {
    fn default() -> Self {
        Interval::Scalar(f64::NEG_INFINITY, f64::INFINITY)
    }
}

impl ValueType for Interval {
    fn from_scalar(val: f64) -> Result<Self, Error> {
        Interval::from_scalar(val, val)
    }

    fn from_boolean(val: bool) -> Result<Self, Error> {
        Ok(Interval::Bool(val, val))
    }

    fn from_value(val: Value) -> Result<Self, Error> {
        match val {
            Value::Bool(val) => Ok(Interval::Bool(val, val)),
            Value::Scalar(val) => Interval::from_scalar(val, val),
        }
    }

    fn unary_op(op: UnaryOp, val: Self) -> Result<Self, Error> {
        match val {
            Interval::Scalar(lo, hi) => match op {
                Negate => Interval::from_scalar(-hi, -lo),
                Sqrt if hi < 0. => Err(Error::InvalidInterval),
                Sqrt if lo < 0. => Interval::from_scalar(0.0, hi.sqrt()),
                Sqrt => Interval::from_scalar(lo.sqrt(), hi.sqrt()),
                Abs if hi <= 0. => Interval::from_scalar((-hi).next_down(), (-lo).next_up()),
                Abs if lo >= 0. => Interval::from_scalar(lo.next_down(), hi.next_up()),
                Abs => Interval::from_scalar(0.0f64.next_down(), lo.abs().max(hi.abs()).next_up()),
                Sin => {
                    let (qlo, qhi) = ((lo / FRAC_PI_2).floor(), (hi / FRAC_PI_2).floor());
                    let n = if lo == hi { 0.0 } else { qhi - qlo };
                    let q = qlo.rem_euclid(4.0);
                    if q == 0.0 && n < 1.0 || q == 3.0 && n < 2.0 {
                        // monotonically increasing
                        Ok(Interval::Scalar(lo.sin().next_down(), hi.sin().next_up()))
                    } else if q == 1.0 && n < 2.0 || q == 2.0 && n < 1.0 {
                        // monotonically decreasing
                        Ok(Interval::Scalar(hi.sin().next_down(), lo.sin().next_up()))
                    } else if q == 0.0 && n < 3.0 || q == 3.0 && n < 4.0 {
                        // increasing, then decreasing
                        Ok(Interval::Scalar(
                            lo.sin().next_down().min(hi.sin().next_down()),
                            1.0,
                        ))
                    } else if q == 1.0 && n < 4.0 || q == 2.0 && n < 3.0 {
                        // decreasing, then increasing
                        Ok(Interval::Scalar(
                            -1.0,
                            lo.sin().next_up().max(hi.sin().next_up()),
                        ))
                    } else {
                        Ok(Interval::Scalar(-1.0, 1.0))
                    }
                }
                Cos => {
                    if hi < lo {
                        Ok(Interval::Scalar(lo, hi))
                    } else {
                        let (qlo, qhi) = ((lo / PI).floor(), (hi / PI).floor());
                        let n = if lo == hi { 0.0 } else { qhi - qlo };
                        let q = if 2.0 * (qlo / 2.0).floor() == qlo {
                            0.0
                        } else {
                            1.0
                        };
                        if n == 0.0 {
                            if q == 0.0 {
                                // monotonically decreasing
                                Ok(Interval::Scalar(hi.cos().next_down(), lo.cos().next_up()))
                            } else {
                                // monotonically increasing
                                Ok(Interval::Scalar(lo.cos().next_down(), hi.cos().next_up()))
                            }
                        } else if n <= 1.0 {
                            if q == 0.0 {
                                // decreasing, then increasing
                                Ok(Interval::Scalar(
                                    -1.0,
                                    lo.cos().next_up().max(hi.cos().next_up()),
                                ))
                            } else {
                                // increasing, then decreasing
                                Ok(Interval::Scalar(
                                    lo.cos().next_down().min(hi.cos().next_down()),
                                    1.0,
                                ))
                            }
                        } else {
                            Ok(Interval::Scalar(-1.0, 1.0))
                        }
                    }
                }
                Tan => {
                    let width = hi - lo;
                    if width >= PI {
                        Interval::from_scalar(f64::NEG_INFINITY, f64::INFINITY)
                    } else {
                        let lo = (lo + FRAC_PI_2).rem_euclid(PI) - FRAC_PI_2;
                        let hi = lo + width;
                        debug_assert!(lo <= hi);
                        debug_assert!(lo >= -FRAC_PI_2);
                        debug_assert!(lo <= FRAC_PI_2);
                        if hi >= FRAC_PI_2 {
                            Interval::from_scalar(f64::NEG_INFINITY, f64::INFINITY)
                        } else {
                            Interval::from_scalar(f64::tan(lo), f64::tan(hi))
                        }
                    }
                }
                Log if hi < 0. => Err(Error::InvalidInterval),
                Log if lo < 0. => Interval::from_scalar(f64::NEG_INFINITY, hi.ln()),
                Log => Interval::from_scalar(lo.ln(), hi.ln()),
                Exp => Interval::from_scalar(lo.exp(), hi.exp()),
                Floor => Interval::from_scalar(lo.floor(), hi.floor()),
                Not => Err(Error::TypeMismatch),
            },
            Interval::Bool(lower, upper) => match op {
                Not => {
                    let (lower, upper) = match (lower, upper) {
                        (true, true) => (false, false),
                        (true, false) | (false, true) => (false, true),
                        (false, false) => (true, true),
                    };
                    Interval::from_boolean(lower, upper)
                }
                Negate | Sqrt | Abs | Sin | Cos | Tan | Log | Exp | Floor => {
                    Err(Error::TypeMismatch)
                }
            },
        }
    }

    fn binary_op(op: BinaryOp, lhs: Self, rhs: Self) -> Result<Self, Error> {
        use Interval::*;
        use std::cmp::Ordering;
        match (lhs, rhs) {
            (Scalar(llo, lhi), Scalar(rlo, rhi)) => match op {
                Add => Interval::from_scalar(llo + rlo, lhi + rhi),
                Subtract => Interval::from_scalar(llo - rhi, lhi - rlo),
                Multiply => {
                    let (lo, hi) = mul((llo, lhi), (rlo, rhi));
                    Interval::from_scalar(lo, hi)
                }
                Divide => {
                    div((llo, lhi), (rlo, rhi)).and_then(|(lo, hi)| Interval::from_scalar(lo, hi))
                }
                Pow if rlo == 2.0 && rhi == 2.0 => {
                    match (llo.total_cmp(&0.0), lhi.total_cmp(&0.0)) {
                        // Squaring
                        (Ordering::Less, Ordering::Greater)
                        | (Ordering::Greater, Ordering::Less) => Ok(Interval::Scalar(
                            0.0,
                            ((lhi * lhi).max(llo * llo)).next_up(),
                        )),
                        _ => Interval::from_scalar((llo * llo).next_down(), (lhi * lhi).next_up()),
                    }
                }
                Pow if rlo == 0.0 && rhi == 0.0 => Ok(Interval::Scalar(1.0, 1.0)),
                Pow if rlo.floor() == rlo && rlo == rhi => {
                    // Singleton integer exponent.
                    if llo.is_nan() && lhi.is_nan() {
                        Ok(Interval::Scalar(f64::NAN, f64::NAN))
                    } else {
                        let rhs = rhi.floor() as i32;
                        if rhs < 0 {
                            if llo == 0.0 && lhi == 0.0 {
                                Ok(Interval::Scalar(f64::NAN, f64::NAN))
                            } else if rhs % 2 == 0 {
                                let (lo, hi) = if lhi <= 0. {
                                    (-lhi, -llo)
                                } else if llo >= 0. {
                                    (llo, lhi)
                                } else {
                                    (0., llo.abs().max(lhi.abs()).next_up())
                                };
                                Interval::from_scalar(
                                    hi.powi(rhs).next_down(),
                                    lo.powi(rhs).next_up(),
                                )
                            } else if llo < 0.0 && lhi > 0.0 {
                                Ok(Interval::default())
                            } else {
                                Interval::from_scalar(
                                    lhi.powi(rhs).next_down(),
                                    llo.powi(rhs).next_up(),
                                )
                            }
                        } else if rhs % 2 == 0 {
                            let (lo, hi) = if lhi <= 0. {
                                (-lhi, -llo)
                            } else if llo >= 0. {
                                (llo, lhi)
                            } else {
                                (0., llo.abs().max(lhi.abs()))
                            };
                            Interval::from_scalar(lo.powi(rhs).next_down(), hi.powi(rhs).next_up())
                        } else {
                            Interval::from_scalar(
                                llo.powi(rhs).next_down(),
                                lhi.powi(rhs).next_up(),
                            )
                        }
                    }
                }
                Pow if llo.is_nan() || lhi.is_nan() || rlo.is_nan() || rhi.is_nan() => {
                    Ok(Interval::Scalar(f64::NAN, f64::NAN))
                }
                Pow => {
                    let (llo, lhi) = intersection((llo, lhi), (0.0, f64::INFINITY));
                    if rhi <= 0.0 {
                        if lhi == 0.0 {
                            Ok(Interval::Scalar(f64::NAN, f64::NAN))
                        } else if lhi < 1.0 {
                            Interval::from_scalar(
                                lhi.powf(rhi).next_down(),
                                llo.powf(rlo).next_up(),
                            )
                        } else if llo > 1.0 {
                            Interval::from_scalar(
                                lhi.powf(rlo).next_down(),
                                llo.powf(rhi).next_up(),
                            )
                        } else {
                            Interval::from_scalar(
                                lhi.powf(rlo).next_down(),
                                llo.powf(rlo).next_up(),
                            )
                        }
                    } else if rlo > 0.0 {
                        if lhi < 1.0 {
                            Interval::from_scalar(
                                llo.powf(rhi).next_down(),
                                lhi.powf(rlo).next_up(),
                            )
                        } else if llo > 1.0 {
                            Interval::from_scalar(
                                llo.powf(rlo).next_down(),
                                lhi.powf(rhi).next_up(),
                            )
                        } else {
                            Interval::from_scalar(
                                llo.powf(rhi).next_down(),
                                lhi.powf(rhi).next_up(),
                            )
                        }
                    } else {
                        Interval::from_scalar(
                            llo.powf(rhi).min(lhi.powf(rlo)).next_down(),
                            llo.powf(rlo).max(lhi.powf(rhi)).next_up(),
                        )
                    }
                }
                Min => Interval::from_scalar(rlo.min(llo), rhi.min(lhi)),
                Max => Interval::from_scalar(rlo.max(llo), rhi.max(lhi)),
                Remainder => {
                    let (mut lo, mut hi) = div((llo, lhi), (rlo, rhi))?;
                    if lo > hi {
                        std::mem::swap(&mut lo, &mut hi);
                    }
                    let (lo, hi) = mul((lo.floor(), hi.floor()), (rlo, rhi));
                    Interval::from_scalar(llo - hi, lhi - lo)
                }
                Less => {
                    let (lo, hi) = if strict_precedes((llo, lhi), (rlo, rhi)) {
                        (true, true)
                    } else if strict_precedes((rlo, rhi), (llo, lhi)) {
                        (false, false)
                    } else {
                        (false, true)
                    };
                    Interval::from_boolean(lo, hi)
                }
                LessOrEqual => {
                    let (lo, hi) = if precedes((llo, lhi), (rlo, rhi)) {
                        (true, true)
                    } else if strict_precedes((rlo, rhi), (llo, lhi)) {
                        (false, false)
                    } else {
                        (false, true)
                    };
                    Interval::from_boolean(lo, hi)
                }
                Equal => {
                    use Overlap::*;
                    let (lo, hi) = match overlap((llo, lhi), (rlo, rhi)) {
                        BothEmpty => (true, true),
                        FirstEmpty | SecondEmpty | Before | After => (false, false),
                        Meets | Overlaps | Starts | ContainedBy | Finishes | StartedBy
                        | FinishedBy | OverlappedBy | Contains | MetBy => (false, true),
                        Matches => {
                            if llo == lhi {
                                (true, true)
                            } else {
                                (false, true)
                            }
                        }
                    };
                    Interval::from_boolean(lo, hi)
                }
                NotEqual => {
                    use Overlap::*;
                    let (lo, hi) = match overlap((llo, lhi), (rlo, rhi)) {
                        BothEmpty => (false, false),
                        FirstEmpty | SecondEmpty | Before | After => (true, true),
                        Meets | Overlaps | Starts | ContainedBy | Finishes | FinishedBy
                        | Contains | StartedBy | OverlappedBy | MetBy => (false, true),
                        Matches => {
                            if llo == lhi {
                                (false, false)
                            } else {
                                (false, true)
                            }
                        }
                    };
                    Interval::from_boolean(lo, hi)
                }
                Greater => {
                    let (lo, hi) = if strict_precedes((rlo, rhi), (llo, lhi)) {
                        (true, true)
                    } else if strict_precedes((llo, lhi), (rlo, rhi)) {
                        (false, false)
                    } else {
                        (false, true)
                    };
                    Interval::from_boolean(lo, hi)
                }
                GreaterOrEqual => {
                    let (lo, hi) = if precedes((rlo, rhi), (llo, lhi)) {
                        (true, true)
                    } else if strict_precedes((llo, lhi), (rlo, rhi)) {
                        (false, false)
                    } else {
                        (false, true)
                    };
                    Interval::from_boolean(lo, hi)
                }
                And | Or => Err(Error::TypeMismatch),
            },
            (Bool(llo, lhi), Bool(rlo, rhi)) => match op {
                Add | Subtract | Multiply | Divide | Pow | Min | Max | Remainder | Less
                | LessOrEqual | Equal | NotEqual | Greater | GreaterOrEqual => {
                    Err(Error::TypeMismatch)
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
            _ => Err(Error::TypeMismatch),
        }
    }

    fn ternary_op(op: TernaryOp, a: Self, b: Self, c: Self) -> Result<Self, Error> {
        use Interval::*;
        match op {
            Choose => match a.boolean()? {
                (true, true) => Ok(b),
                (true, false) | (false, true) => match (b, c) {
                    (Scalar(blo, bhi), Scalar(clo, chi)) => {
                        Interval::from_scalar(blo.min(clo), bhi.max(chi))
                    }
                    (Scalar(_, _), Bool(_, _)) | (Bool(_, _), Scalar(_, _)) => {
                        Err(Error::TypeMismatch)
                    }
                    (Bool(blo, bhi), Bool(clo, chi)) => {
                        if blo == bhi && blo == clo && blo == chi {
                            Interval::from_boolean(false, true)
                        } else {
                            Interval::from_boolean(blo, bhi)
                        }
                    }
                },
                (false, false) => Ok(c),
            },
        }
    }
}

pub type IntervalEvaluator = Evaluator<Interval>;

#[cfg(test)]
mod test {
    use super::*;
    use crate::{ValueEvaluator, assert_float_eq, deftree, test::Sampler, tree::Tree};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    const EPS: f64 = f64::EPSILON * 2.0;

    fn is_common((lo, hi): &(f64, f64)) -> bool {
        lo.is_finite() && hi.is_finite() && lo <= hi
    }

    fn is_empty((lo, hi): &(f64, f64)) -> bool {
        lo.is_nan() && hi.is_nan()
    }

    fn is_entire((lo, hi): &(f64, f64)) -> bool {
        *lo == f64::NEG_INFINITY && *hi == f64::INFINITY
    }

    fn is_subset_of((llo, lhi): &(f64, f64), (rlo, rhi): &(f64, f64)) -> bool {
        (*llo + EPS) >= *rlo && *lhi <= (*rhi + EPS)
    }

    fn contains((lo, hi): &(f64, f64), val: f64) -> bool {
        val.is_finite() && (val + EPS) >= *lo && val <= (*hi + EPS)
    }

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
        let total_range: Vec<(f64, f64)> = {
            for &(label, lower, upper) in vardata {
                ieval.set_value(label, Interval::from_scalar(lower, upper).unwrap());
            }
            ieval
                .run()
                .unwrap()
                .iter()
                .map(|val| {
                    let (lo, hi) = val.scalar().unwrap();
                    assert!(is_common(&(lo, hi)));
                    (lo, hi)
                })
                .collect()
        };
        assert_eq!(total_range.len(), num_roots);
        let mut sampler = Sampler::new(vardata, samples_per_var, 42);
        // When we compute a sub-interval, we will cache the results here.
        let mut computed_intervals = vec![
            (f64::INFINITY, f64::NEG_INFINITY);
            intervals_per_var.pow(vardata.len() as u32) * num_roots
        ];
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
                    intervals.push(Interval::from_scalar(inf, inf + step).unwrap());
                    multiplier *= intervals_per_var;
                    (idx, intervals, multiplier)
                },
            );
            assert!(index < computed.len());
            // Get the interval that is expected to contain the values output by this sample.
            let expected_range = {
                let offset = index * num_roots;
                if !computed[index] {
                    // Evaluate the interval and cache it.
                    for (&label, &ivalue) in symbols.iter().zip(isample.iter()) {
                        ieval.set_value(label, ivalue);
                    }
                    let iresults = ieval.run().unwrap();
                    assert_eq!(iresults.len(), num_roots);
                    for i in 0..num_roots {
                        let iout = iresults[i].scalar().unwrap();
                        assert!(!is_empty(&iout));
                        assert!(!is_entire(&iout));
                        assert!(is_common(&iout));
                        assert!(is_subset_of(&iout, &total_range[i]));
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
                assert!(contains(range, value.scalar().unwrap()));
            }
        }
    }

    #[test]
    fn t_interval_pow() {
        let tree = deftree!(pow 'x 2).unwrap();
        let mut eval = IntervalEvaluator::new(&tree);
        let mut rng = StdRng::seed_from_u64(42);
        const MAX_VAL: f64 = 32.;
        for _ in 0..100 {
            let lo = MAX_VAL * rng.random::<f64>();
            let hi = MAX_VAL * rng.random::<f64>();
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
            assert_float_eq!(val.0, outlo, 1e-12);
            assert_float_eq!(val.1, outhi, 1e-12);
        }
    }

    #[test]
    fn t_interval_sum() {
        check_interval_eval(
            deftree!(+ 'x 'y).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.)],
            10,
            5,
        );
    }

    #[test]
    fn t_interval_tree_1() {
        check_interval_eval(
            deftree!(/ (pow (log (+ (sin 'x) 2.)) 3.) (+ (cos 'x) 2.)).unwrap(),
            &[('x', -2.5, 2.5)],
            100,
            10,
        );
    }

    #[test]
    fn t_interval_squaring() {
        check_interval_eval(
            deftree!(pow 'x 2.).unwrap(),
            &[('x', -10., 10.), ('y', -9., 10.)],
            20,
            5,
        );
        check_interval_eval(
            deftree!(pow (- 'x 1.) 2.).unwrap(),
            &[('x', -10., 10.)],
            20,
            5,
        );
    }

    #[test]
    fn t_interval_cube() {
        check_interval_eval(
            deftree!(pow 'x 3.).unwrap(),
            &[('x', -10., 10.), ('y', -9., 10.)],
            20,
            5,
        );
    }

    #[test]
    fn t_interval_rational_pow() {
        check_interval_eval(deftree!(pow 'x 3.15).unwrap(), &[('x', 0., 10.)], 20, 5);
        check_interval_eval(
            deftree!(pow 'x 2.4556634543).unwrap(),
            &[('x', 2.212, 8.199)],
            20,
            5,
        );
        check_interval_eval(
            deftree!(pow 'x 45.23).unwrap(),
            &[('x', 2.222222, 11.112342)],
            20,
            5,
        );
    }

    #[test]
    fn t_interval_distance_to_point() {
        check_interval_eval(
            deftree!(sqrt (+ (pow (- 'x 2.) 2.) (pow (- 'y 3.) 2.))).unwrap(),
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
                      (- (sqrt (+ (+ (pow (- 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 2.75)
                      (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 4.))
                 (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (+ 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 5.25))
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
                            (log 'x)
                            (+ 'x (pow 'y 2.)))
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
                     (/ (pow (log (+ (sin 'x) 2.)) 3.) (+ (cos 'x) 2.))
                     (+ 'x 'y)
                     ((max (min
                            (- (sqrt (+ (+ (pow (- 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 2.75)
                            (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 4.))
                       (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (+ 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 5.25))
            )).unwrap(),
            &[('x', -10., 10.), ('y', -9., 10.), ('z', -11., 12.)],
            20,
            5
        );
    }

    #[test]
    fn t_interval_choose() {
        check_interval_eval(
            deftree!(if (> 'x 0) 'x (- 'x)).unwrap(),
            &[('x', -10., 10.)],
            100,
            10,
        );
        check_interval_eval(
            deftree!(if (< 'x 0) (- 'x) 'x).unwrap(),
            &[('x', -10., 10.)],
            100,
            10,
        );
    }

    #[test]
    fn t_floor() {
        check_interval_eval(
            deftree!(floor (/ (pow 'x 2) (+ 2 (sin 'x)))).unwrap(),
            &[('x', 1., 5.)],
            100,
            10,
        );
    }

    #[test]
    fn t_remainder() {
        check_interval_eval(
            deftree!(rem (pow 'x 2) (+ 2 (sin 'x))).unwrap(),
            &[('x', 1., 5.)],
            100,
            10,
        );
    }

    #[test]
    fn t_integer_exponents_negative() {
        // Negative even exponent
        check_interval_eval(deftree!(pow 'x (- 2.)).unwrap(), &[('x', 2., 5.)], 20, 5);
        // Negative odd exponent
        check_interval_eval(deftree!(pow 'x (- 3.)).unwrap(), &[('x', 1., 3.)], 20, 5);
        // Negative exponent with interval entirely below 1
        check_interval_eval(deftree!(pow 'x (- 2.)).unwrap(), &[('x', 0.2, 0.8)], 20, 5);
    }

    #[test]
    fn t_pow_crossing_zero_negative_exp() {
        // Odd negative exponent crossing 0 should give ENTIRE
        let tree = deftree!(pow 'x (- 3.)).unwrap();
        let mut eval = IntervalEvaluator::new(&tree);
        eval.set_value('x', Interval::from_scalar(-1., 1.).unwrap());
        let result = eval.run().unwrap()[0].scalar().unwrap();
        assert!(is_entire(&result));
        // Even negative exponent crossing 0
        let tree = deftree!(pow 'x (- 2.)).unwrap();
        let mut eval = IntervalEvaluator::new(&tree);
        eval.set_value('x', Interval::from_scalar(-2., 3.).unwrap());
        let result = eval.run().unwrap()[0].scalar().unwrap();
        // Should get valid result, lower bound from max(abs), upper from min(abs) near 0
        assert!(!is_empty(&result));
    }

    #[test]
    fn t_pow_around_one() {
        // Intervals straddling 1.0 with various exponents
        check_interval_eval(deftree!(pow 'x 2.5).unwrap(), &[('x', 0.5, 1.5)], 20, 5);
        check_interval_eval(deftree!(pow 'x (- 2.5)).unwrap(), &[('x', 0.5, 1.5)], 20, 5);
    }

    #[test]
    fn t_pow_exponent_crossing_zero() {
        // Exponent interval straddling 0
        check_interval_eval(
            deftree!(pow 'x 'y).unwrap(),
            &[('x', 2., 3.), ('y', -1., 1.)],
            15,
            4,
        );
    }

    #[test]
    fn t_tan() {
        check_interval_eval(deftree!(tan 'x).unwrap(), &[('x', -1., 1.)], 50, 10);
        check_interval_eval(deftree!(tan 'x).unwrap(), &[('x', 0., 1.)], 50, 10);
    }

    #[test]
    fn t_exp() {
        check_interval_eval(deftree!(exp 'x).unwrap(), &[('x', -2., 2.)], 50, 10);
    }

    #[test]
    fn t_negate() {
        check_interval_eval(deftree!(- 'x).unwrap(), &[('x', -5., 5.)], 20, 5);
    }

    #[test]
    fn t_interval_abs() {
        // check_interval_eval(deftree!(abs 'x).unwrap(), &[('x', -5., 5.)], 20, 5);
        check_interval_eval(deftree!(abs 'x).unwrap(), &[('x', -3., -1.)], 20, 5);
    }

    #[test]
    fn t_min_max_direct() {
        check_interval_eval(
            deftree!(min 'x 'y).unwrap(),
            &[('x', -5., 5.), ('y', -3., 7.)],
            20,
            5,
        );
        check_interval_eval(
            deftree!(max 'x 'y).unwrap(),
            &[('x', -5., 5.), ('y', -3., 7.)],
            20,
            5,
        );
    }

    #[test]
    fn t_interval_comparisons() {
        check_interval_eval(
            deftree!(if (== 'x 'y) (+ 'x 2.5) (- 'y 1.523)).unwrap(),
            &[('x', 0., 5.), ('y', 3., 8.)],
            20,
            5,
        );
        check_interval_eval(
            deftree!(if (!= 'x 'y) (+ 'x 2.5) (- 'y 1.523)).unwrap(),
            &[('x', 0., 5.), ('y', 3., 8.)],
            20,
            5,
        );
        check_interval_eval(
            deftree!(if (< 'x 'y) (+ 'x 2.5) (- 'y 1.523)).unwrap(),
            &[('x', 0., 5.), ('y', 3., 8.)],
            20,
            5,
        );
        check_interval_eval(
            deftree!(if (<= 'x 'y) (+ 'x 2.5) (- 'y 1.523)).unwrap(),
            &[('x', 0., 5.), ('y', 3., 8.)],
            20,
            5,
        );
    }

    #[test]
    fn t_interval_boolean_ops() {
        check_interval_eval(
            deftree!(if (and (> 'x 0) (< 'y 5)) (- 'x 2.) (+ 'y 1.5)).unwrap(),
            &[('x', -2., 3.), ('y', 2., 7.)],
            20,
            5,
        );
        check_interval_eval(
            deftree!(if (or (> 'x 5) (< 'y 0)) (- 'x 2.) (+ 'y 1.5)).unwrap(),
            &[('x', -2., 3.), ('y', 2., 7.)],
            20,
            5,
        );
        check_interval_eval(
            deftree!(if (not (> 'x 0)) (- 'x 2.) 1.5).unwrap(),
            &[('x', -5., 5.)],
            20,
            5,
        );
    }

    #[test]
    fn t_error_conditions() {
        // Sqrt of negative interval
        let tree = deftree!(sqrt 'x).unwrap();
        let mut eval = IntervalEvaluator::new(&tree);
        eval.set_value('x', Interval::from_scalar(-5., -1.).unwrap());
        assert!(eval.run().is_err());
        // Log of negative interval
        let tree = deftree!(log 'x).unwrap();
        let mut eval = IntervalEvaluator::new(&tree);
        eval.set_value('x', Interval::from_scalar(-5., -1.).unwrap());
        assert!(eval.run().is_err());
        // Division by zero interval
        let tree = deftree!(/ 'x 0.).unwrap();
        let mut eval = IntervalEvaluator::new(&tree);
        eval.set_value('x', Interval::from_scalar(1., 2.).unwrap());
        assert!(eval.run().is_err());
        // 0^(-n) should error
        let tree = deftree!(pow 0. (- 2.)).unwrap();
        let mut eval = IntervalEvaluator::new(&tree);
        assert!(match eval.run().unwrap()[0] {
            Interval::Scalar(lo, hi) => is_empty(&(lo, hi)),
            Interval::Bool(_, _) => false,
        });
    }

    #[test]
    fn t_single_point_intervals() {
        // Single point intervals
        let tree = deftree!(+ 'x 'y).unwrap();
        let mut eval = IntervalEvaluator::new(&tree);
        eval.set_value('x', Interval::from_scalar(2., 2.).unwrap());
        eval.set_value('y', Interval::from_scalar(3., 3.).unwrap());
        let result = eval.run().unwrap()[0].scalar().unwrap();
        assert_float_eq!(result.0, 5., 1e-12);
        assert_float_eq!(result.1, 5., 1e-12);
    }
}
