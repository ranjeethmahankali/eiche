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

fn map2<const DIM: usize>(
    a: [f64; DIM],
    b: [f64; DIM],
    func: impl Fn(f64, f64) -> f64,
) -> [f64; DIM] {
    let mut out = [0.; DIM];
    for i in 0..DIM {
        out[i] = func(a[i], b[i]);
    }
    out
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Dual<const DIM: usize> {
    Scalar(f64, [f64; DIM]),
    Bool(bool),
}

impl<const DIM: usize> Dual<DIM> {
    pub fn scalar(val: f64, idx: Option<usize>) -> Self {
        Dual::Scalar(
            val,
            match idx {
                Some(idx) if idx < DIM => {
                    let mut dual = [0.; DIM];
                    dual[idx] = 1.;
                    dual
                }
                _ => [0.; DIM],
            },
        )
    }
}

impl<const DIM: usize> ValueType for Dual<DIM> {
    fn from_scalar(val: f64) -> Result<Self, Error> {
        Ok(Dual::Scalar(val, [0.; DIM]))
    }

    fn from_boolean(val: bool) -> Result<Self, Error> {
        Ok(Dual::Bool(val))
    }

    fn from_value(val: Value) -> Result<Self, Error> {
        Ok(match val {
            Value::Bool(f) => Dual::Bool(f),
            Value::Scalar(v) => Dual::Scalar(v, [0.; DIM]),
        })
    }

    fn unary_op(op: UnaryOp, val: Self) -> Result<Self, Error> {
        // f(a + bE) = f(a) + f'(a) . bE
        // f(a + bE, c + dE) = f(a, c) + (df(a, b) / da) * bE + (df(a, b) / dc) * dE
        use Dual::*;
        Ok(match val {
            Scalar(real, dual) => match op {
                Negate => Scalar(-real, dual.map(|d| -d)),
                Sqrt => {
                    let real = f64::sqrt(real);
                    Scalar(real, dual.map(|d| 0.5 * d / real))
                }
                Abs => {
                    let (a, b) = if real < 0. {
                        (-real, -1.)
                    } else if real == 0. {
                        (0., 0.)
                    } else {
                        (real, 1.)
                    };
                    Scalar(a, dual.map(|d| b * d))
                }
                Sin => Scalar(f64::sin(real), dual.map(|d| f64::cos(real) * d)),
                Cos => Scalar(f64::cos(real), dual.map(|d| -f64::sin(real) * d)),
                Tan => {
                    let tx = f64::tan(real);
                    Scalar(tx, dual.map(|d| (1. + tx * tx) * d))
                }
                Log => Scalar(f64::ln(real), dual.map(|d| d / real)),
                Exp => Scalar(f64::exp(real), dual.map(|d| f64::exp(real) * d)),
                Floor => Scalar(f64::floor(real), [0.; DIM]),
                Not => return Err(Error::TypeMismatch),
            },
            Bool(flag) => match op {
                Negate | Sqrt | Abs | Sin | Cos | Tan | Log | Exp | Floor => {
                    return Err(Error::TypeMismatch)
                }
                Not => Bool(!flag),
            },
        })
    }

    fn binary_op(op: BinaryOp, lhs: Self, rhs: Self) -> Result<Self, Error> {
        use Dual::*;
        Ok(match (lhs, rhs) {
            (Scalar(a, b), Scalar(c, d)) => match op {
                Add => Scalar(a + c, map2(b, d, |b, d| b + d)),
                Subtract => Scalar(a - c, map2(b, d, |b, d| b - d)),
                Multiply => Scalar(a * c, map2(b, d, |b, d| b * c + a * d)),
                Divide => Scalar(a / c, map2(b, d, |b, d| (b * c - d * a) / (c * c))),
                Pow => {
                    let ac = f64::powf(a, c);
                    Scalar(
                        ac,
                        map2(b, d, |b, d| {
                            c * f64::powf(a, c - 1.) * b
                                + if ac > 0. && d > 0. {
                                    f64::ln(a) * ac * d
                                } else {
                                    0.
                                }
                        }),
                    )
                }
                Min => {
                    // Rust's tuple comparison is lexicographical, which is what we want.
                    if (a, b) < (c, d) {
                        Scalar(a, b)
                    } else {
                        Scalar(c, d)
                    }
                }
                Max => {
                    // Rust's tuple comparison is lexicographical, which is what we want.
                    if (a, b) > (c, d) {
                        Scalar(a, b)
                    } else {
                        Scalar(c, d)
                    }
                }
                Remainder => {
                    let rem = a.rem_euclid(c);
                    Scalar(a - rem * c, map2(b, d, |b, d| b - rem * d))
                }
                Less => Bool((a, b) < (c, d)),
                LessOrEqual => Bool((a, b) <= (c, d)),
                Equal => Bool((a, b) == (c, d)),
                NotEqual => Bool((a, b) != (c, d)),
                Greater => Bool((a, b) > (c, d)),
                GreaterOrEqual => Bool((a, b) >= (c, d)),
                And | Or => return Err(Error::TypeMismatch),
            },
            (Bool(f1), Bool(f2)) => match op {
                Add | Subtract | Multiply | Divide | Pow | Min | Max | Remainder | Less
                | LessOrEqual | Equal | NotEqual | Greater | GreaterOrEqual => {
                    return Err(Error::TypeMismatch)
                }
                And => Bool(f1 && f2),
                Or => Bool(f1 || f2),
            },
            _ => return Err(Error::TypeMismatch),
        })
    }

    fn ternary_op(op: TernaryOp, a: Self, b: Self, c: Self) -> Result<Self, Error> {
        use Dual::*;
        match op {
            Choose => match (a, b, c) {
                (Bool(f), b, c) => {
                    if f {
                        Ok(b)
                    } else {
                        Ok(c)
                    }
                }
                _ => Err(Error::TypeMismatch),
            },
        }
    }
}

pub type DualEvaluator<const DIM: usize> = Evaluator<Dual<DIM>>;

#[cfg(test)]
mod test {
    use crate::{
        deftree,
        dual::Dual,
        eval::ValueEvaluator,
        test::{assert_float_eq, Sampler},
        tree::{Tree, Value},
    };

    use super::DualEvaluator;

    fn check_dual_eval<const DIM: usize>(tree: &Tree, vardata: &[(char, f64, f64)]) {
        let nroots = tree.num_roots();
        let mut dual = DualEvaluator::<DIM>::new(&tree);
        let mut eval = ValueEvaluator::new(&tree);
        let varstr: String = vardata.iter().map(|(c, ..)| *c).collect();
        let nvars = varstr.len();
        let deriv = tree.symbolic_deriv(&varstr).unwrap();
        let mut deval = ValueEvaluator::new(&deriv);
        let mut sampler = Sampler::new(vardata, 10, 42);
        let symbols: Vec<_> = vardata.iter().map(|(c, ..)| *c).collect();
        while let Some(sample) = sampler.next() {
            for (i, (&label, &value)) in symbols.iter().zip(sample.iter()).enumerate() {
                dual.set_value(label, Dual::scalar(value, Some(i)));
                eval.set_value(label, value.into());
                deval.set_value(label, value.into());
            }
            let result_dual = dual.run().unwrap();
            let result_val = eval.run().unwrap();
            let result_deriv = deval.run().unwrap();
            assert_eq!(result_val.len(), nroots);
            assert_eq!(result_dual.len(), nroots);
            assert_eq!(result_deriv.len(), nvars * nroots);
            for (dual, (val, deriv)) in result_dual
                .iter()
                .zip(result_val.iter().zip(result_deriv.iter()))
            {
                for i in 0..nroots {
                    let dual = result_dual[i];
                    let val = result_val[i];
                    let deriv = &result_deriv[(i * nvars)..((i + 1) * nvars)];
                    // Compare the values.
                    match (dual, val) {
                        (Dual::Scalar(real, dual), Value::Scalar(expected)) => {
                            assert_float_eq!(real, expected);
                            for (d, expected) in dual.iter().zip(deriv.iter()) {
                                match expected {
                                    Value::Bool(_) => panic!("Type mismatch"),
                                    Value::Scalar(expected) => assert_float_eq!(d, expected),
                                }
                            }
                        }
                        (Dual::Bool(flag), Value::Bool(expected)) => {
                            assert_eq!(flag, expected)
                        }
                        _ => panic!("Datatype mismatch"),
                    }
                }
                match (dual, val, deriv) {
                    (Dual::Scalar(real, dual), Value::Scalar(val), Value::Scalar(deriv)) => {
                        assert_float_eq!(real, val);
                        assert_float_eq!(dual[0], deriv);
                    }
                    _ => panic!("Unexpected data type"),
                }
            }
        }
    }

    #[test]
    fn t_add_sub() {
        // add
        check_dual_eval::<1>(&deftree!(+ x 1).unwrap(), &[('x', -1., 1.)]);
        check_dual_eval::<1>(&deftree!(+ x y).unwrap(), &[('x', -1., 1.), ('y', -1., 1.)]);
        check_dual_eval::<2>(&deftree!(+ x y).unwrap(), &[('x', -1., 1.), ('y', -1., 1.)]);
        check_dual_eval::<3>(&deftree!(+ x y).unwrap(), &[('x', -1., 1.), ('y', -1., 1.)]);
        // sub
        check_dual_eval::<1>(&deftree!(- x 1).unwrap(), &[('x', -1., 1.)]);
        check_dual_eval::<1>(&deftree!(- x y).unwrap(), &[('x', -1., 1.), ('y', -1., 1.)]);
        check_dual_eval::<2>(&deftree!(- x y).unwrap(), &[('x', -1., 1.), ('y', -1., 1.)]);
        check_dual_eval::<3>(&deftree!(- x y).unwrap(), &[('x', -1., 1.), ('y', -1., 1.)]);
    }

    #[test]
    fn t_mul_div() {
        check_dual_eval::<1>(&deftree!(* x y).unwrap(), &[('x', -1., 1.), ('y', -1., 1.)]);
        check_dual_eval::<2>(&deftree!(* x y).unwrap(), &[('x', -1., 1.), ('y', -1., 1.)]);
        check_dual_eval::<3>(&deftree!(* x y).unwrap(), &[('x', -1., 1.), ('y', -1., 1.)]);
        check_dual_eval::<1>(&deftree!(/ x y).unwrap(), &[('x', -1., 1.), ('y', -1., 1.)]);
        check_dual_eval::<2>(&deftree!(/ x y).unwrap(), &[('x', -1., 1.), ('y', -1., 1.)]);
        check_dual_eval::<3>(&deftree!(/ x y).unwrap(), &[('x', -1., 1.), ('y', -1., 1.)]);
    }
}
