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
                Remainder => Scalar(a.rem_euclid(c), map2(b, d, |b, d| b - (a / c).floor() * d)),
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
    use core::f64;

    use crate::{
        deftree,
        dual::Dual,
        eval::ValueEvaluator,
        test::{assert_float_eq, Sampler},
        tree::{Tree, Value},
    };

    use super::DualEvaluator;

    fn check_dual_eval<const DIM: usize>(tree: &Tree, vardata: &[(char, f64, f64)], eps: f64) {
        let nroots = tree.num_roots();
        let mut dual = DualEvaluator::<DIM>::new(tree);
        let mut eval = ValueEvaluator::new(tree);
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
                    // For this root, iterate over it's partial derivative w.r.t all variables.
                    // Below iterator does that by walking the row of a col-major matrix.
                    let deriv = (0..nvars).map(|vi| &result_deriv[i + nroots * vi]);
                    match (dual, val) {
                        (Dual::Scalar(real, dual), Value::Scalar(expected)) => {
                            assert_float_eq!(real, expected, eps);
                            for (d, expected) in dual.iter().zip(deriv) {
                                match expected {
                                    Value::Bool(_) => panic!("Type mismatch"),
                                    Value::Scalar(expected) => assert_float_eq!(d, expected, eps),
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
                        assert_float_eq!(real, val, eps);
                        assert_float_eq!(dual[0], deriv, eps);
                    }
                    _ => panic!("Unexpected data type"),
                }
            }
        }
    }

    #[test]
    fn t_add_sub() {
        let eps = f64::EPSILON;
        // add
        check_dual_eval::<1>(&deftree!(+ x 1).unwrap(), &[('x', -1., 1.)], eps);
        check_dual_eval::<1>(
            &deftree!(+ x y).unwrap(),
            &[('x', -1., 1.), ('y', -1., 1.)],
            eps,
        );
        check_dual_eval::<2>(
            &deftree!(+ x y).unwrap(),
            &[('x', -1., 1.), ('y', -1., 1.)],
            eps,
        );
        check_dual_eval::<3>(
            &deftree!(+ x y).unwrap(),
            &[('x', -1., 1.), ('y', -1., 1.)],
            eps,
        );
        // sub
        check_dual_eval::<1>(&deftree!(- x 1).unwrap(), &[('x', -1., 1.)], eps);
        check_dual_eval::<1>(
            &deftree!(- x y).unwrap(),
            &[('x', -1., 1.), ('y', -1., 1.)],
            eps,
        );
        check_dual_eval::<2>(
            &deftree!(- x y).unwrap(),
            &[('x', -1., 1.), ('y', -1., 1.)],
            eps,
        );
        check_dual_eval::<3>(
            &deftree!(- x y).unwrap(),
            &[('x', -1., 1.), ('y', -1., 1.)],
            eps,
        );
    }

    #[test]
    fn t_mul_div() {
        let eps = f64::EPSILON;
        check_dual_eval::<1>(
            &deftree!(* x y).unwrap(),
            &[('x', -1., 1.), ('y', -1., 1.)],
            eps,
        );
        check_dual_eval::<2>(
            &deftree!(* x y).unwrap(),
            &[('x', -1., 1.), ('y', -1., 1.)],
            eps,
        );
        check_dual_eval::<3>(
            &deftree!(* x y).unwrap(),
            &[('x', -1., 1.), ('y', -1., 1.)],
            eps,
        );
        check_dual_eval::<1>(
            &deftree!(/ x y).unwrap(),
            &[('x', -1., 1.), ('y', -1., 1.)],
            eps,
        );
        check_dual_eval::<2>(
            &deftree!(/ x y).unwrap(),
            &[('x', -1., 1.), ('y', -1., 1.)],
            eps,
        );
        check_dual_eval::<3>(
            &deftree!(/ x y).unwrap(),
            &[('x', -1., 1.), ('y', -1., 1.)],
            eps,
        );
    }

    #[test]
    fn t_polynomial() {
        check_dual_eval::<1>(&deftree!(pow x 2).unwrap(), &[('x', -10., 10.)], 1e-14);
        check_dual_eval::<2>(&deftree!(pow x 2).unwrap(), &[('x', -10., 10.)], 1e-14);
        check_dual_eval::<1>(&deftree!(pow x 3).unwrap(), &[('x', -10., 10.)], 1e-13);
        check_dual_eval::<1>(
            &deftree!(+ (* 1.5 (pow x 2)) (+ (* 2.3 x) 3.46)).unwrap(),
            &[('x', -10., 10.)],
            1e-14,
        );
        check_dual_eval::<1>(
            &deftree!(+ (* 1.2 (pow x 3)) (+ (* 2.3 (pow x 2)) (+ (* 3.4 x) 4.5))).unwrap(),
            &[('x', -10., 10.)],
            1e-13,
        );
    }

    #[test]
    fn t_gradient_2d() {
        check_dual_eval::<2>(
            &deftree!(- (+ (pow x 2) (pow y 2)) 5).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.)],
            1e-14,
        );
        check_dual_eval::<2>(
            &deftree!(+ (* 2 x) (log y)).unwrap(),
            &[('x', -10., 10.), ('y', 0.1, 10.)],
            1e-14,
        );
    }

    #[test]
    fn t_mat_2x2() {
        check_dual_eval::<2>(
            &deftree!(concat (+ (pow x 2) (pow y 2)) (+ (* 2 x) (log y))).unwrap(),
            &[('x', -10., 10.), ('y', 0.1, 10.)],
            1e-14,
        );
    }

    #[test]
    fn t_trigonometry() {
        check_dual_eval::<1>(&deftree!(pow (sin x) 2).unwrap(), &[('x', -5., 5.)], 1e-15);
        check_dual_eval::<1>(&deftree!(pow (cos x) 2).unwrap(), &[('x', -5., 5.)], 1e-15);
        check_dual_eval::<1>(&deftree!(tan x).unwrap(), &[('x', -1.5, 1.5)], 1e-14);
        check_dual_eval::<1>(&deftree!(sin (pow x 2)).unwrap(), &[('x', -2., 2.)], 1e-14);
    }

    #[test]
    fn t_sqrt() {
        check_dual_eval::<1>(&deftree!(sqrt x).unwrap(), &[('x', 0.01, 10.)], 1e-15);
        check_dual_eval::<1>(&deftree!(* x (sqrt x)).unwrap(), &[('x', 0.01, 10.)], 1e-15);
    }

    #[test]
    fn t_abs() {
        check_dual_eval::<1>(&deftree!(abs x).unwrap(), &[('x', -10., 10.)], 0.);
    }

    #[test]
    fn t_log() {
        check_dual_eval::<1>(
            &deftree!(log (pow x 2)).unwrap(),
            &[('x', 0.01, 10.)],
            1e-14,
        );
    }

    #[test]
    fn t_exp() {
        check_dual_eval::<1>(&deftree!(exp x).unwrap(), &[('x', -10., 10.)], 0.);
        check_dual_eval::<1>(&deftree!(exp (pow x 2)).unwrap(), &[('x', -4., 4.)], 1e-8);
    }

    #[test]
    fn t_min_max() {
        check_dual_eval::<1>(
            &deftree!(min x (pow x 2)).unwrap(),
            &[('x', -3., 3.)],
            1e-14,
        );
        check_dual_eval::<1>(
            &deftree!(max x (pow x 2)).unwrap(),
            &[('x', -3., 3.)],
            1e-14,
        );
    }

    #[test]
    fn t_ternary() {
        check_dual_eval::<1>(
            &deftree!(sderiv (min x (pow x 2)) x).unwrap(),
            &[('x', -3., 5.)],
            1e-15,
        );
    }

    #[test]
    fn t_floor() {
        check_dual_eval::<1>(
            &deftree!(floor (+ x y)).unwrap(),
            &[('x', -1., 1.), ('y', -1., 1.)],
            0.,
        );
    }

    #[test]
    fn t_remainder() {
        check_dual_eval::<1>(
            &deftree!(rem (+ (+ (* (pow x 2) 5) (* x 2)) 3) (* (pow x 2) 3)).unwrap(),
            &[('x', -1., 1.)],
            1e-12,
        )
    }

    #[test]
    fn t_mat_3x3() {
        check_dual_eval::<1>(
            &deftree!(- (sqrt (+ (+ (pow (- x 1) 2) (pow (- y 2) 2)) (pow (- z 3) 2))) 5.).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.), ('z', -10., 10.)],
            1e-14,
        );
        check_dual_eval::<2>(
            &deftree!(- (sqrt (+ (+ (pow (- x 1) 2) (pow (- y 2) 2)) (pow (- z 3) 2))) 5.).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.), ('z', -10., 10.)],
            1e-14,
        );
        check_dual_eval::<3>(
            &deftree!(- (sqrt (+ (+ (pow (- x 1) 2) (pow (- y 2) 2)) (pow (- z 3) 2))) 5.).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.), ('z', -10., 10.)],
            1e-14,
        );
    }
}
