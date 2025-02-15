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

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Dual {
    Scalar(f64, f64),
    Boolean(bool),
}

impl ValueType for Dual {
    fn from_scalar(val: f64) -> Result<Self, Error> {
        Ok(Dual::Scalar(val, 1.))
    }

    fn from_boolean(val: bool) -> Result<Self, Error> {
        Ok(Dual::Boolean(val))
    }

    fn from_value(val: Value) -> Result<Self, Error> {
        Ok(match val {
            Value::Bool(f) => Dual::Boolean(f),
            Value::Scalar(v) => Dual::Scalar(v, 1.),
        })
    }

    fn unary_op(op: UnaryOp, val: Self) -> Result<Self, Error> {
        // f(a + bE) = f(a) + f'(a) . bE
        use Dual::*;
        Ok(match val {
            Scalar(real, dual) => match op {
                Negate => Scalar(-real, -dual),
                Sqrt => {
                    let real = f64::sqrt(real);
                    Scalar(real, 0.5 * dual / real)
                }
                Abs => {
                    let (a, b) = if real < 0. {
                        (-real, -1.)
                    } else if real == 0. {
                        (0., 0.)
                    } else {
                        (real, 1.)
                    };
                    Scalar(a, b * dual)
                }
                Sin => Scalar(f64::sin(real), f64::cos(real) * dual),
                Cos => Scalar(f64::cos(real), -f64::sin(real) * dual),
                Tan => {
                    let tx = f64::tan(real);
                    Scalar(tx, (1. + tx * tx) * dual)
                }
                Log => Scalar(f64::ln(real), dual / real),
                Exp => Scalar(f64::exp(real), f64::exp(real) * dual),
                Floor => Scalar(f64::floor(real), 0.),
                Not => return Err(Error::TypeMismatch),
            },
            Boolean(flag) => match op {
                Negate | Sqrt | Abs | Sin | Cos | Tan | Log | Exp | Floor => {
                    return Err(Error::TypeMismatch)
                }
                Not => Boolean(!flag),
            },
        })
    }

    fn binary_op(op: BinaryOp, lhs: Self, rhs: Self) -> Result<Self, Error> {
        use Dual::*;
        Ok(match (lhs, rhs) {
            (Scalar(a, b), Scalar(c, d)) => match op {
                Add => Scalar(a + c, b + d),
                Subtract => Scalar(a - c, b - d),
                Multiply => Scalar(a * c, b * c + a * d),
                Divide => Scalar(a / c, (b * c - d * a) / (c * c)),
                Pow => {
                    let ac = f64::powf(a, c);
                    Scalar(ac, c * f64::powf(a, c - 1.) * b + f64::ln(a) * ac * d)
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
                    Scalar(a - rem * c, b - rem * d)
                }
                Less => Boolean((a, b) < (c, d)),
                LessOrEqual => Boolean((a, b) <= (c, d)),
                Equal => Boolean((a, b) == (c, d)),
                NotEqual => Boolean((a, b) != (c, d)),
                Greater => Boolean((a, b) > (c, d)),
                GreaterOrEqual => Boolean((a, b) >= (c, d)),
                And | Or => return Err(Error::TypeMismatch),
            },
            (Boolean(f1), Boolean(f2)) => match op {
                Add | Subtract | Multiply | Divide | Pow | Min | Max | Remainder | Less
                | LessOrEqual | Equal | NotEqual | Greater | GreaterOrEqual => {
                    return Err(Error::TypeMismatch)
                }
                And => Boolean(f1 && f2),
                Or => Boolean(f1 || f2),
            },
            _ => return Err(Error::TypeMismatch),
        })
    }

    fn ternary_op(op: TernaryOp, a: Self, b: Self, c: Self) -> Result<Self, Error> {
        use Dual::*;
        match op {
            Choose => match (a, b, c) {
                (Boolean(f), b, c) => {
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

pub type DualEvaluator = Evaluator<Dual>;
