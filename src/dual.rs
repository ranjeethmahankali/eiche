use crate::{
    error::Error,
    eval::ValueType,
    tree::{BinaryOp, BinaryOp::*, TernaryOp, UnaryOp, UnaryOp::*, Value},
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
                    if a < c {
                        Scalar(a, b)
                    } else {
                        Scalar(c, d)
                    }
                }
                Max => {
                    if a > c {
                        Scalar(a, b)
                    } else {
                        Scalar(c, d)
                    }
                }
                Remainder => todo!(),
                Less => todo!(),
                LessOrEqual => todo!(),
                Equal => todo!(),
                NotEqual => todo!(),
                Greater => todo!(),
                GreaterOrEqual => todo!(),
                And => todo!(),
                Or => todo!(),
            },
            (Boolean(f1), Boolean(f2)) => match op {
                Add => todo!(),
                Subtract => todo!(),
                Multiply => todo!(),
                Divide => todo!(),
                Pow => todo!(),
                Min => todo!(),
                Max => todo!(),
                Remainder => todo!(),
                Less => todo!(),
                LessOrEqual => todo!(),
                Equal => todo!(),
                NotEqual => todo!(),
                Greater => todo!(),
                GreaterOrEqual => todo!(),
                And => todo!(),
                Or => todo!(),
            },
            _ => return Err(Error::TypeMismatch),
        })
    }

    fn ternary_op(op: TernaryOp, a: Self, b: Self, c: Self) -> Result<Self, Error> {
        todo!()
    }
}
