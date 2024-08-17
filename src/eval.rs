use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{
    compile::{compile, Instructions},
    error::Error,
    tree::{
        BinaryOp::{self, *},
        Node::{self, *},
        TernaryOp, Tree,
        UnaryOp::{self, *},
        Value,
    },
};

/// Size of a value type must be known at compile time.
pub trait ValueType: Sized + Copy {
    fn from_scalar(val: f64) -> Self;
    fn from_boolean(val: bool) -> Self;
    fn from_value(val: Value) -> Self;
    fn unary_op(op: UnaryOp, val: Self) -> Result<Self, Error>;
    fn binary_op(op: BinaryOp, lhs: Self, rhs: Self) -> Result<Self, Error>;
    fn ternary_op(op: TernaryOp, a: Self, b: Self, c: Self) -> Result<Self, Error>;
}

impl Value {
    pub fn scalar(&self) -> Result<f64, Error> {
        match self {
            Value::Scalar(val) => Ok(*val),
            Value::Bool(_) => Err(Error::TypeMismatch),
        }
    }

    pub fn boolean(&self) -> Result<bool, Error> {
        match self {
            Value::Scalar(_) => Err(Error::TypeMismatch),
            Value::Bool(val) => Ok(*val),
        }
    }
}

impl ValueType for Value {
    fn from_scalar(val: f64) -> Self {
        Value::Scalar(val)
    }

    fn from_boolean(val: bool) -> Self {
        Value::Bool(val)
    }

    fn from_value(val: Value) -> Self {
        val
    }

    fn unary_op(op: UnaryOp, value: Self) -> Result<Self, Error> {
        use Value::*;
        Ok(match op {
            // Scalar
            Negate => Scalar(-value.scalar()?),
            Sqrt => Scalar(f64::sqrt(value.scalar()?)),
            Abs => Scalar(f64::abs(value.scalar()?)),
            Sin => Scalar(f64::sin(value.scalar()?)),
            Cos => Scalar(f64::cos(value.scalar()?)),
            Tan => Scalar(f64::tan(value.scalar()?)),
            Log => Scalar(f64::ln(value.scalar()?)),
            Exp => Scalar(f64::exp(value.scalar()?)),
            // Boolean
            Not => Bool(!value.boolean()?),
        })
    }

    fn binary_op(op: BinaryOp, lhs: Self, rhs: Self) -> Result<Self, Error> {
        use Value::*;
        Ok(match op {
            // Scalar.
            Add => Scalar(lhs.scalar()? + rhs.scalar()?),
            Subtract => Scalar(lhs.scalar()? - rhs.scalar()?),
            Multiply => Scalar(lhs.scalar()? * rhs.scalar()?),
            Divide => Scalar(lhs.scalar()? / rhs.scalar()?),
            Pow => Scalar(f64::powf(lhs.scalar()?, rhs.scalar()?)),
            Min => Scalar(f64::min(lhs.scalar()?, rhs.scalar()?)),
            Max => Scalar(f64::max(lhs.scalar()?, rhs.scalar()?)),
            // Boolean.
            Less => Bool(lhs.scalar()? < rhs.scalar()?),
            LessOrEqual => Bool(lhs.scalar()? <= rhs.scalar()?),
            Equal => Bool(lhs.scalar()? == rhs.scalar()?),
            NotEqual => Bool(lhs.scalar()? != rhs.scalar()?),
            Greater => Bool(lhs.scalar()? > rhs.scalar()?),
            GreaterOrEqual => Bool(lhs.scalar()? >= rhs.scalar()?),
            And => Bool(lhs.boolean()? && rhs.boolean()?),
            Or => Bool(lhs.boolean()? || rhs.boolean()?),
        })
    }

    fn ternary_op(op: TernaryOp, a: Self, b: Self, c: Self) -> Result<Self, Error> {
        Ok(match op {
            TernaryOp::Choose => {
                if a.boolean()? {
                    b
                } else {
                    c
                }
            }
        })
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Interval {
    Scalar(f64, f64),
    Boolean(bool, bool),
}

impl Interval {
    pub fn scalar(&self) -> Result<(f64, f64), Error> {
        match self {
            Interval::Scalar(lower, upper) => Ok((*lower, *upper)),
            _ => Err(Error::TypeMismatch),
        }
    }

    pub fn boolean(&self) -> Result<(bool, bool), Error> {
        match self {
            Interval::Boolean(lower, upper) => Ok((*lower, *upper)),
            _ => Err(Error::TypeMismatch),
        }
    }

    pub fn from_boolean(lower: bool, upper: bool) -> Interval {
        if lower != upper && lower {
            Interval::Boolean(upper, lower)
        } else {
            Interval::Boolean(lower, upper)
        }
    }

    pub fn from_scalar(lower: f64, upper: f64) -> Interval {
        if lower > upper {
            Interval::Scalar(upper, lower)
        } else {
            Interval::Scalar(lower, upper)
        }
    }
}

impl ValueType for Interval {
    fn from_scalar(val: f64) -> Self {
        Interval::Scalar(val, val)
    }

    fn from_boolean(val: bool) -> Self {
        Interval::Boolean(val, val)
    }

    fn from_value(val: Value) -> Self {
        match val {
            Value::Bool(val) => Interval::Boolean(val, val),
            Value::Scalar(val) => Interval::Scalar(val, val),
        }
    }

    fn unary_op(op: UnaryOp, val: Self) -> Result<Self, Error> {
        use inari::interval;
        Ok(match val {
            Interval::Scalar(lower, upper) => {
                let it = interval!(lower, upper).map_err(|_| Error::TypeMismatch)?;
                let out = match op {
                    Negate => it.neg(),
                    Sqrt => it.sqrt(),
                    Abs => it.abs(),
                    Sin => it.sin(),
                    Cos => it.cos(),
                    Tan => it.tan(),
                    Log => it.ln(),
                    Exp => it.exp(),
                    Not => return Err(Error::TypeMismatch),
                };
                Interval::from_scalar(out.inf(), out.sup())
            }
            Interval::Boolean(lower, upper) => match op {
                Not => {
                    let (lower, upper) = match (lower, upper) {
                        (true, true) => (false, false),
                        (true, false) | (false, true) => (false, true),
                        (false, false) => (true, true),
                    };
                    Interval::from_boolean(lower, upper)
                }
                Negate | Sqrt | Abs | Sin | Cos | Tan | Log | Exp => {
                    return Err(Error::TypeMismatch)
                }
            },
        })
    }

    fn binary_op(op: BinaryOp, lhs: Self, rhs: Self) -> Result<Self, Error> {
        use {inari::interval, Interval::*};
        Ok(match (lhs, rhs) {
            (Scalar(llo, lhi), Scalar(rlo, rhi)) => {
                let lhs = interval!(llo, lhi).map_err(|_| Error::TypeMismatch)?;
                let rhs = interval!(rlo, rhi).map_err(|_| Error::TypeMismatch)?;
                let out = match op {
                    Add => lhs.add(rhs),
                    Subtract => lhs.sub(rhs),
                    Multiply => lhs.mul(rhs),
                    Divide => lhs.div(rhs),
                    Pow => lhs.pow(rhs),
                    Min => lhs.min(rhs),
                    Max => lhs.max(rhs),
                    Less => todo!(),
                    LessOrEqual => todo!(),
                    Equal => todo!(),
                    NotEqual => todo!(),
                    Greater => todo!(),
                    GreaterOrEqual => todo!(),
                    And | Or => return Err(Error::TypeMismatch),
                };
                Interval::from_scalar(out.inf(), out.sup())
            }
            (Boolean(llo, lhi), Boolean(rlo, rhi)) => match op {
                Add | Subtract | Multiply | Divide | Pow | Min | Max | Less | LessOrEqual
                | Equal | NotEqual | Greater | GreaterOrEqual => return Err(Error::TypeMismatch),
                And => todo!(),
                Or => todo!(),
            },
            _ => return Err(Error::TypeMismatch),
        })
    }

    fn ternary_op(op: TernaryOp, a: Self, b: Self, c: Self) -> Result<Self, Error> {
        use {inari::interval, Interval::*};
        match op {
            TernaryOp::Choose => Ok(match a.boolean()? {
                (true, true) => b,
                (true, false) | (false, true) => match (b, c) {
                    (Scalar(blo, bhi), Scalar(clo, chi)) => {
                        let b = interval!(blo, bhi).map_err(|_| Error::TypeMismatch)?;
                        let c = interval!(clo, chi).map_err(|_| Error::TypeMismatch)?;
                        let out = b.min(c);
                        Interval::from_scalar(out.inf(), out.sup())
                    }
                    (Scalar(_, _), Boolean(_, _)) | (Boolean(_, _), Scalar(_, _)) => {
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
                (false, false) => c,
            }),
        }
    }
}

impl PartialEq<f64> for Value {
    fn eq(&self, other: &f64) -> bool {
        match self {
            Value::Scalar(val) => val == other,
            Value::Bool(_) => false,
        }
    }
}

impl PartialEq<bool> for Value {
    fn eq(&self, other: &bool) -> bool {
        match self {
            Value::Scalar(_) => false,
            Value::Bool(flag) => flag == other,
        }
    }
}

/// This can be used to compute the value(s) of the tree.
pub struct Evaluator<T>
where
    T: ValueType,
{
    ops: Vec<(Node, usize)>,
    regs: Vec<T>,
    vars: Vec<(char, T)>,
    root_regs: Vec<usize>,
    outputs: Vec<T>,
}

impl<T> Evaluator<T>
where
    T: ValueType,
{
    /// Create a new evaluator for `tree`.
    pub fn new(tree: &Tree) -> Evaluator<T> {
        let Instructions {
            ops,
            num_regs,
            out_regs: root_regs,
        } = compile(tree);
        let num_roots = root_regs.len();
        return Evaluator {
            ops,
            regs: vec![T::from_scalar(0.); num_regs],
            vars: Vec::new(),
            root_regs,
            outputs: vec![T::from_scalar(0.); num_roots],
        };
    }

    /// Get the number of registers used by this evaluator. This is not the same
    /// as the number of nodes in the tree, because registers are allocated as
    /// needed, and reused where possible.
    pub fn num_regs(&self) -> usize {
        return self.regs.len();
    }

    /// Set the value of a scalar variable with the given label. You'd do this
    /// for all the inputs before running the evaluator.
    pub fn set_scalar(&mut self, label: char, value: f64) {
        for (l, v) in self.vars.iter_mut() {
            if *l == label {
                *v = T::from_scalar(value);
                return;
            }
        }
        self.vars.push((label, T::from_scalar(value)));
    }

    /// Run the evaluator and return the result. The result may
    /// contain the output value, or an
    /// error. `Variablenotfound(label)` error means the variable
    /// matching `label` hasn't been assigned a value using `set_scalar`.
    pub fn run(&mut self) -> Result<&[T], Error> {
        for (node, out) in &self.ops {
            self.regs[*out] = match node {
                Constant(val) => T::from_value(*val),
                Symbol(label) => match self.vars.iter().find(|(l, _v)| *l == *label) {
                    Some((_l, v)) => *v,
                    None => return Err(Error::VariableNotFound(*label)),
                },
                Unary(op, input) => T::unary_op(*op, self.regs[*input])?,
                Binary(op, lhs, rhs) => T::binary_op(*op, self.regs[*lhs], self.regs[*rhs])?,
                Ternary(op, a, b, c) => {
                    T::ternary_op(*op, self.regs[*a], self.regs[*b], self.regs[*c])?
                }
            };
        }
        self.outputs.clear();
        self.outputs
            .extend(self.root_regs.iter().map(|r| self.regs[*r]));
        return Ok(&self.outputs);
    }
}

pub type ValueEvaluator = Evaluator<Value>;

pub type IntervalEvaluator = Evaluator<Interval>;

#[cfg(test)]
mod test {
    use super::*;
    use crate::deftree;
    use crate::test::util::{assert_float_eq, check_tree_eval};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn t_constant() {
        let x = deftree!(const std::f64::consts::PI).unwrap();
        assert_eq!(x.roots(), &[Constant(Value::Scalar(std::f64::consts::PI))]);
        let mut eval = ValueEvaluator::new(&x);
        match eval.run() {
            Ok(val) => assert_eq!(val, &[Value::Scalar(std::f64::consts::PI)]),
            _ => assert!(false),
        }
    }

    #[test]
    fn t_pythagoras() {
        const TRIPLETS: [(f64, f64, f64); 6] = [
            (3., 4., 5.),
            (5., 12., 13.),
            (8., 15., 17.),
            (7., 24., 25.),
            (20., 21., 29.),
            (12., 35., 37.),
        ];
        let h = deftree!(sqrt (+ (pow x 2.) (pow y 2.))).unwrap();
        let mut eval = ValueEvaluator::new(&h);
        for (x, y, expected) in TRIPLETS {
            eval.set_scalar('x', x);
            eval.set_scalar('y', y);
            match eval.run() {
                Ok(val) => assert_eq!(val, &[expected]),
                _ => assert!(false),
            }
        }
    }

    #[test]
    fn t_trig_identity() {
        use rand::Rng;
        const PI_2: f64 = 2.0 * std::f64::consts::TAU;
        let sum = deftree!(+ (pow (sin x) 2.) (pow (cos x) 2.)).unwrap();
        let mut eval = ValueEvaluator::new(&sum);
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let x: f64 = PI_2 * rng.gen::<f64>();
            eval.set_scalar('x', x);
            match eval.run() {
                Ok(val) => {
                    assert_eq!(val.len(), 1);
                    assert_float_eq!(val[0].scalar().unwrap(), 1.);
                }
                _ => assert!(false),
            }
        }
    }

    #[test]
    fn t_sum() {
        check_tree_eval(
            deftree!(+ x y).unwrap(),
            |vars: &[f64], output: &mut [f64]| {
                if let [x, y] = vars[..] {
                    output[0] = x + y;
                }
            },
            &[('x', -5., 5.), ('y', -5., 5.)],
            10,
            0.,
        );
    }

    #[test]
    fn t_tree_1() {
        check_tree_eval(
            deftree!(/ (pow (log (+ (sin x) 2.)) 3.) (+ (cos x) 2.)).unwrap(),
            |vars: &[f64], output: &mut [f64]| {
                if let [x] = vars[..] {
                    output[0] = f64::powf(f64::ln(f64::sin(x) + 2.), 3.) / (f64::cos(x) + 2.);
                }
            },
            &[('x', -2.5, 2.5)],
            100,
            0.,
        );
    }

    #[test]
    fn t_tree_2() {
        check_tree_eval(
            deftree!(
                (max (min
                      (- (sqrt (+ (+ (pow (- x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 2.75)
                      (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 4.))
                 (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (+ y 3.) 2.)) (pow (- z 4.) 2.))) 5.25))
            )
            .unwrap(),
            |vars: &[f64], output: &mut [f64]| {
                if let [x, y, z] = vars[..] {
                    let s1 = f64::sqrt(
                        f64::powf(x - 2., 2.) + f64::powf(y - 3., 2.) + f64::powf(z - 4., 2.),
                    ) - 2.75;
                    let s2 = f64::sqrt(
                        f64::powf(x + 2., 2.) + f64::powf(y - 3., 2.) + f64::powf(z - 4., 2.),
                    ) - 4.;
                    let s3 = f64::sqrt(
                        f64::powf(x + 2., 2.) + f64::powf(y + 3., 2.) + f64::powf(z - 4., 2.),
                    ) - 5.25;
                    output[0] = f64::max(f64::min(s1, s2), s3);
                }
            },
            &[('x', -10., 10.), ('y', -9., 10.), ('z', -11., 12.)],
            20,
            1e-14,
        );
    }

    #[test]
    fn t_trees_concat_0() {
        let tree = deftree!(concat
                            (log x)
                            (+ x (pow y 2.)))
        .unwrap();
        println!("{:?}", tree.nodes());
        check_tree_eval(
            tree,
            |vars: &[f64], output: &mut [f64]| {
                if let [x, y] = vars[..] {
                    output[0] = f64::ln(x);
                    output[1] = x + f64::powf(y, 2.);
                }
            },
            &[('x', 1., 10.), ('y', 1., 10.)],
            20,
            1e-14,
        );
    }

    #[test]
    fn t_trees_concat_1() {
        check_tree_eval(
            deftree!(concat
                     (/ (pow (log (+ (sin x) 2.)) 3.) (+ (cos x) 2.))
                     (+ x y)
                     ((max (min
                            (- (sqrt (+ (+ (pow (- x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 2.75)
                            (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 4.))
                       (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (+ y 3.) 2.)) (pow (- z 4.) 2.))) 5.25))
            )).unwrap(),
            |vars: &[f64], output: &mut [f64]| {
                if let [x, y, z] = vars[..] {
                    output[0] = f64::powf(f64::ln(f64::sin(x) + 2.), 3.)
                        / (f64::cos(x) + 2.);
                    output[1] = x + y;
                    output[2] = {
                        let s1 = f64::sqrt(
                            f64::powf(x - 2., 2.) + f64::powf(y - 3., 2.) + f64::powf(z - 4., 2.),
                        ) - 2.75;
                        let s2 = f64::sqrt(
                            f64::powf(x + 2., 2.) + f64::powf(y - 3., 2.) + f64::powf(z - 4., 2.),
                        ) - 4.;
                        let s3 = f64::sqrt(
                            f64::powf(x + 2., 2.) + f64::powf(y + 3., 2.) + f64::powf(z - 4., 2.),
                        ) - 5.25;
                        f64::max(f64::min(s1, s2), s3)
                    };
                }
            },
            &[('x', -10., 10.), ('y', -9., 10.), ('z', -11., 12.)],
            20,
            1e-14,
        );
    }

    #[test]
    fn t_choose() {
        check_tree_eval(
            deftree!(if (> x 0) x (-x)).unwrap(),
            |vars: &[f64], output: &mut [f64]| {
                if let [x] = vars[..] {
                    output[0] = if x < 0. { -x } else { x };
                }
            },
            &[('x', -10., 10.)],
            100,
            0.,
        );
        check_tree_eval(
            deftree!(if (< x 0) (- x) x).unwrap(),
            |vars: &[f64], output: &mut [f64]| {
                if let [x] = vars[..] {
                    output[0] = if x < 0. { -x } else { x };
                }
            },
            &[('x', -10., 10.)],
            100,
            0.,
        );
    }
}
