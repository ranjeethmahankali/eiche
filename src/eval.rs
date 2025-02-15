use std::fmt::Debug;

use crate::{
    compile::{compile, Instructions},
    error::Error,
    tree::{
        BinaryOp::{self, *},
        Node::{self, *},
        TernaryOp::{self, *},
        Tree,
        UnaryOp::{self, *},
        Value,
    },
};

/// Size of a value type must be known at compile time.
pub trait ValueType: Sized + Copy + Debug {
    fn from_scalar(val: f64) -> Result<Self, Error>;
    fn from_boolean(val: bool) -> Result<Self, Error>;
    fn from_value(val: Value) -> Result<Self, Error>;
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
    fn from_scalar(val: f64) -> Result<Self, Error> {
        Ok(Value::Scalar(val))
    }

    fn from_boolean(val: bool) -> Result<Self, Error> {
        Ok(Value::Bool(val))
    }

    fn from_value(val: Value) -> Result<Self, Error> {
        Ok(val)
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
            Floor => Scalar(f64::floor(value.scalar()?)),
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
            Remainder => Scalar(lhs.scalar()?.rem_euclid(rhs.scalar()?)),
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
            Choose => {
                if a.boolean()? {
                    b
                } else {
                    c
                }
            }
        })
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
        Evaluator {
            ops,
            regs: vec![T::from_scalar(0.).unwrap(); num_regs],
            vars: Vec::new(),
            root_regs,
            outputs: vec![T::from_scalar(0.).unwrap(); num_roots],
        }
    }

    /// Get the number of registers used by this evaluator. This is not the same
    /// as the number of nodes in the tree, because registers are allocated as
    /// needed, and reused where possible.
    pub fn num_regs(&self) -> usize {
        self.regs.len()
    }

    /// Set the value of a scalar variable with the given label. You'd do this
    /// for all the inputs before running the evaluator.
    pub fn set_value(&mut self, label: char, value: T) {
        for (l, v) in self.vars.iter_mut() {
            if *l == label {
                *v = value;
                return;
            }
        }
        self.vars.push((label, value));
    }

    /// Run the evaluator and return the result. The result may
    /// contain the output value, or an
    /// error. `Variablenotfound(label)` error means the variable
    /// matching `label` hasn't been assigned a value using `set_scalar`.
    pub fn run(&mut self) -> Result<&[T], Error> {
        for (node, out) in &self.ops {
            self.regs[*out] = match node {
                Constant(val) => T::from_value(*val).unwrap(),
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
        Ok(&self.outputs)
    }
}

pub type ValueEvaluator = Evaluator<Value>;

#[cfg(test)]
mod test {
    use super::*;
    use crate::deftree;
    use crate::test::{assert_float_eq, check_value_eval};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn t_constant() {
        let x = deftree!(const std::f64::consts::PI).unwrap();
        assert_eq!(x.roots(), &[Constant(Value::Scalar(std::f64::consts::PI))]);
        let mut eval = ValueEvaluator::new(&x);
        match eval.run() {
            Ok(val) => assert_eq!(val, &[Value::Scalar(std::f64::consts::PI)]),
            _ => panic!(),
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
            eval.set_value('x', x.into());
            eval.set_value('y', y.into());
            match eval.run() {
                Ok(val) => assert_eq!(val, &[expected]),
                _ => panic!(),
            }
        }
    }

    #[test]
    fn t_trig_identity() {
        const PI_2: f64 = std::f64::consts::TAU;
        let sum = deftree!(+ (pow (sin x) 2.) (pow (cos x) 2.)).unwrap();
        let mut eval = ValueEvaluator::new(&sum);
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let x: f64 = PI_2 * rng.random::<f64>();
            eval.set_value('x', x.into());
            let val = eval.run().unwrap();
            assert_eq!(val.len(), 1);
            assert_float_eq!(val[0].scalar().unwrap(), 1.);
        }
    }

    #[test]
    fn t_sum() {
        check_value_eval(
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
        check_value_eval(
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
        check_value_eval(
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
        check_value_eval(
            deftree!(concat
                            (log x)
                            (+ x (pow y 2.)))
            .unwrap(),
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
        check_value_eval(
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
        check_value_eval(
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
        check_value_eval(
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

    #[test]
    fn t_floor() {
        check_value_eval(
            deftree!(floor (log x)).unwrap(),
            |vars: &[f64], output: &mut [f64]| {
                if let [x] = vars[..] {
                    output[0] = f64::ln(x).floor();
                }
            },
            &[('x', 0.1, 10.)],
            100,
            0.,
        );
    }

    #[test]
    fn t_remainder() {
        check_value_eval(
            deftree!(rem (+ (+ (* (pow x 2) 5) (* x 2)) 3) (* (pow x 2) 3)).unwrap(),
            |vars: &[f64], output: &mut [f64]| {
                if let [x] = vars[..] {
                    output[0] = (5. * x * x + 2. * x + 3.).rem_euclid(3. * x * x);
                }
            },
            &[('x', -1., 1.)],
            100,
            1e-14,
        );
    }

    #[test]
    fn t_bug_repro() {
        let tree = deftree!(concat 1 (const -1.))
            .unwrap()
            .reshape(1, 2)
            .unwrap();
        let mut eval = ValueEvaluator::new(&tree);
        let output = eval.run().unwrap();
        assert_eq!(&[Value::Scalar(1.), Value::Scalar(-1.)], output);
    }
}
