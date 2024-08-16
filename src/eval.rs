use crate::{
    compile::{compile, Instructions},
    error::Error,
    tree::{
        BinaryOp::{self, *},
        Node::{self, *},
        TernaryOp, Tree,
        UnaryOp::{self, *},
        Value::{self, *},
    },
};

impl Value {
    pub fn scalar(&self) -> Result<f64, Error> {
        match self {
            Scalar(val) => Ok(*val),
            Bool(_) => Err(Error::TypeMismatch),
        }
    }

    pub fn boolean(&self) -> Result<bool, Error> {
        match self {
            Scalar(_) => Err(Error::TypeMismatch),
            Bool(val) => Ok(*val),
        }
    }
}

#[derive(Debug, PartialEq)]
struct Interval {
    lower: Value,
    upper: Value,
}

impl Interval {
    pub fn scalar(&self) -> Result<(f64, f64), Error> {
        match (self.lower, self.upper) {
            (Scalar(lower), Scalar(upper)) => Ok((lower, upper)),
            _ => Err(Error::TypeMismatch),
        }
    }

    pub fn boolean(&self) -> Result<(bool, bool), Error> {
        match (self.lower, self.upper) {
            (Bool(lower), Bool(upper)) => Ok((lower, upper)),
            _ => Err(Error::TypeMismatch),
        }
    }
}

impl UnaryOp {
    /// Compute the result of the operation on `value`.
    pub fn apply(&self, value: Value) -> Result<Value, Error> {
        Ok(match self {
            // Scalar
            Negate => Scalar(-value.scalar()?),
            Sqrt => Scalar(f64::sqrt(value.scalar()?)),
            Abs => Scalar(f64::abs(value.scalar()?)),
            Sin => Scalar(f64::sin(value.scalar()?)),
            Cos => Scalar(f64::cos(value.scalar()?)),
            Tan => Scalar(f64::tan(value.scalar()?)),
            Log => Scalar(f64::log(value.scalar()?, std::f64::consts::E)),
            Exp => Scalar(f64::exp(value.scalar()?)),
            // Boolean
            Not => Bool(!value.boolean()?),
        })
    }
}

impl BinaryOp {
    /// Compute the result of the operation on `lhs` and `rhs`.
    pub fn apply(&self, lhs: Value, rhs: Value) -> Result<Value, Error> {
        Ok(match self {
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
}

impl TernaryOp {
    /// Compute the result of the ternary op.
    pub fn apply(&self, a: Value, b: Value, c: Value) -> Result<Value, Error> {
        Ok(match self {
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

impl PartialEq<f64> for Value {
    fn eq(&self, other: &f64) -> bool {
        match self {
            Scalar(val) => val == other,
            Bool(_) => false,
        }
    }
}

impl PartialEq<bool> for Value {
    fn eq(&self, other: &bool) -> bool {
        match self {
            Scalar(_) => false,
            Bool(flag) => flag == other,
        }
    }
}

/// This can be used to compute the value(s) of the tree.
pub struct Evaluator {
    ops: Vec<(Node, usize)>,
    regs: Vec<Value>,
    vars: Vec<(char, Value)>,
    root_regs: Vec<usize>,
    outputs: Vec<Value>,
}

impl Evaluator {
    /// Create a new evaluator for `tree`.
    pub fn new(tree: &Tree) -> Evaluator {
        let Instructions {
            ops,
            num_regs,
            out_regs: root_regs,
        } = compile(tree);
        let num_roots = root_regs.len();
        return Evaluator {
            ops,
            regs: vec![Value::Scalar(0.); num_regs],
            vars: Vec::new(),
            root_regs,
            outputs: vec![Value::Scalar(0.); num_roots],
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
                *v = Value::Scalar(value);
                return;
            }
        }
        self.vars.push((label, Value::Scalar(value)));
    }

    /// Run the evaluator and return the result. The result may
    /// contain the output value, or an
    /// error. `Variablenotfound(label)` error means the variable
    /// matching `label` hasn't been assigned a value using `set_scalar`.
    pub fn run(&mut self) -> Result<&[Value], Error> {
        for (node, out) in &self.ops {
            self.regs[*out] = match node {
                Constant(val) => *val,
                Symbol(label) => match self.vars.iter().find(|(l, _v)| *l == *label) {
                    Some((_l, v)) => *v,
                    None => return Err(Error::VariableNotFound(*label)),
                },
                Unary(op, input) => op.apply(self.regs[*input])?,
                Binary(op, lhs, rhs) => op.apply(self.regs[*lhs], self.regs[*rhs])?,
                Ternary(op, a, b, c) => op.apply(self.regs[*a], self.regs[*b], self.regs[*c])?,
            };
        }
        self.outputs.clear();
        self.outputs
            .extend(self.root_regs.iter().map(|r| self.regs[*r]));
        return Ok(&self.outputs);
    }
}

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
        assert_eq!(x.roots(), &[Constant(Scalar(std::f64::consts::PI))]);
        let mut eval = Evaluator::new(&x);
        match eval.run() {
            Ok(val) => assert_eq!(val, &[Scalar(std::f64::consts::PI)]),
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
        let mut eval = Evaluator::new(&h);
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
        let mut eval = Evaluator::new(&sum);
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
                    output[0] = f64::powf(f64::log(f64::sin(x) + 2., std::f64::consts::E), 3.)
                        / (f64::cos(x) + 2.);
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
                    output[0] = f64::log(x, std::f64::consts::E);
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
                    output[0] = f64::powf(f64::log(f64::sin(x) + 2., std::f64::consts::E), 3.)
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
