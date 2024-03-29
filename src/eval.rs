use crate::{
    error::Error,
    tree::{BinaryOp, BinaryOp::*, Node::*, TernaryOp, Tree, UnaryOp, UnaryOp::*, Value, Value::*},
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
pub struct Evaluator<'a> {
    tree: &'a Tree,
    regs: Box<[Value]>,
}

impl<'a> Evaluator<'a> {
    /// Create a new evaluator for `tree`.
    pub fn new(tree: &'a Tree) -> Evaluator {
        Evaluator {
            tree,
            regs: vec![Scalar(f64::NAN); tree.len()].into_boxed_slice(),
        }
    }

    pub fn set_scalar(&mut self, label: char, value: f64) {
        for (node, reg) in self.tree.nodes().iter().zip(self.regs.iter_mut()) {
            match node {
                Symbol(l) if *l == label => {
                    *reg = Scalar(value);
                }
                _ => {}
            }
        }
    }

    /// Write the `value` into the `index`-th register. The existing
    /// value is overwritten.
    fn write(&mut self, index: usize, value: Value) {
        self.regs[index] = value;
    }

    /// Run the evaluator and return the result. The result may
    /// contain the output value, or an
    /// error. `Variablenotfound(label)` error means the variable
    /// matching `label` hasn't been assigned a value using `set_scalar`.
    pub fn run(&mut self) -> Result<&[Value], Error> {
        for idx in 0..self.tree.len() {
            self.write(
                idx,
                match &self.tree.node(idx) {
                    Constant(val) => *val,
                    Symbol(_) => self.regs[idx],
                    Unary(op, input) => op.apply(self.regs[*input])?,
                    Binary(op, lhs, rhs) => op.apply(self.regs[*lhs], self.regs[*rhs])?,
                    Ternary(op, a, b, c) => {
                        op.apply(self.regs[*a], self.regs[*b], self.regs[*c])?
                    }
                },
            );
        }
        return Ok(&self.regs[self.tree.len() - self.tree.num_roots()..]);
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
