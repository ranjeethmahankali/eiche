pub mod util {
    use crate::tree::Value;
    use crate::{eval::Evaluator, tree::Tree};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    macro_rules! assert_float_eq {
        ($a:expr, $b:expr, $eps:expr, $debug:expr) => {{
            // Make variables to avoid evaluating experssions multiple times.
            let a = $a;
            let b = $b;
            let eps = $eps;
            let error = f64::abs(a - b);
            if error > eps {
                println!("{:?}", $debug);
            }
            assert!(
                error <= eps,
                "Assertion failed: |({}) - ({})| = {:e} <= {:e}",
                a,
                b,
                error,
                eps
            );
        }};
        ($a:expr, $b:expr, $eps:expr) => {
            assert_float_eq!($a, $b, $eps, "")
        };
        ($a:expr, $b:expr) => {
            assert_float_eq!($a, $b, f64::EPSILON)
        };
    }
    pub(crate) use assert_float_eq;

    /// Helper function to evaluate the tree with randomly sampled
    /// variable values and compare the result to the one returned by
    /// the `expectedfn` for the same inputs. The values must be
    /// within `eps` of each other.
    ///
    /// Each variable is sampled within the range indicated by the
    /// corresponding entry in `vardata`. Each entry in vardata
    /// consists of the label of the symbol / variable, lower bound
    /// and upper bound.
    pub fn check_tree_eval<F>(
        tree: Tree,
        mut expectedfn: F,
        vardata: &[(char, f64, f64)],
        samples_per_var: usize,
        eps: f64,
    ) where
        F: FnMut(&[f64], &mut [f64]) -> (),
    {
        use rand::Rng;
        let mut eval = Evaluator::new(&tree);
        let mut expected = vec![f64::NAN; tree.num_roots()];
        let nvars = vardata.len();
        let mut indices = vec![0usize; nvars];
        let mut sample = Vec::<f64>::with_capacity(nvars);
        let mut rng = StdRng::seed_from_u64(42);
        while indices[0] <= samples_per_var {
            let vari = sample.len();
            let (label, lower, upper) = vardata[vari];
            let value = lower + rng.gen::<f64>() * (upper - lower);
            sample.push(value);
            eval.set_scalar(label, value);
            indices[vari] += 1;
            if vari < nvars - 1 {
                continue;
            }
            // We set all the variables. Run the test.
            let results = eval.run().unwrap();
            assert_eq!(results.len(), expected.len());
            expected.fill(f64::NAN);
            expectedfn(&sample, &mut expected);
            for (lhs, rhs) in expected.iter().zip(results.iter()) {
                match rhs {
                    Value::Bool(_) => assert!(false, "Found a boolean when expecting a scalar"),
                    Value::Scalar(rhs) => assert_float_eq!(lhs, rhs, eps),
                }
            }
            // Clean up the index stack.
            sample.pop();
            let mut vari = vari;
            while indices[vari] == samples_per_var && vari > 0 {
                if let Some(_) = sample.pop() {
                    indices[vari] = 0;
                    vari -= 1;
                } else {
                    assert!(false); // To ensure the logic of this test is correct.
                }
            }
        }
    }

    /// Compare `tree1` and `tree2` by evaluating them at randomly sampled
    /// values. The `vardata` slice is expected to contain tuples in the format
    /// (label, min, max), where the label is that of a variable in the tree,
    /// and [min, max] represents the range from which the values for that
    /// variable can be randomly sampled. Each variable will be sampled
    /// `samples_per_var` times, and the trees will be compared at all
    /// combinations of samples. That means, if the trees contain 2 variables
    /// each and `samples_per_var` is 20, then the trees will be evaluated and
    /// compared with 20 ^ 2 = 400 different samples. This test asserts that the
    /// values of the two trees not differ by more than `eps` at all the
    /// samples.
    pub fn compare_trees(
        tree1: &Tree,
        tree2: &Tree,
        vardata: &[(char, f64, f64)],
        samples_per_var: usize,
        eps: f64,
    ) {
        assert!(
            tree1.dims() == tree2.dims(),
            "Trees must have the same dimensions: {:?}, {:?}",
            tree1.dims(),
            tree2.dims()
        );
        use rand::Rng;
        let mut eval1 = Evaluator::new(&tree1);
        let mut eval2 = Evaluator::new(&tree2);
        let nvars = vardata.len();
        let mut indices = vec![0usize; nvars];
        let mut sample = Vec::<f64>::with_capacity(nvars);
        let mut rng = StdRng::seed_from_u64(42);
        while indices[0] <= samples_per_var {
            let vari = sample.len();
            let (label, lower, upper) = vardata[vari];
            let value = lower + rng.gen::<f64>() * (upper - lower);
            sample.push(value);
            eval1.set_scalar(label, value);
            eval2.set_scalar(label, value);
            indices[vari] += 1;
            if vari < nvars - 1 {
                continue;
            }
            let results1 = eval1.run().unwrap();
            let results2 = eval2.run().unwrap();
            assert_eq!(
                results1.len(),
                results2.len(),
                "The results are not of same length"
            );
            for (l, r) in results1.iter().zip(results2.iter()) {
                match (l, r) {
                    (Value::Scalar(a), Value::Scalar(b)) => assert_float_eq!(a, b, eps, sample),
                    (Value::Bool(a), Value::Bool(b)) => assert_eq!(a, b),
                    _ => assert!(false, "Mismatched types"),
                }
            }
            // Clean up the index stack.
            sample.pop();
            let mut vari = vari;
            while indices[vari] == samples_per_var && vari > 0 {
                if let Some(_) = sample.pop() {
                    indices[vari] = 0;
                    vari -= 1;
                } else {
                    assert!(false); // To ensure the logic of this test is correct.
                }
            }
        }
    }
}
