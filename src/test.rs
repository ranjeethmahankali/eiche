use crate::{
    assert_float_eq,
    eval::ValueEvaluator,
    tree::{Tree, Value},
};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Helper for sampling multiple variables at once.
pub(crate) struct Sampler {
    samples_per_var: usize,
    var_samples: Vec<f64>,
    sample: Vec<f64>,
    counter: Vec<usize>,
    done: bool,
}

impl Sampler {
    /**
    Create a sampler for all the variables. `vardata` should contain a
    tuple of (variable label, lower bound, upper bound). The variables
    are sampled between the bounds, `samples_per_var` times.
    */
    pub fn new(vardata: &[(char, f64, f64)], samples_per_var: usize, seed: u64) -> Sampler {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut var_samples = Vec::with_capacity(vardata.len() * samples_per_var);
        for &(_label, lower, upper) in vardata {
            let span = upper - lower;
            for _ in 0..samples_per_var {
                var_samples.push(lower + rng.random::<f64>() * span);
            }
        }
        Sampler {
            samples_per_var,
            var_samples,
            sample: vec![f64::NAN; vardata.len()],
            counter: vec![0; vardata.len()],
            done: false,
        }
    }

    pub fn next(&mut self) -> Option<&[f64]> {
        if self.done {
            return None;
        }
        for (i, c) in self.counter.iter().enumerate() {
            self.sample[i] = self.var_samples[i * self.samples_per_var + *c];
        }
        for c in self.counter.iter_mut() {
            *c += 1;
            if *c < self.samples_per_var {
                break;
            } else {
                *c = 0;
            }
        }
        if self.counter.iter().all(|c| *c == 0) {
            self.done = true;
        }
        Some(&self.sample)
    }
}

/**
Helper function to evaluate the tree with randomly sampled variable values and
compare the result to the one returned by the `expectedfn` for the same
inputs. The values must be within `eps` of each other.

Each variable is sampled within the range indicated by the corresponding entry
in `vardata`. Each entry in vardata consists of the label of the symbol /
variable, lower bound and upper bound.
*/
pub fn check_value_eval<F>(
    tree: Tree,
    mut expectedfn: F,
    vardata: &[(char, f64, f64)],
    samples_per_var: usize,
    eps: f64,
) where
    F: FnMut(&[f64], &mut [f64]),
{
    let mut eval = ValueEvaluator::new(&tree);
    let mut sampler = Sampler::new(vardata, samples_per_var, 42);
    let mut expected = vec![f64::NAN; tree.num_roots()];
    let symbols: Vec<_> = vardata.iter().map(|(label, ..)| *label).collect();
    while let Some(sample) = sampler.next() {
        for (&label, &value) in symbols.iter().zip(sample.iter()) {
            eval.set_value(label, value.into());
        }
        let results = eval.run().unwrap();
        assert_eq!(results.len(), expected.len());
        expected.fill(f64::NAN);
        expectedfn(sample, &mut expected);
        for (lhs, rhs) in expected.iter().zip(results.iter()) {
            match rhs {
                Value::Bool(_) => panic!("Found a boolean when expecting a scalar"),
                Value::Scalar(rhs) => assert_float_eq!(lhs, rhs, eps),
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
    let mut eval1 = ValueEvaluator::new(tree1);
    let mut eval2 = ValueEvaluator::new(tree2);
    let mut sampler = Sampler::new(vardata, samples_per_var, 42);
    let symbols: Vec<_> = vardata.iter().map(|(label, ..)| *label).collect();
    while let Some(sample) = sampler.next() {
        for (&label, &value) in symbols.iter().zip(sample.iter()) {
            eval1.set_value(label, value.into());
            eval2.set_value(label, value.into());
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
                _ => panic!("Mismatched types"),
            }
        }
    }
}
