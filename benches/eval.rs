use criterion::{Criterion, criterion_group, criterion_main};
use eiche::{Deduplicater, Error, Pruner, Tree, ValueEvaluator, deftree, min};
use rand::{SeedableRng, rngs::StdRng};
// use std::hint::black_box;

fn sample_range(range: (f64, f64), rng: &mut StdRng) -> f64 {
    use rand::Rng;
    range.0 + rng.random::<f64>() * (range.1 - range.0)
}

mod spheres {
    use super::*;

    const RADIUS_RANGE: (f64, f64) = (0.2, 2.);
    const X_RANGE: (f64, f64) = (0., 100.);
    const Y_RANGE: (f64, f64) = (0., 100.);
    const Z_RANGE: (f64, f64) = (0., 100.);
    const N_SPHERES: usize = 500;
    const N_QUERIES: usize = 1000;

    fn random_sphere_union() -> Tree {
        let mut rng = StdRng::seed_from_u64(42);
        let mut make_sphere = || -> Result<Tree, Error> {
            deftree!(- (sqrt (+ (+
                                 (pow (- x (const sample_range(X_RANGE, &mut rng))) 2)
                                 (pow (- y (const sample_range(Y_RANGE, &mut rng))) 2))
                              (pow (- z (const sample_range(Z_RANGE, &mut rng))) 2)))
                     (const sample_range(RADIUS_RANGE, &mut rng)))
        };
        let mut tree = make_sphere();
        for _ in 1..N_SPHERES {
            tree = min(tree, make_sphere());
        }
        let tree = tree.unwrap();
        assert_eq!(tree.dims(), (1, 1));
        tree
    }

    fn init_benchmark() -> (Tree, Vec<[f64; 3]>, Vec<f64>) {
        let mut rng = StdRng::seed_from_u64(234);
        (
            {
                let mut dedup = Deduplicater::new();
                let mut pruner = Pruner::new();
                random_sphere_union()
                    .fold()
                    .unwrap()
                    .deduplicate(&mut dedup)
                    .unwrap()
                    .prune(&mut pruner)
                    .unwrap()
            },
            (0..N_QUERIES)
                .map(|_| {
                    [
                        sample_range(X_RANGE, &mut rng),
                        sample_range(Y_RANGE, &mut rng),
                        sample_range(Z_RANGE, &mut rng),
                    ]
                })
                .collect(),
            Vec::with_capacity(N_QUERIES),
        )
    }

    pub mod value_eval {
        use super::*;

        fn with_compile(tree: &Tree, values: &mut Vec<f64>, queries: &[[f64; 3]]) {
            let mut eval = ValueEvaluator::new(tree);
            values.clear();
            values.extend(queries.iter().map(|coords| {
                eval.set_value('x', coords[0].into());
                eval.set_value('y', coords[1].into());
                eval.set_value('z', coords[2].into());
                let results = eval.run().unwrap();
                results[0].scalar().unwrap()
            }))
        }

        fn no_compile(eval: &mut ValueEvaluator, values: &mut Vec<f64>, queries: &[[f64; 3]]) {
            values.clear();
            values.extend(queries.iter().map(|coords| {
                eval.set_value('x', coords[0].into());
                eval.set_value('y', coords[1].into());
                eval.set_value('z', coords[2].into());
                let results = eval.run().unwrap();
                results[0].scalar().unwrap()
            }))
        }

        fn b_with_compile(c: &mut Criterion) {
            let (tree, queries, mut values) = init_benchmark();
            c.bench_function("value-evaluator-with-compilation", |b| {
                b.iter(|| {
                    with_compile(&tree, &mut values, &queries);
                    assert_eq!(values.len(), N_QUERIES);
                })
            });
        }

        fn b_no_compile(c: &mut Criterion) {
            let (tree, queries, mut values) = init_benchmark();
            let mut eval = ValueEvaluator::new(&tree);
            c.bench_function("value-evaluation-no-compilation", |b| {
                b.iter(|| {
                    no_compile(&mut eval, &mut values, &queries);
                    assert_eq!(values.len(), N_QUERIES);
                })
            });
        }

        criterion_group!(bench, b_no_compile, b_with_compile,);
    }

    #[cfg(feature = "llvm-jit")]
    pub mod jit_single {
        use super::*;
        use eiche::{JitContext, JitFn};

        fn with_compilation(tree: &Tree, values: &mut Vec<f64>, queries: &[[f64; 3]]) {
            values.clear();
            let context = JitContext::default();
            let mut eval = tree.jit_compile(&context).unwrap();
            values.extend(queries.iter().map(|coords| {
                let results = eval.run(coords).unwrap();
                results[0]
            }));
        }

        fn no_compilation(eval: &mut JitFn<'_>, values: &mut Vec<f64>, queries: &[[f64; 3]]) {
            values.clear();
            values.extend(queries.iter().map(|coords| {
                let results = eval.run(coords).unwrap();
                results[0]
            }));
        }

        fn b_with_compile(c: &mut Criterion) {
            let (tree, queries, mut values) = init_benchmark();
            c.bench_function("jit-single-eval-with-compile", |b| {
                b.iter(|| {
                    with_compilation(&tree, &mut values, &queries);
                    assert_eq!(values.len(), N_QUERIES);
                })
            });
        }

        fn b_no_compile(c: &mut Criterion) {
            let (tree, queries, mut values) = init_benchmark();
            let context = JitContext::default();
            let mut eval = tree.jit_compile(&context).unwrap();
            c.bench_function("jit-single-eval-no-compile", |b| {
                b.iter(|| {
                    no_compilation(&mut eval, &mut values, &queries);
                    assert_eq!(values.len(), N_QUERIES);
                })
            });
        }

        criterion_group!(bench, b_no_compile, b_with_compile,);
    }
}

#[cfg(not(feature = "llvm-jit"))]
criterion_main!(spheres::value_eval::bench);

#[cfg(feature = "llvm-jit")]
criterion_main!(spheres::value_eval::bench, spheres::jit_single::bench);
