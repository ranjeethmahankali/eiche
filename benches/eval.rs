use criterion::{Criterion, criterion_group, criterion_main};
use eiche::{
    Deduplicater, Error, Interval, Pruner, PruningState, Tree, Value, ValueEvaluator,
    ValuePruningEvaluator, deftree, min,
};
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
    const N_SPHERES: usize = 512;
    const N_QUERIES: usize = 512;

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
            c.bench_function("spheres-value-evaluator-with-compilation", |b| {
                b.iter(|| {
                    with_compile(&tree, &mut values, &queries);
                    assert_eq!(values.len(), N_QUERIES);
                })
            });
        }

        fn b_no_compile(c: &mut Criterion) {
            let (tree, queries, mut values) = init_benchmark();
            let mut eval = ValueEvaluator::new(&tree);
            c.bench_function("spheres-value-evaluation-no-compilation", |b| {
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
            c.bench_function("spheres-jit-single-eval-with-compile", |b| {
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
            c.bench_function("spheres-jit-single-eval-no-compile", |b| {
                b.iter(|| {
                    no_compilation(&mut eval, &mut values, &queries);
                    assert_eq!(values.len(), N_QUERIES);
                })
            });
        }

        criterion_group!(bench, b_no_compile, b_with_compile,);
    }

    #[cfg(feature = "llvm-jit")]
    pub mod jit_simd {
        use super::*;
        use eiche::{JitContext, JitSimdFn};

        fn with_compilation(tree: &Tree, values: &mut Vec<f64>, queries: &[[f64; 3]]) {
            let context = JitContext::default();
            let mut eval = tree.jit_compile_array(&context).unwrap();
            values.clear();
            for q in queries {
                eval.push(q).unwrap();
            }
            eval.run(values);
        }

        fn no_compilation(
            eval: &mut JitSimdFn<'_, f64>,
            values: &mut Vec<f64>,
            queries: &[[f64; 3]],
        ) {
            values.clear();
            for q in queries {
                eval.push(q).unwrap();
            }
            eval.run(values);
        }

        fn b_with_compilation(c: &mut Criterion) {
            let (tree, queries, mut values) = init_benchmark();
            c.bench_function("spheres-jit-simd-with-compilation", |b| {
                b.iter(|| with_compilation(&tree, &mut values, &queries))
            });
        }

        fn b_no_compilation(c: &mut Criterion) {
            let (tree, queries, mut values) = init_benchmark();
            let context = JitContext::default();
            let mut eval = tree.jit_compile_array(&context).unwrap();
            c.bench_function("spheres-jit-simd-with-compilation", |b| {
                b.iter(|| no_compilation(&mut eval, &mut values, &queries))
            });
        }

        criterion_group!(bench, b_no_compilation, b_with_compilation);
    }
}

mod circles {
    use super::*;

    type ImageBuffer = image::ImageBuffer<image::Luma<u8>, Vec<u8>>;

    const PRUNE_DEPTH: usize = 7;
    const DIMS: u32 = 1 << PRUNE_DEPTH;
    const DIMS_F64: f64 = DIMS as f64;
    const RAD_RANGE: (f64, f64) = (0.02 * DIMS_F64, 0.1 * DIMS_F64);

    fn circle(cx: f64, cy: f64, r: f64) -> Result<Tree, Error> {
        deftree!(- (sqrt (+ (pow (- x (const cx)) 2) (pow (- y (const cy)) 2))) (const r))
    }

    fn random_circles(
        xrange: (f64, f64),
        yrange: (f64, f64),
        rad_range: (f64, f64),
        num_circles: usize,
    ) -> Tree {
        let mut rng = StdRng::seed_from_u64(42);
        let mut tree = circle(
            sample_range(xrange, &mut rng),
            sample_range(yrange, &mut rng),
            sample_range(rad_range, &mut rng),
        );
        for _ in 1..num_circles {
            tree = min(
                tree,
                circle(
                    sample_range(xrange, &mut rng),
                    sample_range(yrange, &mut rng),
                    sample_range(rad_range, &mut rng),
                ),
            );
        }
        let tree = tree.unwrap();
        assert_eq!(tree.dims(), (1, 1));
        tree
    }

    fn with_compile(tree: &Tree, image: &mut ImageBuffer) {
        let mut eval = ValueEvaluator::new(tree);
        for y in 0..DIMS {
            eval.set_value('y', Value::Scalar(y as f64 + 0.5));
            for x in 0..DIMS {
                eval.set_value('x', Value::Scalar(x as f64 + 0.5));
                let outputs = eval.run().expect("Failed to run value evaluator");
                assert_eq!(outputs.len(), 1);
                *image.get_pixel_mut(x, y) = match outputs[0] {
                    Value::Bool(_) => panic!("Expecting a scalar"),
                    Value::Scalar(val) => image::Luma([if val < 0. {
                        f64::min((-val / RAD_RANGE.1) * 255., 255.) as u8
                    } else {
                        f64::min(((RAD_RANGE.1 - val) / RAD_RANGE.1) * 255., 255.) as u8
                    }]),
                };
            }
        }
    }

    fn do_pruned_eval(tree: &Tree, image: &mut ImageBuffer) {
        let mut eval = ValuePruningEvaluator::new(
            &tree,
            11,
            [
                ('x', (Interval::from_scalar(0., DIMS_F64).unwrap(), 2)),
                ('y', (Interval::from_scalar(0., DIMS_F64).unwrap(), 2)),
            ]
            .into(),
        );
        const NORM_SAMPLES: [[f64; 2]; 4] =
            [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]];
        let mut state = eval.push_or_pop_to(PRUNE_DEPTH);
        loop {
            match state {
                PruningState::None => break,
                PruningState::Valid(_, _) => {} // Keep going.
                PruningState::Failure(error) => panic!("Error during pruning: {:?}", error),
            }
            for norm in NORM_SAMPLES {
                let mut sample = [0.; 2];
                eval.sample(&norm, &mut sample)
                    .expect("Cannot generate sample");
                eval.set_value('x', Value::Scalar(sample[0]));
                eval.set_value('y', Value::Scalar(sample[1]));
                let outputs = eval.run().expect("Failed to run the pruning evaluator");
                assert_eq!(outputs.len(), 1);
                let coords = sample.map(|c| c as u32);
                *image.get_pixel_mut(coords[0], coords[1]) = match outputs[0] {
                    Value::Bool(_) => panic!("Expecting a scalar"),
                    Value::Scalar(val) => image::Luma([if val < 0. {
                        f64::min((-val / RAD_RANGE.1) * 255., 255.) as u8
                    } else {
                        f64::min(((RAD_RANGE.1 - val) / RAD_RANGE.1) * 255., 255.) as u8
                    }]),
                };
            }
            state = eval.advance(Some(PRUNE_DEPTH));
        }
    }

    fn b_with_compile(c: &mut Criterion) {
        let tree = random_circles((0., DIMS_F64), (0., DIMS_F64), RAD_RANGE, 100);
        let mut image = ImageBuffer::new(DIMS, DIMS);
        c.bench_function("circles-value-eval-with-compilation", |b| {
            b.iter(|| with_compile(&tree, &mut image))
        });
    }

    fn b_pruned_eval(c: &mut Criterion) {
        let tree = random_circles((0., DIMS_F64), (0., DIMS_F64), RAD_RANGE, 100);
        let mut image = ImageBuffer::new(DIMS, DIMS);
        c.bench_function("circles-pruned-eval", |b| {
            b.iter(|| do_pruned_eval(&tree, &mut image))
        });
    }

    criterion_group!(bench, b_with_compile, b_pruned_eval);
}

#[cfg(not(feature = "llvm-jit"))]
criterion_main!(spheres::value_eval::bench, circles::bench);

#[cfg(feature = "llvm-jit")]
criterion_main!(
    spheres::value_eval::bench,
    circles::bench,
    spheres::jit_single::bench,
    spheres::jit_simd::bench,
);
