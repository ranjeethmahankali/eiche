use criterion::{Criterion, criterion_group, criterion_main};
use eiche::{
    Deduplicater, Error, Interval, Pruner, PruningState, Tree, Value, ValueEvaluator,
    ValuePruningEvaluator, deftree, min,
};
use rand::{SeedableRng, rngs::StdRng};
use std::hint::black_box;

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

    pub mod value_eval {
        use super::*;

        /// Includes the compilation times. In this case that is the time spent
        /// creating the ValueEvaluator.
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

        /// Does not include the compilation time, i.e. the time spent creating
        /// the ValueEvaluator.
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
                    with_compile(black_box(&tree), black_box(&mut values), &queries);
                    assert_eq!(values.len(), N_QUERIES);
                })
            });
        }

        fn b_no_compile(c: &mut Criterion) {
            let (tree, queries, mut values) = init_benchmark();
            let mut eval = ValueEvaluator::new(&tree);
            c.bench_function("spheres-value-evaluation-no-compilation", |b| {
                b.iter(|| {
                    no_compile(black_box(&mut eval), black_box(&mut values), &queries);
                    assert_eq!(values.len(), N_QUERIES);
                })
            });
        }

        criterion_group!(bench, b_no_compile, b_with_compile,);
    }

    #[cfg(feature = "llvm-jit")]
    pub mod jit_single {
        use super::*;
        use eiche::{JitContext, JitFn, llvm_jit::NumberType};

        pub fn init_benchmark<T>() -> (Tree, Vec<[T; 3]>, Vec<T>)
        where
            T: NumberType,
        {
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
                            T::from_f64(sample_range(X_RANGE, &mut rng)),
                            T::from_f64(sample_range(Y_RANGE, &mut rng)),
                            T::from_f64(sample_range(Z_RANGE, &mut rng)),
                        ]
                    })
                    .collect(),
                Vec::with_capacity(N_QUERIES),
            )
        }

        /// Includes the time to jit-compile the tree.
        fn with_compilation<T>(tree: &Tree, values: &mut Vec<T>, queries: &[[T; 3]])
        where
            T: NumberType,
        {
            values.clear();
            let context = JitContext::default();
            let mut eval = tree.jit_compile(&context).unwrap();
            values.extend(queries.iter().map(|coords| eval.run_unchecked(coords)[0]));
        }

        /// Does not include the time to jit-compile the tree.
        fn no_compilation<T>(eval: &mut JitFn<'_, T>, values: &mut Vec<T>, queries: &[[T; 3]])
        where
            T: NumberType,
        {
            values.clear();
            values.extend(queries.iter().map(|coords| eval.run_unchecked(coords)[0]));
        }

        fn b_with_compile(c: &mut Criterion) {
            {
                let (tree, queries, mut values) = init_benchmark::<f64>();
                c.bench_function("spheres-jit-f64-single-eval-with-compile", |b| {
                    b.iter(|| {
                        with_compilation(&tree, black_box(&mut values), &queries);
                    })
                });
            }
            {
                let (tree, queries, mut values) = init_benchmark::<f32>();
                c.bench_function("spheres-jit-f32-single-eval-with-compile", |b| {
                    b.iter(|| {
                        with_compilation(&tree, black_box(&mut values), &queries);
                    })
                });
            }
        }

        fn b_no_compile(c: &mut Criterion) {
            {
                let (tree, queries, mut values) = init_benchmark::<f64>();
                let context = JitContext::default();
                let mut eval = tree.jit_compile(&context).unwrap();
                c.bench_function("spheres-jit-f64-single-eval-no-compile", |b| {
                    b.iter(|| {
                        no_compilation(&mut eval, black_box(&mut values), &queries);
                    })
                });
            }
            {
                let (tree, queries, mut values) = init_benchmark::<f32>();
                let context = JitContext::default();
                let mut eval = tree.jit_compile(&context).unwrap();
                c.bench_function("spheres-jit-f32-single-eval-no-compile", |b| {
                    b.iter(|| {
                        no_compilation(&mut eval, black_box(&mut values), &queries);
                    })
                });
            }
        }

        criterion_group!(bench, b_no_compile, b_with_compile);
    }

    #[cfg(feature = "llvm-jit")]
    pub mod jit_simd {
        use super::*;
        use eiche::{JitContext, JitSimdFn, SimdVec, Wfloat};

        fn no_compilation<T>(eval: &mut JitSimdFn<'_, T>, values: &mut Vec<T>)
        where
            Wfloat: SimdVec<T>,
            T: Copy,
        {
            values.clear();
            eval.run(values);
        }

        fn b_no_compilation_f64(c: &mut Criterion) {
            let (tree, queries, mut values) = jit_single::init_benchmark::<f64>();
            let context = JitContext::default();
            let mut eval = tree.jit_compile_array(&context).unwrap();
            for q in queries {
                eval.push(&q).unwrap();
            }
            c.bench_function("spheres-jit-simd-f64-no-compilation", |b| {
                b.iter(|| no_compilation(&mut eval, black_box(&mut values)))
            });
        }

        fn b_no_compilation_f32(c: &mut Criterion) {
            let (tree, queries, mut values) = jit_single::init_benchmark::<f32>();
            let context = JitContext::default();
            let mut eval = tree.jit_compile_array(&context).unwrap();
            for q in queries {
                eval.push(&q).unwrap();
            }
            c.bench_function("spheres-jit-simd-f32-no-compilation", |b| {
                b.iter(|| no_compilation(&mut eval, black_box(&mut values)))
            });
        }

        criterion_group!(bench, b_no_compilation_f64, b_no_compilation_f32);
    }
}

mod circles {
    use super::*;

    type ImageBuffer = image::ImageBuffer<image::Luma<u8>, Vec<u8>>;

    const PRUNE_DEPTH: usize = 9;
    const DIMS: u32 = 1 << PRUNE_DEPTH; // 512 x 512 image.
    const DIMS_F64: f64 = DIMS as f64;
    const RAD_RANGE: (f64, f64) = (0.02 * DIMS_F64, 0.1 * DIMS_F64);
    const N_CIRCLES: usize = 100;

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
        tree.unwrap()
    }

    /// Includes the time to compile, i.e. create the ValueEvaluator.
    fn with_compile(tree: &Tree, image: &mut ImageBuffer) {
        let mut eval = ValueEvaluator::new(tree);
        for y in 0..DIMS {
            eval.set_value('y', Value::Scalar(y as f64 + 0.5));
            for x in 0..DIMS {
                eval.set_value('x', Value::Scalar(x as f64 + 0.5));
                let outputs = eval.run().expect("Failed to run value evaluator");
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

    fn no_compile(eval: &mut ValueEvaluator, image: &mut ImageBuffer) {
        for y in 0..DIMS {
            eval.set_value('y', Value::Scalar(y as f64 + 0.5));
            for x in 0..DIMS {
                eval.set_value('x', Value::Scalar(x as f64 + 0.5));
                let outputs = eval.run().expect("Failed to run value evaluator");
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

    /// Uses the pruning evaluator.
    fn do_pruned_eval(tree: &Tree, image: &mut ImageBuffer) {
        let mut eval = ValuePruningEvaluator::new(
            tree,
            PRUNE_DEPTH + 1,
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
                PruningState::Failure(error) => panic!("Error during pruning: {error:?}"),
            }
            for norm in NORM_SAMPLES {
                let mut sample = [0.; 2];
                eval.sample(&norm, &mut sample)
                    .expect("Cannot generate sample");
                eval.set_value('x', Value::Scalar(sample[0]));
                eval.set_value('y', Value::Scalar(sample[1]));
                let outputs = eval.run().expect("Failed to run the pruning evaluator");
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

    #[cfg(feature = "llvm-jit")]
    pub mod jit_single {
        use super::*;
        use eiche::{JitContext, JitFn};

        fn with_compilation(tree: &Tree, image: &mut ImageBuffer) {
            let context = JitContext::default();
            let mut eval = tree.jit_compile(&context).unwrap();
            let mut coord = [0., 0.];
            for y in 0..DIMS {
                coord[1] = y as f64 + 0.5;
                for x in 0..DIMS {
                    coord[0] = x as f64 + 0.5;
                    let val = eval.run_unchecked(&coord)[0];
                    *image.get_pixel_mut(x, y) = image::Luma([if val < 0. {
                        f64::min((-val / RAD_RANGE.1) * 255., 255.) as u8
                    } else {
                        f64::min(((RAD_RANGE.1 - val) / RAD_RANGE.1) * 255., 255.) as u8
                    }]);
                }
            }
        }

        fn no_compilation(eval: &mut JitFn<'_, f64>, image: &mut ImageBuffer) {
            let mut coord = [0., 0.];
            for y in 0..DIMS {
                coord[1] = y as f64 + 0.5;
                for x in 0..DIMS {
                    coord[0] = x as f64 + 0.5;
                    let val = eval.run_unchecked(&coord)[0];
                    *image.get_pixel_mut(x, y) = image::Luma([if val < 0. {
                        f64::min((-val / RAD_RANGE.1) * 255., 255.) as u8
                    } else {
                        f64::min(((RAD_RANGE.1 - val) / RAD_RANGE.1) * 255., 255.) as u8
                    }]);
                }
            }
        }

        fn b_with_compile(c: &mut Criterion) {
            let tree = random_circles((0., DIMS_F64), (0., DIMS_F64), RAD_RANGE, N_CIRCLES);
            let mut image = ImageBuffer::new(DIMS, DIMS);
            c.bench_function("circles-jit-single-eval-with-compile", |b| {
                b.iter(|| {
                    with_compilation(&tree, black_box(&mut image));
                })
            });
        }

        fn b_no_compile(c: &mut Criterion) {
            let tree = random_circles((0., DIMS_F64), (0., DIMS_F64), RAD_RANGE, N_CIRCLES);
            let mut image = ImageBuffer::new(DIMS, DIMS);
            let context = JitContext::default();
            let mut eval = tree.jit_compile(&context).unwrap();
            c.bench_function("circles-jit-single-eval-no-compile", |b| {
                b.iter(|| {
                    no_compilation(&mut eval, black_box(&mut image));
                })
            });
        }

        criterion_group!(bench, b_no_compile, b_with_compile,);
    }

    fn b_with_compile(c: &mut Criterion) {
        let tree = random_circles((0., DIMS_F64), (0., DIMS_F64), RAD_RANGE, N_CIRCLES);
        let mut image = ImageBuffer::new(DIMS, DIMS);
        c.bench_function("circles-value-eval-with-compilation", |b| {
            b.iter(|| with_compile(black_box(&tree), &mut image))
        });
    }

    fn b_no_compile(c: &mut Criterion) {
        let tree = random_circles((0., DIMS_F64), (0., DIMS_F64), RAD_RANGE, N_CIRCLES);
        let mut image = ImageBuffer::new(DIMS, DIMS);
        let mut eval = ValueEvaluator::new(&tree);
        c.bench_function("circles-value-eval-no-compilation", |b| {
            b.iter(|| no_compile(black_box(&mut eval), &mut image));
        });
    }

    fn b_pruned_eval(c: &mut Criterion) {
        let tree = random_circles((0., DIMS_F64), (0., DIMS_F64), RAD_RANGE, N_CIRCLES);
        let mut image = ImageBuffer::new(DIMS, DIMS);
        c.bench_function("circles-pruned-eval", |b| {
            b.iter(|| do_pruned_eval(black_box(&tree), &mut image))
        });
    }

    criterion_group! {
        name = bench;
        config = Criterion::default().sample_size(10);
        targets = b_with_compile, b_no_compile, b_pruned_eval
    }
}

#[cfg(not(feature = "llvm-jit"))]
criterion_main!(spheres::value_eval::bench, circles::bench);

#[cfg(feature = "llvm-jit")]
criterion_main!(
    spheres::value_eval::bench,
    spheres::jit_single::bench,
    spheres::jit_simd::bench,
    circles::bench,
    circles::jit_single::bench,
);
