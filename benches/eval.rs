use criterion::{Criterion, criterion_group, criterion_main};
use eiche::{
    Deduplicater, Error, IntervalEvaluator, Pruner, Tree, Value, ValueEvaluator, deftree, min,
    test_util,
};
use eiche::{Interval, PruningState, ValuePruningEvaluator};
use rand::Rng;
use rand::{SeedableRng, rngs::StdRng};
use std::hint::black_box;

fn sample_range(range: (f64, f64), rng: &mut StdRng) -> f64 {
    use rand::Rng;
    range.0 + rng.random::<f64>() * (range.1 - range.0)
}

mod spheres {
    use super::*;
    pub const RADIUS_RANGE: (f64, f64) = (0.2, 2.);
    pub const X_RANGE: (f64, f64) = (0., 100.);
    pub const Y_RANGE: (f64, f64) = (0., 100.);
    pub const Z_RANGE: (f64, f64) = (0., 100.);
    pub const N_SPHERES: usize = 512;
    pub const N_QUERIES: usize = 512;

    pub fn random_sphere_union() -> Tree {
        let mut rng = StdRng::seed_from_u64(42);
        let mut make_sphere = || -> Result<Tree, Error> {
            deftree!(- (sqrt (+ (+
                                 (pow (- 'x (const sample_range(X_RANGE, &mut rng))) 2)
                                 (pow (- 'y (const sample_range(Y_RANGE, &mut rng))) 2))
                              (pow (- 'z (const sample_range(Z_RANGE, &mut rng))) 2)))
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
}

mod circles {
    pub type ImageBuffer = image::ImageBuffer<image::Luma<u8>, Vec<u8>>;
    pub const PRUNE_DEPTH: usize = 9;
    pub const DIMS: u32 = 1 << PRUNE_DEPTH; // 512 x 512 image.
    pub const DIMS_F64: f64 = DIMS as f64;
    pub const RAD_RANGE: (f64, f64) = (0.02 * DIMS_F64, 0.1 * DIMS_F64);
    pub const N_CIRCLES: usize = 100;
}

fn b_sphere_value_eval(c: &mut Criterion) {
    let (tree, queries, mut outputs) = {
        let mut rng = StdRng::seed_from_u64(234);
        (
            {
                let mut dedup = Deduplicater::new();
                let mut pruner = Pruner::new();
                spheres::random_sphere_union()
                    .fold()
                    .unwrap()
                    .deduplicate(&mut dedup)
                    .unwrap()
                    .prune(&mut pruner)
                    .unwrap()
            },
            (0..spheres::N_QUERIES)
                .map(|_| {
                    [
                        sample_range(spheres::X_RANGE, &mut rng),
                        sample_range(spheres::Y_RANGE, &mut rng),
                        sample_range(spheres::Z_RANGE, &mut rng),
                    ]
                })
                .collect::<Vec<_>>(),
            Vec::with_capacity(spheres::N_QUERIES),
        )
    };
    let mut eval = ValueEvaluator::new(&tree);
    c.bench_function("spheres-value-eval", |b| {
        b.iter(|| {
            black_box(&mut outputs).clear();
            black_box(&mut outputs).extend(black_box(queries.iter().map(|coords| {
                eval.set_value('x', coords[0].into());
                eval.set_value('y', coords[1].into());
                eval.set_value('z', coords[2].into());
                let results = eval.run().unwrap();
                results[0].scalar().unwrap()
            })))
        })
    });
}

fn b_spheres_interval_eval(c: &mut Criterion) {
    let (tree, queries, mut outputs) = {
        let mut rng = StdRng::seed_from_u64(42);
        let tree = spheres::random_sphere_union()
            .compacted()
            .expect("Cannot compact tree");
        assert_eq!(
            tree.symbols().len(),
            3,
            "The benchmarks make unsafe calls that rely on the number of inputs to this tree being exactly 3."
        );
        let queries: Box<[_]> = (0..spheres::N_QUERIES)
            .map(|_| {
                [spheres::X_RANGE, spheres::Y_RANGE, spheres::Z_RANGE].map(|range| {
                    let mut bounds =
                        [0, 1].map(|_| range.0 + rng.random::<f64>() * (range.1 - range.0));
                    if bounds[0] > bounds[1] {
                        bounds.swap(0, 1);
                    }
                    Interval::Scalar(bounds[0], bounds[1])
                })
            })
            .collect();
        (tree, queries, Vec::with_capacity(spheres::N_QUERIES))
    };
    let mut eval = IntervalEvaluator::new(&tree);
    c.bench_function("spheres-interval-eval", |b| {
        b.iter(|| {
            black_box(&mut outputs).clear();
            black_box(&mut outputs).extend(black_box(queries.iter().map(|sample| {
                let [x, y, z] = &sample;
                black_box(&mut eval).set_value('x', *x);
                black_box(&mut eval).set_value('y', *y);
                black_box(&mut eval).set_value('z', *z);
                let result = black_box(&mut eval).run().unwrap();
                result[0]
            })))
        })
    });
}

fn b_circles_value_eval(c: &mut Criterion) {
    let tree = test_util::random_circles(
        (0., circles::DIMS_F64),
        (0., circles::DIMS_F64),
        circles::RAD_RANGE,
        circles::N_CIRCLES,
    );
    let mut image = circles::ImageBuffer::new(circles::DIMS, circles::DIMS);
    let mut eval = ValueEvaluator::new(&tree);
    let mut group = c.benchmark_group("circles");
    group.sample_size(10);
    group.bench_function("circles-value-eval", |b| {
        b.iter(|| {
            for y in 0..circles::DIMS {
                black_box(&mut eval).set_value('y', Value::Scalar(y as f64 + 0.5));
                for x in 0..circles::DIMS {
                    black_box(&mut eval).set_value('x', Value::Scalar(x as f64 + 0.5));
                    let outputs = black_box(&mut eval)
                        .run()
                        .expect("Failed to run value evaluator");
                    *image.get_pixel_mut(x, y) = match outputs[0] {
                        Value::Bool(_) => panic!("Expecting a scalar"),
                        Value::Scalar(val) => image::Luma([if val < 0. {
                            f64::min((-val / circles::RAD_RANGE.1) * 255., 255.) as u8
                        } else {
                            f64::min(
                                ((circles::RAD_RANGE.1 - val) / circles::RAD_RANGE.1) * 255.,
                                255.,
                            ) as u8
                        }]),
                    };
                }
            }
        });
    });
}

fn b_circles_pruned_eval(c: &mut Criterion) {
    let tree = test_util::random_circles(
        (0., circles::DIMS_F64),
        (0., circles::DIMS_F64),
        circles::RAD_RANGE,
        circles::N_CIRCLES,
    );
    let mut image = circles::ImageBuffer::new(circles::DIMS, circles::DIMS);
    let mut group = c.benchmark_group("circles");
    group.sample_size(10);
    group.bench_function("circles-pruned-eval", |b| {
        b.iter(|| {
            let mut eval = ValuePruningEvaluator::new(
                &tree,
                circles::PRUNE_DEPTH + 1,
                [
                    (
                        'x',
                        (Interval::from_scalar(0., circles::DIMS_F64).unwrap(), 2),
                    ),
                    (
                        'y',
                        (Interval::from_scalar(0., circles::DIMS_F64).unwrap(), 2),
                    ),
                ]
                .into(),
            );
            const NORM_SAMPLES: [[f64; 2]; 4] =
                [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]];
            let mut state = black_box(&mut eval).push_or_pop_to(circles::PRUNE_DEPTH);
            loop {
                match state {
                    PruningState::None => break,
                    PruningState::Valid(_, _) => {} // Keep going.
                    PruningState::Failure(error) => panic!("Error during pruning: {error:?}"),
                }
                for norm in NORM_SAMPLES {
                    let mut sample = [0.; 2];
                    black_box(&mut eval)
                        .sample(&norm, &mut sample)
                        .expect("Cannot generate sample");
                    black_box(&mut eval).set_value('x', Value::Scalar(sample[0]));
                    black_box(&mut eval).set_value('y', Value::Scalar(sample[1]));
                    let outputs = black_box(&mut eval)
                        .run()
                        .expect("Failed to run the pruning evaluator");
                    let coords = sample.map(|c| c as u32);
                    *image.get_pixel_mut(coords[0], coords[1]) = match outputs[0] {
                        Value::Bool(_) => panic!("Expecting a scalar"),
                        Value::Scalar(val) => image::Luma([if val < 0. {
                            f64::min((-val / circles::RAD_RANGE.1) * 255., 255.) as u8
                        } else {
                            f64::min(
                                ((circles::RAD_RANGE.1 - val) / circles::RAD_RANGE.1) * 255.,
                                255.,
                            ) as u8
                        }]),
                    };
                }
                state = eval.advance(Some(circles::PRUNE_DEPTH));
            }
        })
    });
}

fn b_circles_interval_eval(c: &mut Criterion) {
    const N_QUERIES: usize = 512;
    let (tree, queries, mut values) = {
        let mut rng = StdRng::seed_from_u64(42);
        let tree = test_util::random_circles(
            (0., circles::DIMS_F64),
            (0., circles::DIMS_F64),
            circles::RAD_RANGE,
            circles::N_CIRCLES,
        )
        .compacted()
        .expect("Cannot compact tree");
        assert_eq!(
            tree.symbols().len(),
            2,
            "Several unchecked function calls in other benchmarks require the tree to have exactly 2 inputs: x and y"
        );
        // Sample random intervals.
        let samples = (0..N_QUERIES)
            .map(|_| {
                [(0., circles::DIMS_F64), (0., circles::DIMS_F64)].map(|range| {
                    let mut bounds =
                        [0, 1].map(|_| range.0 + rng.random::<f64>() * (range.1 - range.0));
                    if bounds[0] > bounds[1] {
                        bounds.swap(0, 1);
                    }
                    Interval::Scalar(bounds[0], bounds[1])
                })
            })
            .collect::<Vec<_>>();
        (tree, samples, Vec::with_capacity(N_QUERIES))
    };
    let mut eval = IntervalEvaluator::new(&tree);
    c.bench_function("circles-interval-eval", |b| {
        b.iter(|| {
            black_box(&mut values).clear();
            black_box(&mut values).extend(black_box(queries.iter().map(|interval| {
                black_box(&mut eval).set_value('x', interval[0]);
                black_box(&mut eval).set_value('y', interval[1]);
                let result = black_box(&mut eval).run().unwrap();
                result[0]
            })));
        })
    });
}

criterion_group!(
    non_jit,
    b_sphere_value_eval,
    b_spheres_interval_eval,
    b_circles_value_eval,
    b_circles_pruned_eval,
    b_circles_interval_eval
);

#[cfg(feature = "llvm-jit")]
mod jit {
    use super::*;
    use criterion::Criterion;
    use eiche::{
        JitContext, SimdVec, Wide,
        llvm_jit::{NumberType, simd_array::JitSimdBuffers},
    };
    use rand::{SeedableRng, rngs::StdRng};

    fn init_sphere_benchmark<T: NumberType>() -> (Tree, Box<[[T; 3]]>, Vec<T>) {
        let mut rng = StdRng::seed_from_u64(234);
        (
            {
                let mut dedup = Deduplicater::new();
                let mut pruner = Pruner::new();
                let tree = spheres::random_sphere_union()
                    .fold()
                    .unwrap()
                    .deduplicate(&mut dedup)
                    .unwrap()
                    .prune(&mut pruner)
                    .unwrap();
                assert_eq!(
                    tree.symbols().len(),
                    3,
                    "Several unchecked function calls in other benchmarks require the the tree to have exactly three inputs."
                );
                tree
            },
            (0..spheres::N_QUERIES)
                .map(|_| {
                    [
                        T::from_f64(sample_range(spheres::X_RANGE, &mut rng)),
                        T::from_f64(sample_range(spheres::Y_RANGE, &mut rng)),
                        T::from_f64(sample_range(spheres::Z_RANGE, &mut rng)),
                    ]
                })
                .collect::<Box<[_]>>(),
            Vec::with_capacity(spheres::N_QUERIES),
        )
    }

    fn b_spheres_eval<T: NumberType>(c: &mut Criterion) {
        let (tree, queries, mut values) = init_sphere_benchmark();
        let context = JitContext::default();
        let eval = tree.jit_compile(&context, "xyz").unwrap();
        c.bench_function(&format!("spheres-jit-{}", T::type_str()), |b| {
            b.iter(|| {
                black_box(&mut values).clear();
                black_box(&mut values).extend(black_box(queries.iter().map(|coords| {
                    let mut output = [T::nan()];
                    // SAFETY: There is an assert to make sure the tree has 3 input
                    // symbols. That is what the safe version would check for, so we
                    // don't need to check here.
                    unsafe {
                        eval.run_unchecked(coords, &mut output);
                    }
                    output[0]
                })));
            })
        });
    }

    fn b_spheres_simd<T: NumberType>(c: &mut Criterion)
    where
        Wide: SimdVec<T>,
    {
        let (tree, queries, _) = init_sphere_benchmark::<T>();
        let context = JitContext::default();
        let eval = tree.jit_compile_array(&context, "xyz").unwrap();
        let mut buf = JitSimdBuffers::<T>::new(&tree);
        for q in queries {
            buf.pack(&q).unwrap();
        }
        c.bench_function(&format!("spheres-jit-simd-{}", T::type_str()), |b| {
            b.iter(|| {
                buf.clear_outputs();
                black_box(&eval).run(&mut buf);
            })
        });
    }

    fn b_spheres_interval<T: NumberType>(c: &mut Criterion) {
        let (tree, queries, mut outputs) = {
            let mut rng = StdRng::seed_from_u64(42);
            let tree = spheres::random_sphere_union()
                .compacted()
                .expect("Cannot compact tree");
            assert_eq!(
                tree.symbols().len(),
                3,
                "The benchmarks make unsafe calls that rely on the number of inputs to this tree being exactly 3."
            );
            let queries: Box<[_]> = (0..spheres::N_QUERIES)
                .map(|_| {
                    [spheres::X_RANGE, spheres::Y_RANGE, spheres::Z_RANGE].map(|range| {
                        let mut bounds =
                            [0, 1].map(|_| range.0 + rng.random::<f64>() * (range.1 - range.0));
                        if bounds[0] > bounds[1] {
                            bounds.swap(0, 1);
                        }
                        bounds.map(|b| T::from_f64(b))
                    })
                })
                .collect();
            (tree, queries, Vec::with_capacity(spheres::N_QUERIES))
        };
        let context = JitContext::default();
        let eval = tree.jit_compile_interval(&context, "xyz").unwrap();
        c.bench_function(&format!("spheres-jit-interval-{}", T::type_str()), |b| {
            b.iter(|| {
                black_box(&mut outputs).clear();
                black_box(&mut outputs).extend(black_box(queries.iter().map(|coords| {
                    let mut output = [[T::nan(), T::nan()]];
                    // SAFETY: There is an assert to make sure the tree has 3 input
                    // symbols. That is what the safe version would check for, so we
                    // don't need to check here.
                    unsafe {
                        black_box(&eval).run_unchecked(coords.as_ref(), &mut output);
                    }
                    output[0]
                })));
            })
        });
    }

    fn b_circles_interval<T: NumberType>(c: &mut Criterion) {
        const N_QUERIES: usize = 512;
        let (tree, queries, mut outputs) = {
            let mut rng = StdRng::seed_from_u64(42);
            let tree = test_util::random_circles(
                (0., circles::DIMS_F64),
                (0., circles::DIMS_F64),
                circles::RAD_RANGE,
                circles::N_CIRCLES,
            )
            .compacted()
            .expect("Cannot compact tree");
            assert_eq!(
                tree.symbols().len(),
                2,
                "Several unchecked function calls in other benchmarks require the tree to have exactly 2 inputs: x and y"
            );
            // Sample random intervals.
            let queries = (0..N_QUERIES)
                .map(|_| {
                    [(0., circles::DIMS_F64), (0., circles::DIMS_F64)].map(|range| {
                        let mut bounds =
                            [0, 1].map(|_| range.0 + rng.random::<f64>() * (range.1 - range.0));
                        if bounds[0] > bounds[1] {
                            bounds.swap(0, 1);
                        }
                        bounds.map(|b| T::from_f64(b))
                    })
                })
                .collect::<Vec<_>>();
            (tree, queries, Vec::with_capacity(N_QUERIES))
        };
        let context = JitContext::default();
        let eval = tree.jit_compile_interval(&context, "xyz").unwrap();
        c.bench_function(&format!("circles-jit-interval-{}", T::type_str()), |b| {
            b.iter(|| {
                black_box(&mut outputs).clear();
                black_box(&mut outputs).extend(black_box(queries.iter().map(|interval| {
                    let mut output = [[T::nan(); 2]];
                    // SAFETY: There is an assert to make sure the tree has 3 input
                    // symbols. That is what the safe version would check for, so
                    // we don't need to check here.
                    unsafe {
                        black_box(&eval).run_unchecked(interval.as_ref(), &mut output);
                    }
                    output[0]
                })));
            })
        });
    }

    fn b_circles_eval<T: NumberType>(c: &mut Criterion) {
        let tree = test_util::random_circles(
            (0., circles::DIMS_F64),
            (0., circles::DIMS_F64),
            circles::RAD_RANGE,
            circles::N_CIRCLES,
        );
        let mut image = circles::ImageBuffer::new(circles::DIMS, circles::DIMS);
        let context = JitContext::default();
        let eval = tree.jit_compile::<T>(&context, "xy").unwrap();
        c.bench_function(&format!("circles-jit-{}", T::type_str()), |b| {
            b.iter(|| {
                let mut coord = [T::nan(); 2];
                for y in 0..circles::DIMS {
                    coord[1] = T::from_f64(y as f64 + 0.5);
                    for x in 0..circles::DIMS {
                        coord[0] = T::from_f64(x as f64 + 0.5);
                        let mut output = [T::nan()];
                        // SAFETY: Upstream functions assert to make sure the tree
                        // has exactly 3 inputs. This is what the safe version
                        // checks for, so we don't need to check agian.
                        unsafe { eval.run_unchecked(&coord, &mut output) };
                        let val = output[0].to_f64();
                        *image.get_pixel_mut(x, y) = image::Luma([if val < 0. {
                            f64::min((-val / circles::RAD_RANGE.1) * 255., 255.) as u8
                        } else {
                            f64::min(
                                ((circles::RAD_RANGE.1 - val) / circles::RAD_RANGE.1) * 255.,
                                255.,
                            ) as u8
                        }]);
                    }
                }
            })
        });
    }

    criterion_group!(
        jit_group,
        b_spheres_eval::<f32>,
        b_spheres_eval::<f64>,
        b_spheres_simd::<f32>,
        b_spheres_simd::<f64>,
        b_spheres_interval::<f32>,
        b_spheres_interval::<f64>,
        b_circles_interval::<f32>,
        b_circles_interval::<f64>,
        b_circles_eval::<f32>,
        b_circles_eval::<f64>
    );
}

#[cfg(not(feature = "llvm-jit"))]
criterion_main!(non_jit);

#[cfg(feature = "llvm-jit")]
criterion_main!(non_jit, jit::jit_group);
