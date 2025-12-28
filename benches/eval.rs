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

/// Creates a very large tree for benchmarking. This tree is designed to:
/// - Use almost all available operations (arithmetic, trigonometric, comparison, conditional)
/// - Be mathematically valid (sqrt/log inputs are always positive)
/// - Have both depth and breadth for comprehensive evaluation testing
/// - Use multiple input variables (x, y, z, a, b, c) for realistic benchmarking
///
/// The tree is significantly larger than `create_large_tree` in dominator.rs,
/// making it suitable for stress-testing evaluation performance.
pub fn create_small_benchmark_tree() -> Tree {
    // This tree combines multiple mathematical concepts:
    // 1. Distance calculations (always positive for sqrt)
    // 2. Bounded trigonometric expressions
    // 3. Exponential/logarithmic combinations (with positive arguments)
    // 4. Conditional expressions using comparisons
    // 5. Min/max for clamping and selection
    // 6. Floor and remainder for discrete operations
    // All sub-expressions are carefully constructed to ensure mathematical validity:
    // - sqrt inputs are sums of squares or abs values
    // - log inputs are (1 + abs(...)) or exp(...) to ensure positivity
    // - pow with fractional exponents uses abs for base
    deftree!(
        // Top-level: combine multiple large sub-expressions
        min
        (max
            // Branch 1: Complex distance-based expression with trigonometry
            (+
                // Euclidean distance in 3D, scaled by trig functions
                (*
                    (sqrt (+ (+ (pow (- 'x 1.5) 2.) (pow (- 'y 2.3) 2.)) (pow (- 'z 0.7) 2.)))
                    (+ 1. (* 0.5 (sin (* 2. 'x)))))
                // Another distance with cosine modulation
                (*
                    (sqrt (+ (pow (- 'x 3.1) 2.) (pow (- 'y 1.8) 2.)))
                    (+ 1. (* 0.3 (cos (* 1.5 'y))))))
            // Branch 2: Logarithmic expression (log of positive values)
            (+
                (log (+ 1. (+ (pow 'a 2.) (pow 'b 2.))))
                (* 0.5
                    (log (+ 1. (abs (- (* 'a 'b) (* 'x 'y))))))))
        // Second major branch with conditionals and more operations
        (+
            // Conditional: if distance < threshold, use one expression, else another
            (if (< (sqrt (+ (pow 'x 2.) (pow 'y 2.))) 5.)
                // True branch: smooth exponential-based expression
                (* 0.1
                    (exp (- 0. (/ (+ (pow 'x 2.) (pow 'y 2.)) 10.))))
                // False branch: oscillating expression
                (* 0.05
                    (+ (sin (* 0.5 'x)) (cos (* 0.5 'y)))))
            // Add more complexity with nested min/max and floor
            (min
                (max
                    // Expression using floor for discrete steps
                    (/ (floor (* 10. (abs 'z))) 10.)
                    // Clamped tangent (using atan-like bounded expression)
                    (/ (sin 'a) (+ 1. (abs (cos 'a)))))
                // Upper bound with exponential decay
                (+ 2.
                    (exp (- 0. (/ (pow (- 'b 1.) 2.) 2.)))))))
    .unwrap()
    .compacted()
    .unwrap()
}

/// Creates an even larger tree with more operations for stress testing.
/// This tree is designed to exercise every operation type multiple times.
pub fn create_medium_benchmark_tree() -> Tree {
    deftree!(
        // Level 1: Primary min combining two major branches
        min
        // ===== BRANCH A: Distance and trigonometric computations =====
        (max
            // A.1: Multiple nested distance calculations with modulation
            (+
                // 3D distance from point (1.5, 2.3, 0.7)
                (*
                    (sqrt (+ (+ (pow (- 'x 1.5) 2.) (pow (- 'y 2.3) 2.)) (pow (- 'z 0.7) 2.)))
                    (+ 1. (* 0.5 (sin (* 2. 'x)))))
                // Combined with 2D distance from (3.1, 1.8)
                (+
                    (*
                        (sqrt (+ (pow (- 'x 3.1) 2.) (pow (- 'y 1.8) 2.)))
                        (+ 1. (* 0.3 (cos (* 1.5 'y)))))
                    // Additional distance from (0.5, 0.5, 0.5)
                    (*
                        (sqrt (+ (+ (pow (- 'x 0.5) 2.) (pow (- 'y 0.5) 2.)) (pow (- 'z 0.5) 2.)))
                        (+ 1. (* 0.2 (tan (/ 'z (+ 1. (abs 'z)))))))))
            // A.2: Logarithmic expressions with guaranteed positive arguments
            (+
                (+
                    (log (+ 1. (+ (pow 'a 2.) (pow 'b 2.))))
                    (* 0.5 (log (+ 1. (abs (- (* 'a 'b) (* 'x 'y)))))))
                (+
                    (* 0.25 (log (+ 1. (pow (sin 'c) 2.))))
                    (* 0.125 (log (+ 2. (cos (* 0.5 'a))))))))
        // ===== BRANCH B: Conditionals and discrete operations =====
        (+
            // B.1: Nested conditional expressions
            (if (< (sqrt (+ (pow 'x 2.) (pow 'y 2.))) 5.)
                // If close to origin
                (if (> 'z 0.)
                    // z positive: exponential decay
                    (* 0.1 (exp (- 0. (/ (+ (pow 'x 2.) (pow 'y 2.)) 10.))))
                    // z non-positive: different decay
                    (* 0.15 (exp (- 0. (/ (+ (pow 'x 2.) (pow 'z 2.)) 8.)))))
                // If far from origin
                (if (<= 'a 'b)
                    // a <= b: sinusoidal
                    (* 0.05 (+ (sin (* 0.5 'x)) (cos (* 0.5 'y))))
                    // a > b: cosinusoidal
                    (* 0.05 (- (cos (* 0.5 'x)) (sin (* 0.5 'y))))))
            // B.2: More complex nested operations
            (+
                // Floor and remainder operations
                (min
                    (max
                        (/ (floor (* 10. (abs 'z))) 10.)
                        (rem (+ (abs 'x) 1.) 2.))
                    (+ 2. (exp (- 0. (/ (pow (- 'b 1.) 2.) 2.)))))
                // Additional conditional with comparisons
                (if (>= (abs (- 'a 'b)) 1.)
                    // Significant difference
                    (* 0.1 (/ (- 'a 'b) (+ 1. (abs (- 'a 'b)))))
                    // Small difference
                    (* 0.2 (pow (abs (- 'a 'b)) 0.5))))))
    .unwrap()
    .compacted()
    .unwrap()
}

/// Creates the largest benchmark tree for maximum stress testing.
/// This combines multiple complex sub-expressions and uses all operations.
pub fn create_large_benchmark_tree() -> Tree {
    deftree!(
        // Top level: add all computation branches together
        +
        // ===== BRANCHES 1-2: Geometric and analytical computations =====
        (+
            (max
                // 1.1: Signed distance to multiple spheres combined
                (min
                    (min
                        (- (sqrt (+ (+ (pow (- 'x 2.5) 2.) (pow (- 'y 2.5) 2.)) (pow (- 'z 2.5) 2.))) 1.5)
                        (- (sqrt (+ (+ (pow (- 'x 0.) 2.) (pow (- 'y 0.) 2.)) (pow (- 'z 0.) 2.))) 2.0))
                    (min
                        (- (sqrt (+ (+ (pow (- 'x 5.) 2.) (pow (- 'y 0.) 2.)) (pow (- 'z 2.) 2.))) 1.0)
                        (- (sqrt (+ (+ (pow (- 'x 1.) 2.) (pow (- 'y 4.) 2.)) (pow (- 'z 1.) 2.))) 1.8)))
                // 1.2: Gyroid-like periodic structure
                (+
                    (* (sin (* 1.57 'x)) (cos (* 1.57 'y)))
                    (+
                        (* (sin (* 1.57 'y)) (cos (* 1.57 'z)))
                        (* (sin (* 1.57 'z)) (cos (* 1.57 'x))))))
            // 2: Analytical functions
            (+
                // 2.1: Gaussian mixture
                (+
                    (* 2. (exp (- 0. (/ (+ (pow 'x 2.) (pow 'y 2.)) 4.))))
                    (+
                        (* 1.5 (exp (- 0. (/ (+ (pow (- 'x 2.) 2.) (pow (- 'y 2.) 2.)) 2.))))
                        (* 1.0 (exp (- 0. (/ (+ (pow (- 'x 1.) 2.) (pow (- 'y 3.) 2.)) 3.))))))
                // 2.2: Logarithmic combination
                (+
                    (log (+ 1. (+ (+ (pow 'x 2.) (pow 'y 2.)) (pow 'z 2.))))
                    (* 0.5 (log (+ 1. (+ (pow 'a 2.) (pow 'b 2.))))))))
        // ===== BRANCHES 3-5: Conditionals, trig, and comparisons =====
        (+
            // 3: Complex conditionals and discrete ops
            (+
                // 3.1: Multi-level conditional
                (if (< (+ (pow 'x 2.) (pow 'y 2.)) 4.)
                    // Inside circle of radius 2
                    (if (< (abs 'z) 1.)
                        (* 0.5 (- 1. (+ (pow 'x 2.) (pow 'y 2.))))
                        (/ 1. (+ 1. (abs 'z))))
                    // Outside circle
                    (if (> (+ (pow 'x 2.) (pow 'y 2.)) 16.)
                        0.1
                        (/ 1. (sqrt (+ (pow 'x 2.) (pow 'y 2.))))))
                // 3.2: Discrete and modular operations
                (+
                    (* 0.1 (floor (* 5. (+ (sin 'a) 1.))))
                    (+
                        (* 0.05 (rem (abs (+ 'a 'b)) 3.))
                        (* 0.02 (abs (- (sin 'a) (cos 'b)))))))
            // 4-5: Trig and comparisons
            (+
                // 4: Additional trigonometric complexity
                (+
                    (+
                        (* 0.3 (sin (* 2. (+ 'x (cos 'y)))))
                        (+
                            (* 0.2 (cos (* 1.5 (- 'y (sin 'x)))))
                            (* 0.1 (sin (* 3. (+ (sin 'x) (cos 'y)))))))
                    (+
                        (- (sqrt (+ (pow (- (sqrt (+ (pow 'x 2.) (pow 'y 2.))) 3.) 2.) (pow 'z 2.))) 1.)
                        (+
                            (- (sqrt (+ (pow 'x 2.) (pow 'y 2.))) 2.)
                            (abs (- 'z 1.5)))))
                // 5: Comparison-heavy expressions
                (if (>= 'a 0.)
                    (+
                        (if (<= 'b 'c)
                            (* 0.1 (- 'c 'b))
                            (* 0.1 (- 'b 'c)))
                        (min (max 'a 0.5) 2.5))
                    (+
                        (if (> (abs 'a) (abs 'b))
                            (* 0.2 (abs 'a))
                            (* 0.2 (abs 'b)))
                        (max (min 'b 0.5) 0.)))))
    )
    .unwrap()
    .compacted()
    .unwrap()
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
    let mut group = c.benchmark_group("spheres");
    group.sample_size(50);
    group.bench_function("spheres-value-eval", |b| {
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
                        Value::Bool(_) => unreachable!("Expecting a scalar"),
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
                    PruningState::Failure(error) => unreachable!("Error during pruning: {error:?}"),
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
                        Value::Bool(_) => unreachable!("Expecting a scalar"),
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
    let mut group = c.benchmark_group("circles");
    group.sample_size(32);
    group.bench_function("circles-interval-eval", |b| {
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

fn b_small_tree_interval(c: &mut Criterion) {
    let tree = create_small_benchmark_tree();
    const N_QUERIES: usize = 1024;
    const RANGE: (f64, f64) = (100.0, 100.0);
    let mut rng = StdRng::seed_from_u64(42);
    let symbols = tree.symbols();
    let queries: Box<[_]> = (0..(N_QUERIES * symbols.len()))
        .map(|_| {
            let mut bounds = [0, 1].map(|_| RANGE.0 + rng.random::<f64>() * (RANGE.1 - RANGE.0));
            if bounds[0] > bounds[1] {
                bounds.swap(0, 1);
            }
            Interval::Scalar(bounds[0], bounds[1])
        })
        .collect();
    let mut outputs: Vec<Interval> = Vec::with_capacity(N_QUERIES);
    let mut eval = IntervalEvaluator::new(&tree);
    let mut group = c.benchmark_group("small");
    group.sample_size(50);
    group.bench_function("small-tree-interval", |b| {
        b.iter(|| {
            outputs.clear();
            outputs.extend(std::hint::black_box(
                queries.chunks_exact(symbols.len()).map(|inputs| {
                    for (label, input) in symbols.iter().zip(inputs) {
                        eval.set_value(*label, *input);
                    }
                    let result = std::hint::black_box(eval.run().unwrap());
                    result[0]
                }),
            ));
        });
    });
}

fn b_medium_tree_interval(c: &mut Criterion) {
    let tree = create_medium_benchmark_tree();
    const N_QUERIES: usize = 1024;
    const RANGE: (f64, f64) = (100.0, 100.0);
    let mut rng = StdRng::seed_from_u64(42);
    let symbols = tree.symbols();
    let queries: Box<[_]> = (0..(N_QUERIES * symbols.len()))
        .map(|_| {
            let mut bounds = [0, 1].map(|_| RANGE.0 + rng.random::<f64>() * (RANGE.1 - RANGE.0));
            if bounds[0] > bounds[1] {
                bounds.swap(0, 1);
            }
            Interval::Scalar(bounds[0], bounds[1])
        })
        .collect();
    let mut outputs: Vec<Interval> = Vec::with_capacity(N_QUERIES);
    let mut eval = IntervalEvaluator::new(&tree);
    c.bench_function("medium-tree-interval", |b| {
        b.iter(|| {
            outputs.clear();
            outputs.extend(std::hint::black_box(
                queries.chunks_exact(symbols.len()).map(|inputs| {
                    for (label, input) in symbols.iter().zip(inputs) {
                        eval.set_value(*label, *input);
                    }
                    let result = std::hint::black_box(eval.run().unwrap());
                    result[0]
                }),
            ));
        });
    });
}

fn b_large_tree_interval(c: &mut Criterion) {
    let tree = create_large_benchmark_tree();
    const N_QUERIES: usize = 1024;
    const RANGE: (f64, f64) = (100.0, 100.0);
    let mut rng = StdRng::seed_from_u64(42);
    let symbols = tree.symbols();
    let queries: Box<[_]> = (0..(N_QUERIES * symbols.len()))
        .map(|_| {
            let mut bounds = [0, 1].map(|_| RANGE.0 + rng.random::<f64>() * (RANGE.1 - RANGE.0));
            if bounds[0] > bounds[1] {
                bounds.swap(0, 1);
            }
            Interval::Scalar(bounds[0], bounds[1])
        })
        .collect();
    let mut outputs: Vec<Interval> = Vec::with_capacity(N_QUERIES);
    let mut eval = IntervalEvaluator::new(&tree);
    c.bench_function("large-tree-interval", |b| {
        b.iter(|| {
            outputs.clear();
            outputs.extend(std::hint::black_box(
                queries.chunks_exact(symbols.len()).map(|inputs| {
                    for (label, input) in symbols.iter().zip(inputs) {
                        eval.set_value(*label, *input);
                    }
                    let result = std::hint::black_box(eval.run().unwrap());
                    result[0]
                }),
            ));
        });
    });
}

criterion_group!(
    bench_group,
    b_sphere_value_eval,
    b_spheres_interval_eval,
    b_circles_value_eval,
    b_circles_pruned_eval,
    b_circles_interval_eval,
    b_small_tree_interval,
    b_medium_tree_interval,
    b_large_tree_interval
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

    fn b_small_tree_interval<T: NumberType>(c: &mut Criterion) {
        let tree = create_small_benchmark_tree();
        const N_QUERIES: usize = 1024;
        const RANGE: (f64, f64) = (100.0, 100.0);
        let mut rng = StdRng::seed_from_u64(42);
        let symbols = tree.symbols();
        let queries: Box<[_]> = (0..(N_QUERIES * symbols.len()))
            .map(|_| {
                let mut bounds =
                    [0, 1].map(|_| RANGE.0 + rng.random::<f64>() * (RANGE.1 - RANGE.0));
                if bounds[0] > bounds[1] {
                    bounds.swap(0, 1);
                }
                [T::from_f64(bounds[0]), T::from_f64(bounds[1])]
            })
            .collect();
        let mut outputs: Vec<[T; 2]> = Vec::with_capacity(N_QUERIES);
        let params: String = symbols.iter().collect();
        let context = JitContext::default();
        let eval = tree
            .jit_compile_interval::<T>(&context, &params)
            .expect("Cannot compile tree");
        c.bench_function(&format!("interval-jit-small-{}", T::type_str()), |b| {
            b.iter(|| {
                outputs.clear();
                outputs.extend(std::hint::black_box(
                    queries.chunks_exact(symbols.len()).map(|inputs| {
                        let mut out = [[T::nan(); 2]];
                        // # SAFETY: We already checked the sizes.
                        unsafe { eval.run_unchecked(std::hint::black_box(inputs), &mut out) }
                        out[0]
                    }),
                ));
            });
        });
    }

    fn b_medium_tree_interval<T: NumberType>(c: &mut Criterion) {
        let tree = create_medium_benchmark_tree();
        const N_QUERIES: usize = 1024;
        const RANGE: (f64, f64) = (100.0, 100.0);
        let mut rng = StdRng::seed_from_u64(42);
        let symbols = tree.symbols();
        let queries: Box<[_]> = (0..(N_QUERIES * symbols.len()))
            .map(|_| {
                let mut bounds =
                    [0, 1].map(|_| RANGE.0 + rng.random::<f64>() * (RANGE.1 - RANGE.0));
                if bounds[0] > bounds[1] {
                    bounds.swap(0, 1);
                }
                [T::from_f64(bounds[0]), T::from_f64(bounds[1])]
            })
            .collect();
        let mut outputs: Vec<[T; 2]> = Vec::with_capacity(N_QUERIES);
        let params: String = symbols.iter().collect();
        let context = JitContext::default();
        let eval = tree
            .jit_compile_interval::<T>(&context, &params)
            .expect("Cannot compile tree");
        c.bench_function(&format!("interval-jit-medium-{}", T::type_str()), |b| {
            b.iter(|| {
                outputs.clear();
                outputs.extend(std::hint::black_box(
                    queries.chunks_exact(symbols.len()).map(|inputs| {
                        let mut out = [[T::nan(); 2]];
                        // # SAFETY: We already checked the sizes.
                        unsafe { eval.run_unchecked(std::hint::black_box(inputs), &mut out) }
                        out[0]
                    }),
                ));
            });
        });
    }

    fn b_large_tree_interval<T: NumberType>(c: &mut Criterion) {
        let tree = create_large_benchmark_tree();
        const N_QUERIES: usize = 1024;
        const RANGE: (f64, f64) = (100.0, 100.0);
        let mut rng = StdRng::seed_from_u64(42);
        let symbols = tree.symbols();
        let queries: Box<[_]> = (0..(N_QUERIES * symbols.len()))
            .map(|_| {
                let mut bounds =
                    [0, 1].map(|_| RANGE.0 + rng.random::<f64>() * (RANGE.1 - RANGE.0));
                if bounds[0] > bounds[1] {
                    bounds.swap(0, 1);
                }
                [T::from_f64(bounds[0]), T::from_f64(bounds[1])]
            })
            .collect();
        let mut outputs: Vec<[T; 2]> = Vec::with_capacity(N_QUERIES);
        let params: String = symbols.iter().collect();
        let context = JitContext::default();
        let eval = tree
            .jit_compile_interval::<T>(&context, &params)
            .expect("Cannot compile tree");
        c.bench_function(&format!("interval-jit-large-{}", T::type_str()), |b| {
            b.iter(|| {
                outputs.clear();
                outputs.extend(std::hint::black_box(
                    queries.chunks_exact(symbols.len()).map(|inputs| {
                        let mut out = [[T::nan(); 2]];
                        // # SAFETY: We already checked the sizes.
                        unsafe { eval.run_unchecked(std::hint::black_box(inputs), &mut out) }
                        out[0]
                    }),
                ));
            });
        });
    }

    criterion_group!(
        bench_group,
        b_spheres_eval::<f32>,
        b_spheres_eval::<f64>,
        b_spheres_simd::<f32>,
        b_spheres_simd::<f64>,
        b_spheres_interval::<f32>,
        b_spheres_interval::<f64>,
        b_circles_interval::<f32>,
        b_circles_interval::<f64>,
        b_circles_eval::<f32>,
        b_circles_eval::<f64>,
        b_small_tree_interval::<f32>,
        b_small_tree_interval::<f64>,
        b_medium_tree_interval::<f32>,
        b_medium_tree_interval::<f64>,
        b_large_tree_interval::<f32>,
        b_large_tree_interval::<f64>
    );
}

#[cfg(not(feature = "llvm-jit"))]
criterion_main!(bench_group);

#[cfg(feature = "llvm-jit")]
criterion_main!(bench_group, jit::bench_group);
