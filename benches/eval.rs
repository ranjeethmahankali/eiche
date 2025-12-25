use criterion::{Criterion, criterion_group, criterion_main};
use eiche::{Deduplicater, Error, Pruner, Tree, Value, ValueEvaluator, deftree, min};
use eiche::{Interval, PruningState, ValuePruningEvaluator};
use rand::{SeedableRng, rngs::StdRng};
use std::hint::black_box;

/// Creates a very large tree for benchmarking. This tree is designed to:
/// - Use almost all available operations (arithmetic, trigonometric, comparison, conditional)
/// - Be mathematically valid (sqrt/log inputs are always positive)
/// - Have both depth and breadth for comprehensive evaluation testing
/// - Use multiple input variables (x, y, z, a, b, c) for realistic benchmarking
///
/// The tree is significantly larger than `create_large_tree` in dominator.rs,
/// making it suitable for stress-testing evaluation performance.
pub fn create_very_large_benchmark_tree() -> Tree {
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
pub fn create_huge_benchmark_tree() -> Tree {
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
pub fn create_massive_benchmark_tree() -> Tree {
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

    pub mod interval {
        use super::*;
        use eiche::IntervalEvaluator;
        use rand::Rng;

        struct BenchmarkSetup {
            tree: Tree,
            queries: Box<[[Interval; 3]]>,
            outputs: Vec<Interval>,
        }

        fn init_benchmark() -> BenchmarkSetup {
            let mut rng = StdRng::seed_from_u64(42);
            let tree = random_sphere_union()
                .compacted()
                .expect("Cannot compact tree");
            assert_eq!(
                tree.symbols().len(),
                3,
                "The benchmarks make unsafe calls that rely on the number of inputs to this tree being exactly 3."
            );
            let queries: Box<[_]> = (0..N_QUERIES)
                .map(|_| {
                    [X_RANGE, Y_RANGE, Z_RANGE].map(|range| {
                        let mut bounds =
                            [0, 1].map(|_| range.0 + rng.random::<f64>() * (range.1 - range.0));
                        if bounds[0] > bounds[1] {
                            bounds.swap(0, 1);
                        }
                        Interval::Scalar(bounds[0], bounds[1])
                    })
                })
                .collect();
            BenchmarkSetup {
                tree,
                queries,
                outputs: Vec::with_capacity(N_QUERIES),
            }
        }

        fn with_compilation(tree: &Tree, outputs: &mut Vec<Interval>, queries: &[[Interval; 3]]) {
            outputs.clear();
            let mut eval = IntervalEvaluator::new(tree);
            outputs.extend(queries.iter().map(|sample| {
                let [x, y, z] = &sample;
                eval.set_value('x', *x);
                eval.set_value('y', *y);
                eval.set_value('z', *z);
                let result = eval.run().unwrap();
                result[0]
            }))
        }

        /// Does not include the time to jit-compile the tree.
        fn no_compilation(
            eval: &mut IntervalEvaluator,
            outputs: &mut Vec<Interval>,
            queries: &[[Interval; 3]],
        ) {
            outputs.clear();
            outputs.extend(queries.iter().map(|sample| {
                let [x, y, z] = &sample;
                eval.set_value('x', *x);
                eval.set_value('y', *y);
                eval.set_value('z', *z);
                let result = eval.run().unwrap();
                result[0]
            }))
        }

        fn b_with_compile(c: &mut Criterion) {
            let BenchmarkSetup {
                tree,
                queries,
                mut outputs,
            } = init_benchmark();
            let mut group = c.benchmark_group("lower sample count");
            group.sample_size(10);
            group.bench_function("spheres-interval-eval-with-compile", |b| {
                b.iter(|| {
                    with_compilation(&tree, black_box(&mut outputs), &queries);
                })
            });
        }

        fn b_no_compile(c: &mut Criterion) {
            let BenchmarkSetup {
                tree,
                queries,
                mut outputs,
            } = init_benchmark();
            let mut eval = IntervalEvaluator::new(&tree);
            c.bench_function("spheres-interval-eval-no-compile", |b| {
                b.iter(|| {
                    no_compilation(&mut eval, black_box(&mut outputs), &queries);
                })
            });
        }

        criterion_group!(bench, b_no_compile, b_with_compile);
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
                    let tree = random_sphere_union()
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
            T: NumberType + Copy,
        {
            values.clear();
            let context = JitContext::default();
            let eval = tree.jit_compile(&context, "xyz").unwrap();
            values.extend(queries.iter().map(|coords| {
                let mut output = [T::nan()];
                // SAFETY: There is an assert to make sure the tree has 3 input
                // symbols. That is what the safe version would check for, so
                // we don't need to check here.
                unsafe {
                    eval.run_unchecked(coords, &mut output);
                }
                output[0]
            }));
        }

        /// Does not include the time to jit-compile the tree.
        fn no_compilation<T>(eval: &JitFn<'_, T>, values: &mut Vec<T>, queries: &[[T; 3]])
        where
            T: NumberType,
        {
            values.clear();
            values.extend(queries.iter().map(|coords| {
                let mut output = [T::nan()];
                // SAFETY: There is an assert to make sure the tree has 3 input
                // symbols. That is what the safe version would check for, so we
                // don't need to check here.
                unsafe {
                    eval.run_unchecked(coords, &mut output);
                }
                output[0]
            }));
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
                let eval = tree.jit_compile(&context, "xyz").unwrap();
                c.bench_function("spheres-jit-f64-single-eval-no-compile", |b| {
                    b.iter(|| {
                        no_compilation(&eval, black_box(&mut values), &queries);
                    })
                });
            }
            {
                let (tree, queries, mut values) = init_benchmark::<f32>();
                let context = JitContext::default();
                let eval = tree.jit_compile(&context, "xyz").unwrap();
                c.bench_function("spheres-jit-f32-single-eval-no-compile", |b| {
                    b.iter(|| {
                        no_compilation(&eval, black_box(&mut values), &queries);
                    })
                });
            }
        }

        criterion_group!(bench, b_no_compile, b_with_compile);
    }

    #[cfg(feature = "llvm-jit")]
    pub mod jit_simd {
        use super::*;
        use eiche::{
            JitContext, JitSimdFn, SimdVec, Wide,
            llvm_jit::{NumberType, simd_array::JitSimdBuffers},
        };

        fn no_compilation<T>(eval: &mut JitSimdFn<'_, T>, buf: &mut JitSimdBuffers<T>)
        where
            Wide: SimdVec<T>,
            T: Copy + NumberType,
        {
            buf.clear_outputs();
            eval.run(buf);
        }

        fn b_no_compilation_f64(c: &mut Criterion) {
            let (tree, queries, _) = jit_single::init_benchmark::<f64>();
            let context = JitContext::default();
            let mut eval = tree.jit_compile_array(&context, "xyz").unwrap();
            let mut buf = JitSimdBuffers::new(&tree);
            for q in queries {
                buf.pack(&q).unwrap();
            }
            c.bench_function("spheres-jit-simd-f64-no-compilation", |b| {
                b.iter(|| no_compilation(black_box(&mut eval), black_box(&mut buf)))
            });
        }

        fn b_no_compilation_f32(c: &mut Criterion) {
            let (tree, queries, _) = jit_single::init_benchmark::<f32>();
            let context = JitContext::default();
            let mut eval = tree.jit_compile_array(&context, "xyz").unwrap();
            let mut buf = JitSimdBuffers::new(&tree);
            for q in queries {
                buf.pack(&q).unwrap();
            }
            c.bench_function("spheres-jit-simd-f32-no-compilation", |b| {
                b.iter(|| no_compilation(black_box(&mut eval), black_box(&mut buf)))
            });
        }

        criterion_group!(bench, b_no_compilation_f64, b_no_compilation_f32);
    }

    #[cfg(feature = "llvm-jit")]
    pub mod jit_interval {
        use super::*;
        use eiche::{JitContext, JitIntervalFn, llvm_jit::NumberType};
        use rand::Rng;

        struct BenchmarkSetup<T: NumberType> {
            tree: Tree,
            queries: Box<[[[T; 2]; 3]]>,
            outputs: Vec<[T; 2]>,
        }

        fn init_benchmark<T: NumberType>() -> BenchmarkSetup<T> {
            let mut rng = StdRng::seed_from_u64(42);
            let tree = random_sphere_union()
                .compacted()
                .expect("Cannot compact tree");
            assert_eq!(
                tree.symbols().len(),
                3,
                "The benchmarks make unsafe calls that rely on the number of inputs to this tree being exactly 3."
            );
            let queries: Box<[_]> = (0..N_QUERIES)
                .map(|_| {
                    [X_RANGE, Y_RANGE, Z_RANGE].map(|range| {
                        let mut bounds =
                            [0, 1].map(|_| range.0 + rng.random::<f64>() * (range.1 - range.0));
                        if bounds[0] > bounds[1] {
                            bounds.swap(0, 1);
                        }
                        bounds.map(|b| T::from_f64(b))
                    })
                })
                .collect();
            BenchmarkSetup {
                tree,
                queries,
                outputs: Vec::with_capacity(N_QUERIES),
            }
        }

        fn with_compilation<T: NumberType>(
            tree: &Tree,
            values: &mut Vec<[T; 2]>,
            queries: &[[[T; 2]; 3]],
        ) {
            values.clear();
            let context = JitContext::default();
            let eval = tree.jit_compile_interval::<T>(&context, "xyz").unwrap();
            values.extend(queries.iter().map(|coords| {
                let mut output = [[T::nan(), T::nan()]];
                // SAFETY: There is an assert to make sure the tree has 3 input
                // symbols. That is what the safe version would check for, so
                // we don't need to check here.
                unsafe {
                    eval.run_unchecked(coords.as_ref(), &mut output);
                }
                output[0]
            }))
        }

        /// Does not include the time to jit-compile the tree.
        fn no_compilation<T>(
            eval: &JitIntervalFn<'_, T>,
            values: &mut Vec<[T; 2]>,
            queries: &[[[T; 2]; 3]],
        ) where
            T: NumberType,
        {
            values.clear();
            values.extend(queries.iter().map(|coords| {
                let mut output = [[T::nan(), T::nan()]];
                // SAFETY: There is an assert to make sure the tree has 3 input
                // symbols. That is what the safe version would check for, so we
                // don't need to check here.
                unsafe {
                    eval.run_unchecked(coords.as_ref(), &mut output);
                }
                output[0]
            }));
        }

        fn b_with_compile(c: &mut Criterion) {
            {
                let BenchmarkSetup {
                    tree,
                    queries,
                    mut outputs,
                } = init_benchmark::<f64>();
                let mut group = c.benchmark_group("lower sample count");
                group.sample_size(10);
                group.bench_function("spheres-jit-f64-interval-eval-with-compile", |b| {
                    b.iter(|| {
                        with_compilation(&tree, black_box(&mut outputs), &queries);
                    })
                });
            }
            {
                let mut group = c.benchmark_group("lower sample count");
                group.sample_size(10);
                let BenchmarkSetup {
                    tree,
                    queries,
                    mut outputs,
                } = init_benchmark::<f32>();
                group.bench_function("spheres-jit-f32-interval-eval-with-compile", |b| {
                    b.iter(|| {
                        with_compilation(&tree, black_box(&mut outputs), &queries);
                    })
                });
            }
        }

        fn b_no_compile(c: &mut Criterion) {
            {
                let BenchmarkSetup {
                    tree,
                    queries,
                    mut outputs,
                } = init_benchmark::<f64>();
                let context = JitContext::default();
                let eval = tree.jit_compile_interval(&context, "xyz").unwrap();
                c.bench_function("spheres-jit-f64-interval-eval-no-compile", |b| {
                    b.iter(|| {
                        no_compilation(&eval, black_box(&mut outputs), &queries);
                    })
                });
            }
            {
                let BenchmarkSetup {
                    tree,
                    queries,
                    mut outputs,
                } = init_benchmark::<f32>();
                let context = JitContext::default();
                let eval = tree.jit_compile_interval(&context, "xyz").unwrap();
                c.bench_function("spheres-jit-f32-interval-eval-no-compile", |b| {
                    b.iter(|| {
                        no_compilation(&eval, black_box(&mut outputs), &queries);
                    })
                });
            }
        }

        criterion_group!(bench, b_no_compile, b_with_compile);
    }
}

mod circles {
    use super::*;
    use eiche::test_util;

    type ImageBuffer = image::ImageBuffer<image::Luma<u8>, Vec<u8>>;

    const PRUNE_DEPTH: usize = 9;
    const DIMS: u32 = 1 << PRUNE_DEPTH; // 512 x 512 image.
    const DIMS_F64: f64 = DIMS as f64;
    const RAD_RANGE: (f64, f64) = (0.02 * DIMS_F64, 0.1 * DIMS_F64);
    const N_CIRCLES: usize = 100;

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

    fn b_with_compile(c: &mut Criterion) {
        let tree = test_util::random_circles((0., DIMS_F64), (0., DIMS_F64), RAD_RANGE, N_CIRCLES);
        let mut image = ImageBuffer::new(DIMS, DIMS);
        c.bench_function("circles-value-eval-with-compilation", |b| {
            b.iter(|| with_compile(black_box(&tree), &mut image))
        });
    }

    fn b_no_compile(c: &mut Criterion) {
        let tree = test_util::random_circles((0., DIMS_F64), (0., DIMS_F64), RAD_RANGE, N_CIRCLES);
        let mut image = ImageBuffer::new(DIMS, DIMS);
        let mut eval = ValueEvaluator::new(&tree);
        c.bench_function("circles-value-eval-no-compilation", |b| {
            b.iter(|| no_compile(black_box(&mut eval), &mut image));
        });
    }

    fn b_pruned_eval(c: &mut Criterion) {
        let tree = test_util::random_circles((0., DIMS_F64), (0., DIMS_F64), RAD_RANGE, N_CIRCLES);
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

    #[cfg(feature = "llvm-jit")]
    pub mod jit_single {
        use super::*;
        use eiche::{JitContext, JitFn};

        fn with_compilation(tree: &Tree, image: &mut ImageBuffer) {
            let context = JitContext::default();
            let eval = tree.jit_compile(&context, "xy").unwrap();
            let mut coord = [0., 0.];
            for y in 0..DIMS {
                coord[1] = y as f64 + 0.5;
                for x in 0..DIMS {
                    coord[0] = x as f64 + 0.5;
                    let mut output = [0.];
                    // SAFETY: Upstream functions assert to make sure the tree
                    // has exactly 2 inputs. This is what the safe version
                    // checks for, so we don't need to check agian.
                    unsafe { eval.run_unchecked(&coord, &mut output) };
                    let val = output[0];
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
                    let mut output = [0.];
                    // SAFETY: Upstream functions assert to make sure the tree
                    // has exactly 3 inputs. This is what the safe version
                    // checks for, so we don't need to check agian.
                    unsafe { eval.run_unchecked(&coord, &mut output) };
                    let val = output[0];
                    *image.get_pixel_mut(x, y) = image::Luma([if val < 0. {
                        f64::min((-val / RAD_RANGE.1) * 255., 255.) as u8
                    } else {
                        f64::min(((RAD_RANGE.1 - val) / RAD_RANGE.1) * 255., 255.) as u8
                    }]);
                }
            }
        }

        fn b_with_compile(c: &mut Criterion) {
            let tree =
                test_util::random_circles((0., DIMS_F64), (0., DIMS_F64), RAD_RANGE, N_CIRCLES);
            assert_eq!(
                tree.symbols().len(),
                2,
                "Later calls require exactly two inputs."
            );
            let mut image = ImageBuffer::new(DIMS, DIMS);
            c.bench_function("circles-jit-single-eval-with-compile", |b| {
                b.iter(|| {
                    with_compilation(&tree, black_box(&mut image));
                })
            });
        }

        fn b_no_compile(c: &mut Criterion) {
            let tree =
                test_util::random_circles((0., DIMS_F64), (0., DIMS_F64), RAD_RANGE, N_CIRCLES);
            let mut image = ImageBuffer::new(DIMS, DIMS);
            let context = JitContext::default();
            let mut eval = tree.jit_compile(&context, "xy").unwrap();
            c.bench_function("circles-jit-single-eval-no-compile", |b| {
                b.iter(|| {
                    no_compilation(&mut eval, black_box(&mut image));
                })
            });
        }

        criterion_group!(bench, b_no_compile, b_with_compile,);
    }

    pub mod interval {
        use super::*;
        use eiche::IntervalEvaluator;
        use rand::Rng;

        const N_QUERIES: usize = 512;

        fn init_benchmark() -> (Tree, Box<[[Interval; 2]]>, Vec<Interval>) {
            let mut rng = StdRng::seed_from_u64(42);
            let tree =
                test_util::random_circles((0., DIMS_F64), (0., DIMS_F64), RAD_RANGE, N_CIRCLES)
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
                    [(0., DIMS_F64), (0., DIMS_F64)].map(|range| {
                        let mut bounds =
                            [0, 1].map(|_| range.0 + rng.random::<f64>() * (range.1 - range.0));
                        if bounds[0] > bounds[1] {
                            bounds.swap(0, 1);
                        }
                        Interval::Scalar(bounds[0], bounds[1])
                    })
                })
                .collect();
            (tree, samples, Vec::with_capacity(N_QUERIES))
        }

        fn with_compilation(tree: &Tree, values: &mut Vec<Interval>, queries: &[[Interval; 2]]) {
            values.clear();
            let mut eval = IntervalEvaluator::new(tree);
            values.extend(queries.iter().map(|interval| {
                eval.set_value('x', interval[0]);
                eval.set_value('y', interval[1]);
                let result = eval.run().unwrap();
                result[0]
            }));
        }

        fn no_compilation(
            eval: &mut IntervalEvaluator,
            values: &mut Vec<Interval>,
            queries: &[[Interval; 2]],
        ) {
            values.clear();
            values.extend(queries.iter().map(|interval| {
                eval.set_value('x', interval[0]);
                eval.set_value('y', interval[1]);
                let result = eval.run().unwrap();
                result[0]
            }));
        }

        fn b_with_compile(c: &mut Criterion) {
            let (tree, queries, mut values) = init_benchmark();
            c.bench_function("circles-interval-eval-with-compile", |b| {
                b.iter(|| {
                    with_compilation(&tree, black_box(&mut values), &queries);
                })
            });
        }

        fn b_no_compile(c: &mut Criterion) {
            let (tree, queries, mut values) = init_benchmark();
            let mut eval = IntervalEvaluator::new(&tree);
            c.bench_function("circles-interval-eval-no-compile", |b| {
                b.iter(|| {
                    no_compilation(&mut eval, black_box(&mut values), &queries);
                })
            });
        }

        criterion_group!(bench, b_no_compile, b_with_compile);
    }

    #[cfg(feature = "llvm-jit")]
    pub mod jit_interval {
        use super::*;
        use eiche::{JitContext, JitIntervalFn, llvm_jit::NumberType};
        use rand::Rng;

        const N_QUERIES: usize = 512;

        struct BenchmarkSetup<T: NumberType> {
            tree: Tree,
            queries: Box<[[[T; 2]; 2]]>,
            outputs: Vec<[T; 2]>,
        }

        fn init_benchmark<T: NumberType>() -> BenchmarkSetup<T> {
            let mut rng = StdRng::seed_from_u64(42);
            let tree =
                test_util::random_circles((0., DIMS_F64), (0., DIMS_F64), RAD_RANGE, N_CIRCLES)
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
                    [(0., DIMS_F64), (0., DIMS_F64)].map(|range| {
                        let mut bounds =
                            [0, 1].map(|_| range.0 + rng.random::<f64>() * (range.1 - range.0));
                        if bounds[0] > bounds[1] {
                            bounds.swap(0, 1);
                        }
                        bounds.map(|b| T::from_f64(b))
                    })
                })
                .collect();
            BenchmarkSetup {
                tree,
                queries,
                outputs: Vec::with_capacity(N_QUERIES),
            }
        }

        fn with_compilation<T: NumberType>(
            tree: &Tree,
            values: &mut Vec<[T; 2]>,
            queries: &[[[T; 2]; 2]],
        ) {
            values.clear();
            let context = JitContext::default();
            let eval = tree.jit_compile_interval::<T>(&context, "xy").unwrap();
            values.extend(queries.iter().map(|interval| {
                let mut output = [[T::nan(); 2]];
                // SAFETY: There is an assert to make sure the tree has 3 input
                // symbols. That is what the safe version would check for, so
                // we don't need to check here.
                unsafe {
                    eval.run_unchecked(interval.as_ref(), &mut output);
                }
                output[0]
            }));
        }

        fn no_compilation<T: NumberType>(
            eval: &JitIntervalFn<'_, T>,
            values: &mut Vec<[T; 2]>,
            queries: &[[[T; 2]; 2]],
        ) {
            values.clear();
            values.extend(queries.iter().map(|interval| {
                let mut output = [[T::nan(); 2]];
                // SAFETY: There is an assert to make sure the tree has 3 input
                // symbols. That is what the safe version would check for, so
                // we don't need to check here.
                unsafe {
                    eval.run_unchecked(interval.as_ref(), &mut output);
                }
                output[0]
            }));
        }

        fn b_with_compile(c: &mut Criterion) {
            {
                let BenchmarkSetup {
                    tree,
                    queries,
                    mut outputs,
                } = init_benchmark::<f64>();
                c.bench_function("circles-interval-jit-f64-eval-with-compile", |b| {
                    b.iter(|| {
                        with_compilation(&tree, black_box(&mut outputs), &queries);
                    })
                });
            }
            {
                let BenchmarkSetup {
                    tree,
                    queries,
                    mut outputs,
                } = init_benchmark::<f32>();
                c.bench_function("circles-interval-jit-f32-eval-with-compile", |b| {
                    b.iter(|| {
                        with_compilation(&tree, black_box(&mut outputs), &queries);
                    })
                });
            }
        }

        fn b_no_compile(c: &mut Criterion) {
            {
                let BenchmarkSetup {
                    tree,
                    queries,
                    mut outputs,
                } = init_benchmark::<f64>();
                let context = JitContext::default();
                let eval = tree.jit_compile_interval(&context, "xyz").unwrap();
                c.bench_function("circles-interval-jit-f64-eval-no-compile", |b| {
                    b.iter(|| {
                        no_compilation(&eval, black_box(&mut outputs), &queries);
                    })
                });
            }
            {
                let BenchmarkSetup {
                    tree,
                    queries,
                    mut outputs,
                } = init_benchmark::<f32>();
                let context = JitContext::default();
                let eval = tree.jit_compile_interval(&context, "xyz").unwrap();
                c.bench_function("circles-interval-jit-f32-eval-no-compile", |b| {
                    b.iter(|| {
                        no_compilation(&eval, black_box(&mut outputs), &queries);
                    })
                });
            }
        }

        criterion_group!(bench, b_no_compile, b_with_compile);
    }
}

#[cfg(not(feature = "llvm-jit"))]
criterion_main!(
    spheres::value_eval::bench,
    spheres::interval::bench,
    circles::bench,
    circles::interval::bench
);

#[cfg(feature = "llvm-jit")]
criterion_main!(
    spheres::value_eval::bench,
    spheres::interval::bench,
    spheres::jit_single::bench,
    spheres::jit_simd::bench,
    spheres::jit_interval::bench,
    circles::bench,
    circles::interval::bench,
    circles::jit_single::bench,
    circles::jit_interval::bench
);
