use criterion::{Criterion, criterion_group, criterion_main};
use eiche::{Interval, IntervalEvaluator, Tree, deftree};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn b_large_tree_interpreter(c: &mut Criterion) {
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
    c.bench_function("interval-interpreter-large", |b| {
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

fn b_huge_tree_interpreter(c: &mut Criterion) {
    let tree = create_huge_benchmark_tree();
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
    c.bench_function("interval-interpreter-huge", |b| {
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

fn b_massive_tree_interpreter(c: &mut Criterion) {
    let tree = create_massive_benchmark_tree();
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
    c.bench_function("interval-interpreter-massive", |b| {
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
    interpreter,
    b_large_tree_interpreter,
    b_huge_tree_interpreter,
    b_massive_tree_interpreter
);

#[cfg(feature = "llvm-jit")]
mod jit {
    use super::*;
    use eiche::{JitContext, llvm_jit::NumberType};

    fn b_large_tree_jit<T: NumberType>(c: &mut Criterion) {
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

    fn b_huge_tree_jit<T: NumberType>(c: &mut Criterion) {
        let tree = create_huge_benchmark_tree();
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
        c.bench_function(&format!("interval-jit-huge-{}", T::type_str()), |b| {
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

    fn b_massive_tree_jit<T: NumberType>(c: &mut Criterion) {
        let tree = create_massive_benchmark_tree();
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
        c.bench_function(&format!("interval-jit-massive-{}", T::type_str()), |b| {
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
        jit,
        b_large_tree_jit::<f32>,
        b_huge_tree_jit::<f32>,
        b_massive_tree_jit::<f32>,
        b_large_tree_jit::<f64>,
        b_huge_tree_jit::<f64>,
        b_massive_tree_jit::<f64>
    );
}

#[cfg(not(feature = "llvm-jit"))]
criterion_main!(interpreter);

#[cfg(feature = "llvm-jit")]
criterion_main!(interpreter, jit::jit);

/// Creates a very large tree for benchmarking. This tree is designed to:
/// - Use almost all available operations (arithmetic, trigonometric, comparison, conditional)
/// - Be mathematically valid (sqrt/log inputs are always positive)
/// - Have both depth and breadth for comprehensive evaluation testing
/// - Use multiple input variables (x, y, z, a, b, c) for realistic benchmarking
///
/// The tree is significantly larger than `create_large_tree` in dominator.rs,
/// making it suitable for stress-testing evaluation performance.
pub fn create_large_benchmark_tree() -> Tree {
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
