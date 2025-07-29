use criterion::{Criterion, black_box, criterion_group, criterion_main};
use eiche::{Tree, deftree};

fn create_small_tree() -> Tree {
    deftree!(+ (pow (- x 2.95) 2.) (pow (- y 2.05) 2.))
        .unwrap()
        .compacted()
        .unwrap()
}

fn create_medium_tree() -> Tree {
    deftree!(max
             (+ (pow x 2.) (pow y 2.))
             (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.)))
    .unwrap()
    .compacted()
    .unwrap()
}

fn create_large_tree() -> Tree {
    deftree!(min
             (- (log (+
                      (min
                       (+ (sqrt (+ (pow (- x 2.95) 2.) (pow (- y 2.05) 2.))) 3.67)
                       (max
                        (- (sqrt (+ (pow (- x 3.5) 2.) (pow (- y 3.5) 2.))) 2.234)
                        (max
                         (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
                         (- (sqrt (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.))) 5.243))))
                      (exp (pow (min
                                 (- (sqrt (+ (pow (- x 2.95) 2.) (pow (- y 2.05) 2.))) 3.67)
                                 (max
                                  (- (sqrt (+ (pow (- x 3.5) 2.) (pow (- y 3.5) 2.))) 2.234)
                                  (max
                                   (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
                                   (- (sqrt (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.))) 5.243))))
                            2.456))))
              (min
               (/ (+ (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a))
               (/ (- (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a))))
             (+ (log (+
                      (max
                       (- (sqrt (+ (pow (- x 3.95) 2.) (pow (- y 3.05) 2.))) 5.67)
                       (min
                        (- (sqrt (+ (pow (- x 4.51) 2.) (pow (- y 4.51) 2.))) 2.1234)
                        (min
                         (- (sqrt (+ (pow x 2.1) (pow y 2.1))) 4.2432)
                         (- (sqrt (+ (pow (- x 2.512) 2.) (pow (- y 2.512) 2.1))) 5.1243))))
                      (exp (pow (max
                                 (- (sqrt (+ (pow (- x 2.65) 2.) (pow (- y 2.15) 2.))) 3.67)
                                 (min
                                  (- (sqrt (+ (pow (- x 3.65) 2.) (pow (- y 3.75) 2.))) 2.234)
                                  (min
                                   (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
                                   (- (sqrt (+ (pow (- x 2.35) 2.) (pow (- y 2.25) 2.))) 5.1243))))
                            2.1456))))
              (max
               (/ (+ (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a))
               (/ (- (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a)))))
    .unwrap()
    .compacted()
    .unwrap()
}

fn create_deep_chain() -> Tree {
    // Create a very deep chain: sin(cos(tan(log(exp(sqrt(abs(floor(...))))))))
    deftree!(sin (cos (tan (log (exp (sqrt (abs (floor (+ x y))))))))).unwrap()
}

fn create_wide_tree() -> Tree {
    // Create a tree with many parallel branches
    deftree!(+ (+ (+ (+ (+ (+ (+ x y) z) a) b) c) d) (+ (+ (+ e f) g) h)).unwrap()
}

fn create_shared_subtree() -> Tree {
    // Tree with significant subtree sharing
    deftree!(+ (* (+ x y) (+ x y)) (* (+ x y) (- x y)))
        .unwrap()
        .compacted()
        .unwrap()
}

fn create_very_large_tree() -> Tree {
    // Generate a larger tree by composing multiple operations
    deftree!(+
        (max (+ (pow x 2.) (pow y 2.)) (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.)))
        (min
            (sqrt (+ (pow (- x 1.5) 2.) (pow (- y 1.5) 2.)))
            (log (+ 1. (exp (+ (* 2. x) (* 3. y))))))
    )
    .unwrap()
    .compacted()
    .unwrap()
}

fn bench_dominator_small(c: &mut Criterion) {
    let tree = create_small_tree();
    c.bench_function("dominator_small", |b| {
        b.iter(|| {
            let result = black_box(&tree).control_dependence_sorted();
            black_box(result).unwrap()
        })
    });
}

fn bench_dominator_medium(c: &mut Criterion) {
    let tree = create_medium_tree();
    c.bench_function("dominator_medium", |b| {
        b.iter(|| {
            let result = black_box(&tree).control_dependence_sorted();
            black_box(result).unwrap()
        })
    });
}

fn bench_dominator_large(c: &mut Criterion) {
    let tree = create_large_tree();
    c.bench_function("dominator_large", |b| {
        b.iter(|| {
            let result = black_box(&tree).control_dependence_sorted();
            black_box(result).unwrap()
        })
    });
}

fn bench_dominator_deep_chain(c: &mut Criterion) {
    let tree = create_deep_chain();
    c.bench_function("dominator_deep_chain", |b| {
        b.iter(|| {
            let result = black_box(&tree).control_dependence_sorted();
            black_box(result).unwrap()
        })
    });
}

fn bench_dominator_wide_tree(c: &mut Criterion) {
    let tree = create_wide_tree();
    c.bench_function("dominator_wide_tree", |b| {
        b.iter(|| {
            let result = black_box(&tree).control_dependence_sorted();
            black_box(result).unwrap()
        })
    });
}

fn bench_dominator_shared_subtree(c: &mut Criterion) {
    let tree = create_shared_subtree();
    c.bench_function("dominator_shared_subtree", |b| {
        b.iter(|| {
            let result = black_box(&tree).control_dependence_sorted();
            black_box(result).unwrap()
        })
    });
}

fn bench_dominator_very_large(c: &mut Criterion) {
    let tree = create_very_large_tree();
    c.bench_function("dominator_very_large", |b| {
        b.iter(|| {
            let result = black_box(&tree).control_dependence_sorted();
            black_box(result).unwrap()
        })
    });
}

criterion_group!(
    benches,
    bench_dominator_small,
    bench_dominator_medium,
    bench_dominator_large,
    bench_dominator_deep_chain,
    bench_dominator_wide_tree,
    bench_dominator_shared_subtree,
    bench_dominator_very_large
);
criterion_main!(benches);
