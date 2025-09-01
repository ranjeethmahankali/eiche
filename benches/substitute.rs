use criterion::{Criterion, criterion_group, criterion_main};
use eiche::{Error, Tree, deftree, min};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::hint::black_box;

fn sample_range(range: (f64, f64), rng: &mut StdRng) -> f64 {
    use rand::Rng;
    range.0 + rng.random::<f64>() * (range.1 - range.0)
}

/// Create a large complex tree similar to the spheres benchmark but with more variety
fn create_large_tree() -> Tree {
    const RADIUS_RANGE: (f64, f64) = (0.2, 2.);
    const X_RANGE: (f64, f64) = (0., 100.);
    const Y_RANGE: (f64, f64) = (0., 100.);
    const Z_RANGE: (f64, f64) = (0., 100.);
    const N_SPHERES: usize = 256; // Large number for complexity
    let mut rng = StdRng::seed_from_u64(42);
    // Helper function to create a sphere at a random location
    let mut make_sphere = move || -> Result<Tree, Error> {
        deftree!(- (sqrt (+ (+
                             (pow (- 'x (const sample_range(X_RANGE, &mut rng))) 2)
                             (pow (- 'y (const sample_range(Y_RANGE, &mut rng))) 2))
                          (pow (- 'z (const sample_range(Z_RANGE, &mut rng))) 2)))
                 (const sample_range(RADIUS_RANGE, &mut rng)))
    };
    // Helper function to create more complex expressions with various operations
    let mut rng = StdRng::seed_from_u64(42);
    let mut make_complex_expr = || -> Result<Tree, Error> {
        let choice = rng.random_range(0..4);
        match choice {
            0 => deftree!(+ (* 'x (sin 'y)) (cos (* 'z 'x))), // Trigonometric
            1 => deftree!(/ (exp (* 'x 0.1)) (+ 1 (pow 'y 2))), // Exponential
            2 => deftree!(* (log (+ 'x 1)) (sqrt (+ (* 'y 'y) (* 'z 'z)))), // Logarithmic
            _ => deftree!(pow (+ (* 'x 'y) (* 'y 'z)) (/ 'z 10)), // Power
        }
    };
    // Start with a complex expression
    let mut tree = make_complex_expr();
    // Add spheres using min operation
    for _ in 0..N_SPHERES {
        tree = min(tree, make_sphere());
    }
    // Add more complex expressions mixed in
    for _ in 0..32 {
        tree = min(tree, make_complex_expr());
    }
    tree.expect("Cannot create large tree for benchmark")
        .compacted()
        .expect("Cannot compact the large tree")
}

/// Benchmark for substituting a single symbol: x -> (x + 1)
fn b_single_symbol_substitution(c: &mut Criterion) {
    let tree = create_large_tree();
    let old = deftree!('x).unwrap();
    let new = deftree!(+ 'x 1).unwrap();

    c.bench_function("substitute_single_symbol_x_to_x_plus_1", |b| {
        b.iter(|| {
            black_box(tree.clone())
                .substitute(black_box(&old), black_box(&new))
                .unwrap()
        })
    });
}

/// Benchmark for substituting three symbols sequentially: x -> x + 1, y -> y + 1, z -> z + 1
fn b_three_symbol_substitution(c: &mut Criterion) {
    let tree = create_large_tree();
    let old = deftree!(concat 'x 'y 'z).unwrap();
    let new = deftree!(concat (+ 1 'x) (+ 1 'y) (+ 1 'z)).unwrap();
    c.bench_function("substitute_three_symbols_sequential", |b| {
        b.iter(|| {
            black_box(tree.clone())
                .substitute(black_box(&old), black_box(&new))
                .unwrap()
        })
    });
}

/// Additional benchmark: substitute a more complex expression
fn b_complex_substitution(c: &mut Criterion) {
    let tree = create_large_tree();
    let old = deftree!(+ (* 'x 'y) (* 'y 'z)).unwrap(); // A more complex pattern
    let new = deftree!(* (+ 'x 'y) (+ 'y 'z)).unwrap(); // Replace with different structure
    c.bench_function("substitute_complex_expression", |b| {
        b.iter(|| {
            black_box(tree.clone())
                .substitute(black_box(&old), black_box(&new))
                .unwrap()
        })
    });
}

criterion_group!(
    bench,
    b_single_symbol_substitution,
    b_three_symbol_substitution,
    b_complex_substitution
);
criterion_main!(bench);
