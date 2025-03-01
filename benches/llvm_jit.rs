use criterion::{Criterion, criterion_group, criterion_main};
use eiche::{Deduplicater, Error, JitContext, JitFn, Pruner, Tree, ValueEvaluator, deftree, min};
use rand::{SeedableRng, rngs::StdRng};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

fn sample_range(range: (f64, f64), rng: &mut StdRng) -> f64 {
    use rand::Rng;
    range.0 + rng.random::<f64>() * (range.1 - range.0)
}

const _RADIUS_RANGE: (f64, f64) = (0.2, 2.);
const _X_RANGE: (f64, f64) = (0., 100.);
const _Y_RANGE: (f64, f64) = (0., 100.);
const _Z_RANGE: (f64, f64) = (0., 100.);
const _N_SPHERES: usize = 5000;
const _N_QUERIES: usize = 5000;

fn sphere_union() -> Tree {
    let mut rng = StdRng::seed_from_u64(42);
    let mut make_sphere = || -> Result<Tree, Error> {
        deftree!(- (sqrt (+ (+
                                 (pow (- x (const sample_range(_X_RANGE, &mut rng))) 2)
                                 (pow (- y (const sample_range(_Y_RANGE, &mut rng))) 2))
                              (pow (- z (const sample_range(_Z_RANGE, &mut rng))) 2)))
                     (const sample_range(_RADIUS_RANGE, &mut rng)))
    };
    let mut tree = make_sphere();
    for _ in 1.._N_SPHERES {
        tree = min(tree, make_sphere());
    }
    let tree = tree.unwrap();
    assert_eq!(tree.dims(), (1, 1));
    tree
}

fn benchmark_eval(
    values: &mut Vec<f64>,
    queries: &[[f64; 3]],
    eval: &mut ValueEvaluator,
) -> Duration {
    let before = Instant::now();
    values.extend(queries.iter().map(|coords| {
        eval.set_value('x', coords[0].into());
        eval.set_value('y', coords[1].into());
        eval.set_value('z', coords[2].into());
        let results = eval.run().unwrap();
        results[0].scalar().unwrap()
    }));
    Instant::now() - before
}

fn benchmark_jit(values: &mut Vec<f64>, queries: &[[f64; 3]], eval: &mut JitFn) -> Duration {
    let before = Instant::now();
    values.extend(queries.iter().map(|coords| {
        let results = eval.run(coords).unwrap();
        results[0]
    }));
    Instant::now() - before
}

fn b_perft_single() {
    let mut rng = StdRng::seed_from_u64(234);
    let queries: Vec<[f64; 3]> = (0.._N_QUERIES)
        .map(|_| {
            [
                sample_range(_X_RANGE, &mut rng),
                sample_range(_Y_RANGE, &mut rng),
                sample_range(_Z_RANGE, &mut rng),
            ]
        })
        .collect();
    let before = Instant::now();
    let tree = {
        let mut dedup = Deduplicater::new();
        let mut pruner = Pruner::new();
        sphere_union()
            .fold()
            .unwrap()
            .deduplicate(&mut dedup)
            .unwrap()
            .prune(&mut pruner)
            .unwrap()
    };
    println!(
        "Tree creation time: {}ms",
        (Instant::now() - before).as_millis()
    );
    let mut values1: Vec<f64> = Vec::with_capacity(_N_QUERIES);
    let mut eval = ValueEvaluator::new(&tree);
    println!(
        "Tree has {} nodes, evaluator allocated {} registers",
        tree.len(),
        eval.num_regs()
    );
    let evaltime = benchmark_eval(&mut values1, &queries, &mut eval);
    println!("ValueEvaluator time: {}ms", evaltime.as_millis());
    let mut values2: Vec<f64> = Vec::with_capacity(_N_QUERIES);
    let context = JitContext::default();
    let mut jiteval = {
        let before = Instant::now();
        let jiteval = tree.jit_compile(&context).unwrap();
        println!(
            "Compilation time: {}ms",
            (Instant::now() - before).as_millis()
        );
        jiteval
    };
    let jittime = benchmark_jit(&mut values2, &queries, &mut jiteval);
    println!("Jit time: {}ms", jittime.as_millis());
    let ratio = evaltime.as_millis() as f64 / jittime.as_millis() as f64;
    println!("Ratio: {}", ratio);
    assert_eq!(values1.len(), values2.len());
    for (l, r) in values1.iter().zip(values2.iter()) {
        eiche::assert_float_eq!(l, r, 1e-15);
    }
}
