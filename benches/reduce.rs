use criterion::{Criterion, criterion_group, criterion_main};
use eiche::{deftree, reduce};

fn b_hessian(c: &mut Criterion) {
    let tree = deftree!(sderiv (sderiv (- (+ (pow 'x 3) (pow 'y 3)) 5) 'xy) 'xy).unwrap();
    c.bench_function("reduce_hessian_matrix", |b| {
        b.iter(|| reduce(std::hint::black_box(tree.clone()), 256).unwrap())
    });
}

fn b_norm_vec_len(c: &mut Criterion) {
    let tree = deftree!(sqrt (+ (pow (/ 'x (sqrt (+ (pow 'x 2) (pow 'y 2)))) 2)
                              (pow (/ 'y (sqrt (+ (pow 'x 2) (pow 'y 2)))) 2)))
    .unwrap();
    c.bench_function("reduce_normalized_vector_length", |b| {
        b.iter(|| reduce(std::hint::black_box(tree.clone()), 256).unwrap())
    });
}

fn b_circle_gradient(c: &mut Criterion) {
    let tree = deftree!(sderiv (- (sqrt (+ (pow 'x 2) (pow 'y 2))) 3) 'xy).unwrap();
    c.bench_function("reduce_circle_gradient", |b| {
        b.iter(|| reduce(std::hint::black_box(tree.clone()), 256).unwrap())
    });
}

criterion_group!(bench, b_hessian, b_norm_vec_len, b_circle_gradient);
criterion_main!(bench);
