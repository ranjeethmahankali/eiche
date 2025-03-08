use criterion::{Criterion, criterion_group, criterion_main};
use eiche::{deftree, reduce};

fn b_hessian(c: &mut Criterion) {
    let tree = deftree!(sderiv (sderiv (- (+ (pow x 3) (pow y 3)) 5) xy) xy).unwrap();
    c.bench_function("reduce_hessian_matrix", |b| {
        b.iter(|| reduce(std::hint::black_box(tree.clone()), 256).unwrap())
    });
}

criterion_group!(bench, b_hessian);
criterion_main!(bench);
