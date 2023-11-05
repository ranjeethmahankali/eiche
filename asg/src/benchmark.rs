use criterion::{criterion_group, criterion_main, Criterion};

fn tree_parse_perft(c: &mut Criterion) {
    use asg::{deftree, parser::parse_lisp};

    c.bench_function("Parsing large tree lisp", |b| {
        b.iter(|| {
            let _tree = deftree!(
                (- (sqrt (+ (* x x) (* y y))) 5.0)
            );
        })
    });
}

criterion_group!(parser_benches, tree_parse_perft);
criterion_main!(parser_benches);
