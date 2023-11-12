use asg::tree::Tree;
use criterion::{criterion_group, criterion_main, Criterion};

fn parse_large_tree() -> Tree {
    use asg::parsetree;
    parsetree!(
        (pow
         (min
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
                    (- (sqrt (+ (pow (- x 2.95) 2.) (pow (- y 2.05) 2.))) 3.67)
                    (min
                     (- (sqrt (+ (pow (- x 3.5) 2.) (pow (- y 3.5) 2.))) 2.234)
                     (min
                      (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
                      (- (sqrt (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.))) 5.243))))
                   (exp (pow (max
                              (- (sqrt (+ (pow (- x 2.95) 2.) (pow (- y 2.05) 2.))) 3.67)
                              (min
                               (- (sqrt (+ (pow (- x 3.5) 2.) (pow (- y 3.5) 2.))) 2.234)
                               (min
                                (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
                                (- (sqrt (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.))) 5.243))))
                         2.456))))
           (max
            (/ (+ (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a))
            (/ (- (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a)))))
         (min
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
                    (- (sqrt (+ (pow (- x 2.95) 2.) (pow (- y 2.05) 2.))) 3.67)
                    (min
                     (- (sqrt (+ (pow (- x 3.5) 2.) (pow (- y 3.5) 2.))) 2.234)
                     (min
                      (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
                      (- (sqrt (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.))) 5.243))))
                   (exp (pow (max
                              (- (sqrt (+ (pow (- x 2.95) 2.) (pow (- y 2.05) 2.))) 3.67)
                              (min
                               (- (sqrt (+ (pow (- x 3.5) 2.) (pow (- y 3.5) 2.))) 2.234)
                               (min
                                (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
                                (- (sqrt (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.))) 5.243))))
                         2.456))))
           (max
            (/ (+ (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a))
            (/ (- (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a))))))
    )
    .unwrap()
}

fn def_large_tree() -> Tree {
    use asg::deftree;
    deftree!(
        (pow
         (min
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
                    (- (sqrt (+ (pow (- x 2.95) 2.) (pow (- y 2.05) 2.))) 3.67)
                    (min
                     (- (sqrt (+ (pow (- x 3.5) 2.) (pow (- y 3.5) 2.))) 2.234)
                     (min
                      (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
                      (- (sqrt (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.))) 5.243))))
                   (exp (pow (max
                              (- (sqrt (+ (pow (- x 2.95) 2.) (pow (- y 2.05) 2.))) 3.67)
                              (min
                               (- (sqrt (+ (pow (- x 3.5) 2.) (pow (- y 3.5) 2.))) 2.234)
                               (min
                                (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
                                (- (sqrt (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.))) 5.243))))
                         2.456))))
           (max
            (/ (+ (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a))
            (/ (- (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a)))))
         (min
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
                    (- (sqrt (+ (pow (- x 2.95) 2.) (pow (- y 2.05) 2.))) 3.67)
                    (min
                     (- (sqrt (+ (pow (- x 3.5) 2.) (pow (- y 3.5) 2.))) 2.234)
                     (min
                      (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
                      (- (sqrt (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.))) 5.243))))
                   (exp (pow (max
                              (- (sqrt (+ (pow (- x 2.95) 2.) (pow (- y 2.05) 2.))) 3.67)
                              (min
                               (- (sqrt (+ (pow (- x 3.5) 2.) (pow (- y 3.5) 2.))) 2.234)
                               (min
                                (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
                                (- (sqrt (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.))) 5.243))))
                         2.456))))
           (max
            (/ (+ (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a))
            (/ (- (- b) (sqrt (- (pow b 2.) (* 4 (* a c))))) (* 2. a))))))
    )
}

fn tree_parse_perft(c: &mut Criterion) {
    c.bench_function("Parsing a lisp tree with 600 nodes", |b| {
        b.iter(|| {
            let _tree = parse_large_tree();
        })
    });
}

fn tree_build_perft(c: &mut Criterion) {
    c.bench_function("Building a lisp tree with 600 nodes", |b| {
        b.iter(|| {
            let _tree = def_large_tree();
        })
    });
}

criterion_group!(parser_benches, tree_parse_perft, tree_build_perft);
criterion_main!(parser_benches);
