#[cfg(test)]

mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use crate::tree::Node::*;
    use crate::tree::*;

    #[test]
    fn constant() {
        let pi: f64 = 3.14;
        let x: Tree = pi.into();
        match x.root() {
            Constant(val) if *val == pi => (),
            _ => assert!(false),
        }
        let mut eval = Evaluator::new(&x);
        match eval.run() {
            Ok(val) => assert_eq!(val, pi),
            _ => assert!(false),
        }
    }

    #[test]
    fn pythagoras() {
        const TRIPLETS: [(f64, f64, f64); 6] = [
            (3., 4., 5.),
            (5., 12., 13.),
            (8., 15., 17.),
            (7., 24., 25.),
            (20., 21., 29.),
            (12., 35., 37.),
        ];
        for (x, y, expected) in TRIPLETS {
            let h = sqrt(pow(x.into(), 2.0.into()) + pow(y.into(), 2.0.into()));
            let mut eval = Evaluator::new(&h);
            match eval.run() {
                Ok(val) => assert_eq!(val, expected),
                _ => assert!(false),
            }
        }
    }

    #[test]
    fn pythagoras_variables() {
        let x: Tree = 'x'.into();
        let y: Tree = 'y'.into();
        let h = sqrt(pow(x, 2.0.into()) + pow(y, 2.0.into()));
        let mut eval = Evaluator::new(&h);

        const TRIPLETS: [(f64, f64, f64); 6] = [
            (3., 4., 5.),
            (5., 12., 13.),
            (8., 15., 17.),
            (7., 24., 25.),
            (20., 21., 29.),
            (12., 35., 37.),
        ];

        for (xval, yval, expected) in TRIPLETS {
            eval.set_var('x', xval);
            eval.set_var('y', yval);
            match eval.run() {
                Ok(val) => assert_eq!(val, expected),
                _ => assert!(false),
            }
        }
    }

    #[test]
    fn trig_identity() {
        use rand::Rng;
        const PI_2: f64 = 2.0 * std::f64::consts::TAU;

        let sum: Tree = pow(sin('x'.into()), 2.0.into()) + pow(cos('x'.into()), 2.0.into());
        let mut eval = Evaluator::new(&sum);
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let x: f64 = PI_2 * rng.gen::<f64>();
            eval.set_var('x', x);
            match eval.run() {
                Ok(val) => assert!(f64::abs(val - 1.) < 1e-14),
                _ => assert!(false),
            }
        }
    }

    fn eval_test<F>(
        tree: Tree,
        mut expectedfn: F,
        vardata: &[(char, f64, f64)],
        samples_per_var: usize,
        epsilon: f64,
    ) where
        F: FnMut(&[f64]) -> Option<f64>,
    {
        use rand::Rng;
        let mut eval = Evaluator::new(&tree);
        let nvars = vardata.len();
        let mut indices = vec![0usize; nvars];
        let mut sample = Vec::<f64>::with_capacity(nvars);
        let mut rng = StdRng::seed_from_u64(42);
        while indices[0] <= samples_per_var {
            let vari = sample.len();
            let (label, lower, upper) = vardata[vari];
            let value = lower + rng.gen::<f64>() * (upper - lower);
            sample.push(value);
            eval.set_var(label, value);
            indices[vari] += 1;
            if vari < nvars - 1 {
                continue;
            }
            // We set all the variables. Run the test.
            let error = f64::abs(
                eval.run().expect("Unable to compute the actual value.")
                    - expectedfn(&sample[..]).expect("Unable to compute expected value."),
            );
            assert!(error <= epsilon);
            // Clean up the index stack.
            sample.pop();
            let mut vari = vari;
            while indices[vari] == samples_per_var && vari > 0 {
                if let Some(_) = sample.pop() {
                    indices[vari] = 0;
                    vari -= 1;
                } else {
                    assert!(false); // To ensure the logic of this test is correct.
                }
            }
        }
    }

    #[test]
    fn sum_test() {
        eval_test(
            {
                let xtree: Tree = 'x'.into();
                let ytree: Tree = 'y'.into();
                xtree + ytree
            },
            |vars: &[f64]| {
                if let [x, y] = vars[..] {
                    Some(x + y)
                } else {
                    None
                }
            },
            &[('x', -5., 5.), ('y', -5., 5.)],
            10,
            0.,
        );
    }

    #[test]
    fn evaluate_trees() {
        eval_test(
            pow(log(sin('x'.into()) + 2.0.into()), 3.0.into()) / (cos('x'.into()) + 2.0.into()),
            |vars: &[f64]| {
                if let [x] = vars[..] {
                    Some(
                        f64::powf(f64::log(f64::sin(x) + 2., std::f64::consts::E), 3.)
                            / (f64::cos(x) + 2.),
                    )
                } else {
                    None
                }
            },
            &[('x', -2.5, 2.5)],
            100,
            0.,
        );

        eval_test(
            {
                let s1 = {
                    let x: Tree = 'x'.into();
                    let y: Tree = 'y'.into();
                    let z: Tree = 'z'.into();
                    sqrt(
                        pow(x - 2.0.into(), 2.0.into())
                            + pow(y - 3.0.into(), 2.0.into())
                            + pow(z - 4.0.into(), 2.0.into()),
                    ) - 2.75.into()
                };
                let s2 = {
                    let x: Tree = 'x'.into();
                    let y: Tree = 'y'.into();
                    let z: Tree = 'z'.into();
                    sqrt(
                        pow(x + 2.0.into(), 2.0.into())
                            + pow(y - 3.0.into(), 2.0.into())
                            + pow(z - 4.0.into(), 2.0.into()),
                    ) - 4.0.into()
                };
                let s3 = {
                    let x: Tree = 'x'.into();
                    let y: Tree = 'y'.into();
                    let z: Tree = 'z'.into();
                    sqrt(
                        pow(x + 2.0.into(), 2.0.into())
                            + pow(y + 3.0.into(), 2.0.into())
                            + pow(z - 4.0.into(), 2.0.into()),
                    ) - 5.25.into()
                };
                max(min(s1, s2), s3)
            },
            |vars: &[f64]| {
                if let [x, y, z] = vars[..] {
                    let s1 = f64::sqrt(
                        f64::powf(x - 2., 2.) + f64::powf(y - 3., 2.) + f64::powf(z - 4., 2.),
                    ) - 2.75;
                    let s2 = f64::sqrt(
                        f64::powf(x + 2., 2.) + f64::powf(y - 3., 2.) + f64::powf(z - 4., 2.),
                    ) - 4.;
                    let s3 = f64::sqrt(
                        f64::powf(x + 2., 2.) + f64::powf(y + 3., 2.) + f64::powf(z - 4., 2.),
                    ) - 5.25;
                    Some(f64::max(f64::min(s1, s2), s3))
                } else {
                    None
                }
            },
            &[('x', -10., 10.), ('y', -9., 10.), ('z', -11., 12.)],
            20,
            0.,
        );
    }

    #[test]
    fn tree_string_formatting() {
        let tree = {
            let s1 = {
                let x: Tree = 'x'.into();
                let y: Tree = 'y'.into();
                let z: Tree = 'z'.into();
                sqrt(
                    pow(x - 2.0.into(), 2.0.into())
                        + pow(y - 3.0.into(), 2.0.into())
                        + pow(z - 4.0.into(), 2.0.into()),
                ) - 2.75.into()
            };
            let s2 = {
                let x: Tree = 'x'.into();
                let y: Tree = 'y'.into();
                let z: Tree = 'z'.into();
                sqrt(
                    pow(x + 2.0.into(), 2.0.into())
                        + pow(y - 3.0.into(), 2.0.into())
                        + pow(z - 4.0.into(), 2.0.into()),
                ) - 4.0.into()
            };
            let s3 = {
                let x: Tree = 'x'.into();
                let y: Tree = 'y'.into();
                let z: Tree = 'z'.into();
                sqrt(
                    pow(x + 2.0.into(), 2.0.into())
                        + pow(y + 3.0.into(), 2.0.into())
                        + pow(z - 4.0.into(), 2.0.into()),
                ) - 5.25.into()
            };
            max(min(s1, s2), s3)
        };
        assert_eq!(
            format!("{}", tree).trim(),
            "
[61] Max(40, 60)
 ├── [40] Min(19, 39)
 │    ├── [19] Subtract(17, 18)
 │    │    ├── [17] Sqrt(16)
 │    │    │    ├── [16] Add(10, 15)
 │    │    │    │    ├── [10] Add(4, 9)
 │    │    │    │    │    ├── [4] Pow(2, 3)
 │    │    │    │    │    │    ├── [2] Subtract(0, 1)
 │    │    │    │    │    │    │    ├── [0] Symbol(x)
 │    │    │    │    │    │    │    ├── [1] Constant(2)
 │    │    │    │    │    │    ├── [3] Constant(2)
 │    │    │    │    │    ├── [9] Pow(7, 8)
 │    │    │    │    │    │    ├── [7] Subtract(5, 6)
 │    │    │    │    │    │    │    ├── [5] Symbol(y)
 │    │    │    │    │    │    │    ├── [6] Constant(3)
 │    │    │    │    │    │    ├── [8] Constant(2)
 │    │    │    │    ├── [15] Pow(13, 14)
 │    │    │    │    │    ├── [13] Subtract(11, 12)
 │    │    │    │    │    │    ├── [11] Symbol(z)
 │    │    │    │    │    │    ├── [12] Constant(4)
 │    │    │    │    │    ├── [14] Constant(2)
 │    │    ├── [18] Constant(2.75)
 │    ├── [39] Subtract(37, 38)
 │    │    ├── [37] Sqrt(36)
 │    │    │    ├── [36] Add(30, 35)
 │    │    │    │    ├── [30] Add(24, 29)
 │    │    │    │    │    ├── [24] Pow(22, 23)
 │    │    │    │    │    │    ├── [22] Add(20, 21)
 │    │    │    │    │    │    │    ├── [20] Symbol(x)
 │    │    │    │    │    │    │    ├── [21] Constant(2)
 │    │    │    │    │    │    ├── [23] Constant(2)
 │    │    │    │    │    ├── [29] Pow(27, 28)
 │    │    │    │    │    │    ├── [27] Subtract(25, 26)
 │    │    │    │    │    │    │    ├── [25] Symbol(y)
 │    │    │    │    │    │    │    ├── [26] Constant(3)
 │    │    │    │    │    │    ├── [28] Constant(2)
 │    │    │    │    ├── [35] Pow(33, 34)
 │    │    │    │    │    ├── [33] Subtract(31, 32)
 │    │    │    │    │    │    ├── [31] Symbol(z)
 │    │    │    │    │    │    ├── [32] Constant(4)
 │    │    │    │    │    ├── [34] Constant(2)
 │    │    ├── [38] Constant(4)
 ├── [60] Subtract(58, 59)
 │    ├── [58] Sqrt(57)
 │    │    ├── [57] Add(51, 56)
 │    │    │    ├── [51] Add(45, 50)
 │    │    │    │    ├── [45] Pow(43, 44)
 │    │    │    │    │    ├── [43] Add(41, 42)
 │    │    │    │    │    │    ├── [41] Symbol(x)
 │    │    │    │    │    │    ├── [42] Constant(2)
 │    │    │    │    │    ├── [44] Constant(2)
 │    │    │    │    ├── [50] Pow(48, 49)
 │    │    │    │    │    ├── [48] Add(46, 47)
 │    │    │    │    │    │    ├── [46] Symbol(y)
 │    │    │    │    │    │    ├── [47] Constant(3)
 │    │    │    │    │    ├── [49] Constant(2)
 │    │    │    ├── [56] Pow(54, 55)
 │    │    │    │    ├── [54] Subtract(52, 53)
 │    │    │    │    │    ├── [52] Symbol(z)
 │    │    │    │    │    ├── [53] Constant(4)
 │    │    │    │    ├── [55] Constant(2)
 │    ├── [59] Constant(5.25)"
                .trim()
        );
    }

    #[test]
    fn constant_folding() {
        // Basic multiplication.
        let tree = {
            let a: Tree = 2.0.into();
            let b: Tree = 3.0.into();
            a * b
        };
        let tree = tree.fold_constants();
        assert_eq!(tree.len(), 1usize);
        assert_eq!(tree.root(), &Constant(2. * 3.));
        // More complicated tree.
        let tree = {
            let numerator: Tree = {
                let x: Tree = 'x'.into();
                let prod: Tree = {
                    let two: Tree = 2.0.into();
                    let three: Tree = 3.0.into();
                    two * three
                };
                x + prod
            };
            let denom: Tree = {
                let x: Tree = 'x'.into();
                let frac: Tree = {
                    let two: Tree = 2.0.into();
                    let five: Tree = 5.0.into();
                    let three: Tree = 3.0.into();
                    let nine: Tree = 9.0.into();
                    two / min(five, max(three, nine - 5.0.into()))
                };
                log(x + frac)
            };
            numerator / denom
        };
        let expected: Tree = {
            let numerator = {
                let x: Tree = 'x'.into();
                x + 6.0.into()
            };
            let denom = {
                let x: Tree = 'x'.into();
                log(x + 0.5.into())
            };
            numerator / denom
        };
        assert!(tree.len() > expected.len());
        let tree = tree.fold_constants();
        assert_eq!(tree, expected);
    }

    #[test]
    fn deduplication_1() {
        let maketree = || -> Tree {
            let s1 = {
                let x: Tree = 'x'.into();
                let y: Tree = 'y'.into();
                let z: Tree = 'z'.into();
                sqrt(
                    pow(x - 2.0.into(), 2.0.into())
                        + pow(y - 3.0.into(), 2.0.into())
                        + pow(z - 4.0.into(), 2.0.into()),
                ) - 2.75.into()
            };
            let s2 = {
                let x: Tree = 'x'.into();
                let y: Tree = 'y'.into();
                let z: Tree = 'z'.into();
                sqrt(
                    pow(x + 2.0.into(), 2.0.into())
                        + pow(y - 3.0.into(), 2.0.into())
                        + pow(z - 4.0.into(), 2.0.into()),
                ) - 4.0.into()
            };
            let s3 = {
                let x: Tree = 'x'.into();
                let y: Tree = 'y'.into();
                let z: Tree = 'z'.into();
                sqrt(
                    pow(x + 2.0.into(), 2.0.into())
                        + pow(y + 3.0.into(), 2.0.into())
                        + pow(z - 4.0.into(), 2.0.into()),
                ) - 5.25.into()
            };
            max(min(s1, s2), s3)
        };
        let tree = maketree();
        let nodup = maketree().deduplicate();
        assert!(tree.len() > nodup.len());
        let mut eval: Evaluator = Evaluator::new(&tree);
        eval_test(
            nodup,
            move |vars: &[f64]| -> Option<f64> {
                if let [x, y, z] = vars[..] {
                    eval.set_var('x', x);
                    eval.set_var('y', y);
                    eval.set_var('z', z);
                    let result = eval.run();
                    match result {
                        Ok(value) => Some(value),
                        Err(_) => None,
                    }
                } else {
                    None
                }
            },
            &[('x', -10., 10.), ('y', -9., 10.), ('z', -11., 12.)],
            20,
            0.,
        );
    }

    #[test]
    fn deduplication_2() {
        let maketree = || -> Tree {
            pow(log(sin('x'.into()) + 2.0.into()), 3.0.into()) / (cos('x'.into()) + 2.0.into())
        };
        let tree = maketree();
        let nodup = maketree().deduplicate();
        assert!(tree.len() > nodup.len());
        let mut eval: Evaluator = Evaluator::new(&tree);
        eval_test(
            nodup,
            move |vars: &[f64]| -> Option<f64> {
                if let [x] = vars[..] {
                    eval.set_var('x', x);
                    let result = eval.run();
                    match result {
                        Ok(value) => Some(value),
                        Err(_) => None,
                    }
                } else {
                    None
                }
            },
            &[('x', -10., 10.)],
            20,
            0.,
        );
    }

    #[test]
    fn deduplication_3() {
        let maketree = || -> Tree {
            (pow(sin('x'.into()), 2.0.into())
                + pow(cos('x'.into()), 2.0.into())
                + ((cos('x'.into()) * sin('x'.into())) * 2.0.into()))
                / (pow(sin('y'.into()), 2.0.into())
                    + pow(cos('y'.into()), 2.0.into())
                    + ((cos('y'.into()) * sin('y'.into())) * 2.0.into()))
        };
        let tree = maketree();
        let nodup = maketree().deduplicate();
        assert!(tree.len() > nodup.len());
        let mut eval: Evaluator = Evaluator::new(&tree);
        eval_test(
            nodup,
            move |vars: &[f64]| -> Option<f64> {
                if let [x, y] = vars[..] {
                    eval.set_var('x', x);
                    eval.set_var('y', y);
                    let result = eval.run();
                    match result {
                        Ok(value) => Some(value),
                        Err(_) => None,
                    }
                } else {
                    None
                }
            },
            &[('x', -10., 10.), ('y', -9., 10.)],
            20,
            0.,
        );
    }
}
