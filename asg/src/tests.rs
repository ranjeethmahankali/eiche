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

    fn eval_test<F>(tree: Tree, expectedfn: F, vardata: &[(char, f64, f64)], samples_per_var: usize)
    where
        F: Fn(&[f64]) -> Option<f64>,
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
            println!("vars: {:?}", sample);
            assert!(
                f64::abs(
                    dbg!(eval.run().expect("Unable to compute the actual value."))
                        - dbg!(expectedfn(&sample[..]).expect("Unable to compute expected value."))
                ) < 1e-12
            );
            sample.pop();
            if indices[vari] == samples_per_var {
                if let Some(_) = sample.pop() {
                    indices[vari] = 0;
                }
            }
        }
    }

    #[test]
    fn sum_test() {
        let xtree: Tree = 'x'.into();
        let ytree: Tree = 'y'.into();
        eval_test(
            xtree + ytree,
            |vars: &[f64]| {
                if let [x, y] = vars[..] {
                    Some(x + y)
                } else {
                    None
                }
            },
            &[('x', -5., 5.), ('y', -5., 5.)],
            10,
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
        );
    }
}
