#[cfg(test)]

mod tests {
    use std::collections::HashMap;

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
        match eval.run(&HashMap::new()) {
            Ok(val) => assert_eq!(val, pi),
            _ => assert!(false),
        }
    }

    #[test]
    fn pythagoras() {
        let triplets: [(f64, f64, f64); 6] = [
            (3., 4., 5.),
            (5., 12., 13.),
            (8., 15., 17.),
            (7., 24., 25.),
            (20., 21., 29.),
            (12., 35., 37.),
        ];
        for (x, y, expected) in triplets {
            let h = sqrt(pow(x.into(), 2.0.into()) + pow(y.into(), 2.0.into()));
            let mut eval = Evaluator::new(&h);
            match eval.run(&HashMap::new()) {
                Ok(val) => assert_eq!(val, expected),
                _ => assert!(false),
            }
        }
    }
}
