#[cfg(test)]
pub mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use crate::tree::{Node::*, *};
    use crate::{deftree, helper::*, parsetree};

    /// Helper function to evaluate the tree with randomly sampled
    /// variable values and compare the result to the one returned by
    /// the `expectedfn` for the same inputs. The values must be
    /// within `eps` of each other.
    ///
    /// Each variable is sampled within the range indicated by the
    /// corresponding entry in `vardata`. Each entry in vardata
    /// consists of the label of the symbol / variable, lower bound
    /// and upper bound.
    fn check_tree_eval<F>(
        tree: Tree,
        mut expectedfn: F,
        vardata: &[(char, f64, f64)],
        samples_per_var: usize,
        eps: f64,
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
            assert!(error <= eps);
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

    pub fn compare_trees(
        tree1: Tree,
        tree2: Tree,
        vardata: &[(char, f64, f64)],
        samples_per_var: usize,
        eps: f64,
    ) {
        use rand::Rng;
        let mut eval1 = Evaluator::new(&tree1);
        let mut eval2 = Evaluator::new(&tree2);
        let nvars = vardata.len();
        let mut indices = vec![0usize; nvars];
        let mut sample = Vec::<f64>::with_capacity(nvars);
        let mut rng = StdRng::seed_from_u64(42);
        while indices[0] <= samples_per_var {
            let vari = sample.len();
            let (label, lower, upper) = vardata[vari];
            let value = lower + rng.gen::<f64>() * (upper - lower);
            sample.push(value);
            eval1.set_var(label, value);
            eval2.set_var(label, value);
            indices[vari] += 1;
            if vari < nvars - 1 {
                continue;
            }
            let first = eval1.run().expect("Unable to compute the actual value.");
            let second = eval2.run().expect("Unable to compute expected value.");
            // println!("Vars: {:?} | Comparing: {} | {}", sample, first, second);
            // We set all the variables. Run the test.
            let error = f64::abs(first - second);
            assert!(error <= eps);
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
    fn constant() {
        let x: Tree = std::f64::consts::PI.into();
        assert!(matches!(x.root(), Ok(Constant(val)) if *val == std::f64::consts::PI));
        let mut eval = Evaluator::new(&x);
        match eval.run() {
            Ok(val) => assert_eq!(val, std::f64::consts::PI),
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
        let h = deftree!(sqrt (+ (pow x 2.) (pow y 2.)));
        let mut eval = Evaluator::new(&h);
        for (x, y, expected) in TRIPLETS {
            eval.set_var('x', x);
            eval.set_var('y', y);
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

        let sum = deftree!(+ (pow (sin x) 2.) (pow (cos x) 2.));
        let mut eval = Evaluator::new(&sum);
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let x: f64 = PI_2 * rng.gen::<f64>();
            eval.set_var('x', x);
            match eval.run() {
                Ok(val) => assert!(f64::abs(val - 1.) < 1e-14),
                _ => assert!(false),
            }
        }
    }

    #[test]
    fn sum_test() {
        check_tree_eval(
            deftree!(+ x y),
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
        check_tree_eval(
            deftree!(/ (pow (log (+ (sin x) 2.)) 3.) (+ (cos x) 2.)),
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
        check_tree_eval(
            deftree!(
                (max (min
                      (- (sqrt (+ (+ (pow (- x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 2.75)
                      (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 4.))
                 (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (+ y 3.) 2.)) (pow (- z 4.) 2.))) 5.25))
            ),
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
        let tree = deftree!(
            (max (min
                  (- (sqrt (+ (+ (pow (- x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 2.75)
                  (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 4.))
             (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (+ y 3.) 2.)) (pow (- z 4.) 2.))) 5.25))
        );
        assert_eq!(
            format!("{}", tree).trim(),
            "
[61] Max(40, 60)
 ├── [40] Min(19, 39)
 │    ├── [19] Subtract(17, 18)
 │    │    ├── [17] Sqrt(16)
 │    │    │    └── [16] Add(10, 15)
 │    │    │         ├── [10] Add(4, 9)
 │    │    │         │    ├── [4] Pow(2, 3)
 │    │    │         │    │    ├── [2] Subtract(0, 1)
 │    │    │         │    │    │    ├── [0] Symbol(x)
 │    │    │         │    │    │    └── [1] Constant(2)
 │    │    │         │    │    └── [3] Constant(2)
 │    │    │         │    └── [9] Pow(7, 8)
 │    │    │         │         ├── [7] Subtract(5, 6)
 │    │    │         │         │    ├── [5] Symbol(y)
 │    │    │         │         │    └── [6] Constant(3)
 │    │    │         │         └── [8] Constant(2)
 │    │    │         └── [15] Pow(13, 14)
 │    │    │              ├── [13] Subtract(11, 12)
 │    │    │              │    ├── [11] Symbol(z)
 │    │    │              │    └── [12] Constant(4)
 │    │    │              └── [14] Constant(2)
 │    │    └── [18] Constant(2.75)
 │    └── [39] Subtract(37, 38)
 │         ├── [37] Sqrt(36)
 │         │    └── [36] Add(30, 35)
 │         │         ├── [30] Add(24, 29)
 │         │         │    ├── [24] Pow(22, 23)
 │         │         │    │    ├── [22] Add(20, 21)
 │         │         │    │    │    ├── [20] Symbol(x)
 │         │         │    │    │    └── [21] Constant(2)
 │         │         │    │    └── [23] Constant(2)
 │         │         │    └── [29] Pow(27, 28)
 │         │         │         ├── [27] Subtract(25, 26)
 │         │         │         │    ├── [25] Symbol(y)
 │         │         │         │    └── [26] Constant(3)
 │         │         │         └── [28] Constant(2)
 │         │         └── [35] Pow(33, 34)
 │         │              ├── [33] Subtract(31, 32)
 │         │              │    ├── [31] Symbol(z)
 │         │              │    └── [32] Constant(4)
 │         │              └── [34] Constant(2)
 │         └── [38] Constant(4)
 └── [60] Subtract(58, 59)
      ├── [58] Sqrt(57)
      │    └── [57] Add(51, 56)
      │         ├── [51] Add(45, 50)
      │         │    ├── [45] Pow(43, 44)
      │         │    │    ├── [43] Add(41, 42)
      │         │    │    │    ├── [41] Symbol(x)
      │         │    │    │    └── [42] Constant(2)
      │         │    │    └── [44] Constant(2)
      │         │    └── [50] Pow(48, 49)
      │         │         ├── [48] Add(46, 47)
      │         │         │    ├── [46] Symbol(y)
      │         │         │    └── [47] Constant(3)
      │         │         └── [49] Constant(2)
      │         └── [56] Pow(54, 55)
      │              ├── [54] Subtract(52, 53)
      │              │    ├── [52] Symbol(z)
      │              │    └── [53] Constant(4)
      │              └── [55] Constant(2)
      └── [59] Constant(5.25)"
                .trim()
        );
        let tree = tree.deduplicate().unwrap();
        assert_eq!(
            format!("{}", tree).trim(),
            "
[31] Max(23, 30)
 ├── [23] Min(16, 22)
 │    ├── [16] Subtract(14, 15)
 │    │    ├── [14] Sqrt(13)
 │    │    │    └── [13] Add(8, 12)
 │    │    │         ├── [8] Add(3, 7)
 │    │    │         │    ├── [3] Pow(2, 1)
 │    │    │         │    │    ├── [2] Subtract(0, 1)
 │    │    │         │    │    │    ├── [0] Symbol(x)
 │    │    │         │    │    │    └── [1] Constant(2)
 │    │    │         │    │    └── [1] Constant(2)
 │    │    │         │    └── [7] Pow(6, 1)
 │    │    │         │         ├── [6] Subtract(4, 5)
 │    │    │         │         │    ├── [4] Symbol(y)
 │    │    │         │         │    └── [5] Constant(3)
 │    │    │         │         └── [1] Constant(2)
 │    │    │         └── [12] Pow(11, 1)
 │    │    │              ├── [11] Subtract(9, 10)
 │    │    │              │    ├── [9] Symbol(z)
 │    │    │              │    └── [10] Constant(4)
 │    │    │              └── [1] Constant(2)
 │    │    └── [15] Constant(2.75)
 │    └── [22] Subtract(21, 10)
 │         ├── [21] Sqrt(20)
 │         │    └── [20] Add(19, 12)
 │         │         ├── [19] Add(18, 7)
 │         │         │    ├── [18] Pow(17, 1)
 │         │         │    │    ├── [17] Add(0, 1)
 │         │         │    │    │    ├── [0] Symbol(x)
 │         │         │    │    │    └── [1] Constant(2)
 │         │         │    │    └── [1] Constant(2)
 │         │         │    └── [7] Pow(6, 1)
 │         │         │         ├── [6] Subtract(4, 5)
 │         │         │         │    ├── [4] Symbol(y)
 │         │         │         │    └── [5] Constant(3)
 │         │         │         └── [1] Constant(2)
 │         │         └── [12] Pow(11, 1)
 │         │              ├── [11] Subtract(9, 10)
 │         │              │    ├── [9] Symbol(z)
 │         │              │    └── [10] Constant(4)
 │         │              └── [1] Constant(2)
 │         └── [10] Constant(4)
 └── [30] Subtract(28, 29)
      ├── [28] Sqrt(27)
      │    └── [27] Add(26, 12)
      │         ├── [26] Add(18, 25)
      │         │    ├── [18] Pow(17, 1)
      │         │    │    ├── [17] Add(0, 1)
      │         │    │    │    ├── [0] Symbol(x)
      │         │    │    │    └── [1] Constant(2)
      │         │    │    └── [1] Constant(2)
      │         │    └── [25] Pow(24, 1)
      │         │         ├── [24] Add(4, 5)
      │         │         │    ├── [4] Symbol(y)
      │         │         │    └── [5] Constant(3)
      │         │         └── [1] Constant(2)
      │         └── [12] Pow(11, 1)
      │              ├── [11] Subtract(9, 10)
      │              │    ├── [9] Symbol(z)
      │              │    └── [10] Constant(4)
      │              └── [1] Constant(2)
      └── [29] Constant(5.25)"
                .trim()
        );
    }

    #[test]
    fn constant_folding() {
        // Basic multiplication.
        let tree = deftree!(* 2. 3.).fold_constants().unwrap();
        assert_eq!(tree.len(), 1usize);
        assert!(matches!(tree.root(), Ok(Constant(val)) if *val == 2.* 3.));
        // More complicated tree.
        let tree = deftree!(
            (/
             (+ x (* 2. 3.))
             (log (+ x (/ 2. (min 5. (max 3. (- 9. 5.)))))))
        );
        let expected = deftree!(/ (+ x 6.) (log (+ x 0.5)));
        assert!(tree.len() > expected.len());
        let tree = tree.fold_constants().unwrap();
        assert_eq!(tree, expected);
        compare_trees(tree, expected, &[('x', 0.1, 10.)], 100, 0.);
    }

    #[test]
    fn deduplication_1() {
        let tree = deftree!(
            (max (min
                  (- (sqrt (+ (+ (pow (- x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 2.75)
                  (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 4.))
             (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (+ y 3.) 2.)) (pow (- z 4.) 2.))) 5.25))
        );
        let nodup = tree.clone().deduplicate().unwrap();
        assert!(tree.len() > nodup.len());
        assert_eq!(nodup.len(), 32);
        compare_trees(
            tree,
            nodup,
            &[('x', -10., 10.), ('y', -9., 10.), ('z', -11., 12.)],
            20,
            0.,
        );
    }

    #[test]
    fn deduplication_2() {
        let tree = deftree!(/ (pow (log (+ (sin x) 2.)) 3.) (+ (cos x) 2.));
        let nodup = tree.clone().deduplicate().unwrap();
        assert!(tree.len() > nodup.len());
        assert_eq!(nodup.len(), 10);
        compare_trees(tree, nodup, &[('x', -10., 10.)], 400, 0.);
    }

    #[test]
    fn deduplication_3() {
        let tree = deftree!(
            (/
             (+ (pow (sin x) 2.) (+ (pow (cos x) 2.) (* 2. (* (sin x) (cos x)))))
             (+ (pow (sin y) 2.) (+ (pow (cos y) 2.) (* 2. (* (sin y) (cos y))))))
        );
        let nodup = tree.clone().deduplicate().unwrap();
        assert!(tree.len() > nodup.len());
        assert_eq!(nodup.len(), 20);
        compare_trees(tree, nodup, &[('x', -10., 10.), ('y', -9., 10.)], 20, 0.);
    }

    #[test]
    fn depth_traverse() {
        let mut walker = DepthWalker::new();
        {
            let tree = deftree!(+ (pow x 2.) (pow y 2.));
            // Make sure two successive traversal yield the same nodes.
            let a: Vec<_> = walker
                .walk_tree(&tree, true, NodeOrdering::Original)
                .map(|(index, parent)| (index, parent))
                .collect();
            let b: Vec<_> = walker
                .walk_tree(&tree, true, NodeOrdering::Original)
                .map(|(index, parent)| (index, parent))
                .collect();
            assert_eq!(a, b);
        }
        {
            // Make sure the same TraverseDepth can be used on multiple trees.
            let tree = deftree!(+ (pow x 3.) (pow y 3.));
            let a: Vec<_> = walker
                .walk_tree(&tree, true, NodeOrdering::Original)
                .map(|(index, parent)| (index, parent))
                .collect();
            let tree2 = tree.clone();
            let b: Vec<_> = walker
                .walk_tree(&tree2, true, NodeOrdering::Original)
                .map(|(index, parent)| (index, parent))
                .collect();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn tree_from_nodes() {
        use BinaryOp::*;
        use UnaryOp::*;
        // Nodes in order.
        match Tree::from_nodes(vec![
            Symbol('x'),            // 0
            Constant(2.245),        // 1
            Binary(Add, 0, 1),      // 2
            Symbol('y'),            // 3
            Unary(Sqrt, 3),         // 4
            Binary(Multiply, 2, 4), // 5
        ]) {
            Ok(tree) => {
                assert_eq!(tree.len(), 6);
            }
            Err(_) => assert!(false),
        };
        // Nodes out of order.
        assert!(matches!(
            Tree::from_nodes(vec![
                Symbol('x'),            // 0
                Binary(Add, 0, 1),      // 1
                Constant(2.245),        // 2
                Binary(Multiply, 2, 4), // 3
                Symbol('y'),            // 4
                Unary(Sqrt, 3),         // 5
            ]),
            Err(TreeError::WrongNodeOrder)
        ));
    }

    #[test]
    fn recursive_compare() {
        use BinaryOp::*;
        {
            // Check if 'Add' node with mirrored inputs is compared
            // correctly.
            let mut nodes = vec![
                Symbol('y'),            // 0
                Symbol('x'),            // 1
                Binary(Add, 0, 1),      // 2
                Symbol('x'),            // 3
                Symbol('y'),            // 4
                Binary(Add, 3, 4),      // 5
                Binary(Add, 5, 2),      // 6
                Binary(Add, 2, 2),      // 7
                Binary(Multiply, 6, 7), // 8
            ];
            let mut walker1 = DepthWalker::new();
            let mut walker2 = DepthWalker::new();
            fn check_tree(nodes: &Vec<Node>) {
                let tree = Tree::from_nodes(nodes.clone());
                match tree {
                    Ok(tree) => {
                        assert_eq!(tree.len(), nodes.len());
                    }
                    Err(_) => assert!(false),
                };
            }
            check_tree(&nodes);
            assert!(equivalent(2, 5, &nodes, &mut walker1, &mut walker2));
            assert!(equivalent(6, 7, &nodes, &mut walker1, &mut walker2));
            // Try more mirroring
            nodes[6] = Binary(Add, 2, 5);
            check_tree(&nodes);
            assert!(equivalent(2, 5, &nodes, &mut walker1, &mut walker2));
            assert!(equivalent(6, 7, &nodes, &mut walker1, &mut walker2));
            // Multiply node with mirrored inputs.
            nodes[2] = Binary(Multiply, 0, 1);
            nodes[5] = Binary(Multiply, 3, 4);
            check_tree(&nodes);
            assert!(equivalent(2, 5, &nodes, &mut walker1, &mut walker2));
            assert!(equivalent(6, 7, &nodes, &mut walker1, &mut walker2));
            // Min node with mirrored inputs.
            nodes[2] = Binary(Min, 0, 1);
            nodes[5] = Binary(Min, 3, 4);
            check_tree(&nodes);
            assert!(equivalent(2, 5, &nodes, &mut walker1, &mut walker2));
            assert!(equivalent(6, 7, &nodes, &mut walker1, &mut walker2));
            // Max node with mirrored inputs.
            nodes[2] = Binary(Max, 0, 1);
            nodes[5] = Binary(Max, 3, 4);
            check_tree(&nodes);
            assert!(equivalent(2, 5, &nodes, &mut walker1, &mut walker2));
            assert!(equivalent(6, 7, &nodes, &mut walker1, &mut walker2));
            // Subtract node with mirrored inputs.
            nodes[2] = Binary(Subtract, 0, 1);
            nodes[5] = Binary(Subtract, 3, 4);
            check_tree(&nodes);
            assert!(!equivalent(2, 5, &nodes, &mut walker1, &mut walker2));
            assert!(!equivalent(6, 7, &nodes, &mut walker1, &mut walker2));
            // Divide node with mirrored inputs.
            nodes[2] = Binary(Divide, 0, 1);
            nodes[5] = Binary(Divide, 3, 4);
            check_tree(&nodes);
            assert!(!equivalent(2, 5, &nodes, &mut walker1, &mut walker2));
            assert!(!equivalent(6, 7, &nodes, &mut walker1, &mut walker2));
            // Pow node with mirrored inputs.
            nodes[2] = Binary(Pow, 0, 1);
            nodes[5] = Binary(Pow, 3, 4);
            check_tree(&nodes);
            assert!(!equivalent(2, 5, &nodes, &mut walker1, &mut walker2));
            assert!(!equivalent(6, 7, &nodes, &mut walker1, &mut walker2));
        }
    }

    #[test]
    fn tree_parsing() {
        let tree = parsetree!(
            (- (sqrt (+ (pow x 2.) (pow y 2.))) 6.0)
        )
        .unwrap();
        assert_eq!(tree.len(), 10);
        assert_eq!(
            format!("{}", tree).trim(),
            "
[9] Subtract(7, 8)
 ├── [7] Sqrt(6)
 │    └── [6] Add(2, 5)
 │         ├── [2] Pow(0, 1)
 │         │    ├── [0] Symbol(x)
 │         │    └── [1] Constant(2)
 │         └── [5] Pow(3, 4)
 │              ├── [3] Symbol(y)
 │              └── [4] Constant(2)
 └── [8] Constant(6)
"
            .trim()
        );
        // Slightly larger tree written over mutliple lines.
        let tree = parsetree!(
            (min
             (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24)
             (- (sqrt (+ (pow (- x 2.5) 2.) (pow (- y 2.5) 2.))) 5.234)
            )
        )
        .unwrap();
        assert_eq!(tree.len(), 25);
        assert_eq!(
            format!("{}", tree).trim(),
            "
[24] Min(9, 23)
 ├── [9] Subtract(7, 8)
 │    ├── [7] Sqrt(6)
 │    │    └── [6] Add(2, 5)
 │    │         ├── [2] Pow(0, 1)
 │    │         │    ├── [0] Symbol(x)
 │    │         │    └── [1] Constant(2)
 │    │         └── [5] Pow(3, 4)
 │    │              ├── [3] Symbol(y)
 │    │              └── [4] Constant(2)
 │    └── [8] Constant(4.24)
 └── [23] Subtract(21, 22)
      ├── [21] Sqrt(20)
      │    └── [20] Add(14, 19)
      │         ├── [14] Pow(12, 13)
      │         │    ├── [12] Subtract(10, 11)
      │         │    │    ├── [10] Symbol(x)
      │         │    │    └── [11] Constant(2.5)
      │         │    └── [13] Constant(2)
      │         └── [19] Pow(17, 18)
      │              ├── [17] Subtract(15, 16)
      │              │    ├── [15] Symbol(y)
      │              │    └── [16] Constant(2.5)
      │              └── [18] Constant(2)
      └── [22] Constant(5.234)"
                .trim()
        );
    }

    #[test]
    fn parse_tree_with_comments() {
        let tree = parsetree!(
            (min /*IMPORTANT: Do not remove this comment.*/
             (- (sqrt (+ (pow x 2.) (pow y 2.))) 4.24) // IMPORTANT: Do not remove this comment.
             // IMPORTANT: Do not remove this comment.
             (- (sqrt (+ (pow (- x 2.5) 2.) /*IMPORTANT: Do not remove this comment.*/ (pow (- y 2.5) 2.))) 5.234)
            )
        )
        .unwrap();
        assert_eq!(tree.len(), 25);
    }

    #[test]
    fn parse_large_tree() {
        let tree = parsetree!(
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
        )
        .unwrap();
        assert_eq!(
            format!("{}", tree).trim(),
            "
[302] Min(150, 301)
 ├── [150] Subtract(114, 149)
 │    ├── [114] Log(113)
 │    │    └── [113] Add(54, 112)
 │    │         ├── [54] Min(13, 53)
 │    │         │    ├── [13] Add(11, 12)
 │    │         │    │    ├── [11] Sqrt(10)
 │    │         │    │    │    └── [10] Add(4, 9)
 │    │         │    │    │         ├── [4] Pow(2, 3)
 │    │         │    │    │         │    ├── [2] Subtract(0, 1)
 │    │         │    │    │         │    │    ├── [0] Symbol(x)
 │    │         │    │    │         │    │    └── [1] Constant(2.95)
 │    │         │    │    │         │    └── [3] Constant(2)
 │    │         │    │    │         └── [9] Pow(7, 8)
 │    │         │    │    │              ├── [7] Subtract(5, 6)
 │    │         │    │    │              │    ├── [5] Symbol(y)
 │    │         │    │    │              │    └── [6] Constant(2.05)
 │    │         │    │    │              └── [8] Constant(2)
 │    │         │    │    └── [12] Constant(3.67)
 │    │         │    └── [53] Max(27, 52)
 │    │         │         ├── [27] Subtract(25, 26)
 │    │         │         │    ├── [25] Sqrt(24)
 │    │         │         │    │    └── [24] Add(18, 23)
 │    │         │         │    │         ├── [18] Pow(16, 17)
 │    │         │         │    │         │    ├── [16] Subtract(14, 15)
 │    │         │         │    │         │    │    ├── [14] Symbol(x)
 │    │         │         │    │         │    │    └── [15] Constant(3.5)
 │    │         │         │    │         │    └── [17] Constant(2)
 │    │         │         │    │         └── [23] Pow(21, 22)
 │    │         │         │    │              ├── [21] Subtract(19, 20)
 │    │         │         │    │              │    ├── [19] Symbol(y)
 │    │         │         │    │              │    └── [20] Constant(3.5)
 │    │         │         │    │              └── [22] Constant(2)
 │    │         │         │    └── [26] Constant(2.234)
 │    │         │         └── [52] Max(37, 51)
 │    │         │              ├── [37] Subtract(35, 36)
 │    │         │              │    ├── [35] Sqrt(34)
 │    │         │              │    │    └── [34] Add(30, 33)
 │    │         │              │    │         ├── [30] Pow(28, 29)
 │    │         │              │    │         │    ├── [28] Symbol(x)
 │    │         │              │    │         │    └── [29] Constant(2)
 │    │         │              │    │         └── [33] Pow(31, 32)
 │    │         │              │    │              ├── [31] Symbol(y)
 │    │         │              │    │              └── [32] Constant(2)
 │    │         │              │    └── [36] Constant(4.24)
 │    │         │              └── [51] Subtract(49, 50)
 │    │         │                   ├── [49] Sqrt(48)
 │    │         │                   │    └── [48] Add(42, 47)
 │    │         │                   │         ├── [42] Pow(40, 41)
 │    │         │                   │         │    ├── [40] Subtract(38, 39)
 │    │         │                   │         │    │    ├── [38] Symbol(x)
 │    │         │                   │         │    │    └── [39] Constant(2.5)
 │    │         │                   │         │    └── [41] Constant(2)
 │    │         │                   │         └── [47] Pow(45, 46)
 │    │         │                   │              ├── [45] Subtract(43, 44)
 │    │         │                   │              │    ├── [43] Symbol(y)
 │    │         │                   │              │    └── [44] Constant(2.5)
 │    │         │                   │              └── [46] Constant(2)
 │    │         │                   └── [50] Constant(5.243)
 │    │         └── [112] Exp(111)
 │    │              └── [111] Pow(109, 110)
 │    │                   ├── [109] Min(68, 108)
 │    │                   │    ├── [68] Subtract(66, 67)
 │    │                   │    │    ├── [66] Sqrt(65)
 │    │                   │    │    │    └── [65] Add(59, 64)
 │    │                   │    │    │         ├── [59] Pow(57, 58)
 │    │                   │    │    │         │    ├── [57] Subtract(55, 56)
 │    │                   │    │    │         │    │    ├── [55] Symbol(x)
 │    │                   │    │    │         │    │    └── [56] Constant(2.95)
 │    │                   │    │    │         │    └── [58] Constant(2)
 │    │                   │    │    │         └── [64] Pow(62, 63)
 │    │                   │    │    │              ├── [62] Subtract(60, 61)
 │    │                   │    │    │              │    ├── [60] Symbol(y)
 │    │                   │    │    │              │    └── [61] Constant(2.05)
 │    │                   │    │    │              └── [63] Constant(2)
 │    │                   │    │    └── [67] Constant(3.67)
 │    │                   │    └── [108] Max(82, 107)
 │    │                   │         ├── [82] Subtract(80, 81)
 │    │                   │         │    ├── [80] Sqrt(79)
 │    │                   │         │    │    └── [79] Add(73, 78)
 │    │                   │         │    │         ├── [73] Pow(71, 72)
 │    │                   │         │    │         │    ├── [71] Subtract(69, 70)
 │    │                   │         │    │         │    │    ├── [69] Symbol(x)
 │    │                   │         │    │         │    │    └── [70] Constant(3.5)
 │    │                   │         │    │         │    └── [72] Constant(2)
 │    │                   │         │    │         └── [78] Pow(76, 77)
 │    │                   │         │    │              ├── [76] Subtract(74, 75)
 │    │                   │         │    │              │    ├── [74] Symbol(y)
 │    │                   │         │    │              │    └── [75] Constant(3.5)
 │    │                   │         │    │              └── [77] Constant(2)
 │    │                   │         │    └── [81] Constant(2.234)
 │    │                   │         └── [107] Max(92, 106)
 │    │                   │              ├── [92] Subtract(90, 91)
 │    │                   │              │    ├── [90] Sqrt(89)
 │    │                   │              │    │    └── [89] Add(85, 88)
 │    │                   │              │    │         ├── [85] Pow(83, 84)
 │    │                   │              │    │         │    ├── [83] Symbol(x)
 │    │                   │              │    │         │    └── [84] Constant(2)
 │    │                   │              │    │         └── [88] Pow(86, 87)
 │    │                   │              │    │              ├── [86] Symbol(y)
 │    │                   │              │    │              └── [87] Constant(2)
 │    │                   │              │    └── [91] Constant(4.24)
 │    │                   │              └── [106] Subtract(104, 105)
 │    │                   │                   ├── [104] Sqrt(103)
 │    │                   │                   │    └── [103] Add(97, 102)
 │    │                   │                   │         ├── [97] Pow(95, 96)
 │    │                   │                   │         │    ├── [95] Subtract(93, 94)
 │    │                   │                   │         │    │    ├── [93] Symbol(x)
 │    │                   │                   │         │    │    └── [94] Constant(2.5)
 │    │                   │                   │         │    └── [96] Constant(2)
 │    │                   │                   │         └── [102] Pow(100, 101)
 │    │                   │                   │              ├── [100] Subtract(98, 99)
 │    │                   │                   │              │    ├── [98] Symbol(y)
 │    │                   │                   │              │    └── [99] Constant(2.5)
 │    │                   │                   │              └── [101] Constant(2)
 │    │                   │                   └── [105] Constant(5.243)
 │    │                   └── [110] Constant(2.456)
 │    └── [149] Min(131, 148)
 │         ├── [131] Divide(127, 130)
 │         │    ├── [127] Add(116, 126)
 │         │    │    ├── [116] Negate(115)
 │         │    │    │    └── [115] Symbol(b)
 │         │    │    └── [126] Sqrt(125)
 │         │    │         └── [125] Subtract(119, 124)
 │         │    │              ├── [119] Pow(117, 118)
 │         │    │              │    ├── [117] Symbol(b)
 │         │    │              │    └── [118] Constant(2)
 │         │    │              └── [124] Multiply(120, 123)
 │         │    │                   ├── [120] Constant(4)
 │         │    │                   └── [123] Multiply(121, 122)
 │         │    │                        ├── [121] Symbol(a)
 │         │    │                        └── [122] Symbol(c)
 │         │    └── [130] Multiply(128, 129)
 │         │         ├── [128] Constant(2)
 │         │         └── [129] Symbol(a)
 │         └── [148] Divide(144, 147)
 │              ├── [144] Subtract(133, 143)
 │              │    ├── [133] Negate(132)
 │              │    │    └── [132] Symbol(b)
 │              │    └── [143] Sqrt(142)
 │              │         └── [142] Subtract(136, 141)
 │              │              ├── [136] Pow(134, 135)
 │              │              │    ├── [134] Symbol(b)
 │              │              │    └── [135] Constant(2)
 │              │              └── [141] Multiply(137, 140)
 │              │                   ├── [137] Constant(4)
 │              │                   └── [140] Multiply(138, 139)
 │              │                        ├── [138] Symbol(a)
 │              │                        └── [139] Symbol(c)
 │              └── [147] Multiply(145, 146)
 │                   ├── [145] Constant(2)
 │                   └── [146] Symbol(a)
 └── [301] Add(265, 300)
      ├── [265] Log(264)
      │    └── [264] Add(205, 263)
      │         ├── [205] Max(164, 204)
      │         │    ├── [164] Subtract(162, 163)
      │         │    │    ├── [162] Sqrt(161)
      │         │    │    │    └── [161] Add(155, 160)
      │         │    │    │         ├── [155] Pow(153, 154)
      │         │    │    │         │    ├── [153] Subtract(151, 152)
      │         │    │    │         │    │    ├── [151] Symbol(x)
      │         │    │    │         │    │    └── [152] Constant(2.95)
      │         │    │    │         │    └── [154] Constant(2)
      │         │    │    │         └── [160] Pow(158, 159)
      │         │    │    │              ├── [158] Subtract(156, 157)
      │         │    │    │              │    ├── [156] Symbol(y)
      │         │    │    │              │    └── [157] Constant(2.05)
      │         │    │    │              └── [159] Constant(2)
      │         │    │    └── [163] Constant(3.67)
      │         │    └── [204] Min(178, 203)
      │         │         ├── [178] Subtract(176, 177)
      │         │         │    ├── [176] Sqrt(175)
      │         │         │    │    └── [175] Add(169, 174)
      │         │         │    │         ├── [169] Pow(167, 168)
      │         │         │    │         │    ├── [167] Subtract(165, 166)
      │         │         │    │         │    │    ├── [165] Symbol(x)
      │         │         │    │         │    │    └── [166] Constant(3.5)
      │         │         │    │         │    └── [168] Constant(2)
      │         │         │    │         └── [174] Pow(172, 173)
      │         │         │    │              ├── [172] Subtract(170, 171)
      │         │         │    │              │    ├── [170] Symbol(y)
      │         │         │    │              │    └── [171] Constant(3.5)
      │         │         │    │              └── [173] Constant(2)
      │         │         │    └── [177] Constant(2.234)
      │         │         └── [203] Min(188, 202)
      │         │              ├── [188] Subtract(186, 187)
      │         │              │    ├── [186] Sqrt(185)
      │         │              │    │    └── [185] Add(181, 184)
      │         │              │    │         ├── [181] Pow(179, 180)
      │         │              │    │         │    ├── [179] Symbol(x)
      │         │              │    │         │    └── [180] Constant(2)
      │         │              │    │         └── [184] Pow(182, 183)
      │         │              │    │              ├── [182] Symbol(y)
      │         │              │    │              └── [183] Constant(2)
      │         │              │    └── [187] Constant(4.24)
      │         │              └── [202] Subtract(200, 201)
      │         │                   ├── [200] Sqrt(199)
      │         │                   │    └── [199] Add(193, 198)
      │         │                   │         ├── [193] Pow(191, 192)
      │         │                   │         │    ├── [191] Subtract(189, 190)
      │         │                   │         │    │    ├── [189] Symbol(x)
      │         │                   │         │    │    └── [190] Constant(2.5)
      │         │                   │         │    └── [192] Constant(2)
      │         │                   │         └── [198] Pow(196, 197)
      │         │                   │              ├── [196] Subtract(194, 195)
      │         │                   │              │    ├── [194] Symbol(y)
      │         │                   │              │    └── [195] Constant(2.5)
      │         │                   │              └── [197] Constant(2)
      │         │                   └── [201] Constant(5.243)
      │         └── [263] Exp(262)
      │              └── [262] Pow(260, 261)
      │                   ├── [260] Max(219, 259)
      │                   │    ├── [219] Subtract(217, 218)
      │                   │    │    ├── [217] Sqrt(216)
      │                   │    │    │    └── [216] Add(210, 215)
      │                   │    │    │         ├── [210] Pow(208, 209)
      │                   │    │    │         │    ├── [208] Subtract(206, 207)
      │                   │    │    │         │    │    ├── [206] Symbol(x)
      │                   │    │    │         │    │    └── [207] Constant(2.95)
      │                   │    │    │         │    └── [209] Constant(2)
      │                   │    │    │         └── [215] Pow(213, 214)
      │                   │    │    │              ├── [213] Subtract(211, 212)
      │                   │    │    │              │    ├── [211] Symbol(y)
      │                   │    │    │              │    └── [212] Constant(2.05)
      │                   │    │    │              └── [214] Constant(2)
      │                   │    │    └── [218] Constant(3.67)
      │                   │    └── [259] Min(233, 258)
      │                   │         ├── [233] Subtract(231, 232)
      │                   │         │    ├── [231] Sqrt(230)
      │                   │         │    │    └── [230] Add(224, 229)
      │                   │         │    │         ├── [224] Pow(222, 223)
      │                   │         │    │         │    ├── [222] Subtract(220, 221)
      │                   │         │    │         │    │    ├── [220] Symbol(x)
      │                   │         │    │         │    │    └── [221] Constant(3.5)
      │                   │         │    │         │    └── [223] Constant(2)
      │                   │         │    │         └── [229] Pow(227, 228)
      │                   │         │    │              ├── [227] Subtract(225, 226)
      │                   │         │    │              │    ├── [225] Symbol(y)
      │                   │         │    │              │    └── [226] Constant(3.5)
      │                   │         │    │              └── [228] Constant(2)
      │                   │         │    └── [232] Constant(2.234)
      │                   │         └── [258] Min(243, 257)
      │                   │              ├── [243] Subtract(241, 242)
      │                   │              │    ├── [241] Sqrt(240)
      │                   │              │    │    └── [240] Add(236, 239)
      │                   │              │    │         ├── [236] Pow(234, 235)
      │                   │              │    │         │    ├── [234] Symbol(x)
      │                   │              │    │         │    └── [235] Constant(2)
      │                   │              │    │         └── [239] Pow(237, 238)
      │                   │              │    │              ├── [237] Symbol(y)
      │                   │              │    │              └── [238] Constant(2)
      │                   │              │    └── [242] Constant(4.24)
      │                   │              └── [257] Subtract(255, 256)
      │                   │                   ├── [255] Sqrt(254)
      │                   │                   │    └── [254] Add(248, 253)
      │                   │                   │         ├── [248] Pow(246, 247)
      │                   │                   │         │    ├── [246] Subtract(244, 245)
      │                   │                   │         │    │    ├── [244] Symbol(x)
      │                   │                   │         │    │    └── [245] Constant(2.5)
      │                   │                   │         │    └── [247] Constant(2)
      │                   │                   │         └── [253] Pow(251, 252)
      │                   │                   │              ├── [251] Subtract(249, 250)
      │                   │                   │              │    ├── [249] Symbol(y)
      │                   │                   │              │    └── [250] Constant(2.5)
      │                   │                   │              └── [252] Constant(2)
      │                   │                   └── [256] Constant(5.243)
      │                   └── [261] Constant(2.456)
      └── [300] Max(282, 299)
           ├── [282] Divide(278, 281)
           │    ├── [278] Add(267, 277)
           │    │    ├── [267] Negate(266)
           │    │    │    └── [266] Symbol(b)
           │    │    └── [277] Sqrt(276)
           │    │         └── [276] Subtract(270, 275)
           │    │              ├── [270] Pow(268, 269)
           │    │              │    ├── [268] Symbol(b)
           │    │              │    └── [269] Constant(2)
           │    │              └── [275] Multiply(271, 274)
           │    │                   ├── [271] Constant(4)
           │    │                   └── [274] Multiply(272, 273)
           │    │                        ├── [272] Symbol(a)
           │    │                        └── [273] Symbol(c)
           │    └── [281] Multiply(279, 280)
           │         ├── [279] Constant(2)
           │         └── [280] Symbol(a)
           └── [299] Divide(295, 298)
                ├── [295] Subtract(284, 294)
                │    ├── [284] Negate(283)
                │    │    └── [283] Symbol(b)
                │    └── [294] Sqrt(293)
                │         └── [293] Subtract(287, 292)
                │              ├── [287] Pow(285, 286)
                │              │    ├── [285] Symbol(b)
                │              │    └── [286] Constant(2)
                │              └── [292] Multiply(288, 291)
                │                   ├── [288] Constant(4)
                │                   └── [291] Multiply(289, 290)
                │                        ├── [289] Symbol(a)
                │                        └── [290] Symbol(c)
                └── [298] Multiply(296, 297)
                     ├── [296] Constant(2)
                     └── [297] Symbol(a)"
                .trim()
        );
    }
}
