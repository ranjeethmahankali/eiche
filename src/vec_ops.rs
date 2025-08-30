use crate::{
    BinaryOp::*,
    Error,
    Node::{self, *},
    Tree,
    tree::extend_nodes_from_slice,
};

impl Tree {
    pub fn matmul(self, other: Tree) -> Result<Tree, Error> {
        let roots_lt = self.root_indices();
        let roots_rt = other.root_indices();
        let (mut lnodes, ldims) = self.take();
        let (rnodes, rdims) = other.take();
        if ldims.1 != rdims.0 {
            return Err(Error::DimensionMismatch(ldims, rdims));
        }
        if ldims.0 == 0 || ldims.1 == 0 || rdims.0 == 0 || rdims.1 == 0 {
            return Err(Error::InvalidDimensions);
        }
        let offset = extend_nodes_from_slice(&mut lnodes, &rnodes);
        let roots_rt = (roots_rt.start + offset)..(roots_rt.end + offset);
        let (lrows, lcols) = ldims;
        let (rrows, rcols) = rdims;
        let (orows, ocols) = (lrows, rcols);
        let mut newroots: Vec<Node> = Vec::with_capacity(ocols * orows);
        for oc in 0..ocols {
            for or in 0..orows {
                let n_before = lnodes.len();
                let rcol_start = oc * rrows;
                let rcol_idx = rcol_start..(rcol_start + rrows);
                let lrow_idx = (0..lcols).map(|c| or + c * lrows);
                lnodes.extend(
                    lrow_idx
                        .zip(rcol_idx)
                        .map(|(li, ri)| Binary(Multiply, li + roots_lt.start, ri + roots_rt.start)),
                );
                let n_after = lnodes.len();
                let mut total = n_before;
                for curr in (n_before + 1)..n_after {
                    let next = lnodes.len();
                    lnodes.push(Binary(Add, total, curr));
                    total = next;
                }
                if let Some(last) = lnodes.pop() {
                    newroots.push(last);
                }
            }
        }
        lnodes.extend(newroots.drain(..));
        Tree::from_nodes(lnodes, (orows, ocols))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{deftree, test::compare_trees};

    #[test]
    fn t_matmul_2x2_times_2x1_simple() {
        // [a c] * [p] = [a*p + c*q]
        // [b d]   [q]   [b*p + d*q]
        let lhs = deftree!(concat 'a 'b 'c 'd).unwrap().reshape(2, 2).unwrap();
        let rhs = deftree!(concat 'p 'q).unwrap();
        let result = lhs.matmul(rhs).unwrap();
        assert_eq!(result.dims(), (2, 1));
        // Manually construct expected result: [a*p + c*q, b*p + d*q]
        let expected = deftree!(concat
            (+ (* 'a 'p) (* 'c 'q))
            (+ (* 'b 'p) (* 'd 'q))
        )
        .unwrap();
        // Check root nodes match expected structure
        let result_roots = result.roots();
        let expected_roots = expected.roots();
        assert_eq!(result_roots.len(), expected_roots.len());
        // Check tree equivalence
        assert!(
            result.equivalent(&expected),
            "Result tree should be equivalent to expected tree"
        );
        // Numerical verification
        compare_trees(
            &result,
            &expected,
            &[
                ('a', -2., 3.),
                ('b', -1., 4.),
                ('c', -3., 2.),
                ('d', -2., 5.),
                ('p', -1., 2.),
                ('q', -4., 1.),
            ],
            2,
            1e-14,
        );
    }

    #[test]
    fn t_matmul_1x2_times_2x1_expressions() {
        // [sin(x) cos(y)] * [x^2    ] = [sin(x)*x^2 + cos(y)*log(z)]
        //                   [log(z) ]
        let lhs = deftree!(concat (sin 'x) (cos 'y))
            .unwrap()
            .reshape(1, 2)
            .unwrap();
        let rhs = deftree!(concat (pow 'x 2) (log 'z)).unwrap();
        let result = lhs.matmul(rhs).unwrap();
        assert_eq!(result.dims(), (1, 1));
        let expected = deftree!(+ (* (sin 'x) (pow 'x 2)) (* (cos 'y) (log 'z))).unwrap();
        // Verify root structure
        let result_roots = result.roots();
        assert_eq!(result_roots.len(), 1);
        // Check tree equivalence
        assert!(
            result.equivalent(&expected),
            "Result tree should be equivalent to expected tree"
        );
        // Numerical verification with appropriate ranges
        compare_trees(
            &result,
            &expected,
            &[('x', -2., 2.), ('y', -3., 3.), ('z', 0.1, 10.)],
            7,
            1e-13,
        );
    }

    #[test]
    fn t_matmul_2x1_times_1x2_outer_product() {
        // [x+1] * [y-1 y^2] = [(x+1)*(y-1)  (x+1)*y^2]
        // [x^2]               [x^2*(y-1)    x^2*y^2  ]
        let lhs = deftree!(concat (+ 'x 1) (pow 'x 2)).unwrap();
        let rhs = deftree!(concat (- 'y 1) (pow 'y 2))
            .unwrap()
            .reshape(1, 2)
            .unwrap();
        let result = lhs.matmul(rhs).unwrap();
        assert_eq!(result.dims(), (2, 2));
        let expected = deftree!(concat
            (* (+ 'x 1) (- 'y 1))
            (* (pow 'x 2) (- 'y 1))
            (* (+ 'x 1) (pow 'y 2))
            (* (pow 'x 2) (pow 'y 2))
        )
        .unwrap()
        .reshape(2, 2)
        .unwrap();
        // Check dimensions and root count
        let result_roots = result.roots();
        let expected_roots = expected.roots();
        assert_eq!(result_roots.len(), expected_roots.len());
        assert_eq!(result_roots.len(), 4);
        // Check tree equivalence
        assert!(
            result.equivalent(&expected),
            "Result tree should be equivalent to expected tree"
        );
        compare_trees(
            &result,
            &expected,
            &[('x', -3., 3.), ('y', -2., 2.)],
            18,
            1e-14,
        );
    }

    #[test]
    fn t_matmul_1x1_times_1x1_scalar() {
        // [sqrt(x)] * [exp(y)] = [sqrt(x)*exp(y)]
        let lhs = deftree!(sqrt 'x).unwrap().reshape(1, 1).unwrap();
        let rhs = deftree!(exp 'y).unwrap().reshape(1, 1).unwrap();
        let result = lhs.matmul(rhs).unwrap();
        println!("{:?}", result.nodes());
        assert_eq!(result.dims(), (1, 1));
        let expected = deftree!(* (sqrt 'x) (exp 'y)).unwrap();
        let result_roots = result.roots();
        assert_eq!(result_roots.len(), 1);
        match &result_roots[0] {
            Binary(Multiply, _, _) => {}
            _ => panic!("Expected multiplication node at root"),
        }
        // Check tree equivalence
        assert!(
            result.equivalent(&expected),
            "Result tree should be equivalent to expected tree"
        );
        compare_trees(
            &result,
            &expected,
            &[('x', 0.1, 4.), ('y', -2., 2.)],
            18,
            1e-14,
        );
    }

    #[test]
    fn t_matmul_2x3_times_3x2_complex() {
        // [x     sin(y)  2  ] * [a       b^2  ]
        // [x^2   cos(y)  3  ]   [log(a)  c    ] =
        //                       [5       d+1  ]
        //
        // [x*a + sin(y)*log(a) + 2*5       x*b^2 + sin(y)*c + 2*(d+1)   ]
        // [x^2*a + cos(y)*log(a) + 3*5     x^2*b^2 + cos(y)*c + 3*(d+1) ]
        let lhs = deftree!(concat
            'x (pow 'x 2)
            (sin 'y) (cos 'y)
            2 3
        )
        .unwrap()
        .reshape(2, 3)
        .unwrap();
        let rhs = deftree!(concat
            'a (log 'a) 5
            (pow 'b 2) 'c (+ 'd 1)
        )
        .unwrap()
        .reshape(3, 2)
        .unwrap();
        let result = lhs.matmul(rhs).unwrap();
        assert_eq!(result.dims(), (2, 2));
        let expected = deftree!(concat
            (+ (+ (* 'x 'a) (* (sin 'y) (log 'a))) (* 2 5))
            (+ (+ (* (pow 'x 2) 'a) (* (cos 'y) (log 'a))) (* 3 5))
            (+ (+ (* 'x (pow 'b 2)) (* (sin 'y) 'c)) (* 2 (+ 'd 1)))
            (+ (+ (* (pow 'x 2) (pow 'b 2)) (* (cos 'y) 'c)) (* 3 (+ 'd 1)))
        )
        .unwrap()
        .reshape(2, 2)
        .unwrap();
        let result_roots = result.roots();
        let expected_roots = expected.roots();
        assert_eq!(result_roots.len(), expected_roots.len());
        assert_eq!(result_roots.len(), 4);
        // Check tree equivalence
        assert!(
            result.equivalent(&expected),
            "Result tree should be equivalent to expected tree"
        );
        compare_trees(
            &result,
            &expected,
            &[
                ('x', -1., 2.),
                ('y', -1.5, 1.5),
                ('a', 0.1, 3.),
                ('b', -2., 2.),
                ('c', -1., 1.),
                ('d', -2., 2.),
            ],
            2,
            1e-12,
        );
    }

    #[test]
    fn t_matmul_3x1_times_1x3_rank_one() {
        // [x+y ] * [a  b  c] = [(x+y)*a  (x+y)*b  (x+y)*c]
        // [x-y ]              [(x-y)*a  (x-y)*b  (x-y)*c]
        // [x*y ]              [(x*y)*a  (x*y)*b  (x*y)*c]
        let lhs = deftree!(concat (+ 'x 'y) (- 'x 'y) (* 'x 'y)).unwrap();
        let rhs = deftree!(concat 'a 'b 'c).unwrap().reshape(1, 3).unwrap();
        let result = lhs.matmul(rhs).unwrap();
        assert_eq!(result.dims(), (3, 3));
        let expected = deftree!(concat
            (* (+ 'x 'y) 'a) (* (- 'x 'y) 'a) (* (* 'x 'y) 'a)
            (* (+ 'x 'y) 'b) (* (- 'x 'y) 'b) (* (* 'x 'y) 'b)
            (* (+ 'x 'y) 'c) (* (- 'x 'y) 'c) (* (* 'x 'y) 'c)
        )
        .unwrap()
        .reshape(3, 3)
        .unwrap();
        assert_eq!(result.roots().len(), 9);
        assert_eq!(expected.roots().len(), 9);
        // Check tree equivalence
        assert!(
            result.equivalent(&expected),
            "Result tree should be equivalent to expected tree"
        );
        compare_trees(
            &result,
            &expected,
            &[
                ('x', -2., 2.),
                ('y', -1., 3.),
                ('a', -1., 1.),
                ('b', -2., 2.),
                ('c', -1., 1.),
            ],
            3,
            1e-14,
        );
    }

    #[test]
    fn t_matmul_constants_and_expressions() {
        // [2    x^2 ] * [y  ] = [2*y + x^2*3]
        // [x+1  5   ]   [3  ]   [(x+1)*y + 5*3]
        let lhs = deftree!(concat 2 (+ 'x 1) (pow 'x 2) 5)
            .unwrap()
            .reshape(2, 2)
            .unwrap();
        let rhs = deftree!(concat 'y 3).unwrap();
        let result = lhs.matmul(rhs).unwrap();
        assert_eq!(result.dims(), (2, 1));
        let expected = deftree!(concat
            (+ (* 2 'y) (* (pow 'x 2) 3))
            (+ (* (+ 'x 1) 'y) (* 5 3))
        )
        .unwrap();
        assert_eq!(result.roots().len(), 2);
        assert_eq!(expected.roots().len(), 2);
        // Check tree equivalence
        assert!(
            result.equivalent(&expected),
            "Result tree should be equivalent to expected tree"
        );
        compare_trees(
            &result,
            &expected,
            &[('x', -2., 2.), ('y', -1., 1.)],
            18,
            1e-14,
        );
    }

    #[test]
    fn t_matmul_dimension_mismatch() {
        // 2x2 matrix times 3x2 matrix should fail (2 != 3)
        let lhs = deftree!(concat 'a 'b 'c 'd).unwrap().reshape(2, 2).unwrap();
        let rhs = deftree!(concat 'p 'q 'r 's 't 'u)
            .unwrap()
            .reshape(3, 2)
            .unwrap();
        match lhs.matmul(rhs) {
            Err(Error::DimensionMismatch((2, 2), (3, 2))) => {}
            other => panic!("Expected DimensionMismatch error, got: {:?}", other),
        }
    }

    #[test]
    fn t_matmul_dimension_mismatch_2() {
        // 3x1 matrix times 2x1 matrix should fail (1 != 2)
        let lhs = deftree!(concat 'a 'b 'c).unwrap();
        let rhs = deftree!(concat 'p 'q).unwrap();
        match lhs.matmul(rhs) {
            Err(Error::DimensionMismatch((3, 1), (2, 1))) => {}
            other => panic!("Expected DimensionMismatch error, got: {:?}", other),
        }
    }

    #[test]
    fn t_matmul_dimension_mismatch_3() {
        // 1x4 matrix times 2x1 matrix should fail (4 != 2)
        let lhs = deftree!(concat 'a 'b 'c 'd).unwrap().reshape(1, 4).unwrap();
        let rhs = deftree!(concat 'p 'q).unwrap();
        match lhs.matmul(rhs) {
            Err(Error::DimensionMismatch((1, 4), (2, 1))) => {}
            other => panic!("Expected DimensionMismatch error, got: {:?}", other),
        }
    }

    #[test]
    fn t_matmul_zero_dimensions() {
        // Test matrices with zero dimensions
        let valid = deftree!(concat 'a 'b).unwrap();
        // These reshape operations should fail, but if they succeed by some means,
        // matmul should handle them appropriately
        if let Ok(zero_row) = valid.clone().reshape(0, 2) {
            if let Ok(valid_mat) = valid.clone().reshape(2, 1) {
                match zero_row.matmul(valid_mat) {
                    Err(Error::InvalidDimensions) => {}
                    other => panic!("Expected InvalidDimensions error, got: {:?}", other),
                }
            }
        }
        if let Ok(zero_col) = valid.clone().reshape(2, 0) {
            if let Ok(valid_mat) = valid.clone().reshape(1, 2) {
                match valid_mat.matmul(zero_col) {
                    Err(Error::InvalidDimensions) => {}
                    other => panic!("Expected InvalidDimensions error, got: {:?}", other),
                }
            }
        }
    }

    #[test]
    fn t_matmul_trigonometric_expressions() {
        // [sin(x)  cos(x)] * [tan(y)] = [sin(x)*tan(y) + cos(x)*1/tan(y)]
        // [1       0     ]   [1/tan(y)]  [1*tan(y) + 0*1/tan(y)]
        let lhs = deftree!(concat (sin 'x) 1 (cos 'x) 0)
            .unwrap()
            .reshape(2, 2)
            .unwrap();
        let rhs = deftree!(concat (tan 'y) (/ 1 (tan 'y))).unwrap();
        let result = lhs.matmul(rhs).unwrap();
        assert_eq!(result.dims(), (2, 1));
        let expected = deftree!(concat
            (+ (* (sin 'x) (tan 'y)) (* (cos 'x) (/ 1 (tan 'y))))
            (+ (* 1 (tan 'y)) (* 0 (/ 1 (tan 'y))))
        )
        .unwrap();
        assert_eq!(result.roots().len(), 2);
        // Check tree equivalence
        assert!(
            result.equivalent(&expected),
            "Result tree should be equivalent to expected tree"
        );
        // Use smaller range for tangent to avoid singularities
        compare_trees(
            &result,
            &expected,
            &[('x', -1., 1.), ('y', -1., 1.)],
            18,
            1e-12,
        );
    }
}
