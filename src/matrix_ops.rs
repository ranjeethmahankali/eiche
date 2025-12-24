use crate::{
    BinaryOp::*,
    Error,
    Node::{self, *},
    TernaryOp::*,
    Tree,
    UnaryOp::*,
    Value,
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
                let prod_range = n_before..n_after;
                lnodes.extend(
                    std::iter::once(prod_range.start)
                        .chain(n_after..(n_after + prod_range.len().saturating_sub(2)))
                        .zip(prod_range.skip(1))
                        .map(|(l, r)| Binary(Add, l, r)),
                );
                if let Some(last) = lnodes.pop() {
                    newroots.push(last);
                }
            }
        }
        lnodes.append(&mut newroots);
        Tree::from_nodes(lnodes, (orows, ocols))
    }

    pub fn transpose(self) -> Result<Tree, Error> {
        let nroots = self.num_roots();
        let (mut nodes, dims) = self.take();
        let (rows, cols) = dims;
        let ntotal = nodes.len();
        let roots = &mut nodes[(ntotal - nroots)..];
        match (rows, cols) {
            (1, 1) => {} // Do nothing
            (r, c) if r == c => {
                // Square case: do in place
                for i in 0..rows {
                    for j in (i + 1)..cols {
                        roots.swap(i * cols + j, j * cols + i);
                    }
                }
            }
            _ => {
                // Rectangular case: allocate new matrix
                let mut newroots = roots.to_vec();
                for i in 0..rows {
                    for j in 0..cols {
                        newroots[i * cols + j] = roots[j * rows + i];
                    }
                }
                roots.copy_from_slice(&newroots);
            }
        }
        Tree::from_nodes(nodes, (cols, rows))
    }

    pub fn dot_product(self, other: Tree) -> Result<Tree, Error> {
        let ldims = self.dims();
        let rdims = other.dims();
        match (ldims, rdims) {
            ((lr, 1), (rr, 1)) if lr == rr => self.transpose()?.matmul(other),
            ((1, lc), (1, rc)) if lc == rc => self.matmul(other.transpose()?),
            ((lr, 1), (1, rc)) if lr == rc => other.matmul(self),
            ((1, lc), (rr, 1)) if lc == rr => self.matmul(other),
            _ => Err(Error::InvalidDimensions),
        }
    }

    pub fn extract(self, indices: &[(usize, usize)]) -> Result<Tree, Error> {
        let n_roots_new = indices.len();
        if n_roots_new == 0 {
            return Err(Error::InvalidDimensions);
        }
        let n_roots_old = self.num_roots();
        let (mut nodes, dims) = self.take();
        let (rows, cols) = dims;
        let n_nodes_old = nodes.len();
        let n_keep = n_nodes_old - n_roots_old;
        let old_roots = &nodes[n_keep..];
        if let Some((r, c)) = indices.iter().find(|(r, c)| !(*r < rows && *c < cols)) {
            return Err(Error::IndexOutOfBounds(*r, *c));
        }
        let mut new_roots: Vec<Node> = indices
            .iter()
            .map(|(r, c)| old_roots[c * rows + r])
            .collect();
        nodes.truncate(n_keep);
        nodes.append(&mut new_roots);
        Tree::from_nodes(nodes, (n_roots_new, 1))
    }

    pub fn l2norm(self) -> Result<Tree, Error> {
        let n_roots = self.num_roots();
        if n_roots == 0 {
            return Err(Error::InvalidDimensions);
        }
        let (mut nodes, dims) = self.take();
        match dims {
            (1, 1) => crate::abs(Tree::from_nodes(nodes, dims)),
            (1, _) | (_, 1) => {
                let n_before = nodes.len();
                nodes.extend(((n_before - n_roots)..n_before).map(|ri| Binary(Multiply, ri, ri)));
                let n_total = nodes.len();
                let sqrange = (n_total - n_roots)..n_total;
                nodes.extend(
                    std::iter::once(sqrange.start)
                        .chain(n_total..(n_total + sqrange.len().saturating_sub(2)))
                        .zip(sqrange.skip(1))
                        .map(|(l, r)| Binary(Add, l, r)),
                );
                nodes.push(Unary(Sqrt, nodes.len() - 1));
                Tree::from_nodes(nodes, (1, 1))
            }
            _ => Err(Error::InvalidDimensions),
        }
    }

    pub fn normalize(self) -> Result<Tree, Error> {
        let n_roots = self.num_roots();
        if n_roots == 0 {
            return Err(Error::InvalidDimensions);
        }
        let (mut nodes, dims) = self.take();
        match dims {
            (1, 1) => {
                let val = nodes.len() - 1;
                let zero = {
                    let out = nodes.len();
                    nodes.push(Constant(Value::Scalar(0.)));
                    out
                };
                let ltz = {
                    let out = nodes.len();
                    nodes.push(Binary(Less, val, zero));
                    out
                };
                let gtz = {
                    let out = nodes.len();
                    nodes.push(Binary(Greater, val, zero));
                    out
                };
                let one = {
                    let out = nodes.len();
                    nodes.push(Constant(Value::Scalar(1.)));
                    out
                };
                let neg_one = {
                    let out = nodes.len();
                    nodes.push(Constant(Value::Scalar(-1.)));
                    out
                };
                let first_choice = {
                    let out = nodes.len();
                    nodes.push(Ternary(Choose, gtz, one, zero));
                    out
                };
                nodes.push(Ternary(Choose, ltz, neg_one, first_choice));
                Tree::from_nodes(nodes, (1, 1))
            }
            (1, _) | (_, 1) => {
                let n_before = nodes.len();
                let comp_range = (n_before - n_roots)..n_before; // Components of the vector.
                nodes.extend(comp_range.clone().map(|ri| Binary(Multiply, ri, ri)));
                let n_total = nodes.len();
                let sqrange = (n_total - n_roots)..n_total; // Squares of the components.
                nodes.extend(
                    std::iter::once(sqrange.start)
                        .chain(n_total..(n_total + sqrange.len().saturating_sub(2)))
                        .zip(sqrange.skip(1))
                        .map(|(l, r)| Binary(Add, l, r)),
                );
                let vlen = nodes.len(); // l2 norm of the vector.
                nodes.push(Unary(Sqrt, nodes.len() - 1));
                nodes.extend(comp_range.map(|i| Binary(Divide, i, vlen)));
                Tree::from_nodes(nodes, dims)
            }
            _ => Err(Error::InvalidDimensions),
        }
    }
}

#[cfg(test)]
mod test {
    use core::f64;

    use super::*;
    use crate::{deftree, test_util::compare_trees};

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
        if let Ok(zero_row) = valid.clone().reshape(0, 2)
            && let Ok(valid_mat) = valid.clone().reshape(2, 1)
        {
            match zero_row.matmul(valid_mat) {
                Err(Error::InvalidDimensions) => {}
                other => panic!("Expected InvalidDimensions error, got: {:?}", other),
            }
        }
        if let Ok(zero_col) = valid.clone().reshape(2, 0)
            && let Ok(valid_mat) = valid.clone().reshape(1, 2)
        {
            match valid_mat.matmul(zero_col) {
                Err(Error::InvalidDimensions) => {}
                other => panic!("Expected InvalidDimensions error, got: {:?}", other),
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

    #[test]
    fn t_transpose_vectors() {
        // Test row vectors
        {
            // [a b c] -> [a]
            //            [b]
            //            [c]
            let matrix = deftree!(concat 'a 'b 'c).unwrap().reshape(1, 3).unwrap();
            let result = matrix.transpose().unwrap();
            assert_eq!(result.dims(), (3, 1));
            let expected = deftree!(concat 'a 'b 'c).unwrap(); // column vector (3, 1)
            assert!(
                result.equivalent(&expected),
                "Row vector transpose should be equivalent to column vector"
            );
        }
        {
            // [a b c d] -> [a]
            //              [b]
            //              [c]
            //              [d]
            let matrix = deftree!(concat 'a 'b 'c 'd).unwrap().reshape(1, 4).unwrap();
            let result = matrix.transpose().unwrap();
            assert_eq!(result.dims(), (4, 1));
            let expected = deftree!(concat 'a 'b 'c 'd).unwrap(); // column vector (4, 1)
            assert!(result.equivalent(&expected), "1x4 transpose should be 4x1");
        }
        // Test column vectors
        {
            // [a] -> [a b c]
            // [b]
            // [c]
            let matrix = deftree!(concat 'a 'b 'c).unwrap(); // column vector (3, 1)
            let result = matrix.transpose().unwrap();
            assert_eq!(result.dims(), (1, 3));
            let expected = deftree!(concat 'a 'b 'c).unwrap().reshape(1, 3).unwrap();
            assert!(
                result.equivalent(&expected),
                "Column vector transpose should be equivalent to row vector"
            );
        }
        {
            // [a] -> [a b c d]
            // [b]
            // [c]
            // [d]
            let matrix = deftree!(concat 'a 'b 'c 'd).unwrap(); // column vector (4, 1)
            let result = matrix.transpose().unwrap();
            assert_eq!(result.dims(), (1, 4));
            let expected = deftree!(concat 'a 'b 'c 'd).unwrap().reshape(1, 4).unwrap();
            assert!(result.equivalent(&expected), "4x1 transpose should be 1x4");
        }
    }

    #[test]
    fn t_transpose_square_matrices() {
        // Test 1x1 scalar
        {
            // [a] -> [a] (no change)
            let matrix = deftree!('a).unwrap().reshape(1, 1).unwrap();
            let result = matrix.transpose().unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!('a).unwrap().reshape(1, 1).unwrap();
            assert!(
                result.equivalent(&expected),
                "1x1 transpose should be equivalent to original"
            );
        }
        // Test 2x2 matrix
        {
            // [a c] -> [a b]
            // [b d]    [c d]
            let matrix = deftree!(concat 'a 'b 'c 'd).unwrap().reshape(2, 2).unwrap();
            let result = matrix.transpose().unwrap();
            assert_eq!(result.dims(), (2, 2));
            let expected = deftree!(concat 'a 'c 'b 'd).unwrap().reshape(2, 2).unwrap();
            assert!(
                result.equivalent(&expected),
                "2x2 transpose should swap off-diagonal elements"
            );
        }
        // Test 3x3 matrix
        {
            // [a d g] -> [a b c]
            // [b e h]    [d e f]
            // [c f i]    [g h i]
            let matrix = deftree!(concat 'a 'b 'c 'd 'e 'f 'g 'h 'i)
                .unwrap()
                .reshape(3, 3)
                .unwrap();
            let result = matrix.transpose().unwrap();
            assert_eq!(result.dims(), (3, 3));
            let expected = deftree!(concat 'a 'd 'g 'b 'e 'h 'c 'f 'i)
                .unwrap()
                .reshape(3, 3)
                .unwrap();
            assert!(
                result.equivalent(&expected),
                "3x3 transpose should reflect across diagonal"
            );
        }
        // Test diagonal matrix (should be symmetric)
        {
            // [a 0 0] -> [a 0 0]
            // [0 b 0]    [0 b 0]
            // [0 0 c]    [0 0 c]
            let matrix = deftree!(concat 'a 0 0 0 'b 0 0 0 'c)
                .unwrap()
                .reshape(3, 3)
                .unwrap();
            let result = matrix.clone().transpose().unwrap();
            assert_eq!(result.dims(), (3, 3));
            assert!(
                result.equivalent(&matrix),
                "Diagonal matrix should be equivalent to its transpose"
            );
        }
        // Test identity matrix (should be symmetric)
        {
            // [1 0] -> [1 0]
            // [0 1]    [0 1]
            let matrix = deftree!(concat 1 0 0 1).unwrap().reshape(2, 2).unwrap();
            let result = matrix.clone().transpose().unwrap();
            assert_eq!(result.dims(), (2, 2));
            assert!(
                result.equivalent(&matrix),
                "Identity matrix should be equivalent to its transpose"
            );
        }
    }

    #[test]
    fn t_transpose_rectangular_matrices() {
        // Test 2x3 matrix
        {
            // [a c e] -> [a b]
            // [b d f]    [c d]
            //            [e f]
            let matrix = deftree!(concat 'a 'b 'c 'd 'e 'f)
                .unwrap()
                .reshape(2, 3)
                .unwrap();
            let result = matrix.transpose().unwrap();
            assert_eq!(result.dims(), (3, 2));
            let expected = deftree!(concat 'a 'c 'e 'b 'd 'f)
                .unwrap()
                .reshape(3, 2)
                .unwrap();
            println!("{:?}", result);
            println!("{:?}", expected);
            assert!(
                result.equivalent(&expected),
                "2x3 transpose should become 3x2"
            );
        }
        // Test 3x2 matrix
        {
            // [a d] -> [a b c]
            // [b e]    [d e f]
            // [c f]
            let matrix = deftree!(concat 'a 'b 'c 'd 'e 'f)
                .unwrap()
                .reshape(3, 2)
                .unwrap();
            let result = matrix.transpose().unwrap();
            assert_eq!(result.dims(), (2, 3));
            let expected = deftree!(concat 'a 'd 'b 'e 'c 'f)
                .unwrap()
                .reshape(2, 3)
                .unwrap();
            assert!(
                result.equivalent(&expected),
                "3x2 transpose should become 2x3"
            );
        }
        // Test with constants
        {
            // [1 3 5] -> [1 2]
            // [2 4 6]    [3 4]
            //            [5 6]
            let matrix = deftree!(concat 1 2 3 4 5 6).unwrap().reshape(2, 3).unwrap();
            let result = matrix.transpose().unwrap();
            assert_eq!(result.dims(), (3, 2));
            let expected = deftree!(concat 1 3 5 2 4 6).unwrap().reshape(3, 2).unwrap();
            assert!(
                result.equivalent(&expected),
                "Transpose should work with constants"
            );
        }
    }

    #[test]
    fn t_transpose_with_expressions() {
        // Test with complex expressions
        {
            // [x+1  sin(y)] -> [x+1   x^2]
            // [x^2  cos(z)]    [sin(y) cos(z)]
            let matrix = deftree!(concat (+ 'x 1) (pow 'x 2) (sin 'y) (cos 'z))
                .unwrap()
                .reshape(2, 2)
                .unwrap();
            let result = matrix.transpose().unwrap();
            assert_eq!(result.dims(), (2, 2));
            let expected = deftree!(concat (+ 'x 1) (sin 'y) (pow 'x 2) (cos 'z))
                .unwrap()
                .reshape(2, 2)
                .unwrap();
            assert!(
                result.equivalent(&expected),
                "Transpose should work with complex expressions"
            );
        }
        // Test with mixed expressions
        {
            // [x   2  ] -> [x    sin(y)]
            // [sin(y) 3]    [2    3]
            let matrix = deftree!(concat 'x (sin 'y) 2 3)
                .unwrap()
                .reshape(2, 2)
                .unwrap();
            let result = matrix.transpose().unwrap();
            assert_eq!(result.dims(), (2, 2));
            let expected = deftree!(concat 'x 2 (sin 'y) 3)
                .unwrap()
                .reshape(2, 2)
                .unwrap();
            assert!(
                result.equivalent(&expected),
                "Transpose should work with mixed symbols and constants"
            );
        }
    }

    #[test]
    fn t_transpose_properties() {
        // Test double transpose: (A^T)^T = A
        {
            let original = deftree!(concat 'a 'b 'c 'd 'e 'f)
                .unwrap()
                .reshape(2, 3)
                .unwrap();
            let once_transposed = original.clone().transpose().unwrap();
            let twice_transposed = once_transposed.transpose().unwrap();
            assert_eq!(twice_transposed.dims(), original.dims());
            assert!(
                twice_transposed.equivalent(&original),
                "Double transpose should return to original"
            );
        }
        // Test with different dimensions
        {
            let original = deftree!(concat 'a 'b 'c 'd).unwrap().reshape(1, 4).unwrap();
            let twice_transposed = original.clone().transpose().unwrap().transpose().unwrap();
            assert_eq!(twice_transposed.dims(), original.dims());
            assert!(
                twice_transposed.equivalent(&original),
                "Double transpose should return to original for vectors"
            );
        }
    }

    #[test]
    fn t_dot_product_same_orientation() {
        // Test column vectors
        {
            // [a] · [x] = a*x + b*y + c*z
            // [b]   [y]
            // [c]   [z]
            let u = deftree!(concat 'a 'b 'c).unwrap(); // (3,1)
            let v = deftree!(concat 'x 'y 'z).unwrap(); // (3,1)
            let result = u.dot_product(v).unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(+ (+ (* 'a 'x) (* 'b 'y)) (* 'c 'z)).unwrap();
            assert!(
                result.equivalent(&expected),
                "Column vector dot product should work"
            );
        }
        // Test row vectors
        {
            // [a b c] · [x y z] = a*x + b*y + c*z
            let u = deftree!(concat 'a 'b 'c).unwrap().reshape(1, 3).unwrap(); // (1,3)
            let v = deftree!(concat 'x 'y 'z).unwrap().reshape(1, 3).unwrap(); // (1,3)
            let result = u.dot_product(v).unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(+ (+ (* 'a 'x) (* 'b 'y)) (* 'c 'z)).unwrap();
            assert!(
                result.equivalent(&expected),
                "Row vector dot product should work"
            );
        }
        // Test 2D vectors
        {
            // [x+1] · [y-1] = (x+1)*(y-1) + x^2*y^2
            // [x^2]   [y^2]
            let u = deftree!(concat (+ 'x 1) (pow 'x 2)).unwrap(); // (2,1)
            let v = deftree!(concat (- 'y 1) (pow 'y 2)).unwrap(); // (2,1)
            let result = u.dot_product(v).unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(+ (* (+ 'x 1) (- 'y 1)) (* (pow 'x 2) (pow 'y 2))).unwrap();
            assert!(
                result.equivalent(&expected),
                "Expression vector dot product should work"
            );
        }
        // Test scalar (1x1)
        {
            // [a] · [b] = a*b
            let u = deftree!('a).unwrap().reshape(1, 1).unwrap(); // (1,1)
            let v = deftree!('b).unwrap().reshape(1, 1).unwrap(); // (1,1)
            let result = u.dot_product(v).unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(* 'a 'b).unwrap();
            assert!(
                result.equivalent(&expected),
                "1x1 dot product should be multiplication"
            );
        }
    }

    #[test]
    fn t_dot_product_mixed_orientation() {
        // Test column · row
        {
            // [a] · [x y z] = a*x + b*y + c*z
            // [b]
            // [c]
            let u = deftree!(concat 'a 'b 'c).unwrap(); // (3,1)
            let v = deftree!(concat 'x 'y 'z).unwrap().reshape(1, 3).unwrap(); // (1,3)
            let result = u.dot_product(v).unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(+ (+ (* 'a 'x) (* 'b 'y)) (* 'c 'z)).unwrap();
            assert!(
                result.equivalent(&expected),
                "Column · row dot product should work"
            );
        }
        // Test row · column
        {
            // [a b c] · [x] = a*x + b*y + c*z
            //           [y]
            //           [z]
            let u = deftree!(concat 'a 'b 'c).unwrap().reshape(1, 3).unwrap(); // (1,3)
            let v = deftree!(concat 'x 'y 'z).unwrap(); // (3,1)
            let result = u.dot_product(v).unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(+ (+ (* 'a 'x) (* 'b 'y)) (* 'c 'z)).unwrap();
            assert!(
                result.equivalent(&expected),
                "Row · column dot product should work"
            );
        }
        // Test with expressions
        {
            // [sin(x) cos(y)] · [tan(z)] = sin(x)*tan(z) + cos(y)*exp(w)
            //                   [exp(w)]
            let u = deftree!(concat (sin 'x) (cos 'y))
                .unwrap()
                .reshape(1, 2)
                .unwrap(); // (1,2)
            let v = deftree!(concat (tan 'z) (exp 'w)).unwrap(); // (2,1)
            let result = u.dot_product(v).unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(+ (* (sin 'x) (tan 'z)) (* (cos 'y) (exp 'w))).unwrap();
            assert!(
                result.equivalent(&expected),
                "Mixed orientation with expressions should work"
            );
        }
    }

    #[test]
    fn t_dot_product_errors() {
        // Test dimension mismatch
        {
            let u = deftree!(concat 'a 'b).unwrap(); // (2,1)
            let v = deftree!(concat 'x 'y 'z).unwrap(); // (3,1)
            match u.dot_product(v) {
                Err(Error::InvalidDimensions) => {}
                other => panic!("Expected InvalidDimensions error, got: {:?}", other),
            }
        }
        // Test non-vector operands (matrix)
        {
            let matrix = deftree!(concat 'a 'b 'c 'd).unwrap().reshape(2, 2).unwrap(); // (2,2)
            let vector = deftree!(concat 'x 'y).unwrap(); // (2,1)
            match matrix.dot_product(vector) {
                Err(Error::InvalidDimensions) => {}
                other => panic!("Expected InvalidDimensions error, got: {:?}", other),
            }
        }
        // Test mixed vector/matrix mismatch
        {
            let vector = deftree!(concat 'a 'b 'c).unwrap().reshape(1, 3).unwrap(); // (1,3)
            let matrix = deftree!(concat 'x 'y 'z 'w 'p 'q)
                .unwrap()
                .reshape(2, 3)
                .unwrap(); // (2,3)
            match vector.dot_product(matrix) {
                Err(Error::InvalidDimensions) => {}
                other => panic!("Expected InvalidDimensions error, got: {:?}", other),
            }
        }
        // Test row vector dimension mismatch
        {
            let u = deftree!(concat 'a 'b).unwrap().reshape(1, 2).unwrap(); // (1,2)
            let v = deftree!(concat 'x 'y 'z 'w).unwrap().reshape(1, 4).unwrap(); // (1,4)
            match u.dot_product(v) {
                Err(Error::InvalidDimensions) => {}
                other => panic!("Expected InvalidDimensions error, got: {:?}", other),
            }
        }
    }

    #[test]
    fn t_dot_product_properties() {
        // Test commutativity: u · v = v · u
        {
            let u = deftree!(concat 'a 'b 'c).unwrap(); // (3,1)
            let v = deftree!(concat 'x 'y 'z).unwrap(); // (3,1)
            let uv = u.clone().dot_product(v.clone()).unwrap();
            let vu = v.dot_product(u).unwrap();
            assert!(uv.equivalent(&vu), "Dot product should be commutative");
        }
        // Test commutativity with mixed orientations
        {
            let u = deftree!(concat 'a 'b).unwrap(); // (2,1)
            let v = deftree!(concat 'x 'y).unwrap().reshape(1, 2).unwrap(); // (1,2)
            let uv = u.clone().dot_product(v.clone()).unwrap();
            let vu = v.dot_product(u).unwrap();
            assert!(
                uv.equivalent(&vu),
                "Dot product should be commutative across orientations"
            );
        }
        // Test with constants
        {
            let u = deftree!(concat 1 2 3).unwrap(); // (3,1)
            let v = deftree!(concat 4 5 6).unwrap(); // (3,1)
            let result = u.dot_product(v).unwrap();
            // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
            let expected = deftree!(+ (+ (* 1 4) (* 2 5)) (* 3 6)).unwrap();
            assert!(
                result.equivalent(&expected),
                "Dot product with constants should work"
            );
        }
    }

    #[test]
    fn t_extract_basic_cases() {
        // Single element extraction
        {
            // [a c e]  -> extract (1,2) -> [f]
            // [b d f]
            let matrix = deftree!(concat 'a 'b 'c 'd 'e 'f)
                .unwrap()
                .reshape(2, 3)
                .unwrap();
            let result = matrix.extract(&[(1, 2)]).unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!('f).unwrap();
            assert!(
                result.equivalent(&expected),
                "Single element extraction should work"
            );
        }
        // Multiple elements in order
        {
            // [a c e]  -> extract [(0,0), (1,1), (0,2)] -> [a]
            // [b d f]                                        [d]
            //                                                [e]
            let matrix = deftree!(concat 'a 'b 'c 'd 'e 'f)
                .unwrap()
                .reshape(2, 3)
                .unwrap();
            let result = matrix.extract(&[(0, 0), (1, 1), (0, 2)]).unwrap();
            assert_eq!(result.dims(), (3, 1));
            let expected = deftree!(concat 'a 'd 'e).unwrap();
            assert!(
                result.equivalent(&expected),
                "Multiple element extraction should work"
            );
        }
        // Column vector extraction
        {
            // [a]  -> extract [(1,0), (0,0), (2,0)] -> [b]
            // [b]                                        [a]
            // [c]                                        [c]
            let vector = deftree!(concat 'a 'b 'c).unwrap(); // (3,1)
            let result = vector.extract(&[(1, 0), (0, 0), (2, 0)]).unwrap();
            assert_eq!(result.dims(), (3, 1));
            let expected = deftree!(concat 'b 'a 'c).unwrap();
            assert!(
                result.equivalent(&expected),
                "Column vector reordering should work"
            );
        }
        // Row vector extraction
        {
            // [a b c d]  -> extract [(0,3), (0,1), (0,0)] -> [d]
            //                                                [b]
            //                                                [a]
            let vector = deftree!(concat 'a 'b 'c 'd).unwrap().reshape(1, 4).unwrap(); // (1,4)
            let result = vector.extract(&[(0, 3), (0, 1), (0, 0)]).unwrap();
            assert_eq!(result.dims(), (3, 1));
            let expected = deftree!(concat 'd 'b 'a).unwrap();
            assert!(
                result.equivalent(&expected),
                "Row vector extraction should work"
            );
        }
    }

    #[test]
    fn t_extract_with_expressions() {
        // Extract from matrix with expressions
        {
            // [x+1   sin(y)]  -> extract [(1,0), (0,1)] -> [x^2  ]
            // [x^2   cos(z)]                               [sin(y)]
            let matrix = deftree!(concat (+ 'x 1) (pow 'x 2) (sin 'y) (cos 'z))
                .unwrap()
                .reshape(2, 2)
                .unwrap();
            let result = matrix.extract(&[(1, 0), (0, 1)]).unwrap();
            assert_eq!(result.dims(), (2, 1));
            let expected = deftree!(concat (pow 'x 2) (sin 'y)).unwrap();
            assert!(
                result.equivalent(&expected),
                "Expression extraction should work"
            );
        }
        // Extract with duplicates (same element multiple times)
        {
            // [a b]  -> extract [(0,0), (1,1), (0,0)] -> [a]
            // [c d]                                      [d]
            //                                            [a]
            let matrix = deftree!(concat 'a 'c 'b 'd).unwrap().reshape(2, 2).unwrap();
            let result = matrix.extract(&[(0, 0), (1, 1), (0, 0)]).unwrap();
            assert_eq!(result.dims(), (3, 1));
            let expected = deftree!(concat 'a 'd 'a).unwrap();
            assert!(
                result.equivalent(&expected),
                "Duplicate extraction should work"
            );
        }
        // Extract all elements (full matrix flattening)
        {
            // [a c]  -> extract [(0,0), (1,0), (0,1), (1,1)] -> [a]
            // [b d]                                             [b]
            //                                                   [c]
            //                                                   [d]
            let matrix = deftree!(concat 'a 'b 'c 'd).unwrap().reshape(2, 2).unwrap();
            let result = matrix.extract(&[(0, 0), (1, 0), (0, 1), (1, 1)]).unwrap();
            assert_eq!(result.dims(), (4, 1));
            let expected = deftree!(concat 'a 'b 'c 'd).unwrap();
            assert!(
                result.equivalent(&expected),
                "Full matrix extraction should work"
            );
        }
    }

    #[test]
    fn t_extract_edge_cases() {
        // 1x1 matrix (scalar)
        {
            let scalar = deftree!('x).unwrap().reshape(1, 1).unwrap();
            let result = scalar.extract(&[(0, 0)]).unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!('x).unwrap();
            assert!(
                result.equivalent(&expected),
                "Scalar extraction should work"
            );
        }
        // Large matrix with selective extraction
        {
            // Extract corners from 3x3 matrix
            // [a d g]  -> extract [(0,0), (0,2), (2,0), (2,2)] -> [a]
            // [b e h]                                             [g]
            // [c f i]                                             [c]
            //                                                     [i]
            let matrix = deftree!(concat 'a 'b 'c 'd 'e 'f 'g 'h 'i)
                .unwrap()
                .reshape(3, 3)
                .unwrap();
            let result = matrix.extract(&[(0, 0), (0, 2), (2, 0), (2, 2)]).unwrap();
            assert_eq!(result.dims(), (4, 1));
            let expected = deftree!(concat 'a 'g 'c 'i).unwrap();
            assert!(
                result.equivalent(&expected),
                "Corner extraction should work"
            );
        }
        // Reverse order extraction
        {
            // [a b c]  -> extract [(0,2), (0,1), (0,0)] -> [c]
            //                                              [b]
            //                                              [a]
            let vector = deftree!(concat 'a 'b 'c).unwrap().reshape(1, 3).unwrap();
            let result = vector.extract(&[(0, 2), (0, 1), (0, 0)]).unwrap();
            assert_eq!(result.dims(), (3, 1));
            let expected = deftree!(concat 'c 'b 'a).unwrap();
            assert!(
                result.equivalent(&expected),
                "Reverse extraction should work"
            );
        }
    }

    #[test]
    fn t_extract_error_cases() {
        // Empty indices should return InvalidDimensions error
        {
            let matrix = deftree!(concat 'a 'b 'c 'd).unwrap().reshape(2, 2).unwrap();
            let result = matrix.extract(&[]);
            match result {
                Err(Error::InvalidDimensions) => {}
                other => panic!("Expected InvalidDimensions error, got: {:?}", other),
            }
        }
        // Out of bounds access should return IndexOutOfBounds error
        {
            let matrix = deftree!(concat 'a 'b 'c 'd).unwrap().reshape(2, 2).unwrap(); // Valid indices: (0,0), (0,1), (1,0), (1,1)
            // Try to access (2, 0) which is out of bounds for rows
            let result = matrix.extract(&[(2, 0)]);
            match result {
                Err(Error::IndexOutOfBounds(2, 0)) => {}
                other => panic!("Expected IndexOutOfBounds(2, 0) error, got: {:?}", other),
            }
        }
        // Out of bounds column access
        {
            let matrix = deftree!(concat 'a 'b 'c 'd).unwrap().reshape(2, 2).unwrap(); // Valid indices: (0,0), (0,1), (1,0), (1,1)
            // Try to access (0, 3) which is out of bounds for columns
            let result = matrix.extract(&[(0, 3)]);
            match result {
                Err(Error::IndexOutOfBounds(0, 3)) => {}
                other => panic!("Expected IndexOutOfBounds(0, 3) error, got: {:?}", other),
            }
        }
        // Mixed valid and invalid indices
        {
            let matrix = deftree!(concat 'a 'b 'c 'd).unwrap().reshape(2, 2).unwrap();
            // First index valid, second invalid
            let result = matrix.extract(&[(0, 0), (3, 1)]);
            match result {
                Err(Error::IndexOutOfBounds(3, 1)) => {}
                other => panic!("Expected IndexOutOfBounds(3, 1) error, got: {:?}", other),
            }
        }
    }

    #[test]
    fn t_l2norm_basic_cases() {
        // Scalar (1x1) case
        {
            let scalar = deftree!(5).unwrap().reshape(1, 1).unwrap();
            let result = scalar.l2norm().unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(abs 5).unwrap();
            assert!(
                result.equivalent(&expected),
                "L2 norm of scalar should be absolute value"
            );
        }
        // 2D column vector
        {
            // ||[3, 4]|| = sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5
            let vector = deftree!(concat 3 4).unwrap(); // (2,1)
            let result = vector.l2norm().unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(sqrt (+ (* 3 3) (* 4 4))).unwrap();
            assert!(
                result.equivalent(&expected),
                "L2 norm should work for 2D column vector"
            );
        }
        // 2D row vector
        {
            // ||[3, 4]|| = sqrt(3² + 4²)
            let vector = deftree!(concat 3 4).unwrap().reshape(1, 2).unwrap(); // (1,2)
            let result = vector.l2norm().unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(sqrt (+ (* 3 3) (* 4 4))).unwrap();
            assert!(
                result.equivalent(&expected),
                "L2 norm should work for 2D row vector"
            );
        }
        // 3D column vector with expressions
        {
            // ||[x, sin(y), 2]|| = sqrt(x² + sin²(y) + 4)
            let vector = deftree!(concat 'x (sin 'y) 2).unwrap(); // (3,1)
            let result = vector.l2norm().unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(sqrt (+ (+
                                             (* 'x 'x)
                                             (* (sin 'y) (sin 'y)))
                                          (* 2 2)))
            .unwrap();
            assert!(
                result.equivalent(&expected),
                "L2 norm should work with expressions"
            );
        }
        // Single element column vector
        {
            // ||[a]|| = sqrt(a²) = |a|
            let vector = deftree!('a).unwrap(); // (1,1) treated as vector
            let result = vector.l2norm().unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(abs 'a).unwrap();
            assert!(
                result.equivalent(&expected),
                "L2 norm of single element should be absolute value"
            );
        }
    }

    #[test]
    fn t_l2norm_larger_vectors() {
        // 4D column vector
        {
            // ||[1, 2, 3, 4]|| = sqrt(1² + 2² + 3² + 4²) = sqrt(30)
            let vector = deftree!(concat 1 2 3 4).unwrap(); // (4,1)
            let result = vector.l2norm().unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(sqrt (+ (+
                                             (+ (* 1 1) (* 2 2))
                                             (* 3 3))
                                          (* 4 4)))
            .unwrap();
            assert!(
                result.equivalent(&expected),
                "L2 norm should work for 4D vector"
            );
        }
        // 5D row vector with mixed expressions
        {
            // ||[cos(a), x+1, exp(b), 0, sqrt(c)]||
            let vector = deftree!(concat (cos 'a) (+ 'x 1) (exp 'b) 0 (sqrt 'c))
                .unwrap()
                .reshape(1, 5)
                .unwrap(); // (1,5)
            let result = vector.l2norm().unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(sqrt (
                + (+ (+
                      (+ (* (cos 'a) (cos 'a)) (* (+ 'x 1) (+ 'x 1)))
                      (* (exp 'b) (exp 'b)))
                   (* 0 0)) (* (sqrt 'c) (sqrt 'c))
            ))
            .unwrap();
            assert!(
                result.equivalent(&expected),
                "L2 norm should work for 5D vector with mixed expressions"
            );
        }
        // 6D vector with repeated elements
        {
            // ||[a, a, b, b, c, c]|| = sqrt(2a² + 2b² + 2c²)
            let vector = deftree!(concat 'a 'a 'b 'b 'c 'c).unwrap(); // (6,1)
            let result = vector.l2norm().unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(sqrt (
                + (+ (+
                      (+ (+ (* 'a 'a) (* 'a 'a)) (* 'b 'b))
                      (* 'b 'b))
                   (* 'c 'c)) (* 'c 'c)
            ))
            .unwrap();
            assert!(
                result.equivalent(&expected),
                "L2 norm should work for vector with repeated elements"
            );
        }
    }

    #[test]
    fn t_l2norm_complex_expressions() {
        // Vector with nested mathematical expressions
        {
            // ||[sin(x)·cos(y), tan(z), log(w)]||
            let vector = deftree!(concat (* (sin 'x) (cos 'y)) (tan 'z) (log 'w)).unwrap(); // (3,1)
            let result = vector.l2norm().unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(sqrt (
                + (+ (* (* (sin 'x) (cos 'y)) (* (sin 'x) (cos 'y))) (* (tan 'z) (tan 'z))) (* (log 'w) (log 'w))
            )).unwrap();
            assert!(
                result.equivalent(&expected),
                "L2 norm should work with nested expressions"
            );
        }
        // Vector with polynomial expressions
        {
            // ||[x², x+1, x-1]|| = sqrt((x²)² + (x+1)² + (x-1)²)
            let vector = deftree!(concat (pow 'x 2) (+ 'x 1) (- 'x 1)).unwrap(); // (3,1)
            let result = vector.l2norm().unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(sqrt (
                + (+ (* (pow 'x 2) (pow 'x 2)) (* (+ 'x 1) (+ 'x 1))) (* (- 'x 1) (- 'x 1))
            ))
            .unwrap();
            assert!(
                result.equivalent(&expected),
                "L2 norm should work with polynomial expressions"
            );
        }
        // Vector with constants and variables mixed
        {
            // ||[π, e, x, 42]|| where π and e are constants
            let vector = deftree!(concat
                                  (const f64::consts::PI)
                                  (const f64::consts::E)
                                  'x
                                  42)
            .unwrap(); // (4,1)
            let result = vector.l2norm().unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(sqrt (
                + (+ (+
                      (* (const f64::consts::PI) (const f64::consts::PI))
                      (* (const f64::consts::E) (const f64::consts::E)))
                   (* 'x 'x)) (* 42 42)
            ))
            .unwrap();
            assert!(
                result.equivalent(&expected),
                "L2 norm should work with mixed constants and variables"
            );
        }
    }

    #[test]
    fn t_l2norm_edge_cases() {
        // Vector with zero elements
        {
            // ||[0, 0, 0]|| = sqrt(0 + 0 + 0) = 0
            let vector = deftree!(concat 0 0 0).unwrap(); // (3,1)
            let result = vector.l2norm().unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(sqrt (+ (+ (* 0 0) (* 0 0)) (* 0 0))).unwrap();
            assert!(
                result.equivalent(&expected),
                "L2 norm of zero vector should work"
            );
        }
        // Very long vector (testing the chaining logic)
        {
            // ||[1, 1, 1, 1, 1, 1, 1]|| = sqrt(7)
            let vector = deftree!(concat 1 1 1 1 1 1 1).unwrap(); // (7,1)
            let result = vector.l2norm().unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(sqrt (
                + (+ (+
                      (+ (+ (+ (* 1 1) (* 1 1)) (* 1 1)) (* 1 1))
                      (* 1 1))
                   (* 1 1)) (* 1 1)
            ))
            .unwrap();
            assert!(
                result.equivalent(&expected),
                "L2 norm should work for long vectors"
            );
        }
        // Negative values (should be squared away)
        {
            // ||[-3, 4, -5]|| = sqrt(9 + 16 + 25) = sqrt(50)
            let vector = deftree!(concat (- 3) 4 (- 5)).unwrap(); // (3,1)
            let result = vector.l2norm().unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(sqrt (
                + (+ (* (- 3) (- 3)) (* 4 4)) (* (- 5) (- 5))
            ))
            .unwrap();
            assert!(
                result.equivalent(&expected),
                "L2 norm should work with negative values"
            );
        }
    }

    #[test]
    fn t_l2norm_error_cases() {
        // Matrix (not a vector) should fail
        {
            let matrix = deftree!(concat 'a 'b 'c 'd).unwrap().reshape(2, 2).unwrap();
            let result = matrix.l2norm();
            match result {
                Err(Error::InvalidDimensions) => {}
                other => panic!("Expected InvalidDimensions error, got: {:?}", other),
            }
        }
        // 3x2 matrix should fail
        {
            let matrix = deftree!(concat 'a 'b 'c 'd 'e 'f)
                .unwrap()
                .reshape(3, 2)
                .unwrap();
            let result = matrix.l2norm();
            match result {
                Err(Error::InvalidDimensions) => {}
                other => panic!("Expected InvalidDimensions error, got: {:?}", other),
            }
        }
    }

    #[test]
    fn t_normalized_vec_scalar_cases() {
        // Positive scalar (1x1) case
        {
            let scalar = deftree!('x).unwrap().reshape(1, 1).unwrap();
            let result = scalar.normalize().unwrap();
            assert_eq!(result.dims(), (1, 1));
            // Expected: if x > 0 then 1 else if x < 0 then -1 else 0
            let expected = deftree!(if (< 'x 0) (const -1.0) (if (> 'x 0) 1 0)).unwrap();
            assert!(
                result.equivalent(&expected),
                "Scalar normalization should use ternary conditions"
            );
            compare_trees(&result, &expected, &[('x', -5.0, 5.0)], 20, 1e-15);
        }
    }

    #[test]
    fn t_normalized_vec_basic_vectors() {
        // 2D column vector
        {
            // Normalize [3, 4] -> [3/5, 4/5] since ||[3,4]|| = 5
            let vector = deftree!(concat 3 4).unwrap(); // (2,1)
            let result = vector.normalize().unwrap();
            assert_eq!(result.dims(), (2, 1));
            // Expected: [3/sqrt(9+16), 4/sqrt(9+16)] = [3/sqrt(25), 4/sqrt(25)]
            let expected = deftree!(concat
                (/ 3 (sqrt (+ (* 3 3) (* 4 4))))
                (/ 4 (sqrt (+ (* 3 3) (* 4 4))))
            )
            .unwrap();
            assert!(
                result.equivalent(&expected),
                "2D vector normalization should work"
            );
            compare_trees(
                &result,
                &expected,
                &[],
                1, // No variables, just constants
                1e-14,
            );
        }
        // 2D row vector with variables
        {
            // Normalize [x, y]
            let vector = deftree!(concat 'x 'y).unwrap().reshape(1, 2).unwrap(); // (1,2)
            let result = vector.normalize().unwrap();
            assert_eq!(result.dims(), (1, 2));
            let expected = deftree!(concat
                (/ 'x (sqrt (+ (* 'x 'x) (* 'y 'y))))
                (/ 'y (sqrt (+ (* 'x 'x) (* 'y 'y))))
            )
            .unwrap()
            .reshape(1, 2)
            .unwrap();
            assert!(
                result.equivalent(&expected),
                "Row vector normalization should work"
            );
            compare_trees(
                &result,
                &expected,
                &[('x', -3.0, 3.0), ('y', -3.0, 3.0)],
                20, // 20^2 = 400 samples
                1e-13,
            );
        }
    }

    #[test]
    fn t_normalized_vec_expression_vectors() {
        // 3D vector with expressions
        {
            // Normalize [sin(x), cos(y), 2]
            let vector = deftree!(concat (sin 'x) (cos 'y) 2).unwrap(); // (3,1)
            let result = vector.normalize().unwrap();
            assert_eq!(result.dims(), (3, 1));
            let expected = deftree!(concat
                (/ (sin 'x) (sqrt (+ (+ (* (sin 'x) (sin 'x)) (* (cos 'y) (cos 'y))) (* 2 2))))
                (/ (cos 'y) (sqrt (+ (+ (* (sin 'x) (sin 'x)) (* (cos 'y) (cos 'y))) (* 2 2))))
                (/ 2 (sqrt (+ (+ (* (sin 'x) (sin 'x)) (* (cos 'y) (cos 'y))) (* 2 2))))
            )
            .unwrap();
            assert!(
                result.equivalent(&expected),
                "Expression vector normalization should work"
            );
            compare_trees(
                &result,
                &expected,
                &[('x', -1.5, 1.5), ('y', -1.5, 1.5)],
                20, // 20^2 = 400 samples
                1e-12,
            );
        }
        // 4D vector with mixed expressions
        {
            // Normalize [x+1, x-1, x^2, exp(y)]
            let vector = deftree!(concat (+ 'x 1) (- 'x 1) (pow 'x 2) (exp 'y)).unwrap(); // (4,1)
            let result = vector.normalize().unwrap();
            assert_eq!(result.dims(), (4, 1));
            let expected = deftree!(concat
                (/ (+ 'x 1)
                 (sqrt (+ (+ (+
                              (* (+ 'x 1) (+ 'x 1))
                              (* (- 'x 1) (- 'x 1)))
                           (* (pow 'x 2) (pow 'x 2)))
                        (* (exp 'y) (exp 'y)))))
                (/ (- 'x 1)
                 (sqrt (+ (+ (+
                              (* (+ 'x 1) (+ 'x 1))
                              (* (- 'x 1) (- 'x 1)))
                           (* (pow 'x 2) (pow 'x 2)))
                        (* (exp 'y) (exp 'y)))))
                (/ (pow 'x 2)
                 (sqrt (+ (+ (+
                              (* (+ 'x 1) (+ 'x 1))
                              (* (- 'x 1) (- 'x 1)))
                           (* (pow 'x 2) (pow 'x 2)))
                        (* (exp 'y) (exp 'y)))))
                (/ (exp 'y)
                 (sqrt (+ (+ (+
                              (* (+ 'x 1) (+ 'x 1))
                              (* (- 'x 1) (- 'x 1)))
                           (* (pow 'x 2) (pow 'x 2)))
                        (* (exp 'y) (exp 'y)))))
            )
            .unwrap();
            assert!(
                result.equivalent(&expected),
                "Mixed expression vector normalization should work"
            );
            compare_trees(
                &result,
                &expected,
                &[('x', -2.0, 2.0), ('y', -1.0, 1.0)], // Keep y small to prevent exp overflow
                18,                                    // 18^2 = 324 samples
                1e-12,
            );
        }
    }

    #[test]
    fn t_normalized_vec_edge_cases() {
        // Single element vector (should behave like scalar)
        {
            let vector = deftree!('a).unwrap(); // (1,1) but treated as vector
            let result = vector.normalize().unwrap();
            assert_eq!(result.dims(), (1, 1));
            let expected = deftree!(if (< 'a 0) (const -1.) (if (> 'a 0) 1 0)).unwrap();
            assert!(
                result.equivalent(&expected),
                "Single element vector should use scalar normalization"
            );
            compare_trees(&result, &expected, &[('a', -4.0, 4.0)], 20, 1e-15);
        }
        // Vector with zeros and negatives
        {
            // Normalize [0, -3, 4] -> [0, -3/5, 4/5]
            let vector = deftree!(concat 0 (- 3) 4).unwrap(); // (3,1)
            let result = vector.normalize().unwrap();
            assert_eq!(result.dims(), (3, 1));
            let expected = deftree!(concat
                (/ 0 (sqrt (+ (+ (* 0 0) (* (- 3) (- 3))) (* 4 4))))
                (/ (- 3) (sqrt (+ (+ (* 0 0) (* (- 3) (- 3))) (* 4 4))))
                (/ 4 (sqrt (+ (+ (* 0 0) (* (- 3) (- 3))) (* 4 4))))
            )
            .unwrap();
            assert!(
                result.equivalent(&expected),
                "Vector with zeros and negatives should normalize correctly"
            );
            compare_trees(
                &result,
                &expected,
                &[],
                1, // No variables
                1e-14,
            );
        }
        // Longer vector (5D) to test chaining logic
        {
            // Normalize [1, 1, 1, 1, 1] -> all components become 1/sqrt(5)
            let vector = deftree!(concat 1 1 1 1 1).unwrap(); // (5,1)
            let result = vector.normalize().unwrap();
            assert_eq!(result.dims(), (5, 1));
            let expected = deftree!(concat
                (/ 1 (sqrt (+ (+ (+ (+ (* 1 1) (* 1 1)) (* 1 1)) (* 1 1)) (* 1 1))))
                (/ 1 (sqrt (+ (+ (+ (+ (* 1 1) (* 1 1)) (* 1 1)) (* 1 1)) (* 1 1))))
                (/ 1 (sqrt (+ (+ (+ (+ (* 1 1) (* 1 1)) (* 1 1)) (* 1 1)) (* 1 1))))
                (/ 1 (sqrt (+ (+ (+ (+ (* 1 1) (* 1 1)) (* 1 1)) (* 1 1)) (* 1 1))))
                (/ 1 (sqrt (+ (+ (+ (+ (* 1 1) (* 1 1)) (* 1 1)) (* 1 1)) (* 1 1))))
            )
            .unwrap();
            assert!(
                result.equivalent(&expected),
                "5D vector normalization should work"
            );
            compare_trees(
                &result,
                &expected,
                &[],
                1, // No variables
                1e-14,
            );
        }
    }

    #[test]
    fn t_normalized_vec_complex_expressions() {
        // Vector with trigonometric expressions
        {
            // Normalize [cos(x), sin(x)] -> should have unit norm for any x
            let vector = deftree!(concat (cos 'x) (sin 'x)).unwrap(); // (2,1)
            let result = vector.normalize().unwrap();
            assert_eq!(result.dims(), (2, 1));
            let expected = deftree!(concat
                (/ (cos 'x) (sqrt (+ (* (cos 'x) (cos 'x)) (* (sin 'x) (sin 'x)))))
                (/ (sin 'x) (sqrt (+ (* (cos 'x) (cos 'x)) (* (sin 'x) (sin 'x)))))
            )
            .unwrap();
            assert!(
                result.equivalent(&expected),
                "Trigonometric vector normalization should work"
            );
            compare_trees(&result, &expected, &[('x', -2.0, 2.0)], 20, 1e-13);
        }
        // Vector with polynomial and logarithmic terms
        {
            // Normalize [x^2, log(y), sqrt(z)]
            let vector = deftree!(concat (pow 'x 2) (log 'y) (sqrt 'z)).unwrap(); // (3,1)
            let result = vector.normalize().unwrap();
            assert_eq!(result.dims(), (3, 1));
            let expected = deftree!(concat
                                    (/ (pow 'x 2)
                                     (sqrt (+ (+
                                               (* (pow 'x 2) (pow 'x 2))
                                               (* (log 'y) (log 'y)))
                                            (* (sqrt 'z) (sqrt 'z)))))
                                    (/ (log 'y)
                                     (sqrt (+ (+
                                               (* (pow 'x 2) (pow 'x 2))
                                               (* (log 'y) (log 'y)))
                                            (* (sqrt 'z) (sqrt 'z)))))
                                    (/ (sqrt 'z)
                                     (sqrt (+ (+
                                               (* (pow 'x 2) (pow 'x 2))
                                               (* (log 'y) (log 'y)))
                                            (* (sqrt 'z) (sqrt 'z)))))
            )
            .unwrap();
            assert!(
                result.equivalent(&expected),
                "Complex expression vector normalization should work"
            );
            compare_trees(
                &result,
                &expected,
                &[('x', -2.0, 2.0), ('y', 0.1, 5.0), ('z', 0.1, 5.0)], // Positive y,z for log/sqrt
                7,                                                     // 7^3 = 343 samples
                1e-11,
            );
        }
    }

    #[test]
    fn t_normalized_vec_error_cases() {
        // Matrix (not a vector) should fail
        {
            let matrix = deftree!(concat 'a 'b 'c 'd).unwrap().reshape(2, 2).unwrap();
            let result = matrix.normalize();
            match result {
                Err(Error::InvalidDimensions) => {}
                other => panic!("Expected InvalidDimensions error, got: {:?}", other),
            }
        }
        // 3x2 matrix should fail
        {
            let matrix = deftree!(concat 'a 'b 'c 'd 'e 'f)
                .unwrap()
                .reshape(3, 2)
                .unwrap();
            let result = matrix.normalize();
            match result {
                Err(Error::InvalidDimensions) => {}
                other => panic!("Expected InvalidDimensions error, got: {:?}", other),
            }
        }
        // Empty vector should fail (though this might not be constructible)
        {
            // This tests the n_roots == 0 check
            // Creating such a case might be impossible through normal API,
            // but the check exists for safety
        }
    }

    #[test]
    fn t_normalized_vec_properties() {
        // Test that normalization preserves direction and makes unit vectors
        // This is mainly tested through numerical comparison, but we can
        // verify some algebraic properties
        // Normalizing a unit vector should be close to identity (for cos/sin case)
        {
            // [cos(x), sin(x)] is already unit, so normalizing should give same result
            let unit_vector = deftree!(concat (cos 'x) (sin 'x)).unwrap();
            let normalized = unit_vector.clone().normalize().unwrap();
            // The normalized version should be very close to the original
            // since cos²(x) + sin²(x) = 1
            compare_trees(
                &normalized,
                &unit_vector,
                &[('x', -3.0, 3.0)],
                20,
                1e-12, // Should be very close since original is already normalized
            );
        }
        // Double normalization should be idempotent (normalizing normalized vector)
        {
            let vector = deftree!(concat 'x 'y 'z).unwrap();
            let once_normalized = vector.normalize().unwrap();
            let twice_normalized = once_normalized.clone().normalize().unwrap();
            compare_trees(
                &once_normalized,
                &twice_normalized,
                &[('x', -2.0, 2.0), ('y', -2.0, 2.0), ('z', -2.0, 2.0)],
                7,     // 7^3 = 343 samples
                1e-10, // Slightly looser tolerance due to double computation
            );
        }
    }
}
