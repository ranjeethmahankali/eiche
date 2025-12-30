#[macro_export]
macro_rules! concat_trees {
    ($tree:tt) => {
        $crate::deftree!($tree)
    };
    ($lhs:tt $($rhs:tt) +) => {
        $crate::Tree::concat($crate::deftree!($lhs), $crate::concat_trees!($($rhs) +))
    };
}

/// Construct a tree from the lisp expresion.
#[macro_export]
macro_rules! deftree {
    () => {}; // empty;
    (let (($name:ident $val:tt)) $body:tt) => {{
        let $name = $crate::deftree!($val);
        $crate::deftree!($body)
    }};
    (let (($name:ident $val:tt) $($rest:tt)+) $body:tt) => {{
        let $name = $crate::deftree!($val);
        $crate::deftree!(let ($($rest)+) $body)
    }};
    (($($a:tt)*)) => { // Unwrap redundant parens.
        $crate::deftree!($($a)*)
    };
    ($a:block) => { // Block expressions.
        $a
    };    // Concat
    (concat $($trees:tt) +) => {
        $crate::concat_trees!($($trees) +)
    };
    // Derivatives.
    (sderiv $tree:tt $params:lifetime) => {{
        let params = stringify!($params);
        $crate::symbolic_deriv($crate::deftree!($tree), &params[1..])
    }};
    (nderiv $tree:tt $params:lifetime $eps:literal) => {{
        let params = stringify!($params);
        $crate::numerical_deriv($crate::deftree!($tree), &params[1..], $eps)
    }};
    // Matrix and vector operations that are not unary and binary ops.
    (reshape $tree:tt $rows:literal $cols:literal) => {
        $crate::reshape($crate::deftree!($tree), $rows, $cols)
    };
    (extract $tree:tt ($(($i:literal $j:literal))+)) => {
        $crate::extract($crate::deftree!($tree), &[$(($i, $j)),+])
    };
    // Constants.
    (const $tt:expr) => {{
        let out: Result<$crate::Tree, $crate::Error>  = Ok($crate::Tree::constant({$tt}.into()));
        out
    }};
    // Unary ops with functions names.
    ($unary_op:ident $a:tt) => {
        $crate::$unary_op($crate::deftree!($a))
    };
    // Binary ops with function names.
    ($binary_op:ident $a:tt $b:tt) => {
        $crate::$binary_op($crate::deftree!($a), $crate::deftree!($b))
    };
    // Operators.
    (- $a:tt) => {
        $crate::negate($crate::deftree!($a))
    };
    (- $a:tt $b:tt) => {
        $crate::sub($crate::deftree!($a), $crate::deftree!($b))
    };
    (+ $a:tt $b:tt) => {
        $crate::add($crate::deftree!($a), $crate::deftree!($b))
    };
    (/ $a:tt $b:tt) => {
        $crate::div($crate::deftree!($a), $crate::deftree!($b))
    };
    (* $a:tt $b:tt) => {
        $crate::mul($crate::deftree!($a), $crate::deftree!($b))
    };
    (< $a:tt $b:tt) => {
      $crate::less($crate::deftree!($a), $crate::deftree!($b))
    };
    (<= $a:tt $b:tt) => {
      $crate::leq($crate::deftree!($a), $crate::deftree!($b))
    };
    (== $a:tt $b:tt) => {
      $crate::equals($crate::deftree!($a), $crate::deftree!($b))
    };
    (!= $a:tt $b:tt) => {
      $crate::neq($crate::deftree!($a), $crate::deftree!($b))
    };
    (> $a:tt $b:tt) => {
      $crate::greater($crate::deftree!($a), $crate::deftree!($b))
    };
    (>= $a:tt $b:tt) => {
      $crate::geq($crate::deftree!($a), $crate::deftree!($b))
    };
    // Constants
    ($a:literal) => {{
        let out: Result<$crate::Tree, $crate::Error> = Ok($crate::Tree::constant(($a).into()));
        out
    }};
    // Rust variables.
    ($var:ident) => {
        $var.clone()
    };
    // Conditional / piecewise
    (if $cond:tt $a:tt $b:tt) => {
        $crate::Tree::piecewise($crate::deftree!($cond), $crate::deftree!($a), $crate::deftree!($b))
    };
    // Symbols
    ($a:lifetime) => {{
        const LABEL: &str = {stringify!($a)};
        // Expecting a single character lifetime like 'x. So length should be 2.
        const {assert!(LABEL.len() == 2, "Symbols can only have a single character as an identifier.")};
        let out: Result<$crate::Tree, $crate::Error> = Ok($crate::Tree::symbol(LABEL.chars().last().unwrap()));
        out
    }};
}

/// Assert that the floating point numbers are equal within the given epsilon.
#[macro_export]
macro_rules! assert_float_eq {
    ($a:expr, $b:expr, $eps:expr, $debug:expr) => {{
        // Make variables to avoid evaluating experssions multiple times.
        let a = $a;
        let b = $b;
        let eps = $eps;
        let error = (a - b).abs();
        if error > eps {
            eprintln!("{:?}", $debug);
        }
        assert!(
            error <= eps,
            "Assertion failed: |({}) - ({})| = {:e} <= {:e}",
            a,
            b,
            error,
            eps
        );
    }};
    ($a:expr, $b:expr, $eps:expr) => {
        $crate::assert_float_eq!($a, $b, $eps, "")
    };
    ($a:expr, $b:expr) => {
        $crate::assert_float_eq!($a, $b, f64::EPSILON)
    };
}

#[cfg(test)]
mod test {
    use crate::tree::{BinaryOp::*, Node::*, UnaryOp::*, Value::*};

    #[test]
    fn t_symbol_deftree() {
        let tree = deftree!('x).unwrap();
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.roots(), &[Symbol('x')]);
    }

    #[test]
    fn t_constant_deftree() {
        let tree = deftree!(2.).unwrap();
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.roots(), &[Constant(Scalar(2.))]);
    }

    #[test]
    fn t_negate_deftree() {
        let tree = deftree!(- 'x).unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Negate, 0)]);
    }

    #[test]
    fn t_sqrt_deftree() {
        let tree = deftree!(sqrt 'x).unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Sqrt, 0)]);
    }

    #[test]
    fn t_abs_deftree() {
        let tree = deftree!(abs 'x).unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Abs, 0)]);
    }

    #[test]
    fn t_sin_deftree() {
        let tree = deftree!(sin 'x).unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Sin, 0)]);
    }

    #[test]
    fn t_cos_deftree() {
        let tree = deftree!(cos 'x).unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Cos, 0)]);
    }

    #[test]
    fn t_tan_deftree() {
        let tree = deftree!(tan 'x).unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Tan, 0)]);
    }

    #[test]
    fn t_log_deftree() {
        let tree = deftree!(log 'x).unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Log, 0)]);
    }

    #[test]
    fn t_exp_deftree() {
        let tree = deftree!(exp 'x).unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Exp, 0)]);
    }

    #[test]
    fn t_add_deftree() {
        let tree = deftree!(+ 'x 'y).unwrap();
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.nodes(), &[Symbol('x'), Symbol('y'), Binary(Add, 0, 1)]);
        let tree = deftree!(+ 2. (- 'x)).unwrap();
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &[
                Constant(Scalar(2.)),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Add, 0, 2)
            ]
        );
    }

    #[test]
    fn t_subtract_deftree() {
        let tree = deftree!(- 'x 'y).unwrap();
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &[Symbol('x'), Symbol('y'), Binary(Subtract, 0, 1)]
        );
        let tree = deftree!(-2.(- 'x)).unwrap();
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &[
                Constant(Scalar(2.)),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Subtract, 0, 2)
            ]
        );
    }

    #[test]
    fn t_multiply_deftree() {
        let tree = deftree!(* 'x 'y).unwrap();
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &[Symbol('x'), Symbol('y'), Binary(Multiply, 0, 1)]
        );
        let tree = deftree!(*(2.)(- 'x)).unwrap();
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &[
                Constant(Scalar(2.)),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Multiply, 0, 2)
            ]
        );
    }

    #[test]
    fn t_divide_deftree() {
        let tree = deftree!(/ 'x 'y).unwrap();
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &[Symbol('x'), Symbol('y'), Binary(Divide, 0, 1)]
        );
        let tree = deftree!(/ 2. (- 'x)).unwrap();
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &[
                Constant(Scalar(2.)),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Divide, 0, 2)
            ]
        );
    }

    #[test]
    fn t_pow_deftree() {
        let tree = deftree!(pow 'x 'y).unwrap();
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.nodes(), &[Symbol('x'), Symbol('y'), Binary(Pow, 0, 1)]);
        let tree = deftree!(pow 2. (- 'x)).unwrap();
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &[
                Constant(Scalar(2.)),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Pow, 0, 2)
            ]
        );
    }

    #[test]
    fn t_min_deftree() {
        let tree = deftree!(min 'x 'y).unwrap();
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.nodes(), &[Symbol('x'), Symbol('y'), Binary(Min, 0, 1)]);
        let tree = deftree!(min 2. (- 'x)).unwrap();
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &[
                Constant(Scalar(2.)),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Min, 0, 2)
            ]
        );
    }

    #[test]
    fn t_max_deftree() {
        let tree = deftree!(max 'x 'y).unwrap();
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.nodes(), &[Symbol('x'), Symbol('y'), Binary(Max, 0, 1)]);
        let tree = deftree!(max 2. (- 'x)).unwrap();
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &[
                Constant(Scalar(2.)),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Max, 0, 2)
            ]
        );
    }

    #[test]
    fn t_concat_deftree() {
        let tree = deftree!(concat 'a 'b 'c 'd).unwrap();
        assert_eq!(
            tree.nodes(),
            &[Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d')]
        );
        assert_eq!(tree.dims(), (4, 1));
    }

    #[test]
    fn t_deftree_bool() {
        let tree = concat_trees!(true false).unwrap();
        assert_eq!(tree.nodes(), &[Constant(Bool(true)), Constant(Bool(false))]);
    }

    #[test]
    fn t_let_single_binding() {
        let tree = deftree!(let ((numer (+ 1 'x)))
                            (/ numer 2))
        .unwrap();
        assert_eq!(tree.dims(), (1, 1));
        assert_eq!(
            tree.nodes(),
            &[
                Constant(Scalar(1.0)),
                Symbol('x'),
                Binary(Add, 0, 1),
                Constant(Scalar(2.0)),
                Binary(Divide, 2, 3)
            ]
        );
    }

    #[test]
    fn t_let_multiple_bindings() {
        let tree = deftree!(let ((numer (log (+ 1 (pow 'x 2))))
                                 (denom (+ 1 (pow 'x 2))))
                            (/ numer denom))
        .unwrap();
        println!("{:?}", &tree);
        assert_eq!(tree.dims(), (1, 1));
        assert_eq!(
            tree.nodes(),
            &[
                Constant(Scalar(1.0)),
                Symbol('x'),
                Constant(Scalar(2.0)),
                Binary(Pow, 1, 2),
                Binary(Add, 0, 3),
                Unary(Log, 4),
                Constant(Scalar(1.0)),
                Symbol('x'),
                Constant(Scalar(2.0)),
                Binary(Pow, 7, 8),
                Binary(Add, 6, 9),
                Binary(Divide, 5, 10)
            ]
        );
    }

    #[test]
    fn t_let_multiple_bindings_multiple_uses() {
        let tree = deftree!(let ((x (+ 1 'p))
                                 (y (- 1 'q)))
                            (/ (+ x y) (- x y)))
        .unwrap();
        assert_eq!(
            tree.nodes(),
            &[
                Constant(Scalar(1.0)),
                Symbol('p'),
                Binary(Add, 0, 1),
                Constant(Scalar(1.0)),
                Symbol('q'),
                Binary(Subtract, 3, 4),
                Binary(Add, 2, 5),
                Constant(Scalar(1.0)),
                Symbol('p'),
                Binary(Add, 7, 8),
                Constant(Scalar(1.0)),
                Symbol('q'),
                Binary(Subtract, 10, 11),
                Binary(Subtract, 9, 12),
                Binary(Divide, 6, 13)
            ]
        );
    }

    // Matrix operation tests
    #[test]
    fn t_transpose_deftree() {
        let tree = deftree!(transpose (concat 'x 'y 'z)).unwrap();
        assert_eq!(tree.dims(), (1, 3));
        // Should be equivalent to reshaping the vector
        let expected = deftree!(reshape (concat 'x 'y 'z) 1 3).unwrap();
        assert!(tree.equivalent(&expected));
    }

    #[test]
    fn t_l2norm_deftree() {
        let tree = deftree!(l2norm (concat 'x 'y)).unwrap();
        assert_eq!(tree.dims(), (1, 1));
        // Should compute sqrt(x² + y²)
        let expected = deftree!(sqrt (+ (* 'x 'x) (* 'y 'y))).unwrap();
        assert!(tree.equivalent(&expected));
    }

    #[test]
    fn t_dot_deftree() {
        let tree = deftree!(let ((vec1 (concat 'a 'b))
                                 (vec2 (concat 'x 'y)))
                            (dot vec1 vec2))
        .unwrap();
        assert_eq!(tree.dims(), (1, 1));
        // Should compute a*x + b*y
        let expected = deftree!(+ (* 'a 'x) (* 'b 'y)).unwrap();
        assert!(tree.equivalent(&expected));
    }

    #[test]
    fn t_matmul_deftree() {
        let tree = deftree!(let ((mat1 (reshape (concat 'a 'b 'c 'd) 2 2))
                                 (mat2 (concat 'x 'y)))
                            (matmul mat1 mat2))
        .unwrap();
        assert_eq!(tree.dims(), (2, 1));
        // Should compute matrix multiplication: [a c; b d] * [x; y] = [a*x + c*y; b*x + d*y]
        let expected = deftree!(concat
            (+ (* 'a 'x) (* 'c 'y))
            (+ (* 'b 'x) (* 'd 'y))
        )
        .unwrap();
        assert!(tree.equivalent(&expected));
    }

    #[test]
    fn t_normalize_deftree() {
        let tree = deftree!(normalize (concat 'x 'y)).unwrap();
        assert_eq!(tree.dims(), (2, 1));
        // Should compute [x/sqrt(x²+y²), y/sqrt(x²+y²)]
        let expected = deftree!(concat
            (/ 'x (sqrt (+ (* 'x 'x) (* 'y 'y))))
            (/ 'y (sqrt (+ (* 'x 'x) (* 'y 'y))))
        )
        .unwrap();
        assert!(tree.equivalent(&expected));
    }

    #[test]
    fn t_extract_deftree() {
        // Test extracting elements from a 2x2 matrix
        let tree = deftree!(let ((mat (reshape (concat 'a 'b 'c 'd) 2 2)))
                            (extract mat ((0 0) (1 1) (0 1))))
        .unwrap();
        assert_eq!(tree.dims(), (3, 1));
        // Should extract elements [a, d, c] (positions (0,0), (1,1), (0,1))
        let expected = deftree!(concat 'a 'd 'c).unwrap();
        assert!(tree.equivalent(&expected));
    }

    #[test]
    fn t_matrix_operations_composition() {
        // Test composition of matrix operations
        let tree = deftree!(let ((vec (concat 'x 'y 'z)))
                            (l2norm (transpose vec)))
        .unwrap();
        assert_eq!(tree.dims(), (1, 1));
        // Should be equivalent to l2norm of the original vector since transpose of column vector is row vector
        let expected = deftree!(l2norm (concat 'x 'y 'z)).unwrap();
        assert!(tree.equivalent(&expected));
        // Test normalize composition
        let normalized_tree = deftree!(let ((vec (concat 'a 'b)))
                                       (l2norm (normalize vec)))
        .unwrap();
        assert_eq!(normalized_tree.dims(), (1, 1));
    }
}
