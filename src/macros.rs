#[macro_export]
macro_rules! concat_trees {
    ($tree:tt) => {
        $crate::deftree!($tree)
    };
    ($lhs:tt $($rhs:tt) +) => {
        $crate::tree::Tree::concat($crate::deftree!($lhs), $crate::concat_trees!($($rhs) +))
    };
}

/// Construct a tree from the lisp expresion.
#[macro_export]
macro_rules! deftree {
    () => {}; // empty;
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
    (sderiv $tree:tt $params:ident) => {
        $crate::derivative::symbolic_deriv($crate::deftree!($tree), stringify!($params))
    };
    (nderiv $tree:tt $params:ident $eps:literal) => {
        $crate::derivative::numerical_deriv($crate::deftree!($tree), stringify!($params), $eps)
    };
    // Reshape
    (reshape $tree:tt $rows:literal $cols:literal) => {
        $crate::tree::reshape($crate::deftree!($tree), $rows, $cols)
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
    // Conditional / piecewise
    (if $cond:tt $a:tt $b:tt) => {
        $crate::Tree::piecewise($crate::deftree!($cond), $crate::deftree!($a), $crate::deftree!($b))
    };
    // Symbols
    ($a:ident) => {{
        const LABEL: &str = {stringify!($a)};
        const {assert!(LABEL.len() == 1, "Symbols can only have a single character as an identifier.")};
        let out: Result<$crate::Tree, $crate::Error> = Ok($crate::Tree::symbol(LABEL.chars().next().unwrap()));
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
        let error = f64::abs(a - b);
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
        assert_float_eq!($a, $b, $eps, "")
    };
    ($a:expr, $b:expr) => {
        assert_float_eq!($a, $b, f64::EPSILON)
    };
}

#[cfg(test)]
mod test {
    use crate::tree::{BinaryOp::*, Node::*, UnaryOp::*, Value::*};

    #[test]
    fn t_symbol_deftree() {
        let tree = deftree!(x).unwrap();
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
        let tree = deftree!(-x).unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Negate, 0)]);
    }

    #[test]
    fn t_sqrt_deftree() {
        let tree = deftree!(sqrt x).unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Sqrt, 0)]);
    }

    #[test]
    fn t_abs_deftree() {
        let tree = deftree!(abs x).unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Abs, 0)]);
    }

    #[test]
    fn t_sin_deftree() {
        let tree = deftree!(sin x).unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Sin, 0)]);
    }

    #[test]
    fn t_cos_deftree() {
        let tree = deftree!(cos x).unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Cos, 0)]);
    }

    #[test]
    fn t_tan_deftree() {
        let tree = deftree!(tan x).unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Tan, 0)]);
    }

    #[test]
    fn t_log_deftree() {
        let tree = deftree!(log x).unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Log, 0)]);
    }

    #[test]
    fn t_exp_deftree() {
        let tree = deftree!(exp x).unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Exp, 0)]);
    }

    #[test]
    fn t_add_deftree() {
        let tree = deftree!(+ x y).unwrap();
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.nodes(), &[Symbol('x'), Symbol('y'), Binary(Add, 0, 1)]);
        let tree = deftree!(+ 2. (-x)).unwrap();
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
        let tree = deftree!(- x y).unwrap();
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &[Symbol('x'), Symbol('y'), Binary(Subtract, 0, 1)]
        );
        let tree = deftree!(-2.(-x)).unwrap();
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
        let tree = deftree!(* x y).unwrap();
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &[Symbol('x'), Symbol('y'), Binary(Multiply, 0, 1)]
        );
        let tree = deftree!(*(2.)(-x)).unwrap();
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
        let tree = deftree!(/ x y).unwrap();
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &[Symbol('x'), Symbol('y'), Binary(Divide, 0, 1)]
        );
        let tree = deftree!(/ 2. (-x)).unwrap();
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
        let tree = deftree!(pow x y).unwrap();
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.nodes(), &[Symbol('x'), Symbol('y'), Binary(Pow, 0, 1)]);
        let tree = deftree!(pow 2. (-x)).unwrap();
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
        let tree = deftree!(min x y).unwrap();
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.nodes(), &[Symbol('x'), Symbol('y'), Binary(Min, 0, 1)]);
        let tree = deftree!(min 2. (-x)).unwrap();
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
        let tree = deftree!(max x y).unwrap();
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.nodes(), &[Symbol('x'), Symbol('y'), Binary(Max, 0, 1)]);
        let tree = deftree!(max 2. (-x)).unwrap();
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
        let tree = deftree!(concat a b c d).unwrap();
        assert_eq!(
            tree.nodes(),
            &[Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d')]
        );
        assert_eq!(tree.dims(), (4, 1));
    }

    #[test]
    fn t_deftree_bool() {
        let tree = deftree!(concat true false).unwrap();
        assert_eq!(tree.nodes(), &[Constant(Bool(true)), Constant(Bool(false))]);
    }
}
