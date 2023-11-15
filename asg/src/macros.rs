/// Assert at compile time using this macro. This is similar to
/// static_assert in C++.
#[macro_export]
macro_rules! const_assert {
    ($message: literal, $($tt:tt)*) => {
        const _: () = assert!($($tt)*, $message);
    }
}

/// Construct a tree from the lisp expresion.
#[macro_export]
macro_rules! deftree {
    () => {}; // empty;
    (($($a:tt)*)) => { // Unwrap redundant parens.
        $crate::deftree!($($a)*)
    };
    ({$a:expr}) => { // Unwrap curly braces to variables.
        $a
    };
    ({$($a:tt)*}) => { // More complex expressions.
        {$($a)*}
    };
    // Unary ops.
    (- $a:tt) => {
        -$crate::deftree!($a)
    };
    (sqrt $a:tt) => {
        $crate::tree::sqrt($crate::deftree!($a))
    };
    (abs $a:tt) => {
        $crate::tree::abs($crate::deftree!($a))
    };
    (sin $a:tt) => {
        $crate::tree::sin($crate::deftree!($a))
    };
    (cos $a:tt) => {
        $crate::tree::cos($crate::deftree!($a))
    };
    (tan $a:tt) => {
        $crate::tree::tan($crate::deftree!($a))
    };
    (log $a:tt) => {
        $crate::tree::log($crate::deftree!($a))
    };
    (exp $a:tt) => {
        $crate::tree::exp($crate::deftree!($a))
    };
    // Binary ops.
    (+ $a:tt $b:tt) => {
        $crate::deftree!($a) + $crate::deftree!($b)
    };
    (- $a:tt $b:tt) => {
        $crate::deftree!($a) - $crate::deftree!($b)
    };
    (* $a:tt $b:tt) => {
        $crate::deftree!($a) * $crate::deftree!($b)
    };
    (/ $a:tt $b:tt) => {
        $crate::deftree!($a) / $crate::deftree!($b)
    };
    (pow $a:tt $b: tt) => {
        $crate::tree::pow($crate::deftree!($a), $crate::deftree!($b))
    };
    (min $a:tt $b: tt) => {
        $crate::tree::min($crate::deftree!($a), $crate::deftree!($b))
    };
    (max $a:tt $b: tt) => {
        $crate::tree::max($crate::deftree!($a), $crate::deftree!($b))
    };
    // Symbols.
    ($a:ident) => {{
        const LABEL: &str = stringify!($a);
        $crate::const_assert!(
            "Symbols can only have a single character as an identifier.",
            LABEL.len() == 1
        );
        $crate::tree::Tree::symbol(LABEL.chars().next().unwrap())
    }};
    // Float constants.
    (const $tt:expr) => {
        $crate::tree::Tree::constant($tt)
    };
    ($a:literal) => {
        $crate::tree::Tree::constant($a as f64)
    };
}

/// Convert the lisp expression into a string and parse the string
/// into a tree.
#[macro_export]
macro_rules! parsetree {
    ($($exp:tt) *) => {
        $crate::lisp::parse_tree(stringify!($($exp) *))
    };
}

#[cfg(test)]
mod tests {
    use crate::tree::{BinaryOp::*, Node::*, UnaryOp::*};

    #[test]
    fn t_symbol_deftree() {
        let tree = deftree!(x);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.root(), &Symbol('x'));
    }

    #[test]
    fn t_constant_deftree() {
        let tree = deftree!(2.);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.root(), &Constant(2.));
    }

    #[test]
    fn t_negate_deftree() {
        let tree = deftree!(-x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &vec![Symbol('x'), Unary(Negate, 0)]);
    }

    #[test]
    fn t_sqrt_deftree() {
        let tree = deftree!(sqrt x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &vec![Symbol('x'), Unary(Sqrt, 0)]);
    }

    #[test]
    fn t_abs_deftree() {
        let tree = deftree!(abs x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &vec![Symbol('x'), Unary(Abs, 0)]);
    }

    #[test]
    fn t_sin_deftree() {
        let tree = deftree!(sin x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &vec![Symbol('x'), Unary(Sin, 0)]);
    }

    #[test]
    fn t_cos_deftree() {
        let tree = deftree!(cos x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &vec![Symbol('x'), Unary(Cos, 0)]);
    }

    #[test]
    fn t_tan_deftree() {
        let tree = deftree!(tan x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &vec![Symbol('x'), Unary(Tan, 0)]);
    }

    #[test]
    fn t_log_deftree() {
        let tree = deftree!(log x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &vec![Symbol('x'), Unary(Log, 0)]);
    }

    #[test]
    fn t_exp_deftree() {
        let tree = deftree!(exp x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &vec![Symbol('x'), Unary(Exp, 0)]);
    }

    #[test]
    fn t_add_deftree() {
        let tree = deftree!(+ x y);
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &vec![Symbol('x'), Symbol('y'), Binary(Add, 0, 1)]
        );
        let tree = deftree!(+ 2. (-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &vec![
                Constant(2.),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Add, 0, 2)
            ]
        );
    }

    #[test]
    fn t_subtract_deftree() {
        let tree = deftree!(- x y);
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &vec![Symbol('x'), Symbol('y'), Binary(Subtract, 0, 1)]
        );
        let tree = deftree!(-2.(-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &vec![
                Constant(2.),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Subtract, 0, 2)
            ]
        );
    }

    #[test]
    fn t_multiply_deftree() {
        let tree = deftree!(* x y);
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &vec![Symbol('x'), Symbol('y'), Binary(Multiply, 0, 1)]
        );
        let tree = deftree!(*(2.)(-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &vec![
                Constant(2.),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Multiply, 0, 2)
            ]
        );
    }

    #[test]
    fn t_divide_deftree() {
        let tree = deftree!(/ x y);
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &vec![Symbol('x'), Symbol('y'), Binary(Divide, 0, 1)]
        );
        let tree = deftree!(/ 2. (-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &vec![
                Constant(2.),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Divide, 0, 2)
            ]
        );
    }

    #[test]
    fn t_pow_deftree() {
        let tree = deftree!(pow x y);
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &vec![Symbol('x'), Symbol('y'), Binary(Pow, 0, 1)]
        );
        let tree = deftree!(pow 2. (-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &vec![
                Constant(2.),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Pow, 0, 2)
            ]
        );
    }

    #[test]
    fn t_min_deftree() {
        let tree = deftree!(min x y);
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &vec![Symbol('x'), Symbol('y'), Binary(Min, 0, 1)]
        );
        let tree = deftree!(min 2. (-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &vec![
                Constant(2.),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Min, 0, 2)
            ]
        );
    }

    #[test]
    fn t_max_deftree() {
        let tree = deftree!(max x y);
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.nodes(),
            &vec![Symbol('x'), Symbol('y'), Binary(Max, 0, 1)]
        );
        let tree = deftree!(max 2. (-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &vec![
                Constant(2.),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Max, 0, 2)
            ]
        );
    }
}
