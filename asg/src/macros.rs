/// Assert at compile time using this macro. This is similar to
/// static_assert in C++.
#[macro_export]
macro_rules! const_assert {
    ($message: literal, $($tt:tt)*) => {
        const _: () = assert!($($tt)*, $message);
    }
}

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
    };
    // Concat
    (concat $($trees:tt) +) => {
        $crate::concat_trees!($($trees) +)
    };
    // Unary ops with functions names.
    ($unary_op:ident $a:tt) => {
        $crate::tree::$unary_op($crate::deftree!($a))
    };
    // Binary ops with function names.
    ($binary_op:ident $a:tt $b:tt) => {
        $crate::tree::$binary_op($crate::deftree!($a), $crate::deftree!($b))
    };
    // Negate operator
    (- $a:tt) => {
        -$crate::deftree!($a)
    };
    // Binary ops with operators
    ($op:tt $lhs:tt $rhs:tt) => {
        $crate::deftree!($lhs) $op $crate::deftree!($rhs)
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

#[cfg(test)]
mod test {
    use crate::tree::{BinaryOp::*, Node::*, UnaryOp::*};

    #[test]
    fn t_symbol_deftree() {
        let tree = deftree!(x);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.roots(), &[Symbol('x')]);
    }

    #[test]
    fn t_constant_deftree() {
        let tree = deftree!(2.);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.roots(), &[Constant(2.)]);
    }

    #[test]
    fn t_negate_deftree() {
        let tree = deftree!(-x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Negate, 0)]);
    }

    #[test]
    fn t_sqrt_deftree() {
        let tree = deftree!(sqrt x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Sqrt, 0)]);
    }

    #[test]
    fn t_abs_deftree() {
        let tree = deftree!(abs x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Abs, 0)]);
    }

    #[test]
    fn t_sin_deftree() {
        let tree = deftree!(sin x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Sin, 0)]);
    }

    #[test]
    fn t_cos_deftree() {
        let tree = deftree!(cos x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Cos, 0)]);
    }

    #[test]
    fn t_tan_deftree() {
        let tree = deftree!(tan x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Tan, 0)]);
    }

    #[test]
    fn t_log_deftree() {
        let tree = deftree!(log x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Log, 0)]);
    }

    #[test]
    fn t_exp_deftree() {
        let tree = deftree!(exp x);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes(), &[Symbol('x'), Unary(Exp, 0)]);
    }

    #[test]
    fn t_add_deftree() {
        let tree = deftree!(+ x y);
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.nodes(), &[Symbol('x'), Symbol('y'), Binary(Add, 0, 1)]);
        let tree = deftree!(+ 2. (-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &[
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
            &[Symbol('x'), Symbol('y'), Binary(Subtract, 0, 1)]
        );
        let tree = deftree!(-2.(-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &[
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
            &[Symbol('x'), Symbol('y'), Binary(Multiply, 0, 1)]
        );
        let tree = deftree!(*(2.)(-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &[
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
            &[Symbol('x'), Symbol('y'), Binary(Divide, 0, 1)]
        );
        let tree = deftree!(/ 2. (-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &[
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
        assert_eq!(tree.nodes(), &[Symbol('x'), Symbol('y'), Binary(Pow, 0, 1)]);
        let tree = deftree!(pow 2. (-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &[
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
        assert_eq!(tree.nodes(), &[Symbol('x'), Symbol('y'), Binary(Min, 0, 1)]);
        let tree = deftree!(min 2. (-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &[
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
        assert_eq!(tree.nodes(), &[Symbol('x'), Symbol('y'), Binary(Max, 0, 1)]);
        let tree = deftree!(max 2. (-x));
        assert_eq!(tree.len(), 4);
        assert_eq!(
            tree.nodes(),
            &[
                Constant(2.),
                Symbol('x'),
                Unary(Negate, 1),
                Binary(Max, 0, 2)
            ]
        );
    }

    #[test]
    fn t_concat_deftree() {
        let tree = deftree!(concat a b c d);
        assert_eq!(
            tree.nodes(),
            &[Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d')]
        );
        assert_eq!(tree.dims(), (4, 1));
    }
}
