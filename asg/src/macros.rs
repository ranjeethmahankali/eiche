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
        <$crate::tree::Node as Into<$crate::tree::Tree>>::into($crate::tree::Node::Symbol(LABEL.chars().next().unwrap()))
    }};
    // Float constants.
    (const $tt:expr) => {
        <$crate::tree::Node as Into<$crate::tree::Tree>>::into($crate::tree::Node::Constant($tt))
    };
    ($a:literal) => {
        <$crate::tree::Node as Into<$crate::tree::Tree>>::into($crate::tree::Node::Constant($a as f64))
    };
}

/// Convert the lisp expression into a string and parse the string
/// into a tree.
#[macro_export]
macro_rules! parsetree {
    ($($exp:tt) *) => {
        $crate::parser::parse_tree(stringify!($($exp) *))
    };
}
