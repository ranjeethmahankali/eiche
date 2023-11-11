#[macro_export]
macro_rules! const_assert {
    ($message: literal, $($tt:tt)*) => {
        const _: () = assert!($($tt)*, $message);
    }
}

#[macro_export]
macro_rules! deftree {
    () => {}; // empty;
    (($($a:tt)*)) => { // Unwrap redundant parens.
        deftree!($($a)*)
    };
    ({$a:expr}) => { // Unwrap curly braces to variables.
        $a
    };
    ({$($a:tt)*}) => { // More complex expressions.
        {$($a)*}
    };
    // Unary ops.
    (- $a:tt) => {
        -deftree!($a)
    };
    (sqrt $a:tt) => {
        $crate::tree::sqrt(deftree!($a))
    };
    (abs $a:tt) => {
        $crate::tree::abs(deftree!($a))
    };
    (sin $a:tt) => {
        $crate::tree::sin(deftree!($a))
    };
    (cos $a:tt) => {
        $crate::tree::cos(deftree!($a))
    };
    (tan $a:tt) => {
        $crate::tree::tan(deftree!($a))
    };
    (log $a:tt) => {
        $crate::tree::log(deftree!($a))
    };
    (exp $a:tt) => {
        $crate::tree::exp(deftree!($a))
    };
    // Binary ops.
    (+ $a:tt $b:tt) => {
        deftree!($a) + deftree!($b)
    };
    (- $a:tt $b:tt) => {
        deftree!($a) - deftree!($b)
    };
    (* $a:tt $b:tt) => {
        deftree!($a) * deftree!($b)
    };
    (/ $a:tt $b:tt) => {
        deftree!($a) / deftree!($b)
    };
    (pow $a:tt $b: tt) => {
        $crate::tree::pow(deftree!($a), deftree!($b))
    };
    (min $a:tt $b: tt) => {
        $crate::tree::min(deftree!($a), deftree!($b))
    };
    (max $a:tt $b: tt) => {
        $crate::tree::max(deftree!($a), deftree!($b))
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

#[macro_export]
macro_rules! parsetree {
    ($($exp:tt) *) => {
        $crate::parser::parse_tree(stringify!($($exp) *))
    };
}
