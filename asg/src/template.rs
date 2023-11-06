use lazy_static::lazy_static;

use crate::{
    deftemplate,
    parser::parse_template,
    tree::{BinaryOp, Node, UnaryOp},
};

pub struct Template {
    ping: Vec<Node>,
    pong: Vec<Node>,
}

lazy_static! {
    static ref TEMPLATES: Vec<Template> = vec![

        // Factoring a multiplication out of addition.
        deftemplate!(
            (_ping (+ (* k a) (* k b))
             _pong (* k (+ a b)))
        ).unwrap(),
        // Min of two square-roots.
        deftemplate!(
            (_ping (min (sqrt a) (sqrt b))
             _pong (sqrt (min a b)))
        ).unwrap(),
        // Interchangeable fractions.
        deftemplate!(
            (_ping (* (/ a b) (/ x y))
             _pong (* (/ a y) (/ x b)))
        ).unwrap(),
        // Cancelling division.
        deftemplate!(
            (_ping (/ a a)
             _pong 1.0)
        ).unwrap(),
        // Distributing pow over division.
        deftemplate!(
            (_ping (pow (/ a b) 2.)
             _pong (/ (pow a 2.) (pow b 2.)))
        ).unwrap(),
        // Distribute pow over multiplication.
        deftemplate!(
            (_ping (pow (* a b) 2.)
             _pong (* (pow a 2.) (pow b 2.)))
        ).unwrap(),
        // Square of square-root.
        deftemplate!(
            (_ping (pow (sqrt a) 2.)
             _pong a)
        ).unwrap(),
        // Square root of square.
        deftemplate!(
            (_ping (sqrt (pow a 2.))
             _pong a)
        ).unwrap(),
        // Combine exponents.
        deftemplate!(
            (_ping (pow (pow a x) y)
             _pong (pow a (* x y)))
        ).unwrap(),
        // Adding fractions.
        deftemplate!(
            (_ping (+ (/ a d) (/ b d))
             _pong (/ (+ a b) d))
        ).unwrap(),

        // ====== Identity operations ======

        // Add zero.
        deftemplate!(
            (_ping (+ x 0.)
             _pong x)
        ).unwrap(),
        // Subtract zero.
        deftemplate!(
          (_ping (- x 0) _pong x)
        ).unwrap(),
        // Multiply by 1.
        deftemplate!(
          (_ping (* x 1.) _pong x)
        ).unwrap(),
        // Raised to the power of 1.
        deftemplate!(
            (_ping (pow x 1.) _pong x)
        ).unwrap(),

        // ====== Other templates =======

        // Multiply by zero.
        deftemplate!(
            (_ping (* x 0.) _pong 0.)
        ).unwrap(),
        // Raised to the power of zero.
        deftemplate!(
            (_ping (pow x 0.) _pong 1.)
        ).unwrap(),
    ];
}

impl Template {
    pub fn from(ping: Vec<Node>, pong: Vec<Node>) -> Template {
        Template { ping, pong }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_templates() {
        assert!(!TEMPLATES.is_empty());
    }
}
