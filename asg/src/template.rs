use lazy_static::lazy_static;

use crate::tree::{Node, Tree, TreeError};

pub struct Template {
    name: &'static str,
    ping: Vec<Node>,
    pong: Vec<Node>,
}

#[macro_export]
macro_rules! parsetemplate {
    (($($tt:tt)*)) => {
        parsetemplate!($($tt)*)
    };
    ($name: ident ping ($($ping:tt) *) pong ($($pong:tt) *)) => {
        Template::from(
            stringify!($name),
            $crate::parser::parse_nodes(stringify!(($($ping) *))).unwrap(),
            $crate::parser::parse_nodes(stringify!(($($pong) *))).unwrap()
        )
    };
}

impl Template {
    pub fn from(
        name: &'static str,
        ping: Vec<Node>,
        pong: Vec<Node>,
    ) -> Result<Template, TreeError> {
        Ok(Template {
            name,
            ping: Tree::validate(ping)?,
            pong: Tree::validate(pong)?,
        })
    }

    pub fn mirror_templates(mut templates: Vec<Template>) -> Vec<Template> {
        let num = templates.len();
        for i in 0..num {
            let t = &templates[i];
            templates.push(Template {
                name: t.name,
                ping: t.pong.clone(),
                pong: t.ping.clone(),
            });
        }
        return templates;
    }
}

lazy_static! {
    static ref TEMPLATES: Vec<Template> = Template::mirror_templates(vec![

        parsetemplate!( DISTRIBUTE_MUL
            ping (+ (* k a) (* k b)) pong (* k (+ a b))
        ).unwrap(),
        parsetemplate!( MIN_OF_SQRT
            ping (min (sqrt a) (sqrt b))
                pong (sqrt (min a b))
        ).unwrap(),
        parsetemplate!( REARRANGE_FRAC
            ping (* (/ a b) (/ x y))
                pong (* (/ a y) (/ x b))
        ).unwrap(),
        parsetemplate!( DIVIDE_BY_SELF
            ping (/ a a)
                pong (1.0)
        ).unwrap(),
        parsetemplate!( DISTRIBUTE_POW_DIV
            ping (pow (/ a b) 2.)
                pong (/ (pow a 2.) (pow b 2.))
        ).unwrap(),
        parsetemplate!( DISTRIBUTE_POW_MUL
            ping (pow (* a b) 2.)
                pong (* (pow a 2.) (pow b 2.))
        ).unwrap(),
        parsetemplate!( SQUARE_SQRT
            ping (pow (sqrt a) 2.)
                pong (a)
        ).unwrap(),
        parsetemplate!( SQRT_SQUARE
            ping (sqrt (pow a 2.))
                pong (a)
        ).unwrap(),
        parsetemplate!( ADD_EXPONENTS
            ping (pow (pow a x) y)
                pong (pow a (* x y))
        ).unwrap(),
        parsetemplate!( ADD_FRAC
            ping (+ (/ a d) (/ b d))
                pong (/ (+ a b) d)
        ).unwrap(),

        // ====== Identity operations ======

        parsetemplate!( ADD_ZERO
            ping (+ x 0.)
                pong (x)
        ).unwrap(),
        parsetemplate!( SUB_ZERO
            ping (- x 0) pong (x)
        ).unwrap(),
        parsetemplate!( MUL_1
            ping (* x 1.) pong (x)
        ).unwrap(),
        parsetemplate!( POW_1
            ping (pow x 1.) pong (x)
        ).unwrap(),

        // ====== Other templates =======

        parsetemplate!( MUL_0
            ping (* x 0.) pong (0.)
        ).unwrap(),
        parsetemplate!( POW_0
            ping (pow x 0.) pong (1.)
        ).unwrap(),
        // Min and max simplifications from:
        // https://math.stackexchange.com/questions/1195917/simplifying-a-function-that-has-max-and-min-expressions
        parsetemplate!( MIN_EXPAND
            ping (min a b)
                pong (/ (+ (+ a b) (abs (- b a))) 2.)
        ).unwrap(),
        parsetemplate!( MAX_EXPAND
            ping (min a b)
                pong (/ (- (+ a b) (abs (- b a))) 2.)
        ).unwrap(),
    ]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_templates() {
        // Just to make sure all the templates are valid and load
        // correctly.
        assert!(!TEMPLATES.is_empty());
        assert!(TEMPLATES.len() >= 36);
        // All templates should be mirrored.
        assert_eq!(TEMPLATES.len() % 2, 0);
    }

    // #[test]
    // fn check_templates() {
    //     let filter_symbols = |node: &Node| -> Option<char> {
    //         if let Node::Symbol(c) = node {
    //             Some(*c)
    //         } else {
    //             None
    //         }
    //     };
    //     for template in TEMPLATES.iter() {
    //         use crate::tests::tests::compare_trees;
    //         // Check if valid trees can be made from the templates.
    //         let ping = Tree::from_nodes(template.ping.clone()).unwrap();
    //         let pong = Tree::from_nodes(template.pong.clone()).unwrap();
    //         let vardata: Vec<(char, f64, f64)> = {
    //             let mut vars: Vec<_> = ping
    //                 .nodes()
    //                 .iter()
    //                 .chain(pong.nodes().iter())
    //                 .filter_map(filter_symbols)
    //                 .collect();
    //             vars.sort();
    //             vars.dedup();
    //             vars.iter().map(|c| (*c, -10., 10.)).collect()
    //         };
    //         println!(
    //             "=================\nPing: {:?}\nPong: {:?}\nVariables: {:?}\n===============",
    //             ping, pong, vardata
    //         );
    //         compare_trees(ping, pong, &vardata, 40, 1e-12);
    //     }
    // }
}
