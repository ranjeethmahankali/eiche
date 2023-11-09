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

    fn match_node(&self, node: &Node, capture: &mut Capture) -> bool {
        todo!();
    }

    fn match_from(&self, index: usize, tree: &Tree, capture: &mut Capture) {
        for i in (index + 1)..tree.len() {
            if self.match_node(tree.node(i), capture) {
                return;
            }
        }
    }

    pub fn first_match(&self, tree: &Tree, capture: &mut Capture) {
        self.match_from(0, tree, capture);
    }

    pub fn next_match(&self, tree: &Tree, capture: &mut Capture) {
        match capture.node_index {
            Some(i) => self.match_from(i, tree, capture),
            None => return,
        }
    }
}

pub struct Capture {
    node_index: Option<usize>,
    bindings: Vec<(char, usize)>,
}

impl Capture {
    pub fn new() -> Capture {
        Capture {
            node_index: None,
            bindings: vec![],
        }
    }

    pub fn is_valid(&self) -> bool {
        return self.node_index.is_some();
    }

    pub fn apply(&self, tree: Tree) -> Tree {
        todo!();
    }
}

lazy_static! {
    static ref TEMPLATES: Vec<Template> = vec![

        parsetemplate!(distribute_mul
                       ping (+ (* k a) (* k b))
                       pong (* k (+ a b))
        ).unwrap(),
        parsetemplate!(min_of_sqrt
                       ping (min (sqrt a) (sqrt b))
                       pong (sqrt (min a b))
        ).unwrap(),
        parsetemplate!(rearrange_frac
                       ping (* (/ a b) (/ x y))
                       pong (* (/ a y) (/ x b))
        ).unwrap(),
        parsetemplate!(divide_by_self
                       ping (/ a a)
                       pong (1.0)
        ).unwrap(),
        parsetemplate!(distribute_pow_div
                       ping (pow (/ a b) k)
                       pong (/ (pow a k) (pow b k))
        ).unwrap(),
        parsetemplate!(distribute_pow_mul
                       ping (pow (* a b) k)
                       pong (* (pow a k) (pow b k))
        ).unwrap(),
        parsetemplate!(square_sqrt
                       ping (pow (sqrt a) 2.)
                       pong (a)
        ).unwrap(),
        parsetemplate!(sqrt_square
                       ping (sqrt (pow a 2.))
                       pong (abs a)
        ).unwrap(),
        parsetemplate!(square_abs
                       ping (pow (abs x) 2.)
                       pong (pow x 2.)
        ).unwrap(),
        parsetemplate!(mul_exponents
                       ping (pow (pow a x) y)
                       pong (pow a (* x y))
        ).unwrap(),
        parsetemplate!(add_exponents
                       ping (* (pow a x) (pow a y))
                       pong (pow a (+ x y))
        ).unwrap(),
        parsetemplate!(add_frac
                       ping (+ (/ a d) (/ b d))
                       pong (/ (+ a b) d)
        ).unwrap(),

        // ====== Identity operations ======

        parsetemplate!(add_zero
                       ping (+ x 0.)
                       pong (x)
        ).unwrap(),
        parsetemplate!(sub_zero
                       ping (- x 0)
                       pong (x)
        ).unwrap(),
        parsetemplate!(mul_1
                       ping (* x 1.)
                       pong (x)
        ).unwrap(),
        parsetemplate!(pow_1
                       ping (pow x 1.)
                       pong (x)
        ).unwrap(),

        // ====== Other templates =======

        parsetemplate!(mul_0
                       ping (* x 0.)
                       pong (0.)
        ).unwrap(),
        parsetemplate!(pow_0
                       ping (pow x 0.)
                       pong (1.)
        ).unwrap(),
        // Min and max simplifications from:
        // https://math.stackexchange.com/questions/1195917/simplifying-a-function-that-has-max-and-min-expressions
        parsetemplate!(min_expand
                       ping (min a b)
                       pong (/ (- (+ a b) (abs (- b a))) 2.)
        ).unwrap(),
        parsetemplate!(max_expand
                       ping (max a b)
                       pong (/ (+ (+ a b) (abs (- b a))) 2.)
        ).unwrap(),
    ];
}

pub fn get_templates() -> &'static Vec<Template> {
    &TEMPLATES
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn load_templates() {
        // Just to make sure all the templates are valid and load
        // correctly.
        assert!(!TEMPLATES.is_empty());
        // Make sure templates have unique names.
        let mut names: HashSet<&str> = HashSet::with_capacity(TEMPLATES.len());
        for t in TEMPLATES.iter() {
            assert!(names.insert(t.name), "Duplicate template found.");
        }
    }

    #[test]
    fn check_templates() {
        let mut checked: HashSet<&str> = HashSet::with_capacity(TEMPLATES.len());
        let mut check_one = |name: &'static str, vardata: &[(char, f64, f64)], eps: f64| {
            use crate::tests::tests::compare_trees;
            // Find template by name.
            let template = TEMPLATES
                .iter()
                .find(|t| t.name == name)
                .expect(format!("No template found with name: {}", name).as_str());
            // Check if valid trees can be made from the templates.
            let ping = Tree::from_nodes(template.ping.clone()).unwrap();
            let pong = Tree::from_nodes(template.pong.clone()).unwrap();
            print!("{}   ... ", name);
            compare_trees(ping, pong, vardata, 20, eps);
            println!("✔ Passed.");
            assert!(
                checked.insert(name),
                "This template has already been checked. Remove this redundancy."
            );
        };
        // Check each template. This is necessary they need different
        // vardata and ranges. e.g. You can't use negative values in
        // sqrt.
        {
            check_one(
                "distribute_mul",
                &[('k', -10., 10.), ('a', -10., 10.), ('b', -10., 10.)],
                1e-12,
            );
            check_one("min_of_sqrt", &[('a', -10., 10.), ('b', -10., 10.)], 1e-12);
            check_one(
                "rearrange_frac",
                &[
                    ('a', -10., 10.),
                    ('b', -10., 10.),
                    ('x', -10., 10.),
                    ('y', -10., 10.),
                ],
                1e-10,
            );
            check_one("divide_by_self", &[('a', -10., 10.)], 1e-12);
            check_one(
                "distribute_pow_div",
                &[('a', 1., 10.), ('b', 1., 10.), ('k', 0.1, 5.)],
                1e-10,
            );
            check_one(
                "distribute_pow_mul",
                &[('a', 1., 5.), ('b', 1., 5.), ('k', 0.5, 3.)],
                1e-10,
            );
            check_one("square_sqrt", &[('a', -10., 10.)], 1e-12);
            check_one("sqrt_square", &[('a', -10., 10.)], 1e-12);
            check_one("square_abs", &[('x', -10., 10.)], 0.);
            check_one(
                "mul_exponents",
                &[('a', 1., 5.), ('x', 0.5, 3.), ('y', 0.5, 2.)],
                1e-9,
            );
            check_one(
                "add_exponents",
                &[('a', 1., 5.), ('x', 0.5, 3.), ('y', 0.5, 2.)],
                1e-12,
            );
            check_one(
                "add_frac",
                &[('a', -10., 10.), ('b', -10., 10.), ('d', -10., 10.)],
                1e-12,
            );
        }
        {
            // === Identity operations ===
            check_one("add_zero", &[('x', -10., 10.)], 0.);
            check_one("sub_zero", &[('x', -10., 10.)], 0.);
            check_one("mul_1", &[('x', -10., 10.)], 0.);
            check_one("pow_1", &[('x', -10., 10.)], 0.);
        }
        {
            // === Other templates ===
            check_one("mul_0", &[('x', -10., 10.)], 0.);
            check_one("pow_0", &[('x', -10., 10.)], 0.);
            check_one("min_expand", &[('a', -10., 10.), ('b', -10., 10.)], 1e-14);
            check_one("max_expand", &[('a', -10., 10.), ('b', -10., 10.)], 1e-14);
        }
        {
            // Make sure all templates have been checked.
            let unchecked = TEMPLATES
                .iter()
                .map(|t| t.name)
                .filter(|name| !checked.contains(name))
                .collect::<Vec<&str>>()
                .join("\n");
            if !unchecked.is_empty() {
                panic!(
                    "\nThe following templates have not been tested:\n{}\n",
                    unchecked
                );
            }
        }
    }
}
