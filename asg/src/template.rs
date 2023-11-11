use lazy_static::lazy_static;

use crate::tree::Tree;

#[derive(Clone)]
pub struct Template {
    name: String,
    ping: Tree,
    pong: Tree,
    dof_ping: Box<[usize]>,
    dof_pong: Box<[usize]>,
}

/// This macro is only meant for use within this module.
macro_rules! deftemplate {
    (($($tt:tt)*)) => { // Unwrap parens.
        parsetemplate!($($tt)*)
    };
    ($name: ident ping ($($ping:tt) *) pong ($($pong:tt) *)) => {
        Template::from(
            stringify!($name),
            $crate::deftree!(($($ping) *)),
            $crate::deftree!(($($pong) *))
        )
    };
}

impl Template {
    pub fn from(name: &str, ping: Tree, pong: Tree) -> Template {
        let pinglen = ping.len();
        let ponglen = pong.len();
        let mut template = Template {
            name: name.to_string(),
            ping,
            pong,
            dof_ping: vec![0; pinglen].into_boxed_slice(),
            dof_pong: vec![0; ponglen].into_boxed_slice(),
        };
        calc_dof(
            &template.ping,
            template.ping.root_index(),
            &mut template.dof_ping,
        );
        calc_dof(
            &template.pong,
            template.pong.root_index(),
            &mut template.dof_pong,
        );
        return template;
    }

    pub fn mirrored(&self) -> Template {
        Template {
            name: {
                const REV: &str = "rev_";
                match self.name.strip_prefix(REV) {
                    Some(stripped) => stripped.to_string(),
                    None => REV.to_string() + &self.name.to_string(),
                }
            },
            ping: self.pong.clone(),
            pong: self.ping.clone(),
            dof_ping: self.dof_ping.clone(),
            dof_pong: self.dof_pong.clone(),
        }
    }

    pub fn ping(&self) -> &Tree {
        &self.ping
    }
}

fn calc_dof(tree: &Tree, root: usize, dofs: &mut Box<[usize]>) {
    use crate::tree::Node::*;
    dofs[root] = match tree.node(root) {
        Constant(_) | Symbol(_) => 0,
        Unary(_op, input) => {
            calc_dof(tree, *input, dofs);
            dofs[*input]
        }
        Binary(op, lhs, rhs) => {
            calc_dof(tree, *lhs, dofs);
            calc_dof(tree, *rhs, dofs);
            dofs[*lhs] + dofs[*rhs] + {
                if op.is_commutative() {
                    1
                } else {
                    0
                }
            }
        }
    };
}

lazy_static! {
    static ref TEMPLATES: Vec<Template> = vec![
        deftemplate!(distribute_mul
                     ping (+ (* k a) (* k b))
                     pong (* k (+ a b))
        ),
        deftemplate!(min_of_sqrt
                     ping (min (sqrt a) (sqrt b))
                     pong (sqrt (min a b))
        ),
        deftemplate!(rearrange_frac
                     ping (* (/ a b) (/ x y))
                     pong (* (/ a y) (/ x b))
        ),
        deftemplate!(divide_by_self
                     ping (/ a a)
                     pong (1.0)
        ),
        deftemplate!(distribute_pow_div
                     ping (pow (/ a b) k)
                     pong (/ (pow a k) (pow b k))
        ),
        deftemplate!(distribute_pow_mul
                     ping (pow (* a b) k)
                     pong (* (pow a k) (pow b k))
        ),
        deftemplate!(square_sqrt
                     ping (pow (sqrt a) 2.)
                     pong (a)
        ),
        deftemplate!(sqrt_square
                     ping (sqrt (pow a 2.))
                     pong (abs a)
        ),
        deftemplate!(square_abs
                     ping (pow (abs x) 2.)
                     pong (pow x 2.)
        ),
        deftemplate!(mul_exponents
                     ping (pow (pow a x) y)
                     pong (pow a (* x y))
        ),
        deftemplate!(add_exponents
                     ping (* (pow a x) (pow a y))
                     pong (pow a (+ x y))
        ),
        deftemplate!(add_frac
                     ping (+ (/ a d) (/ b d))
                     pong (/ (+ a b) d)
        ),

        // ====== Identity operations ======

        deftemplate!(add_zero
                     ping (+ x 0.)
                     pong (x)
        ),
        deftemplate!(sub_zero
                     ping (- x 0)
                     pong (x)
        ),
        deftemplate!(mul_1
                     ping (* x 1.)
                     pong (x)
        ),
        deftemplate!(pow_1
                     ping (pow x 1.)
                     pong (x)
        ),
        deftemplate!(div_1
                     ping (/ x 1.)
                     pong (x)
        ),

        // ====== Other templates =======

        deftemplate!(mul_0
                     ping (* x 0.)
                     pong (0.)
        ),
        deftemplate!(pow_0
                     ping (pow x 0.)
                     pong (1.)
        ),
        // ====== Min and max simplifications ======

        // https://math.stackexchange.com/questions/1195917/simplifying-a-function-that-has-max-and-min-expressions
        deftemplate!(min_expand
                     ping (min a b)
                     pong (/ (- (+ a b) (abs (- b a))) 2.)
        ),
        deftemplate!(max_expand
                     ping (max a b)
                     pong (/ (+ (+ a b) (abs (- b a))) 2.)

        ),
        deftemplate!(max_of_sub
                     ping (min (- a x) (- a y))
                     pong (- a (max x y))
        ),
    ];

    static ref MIRRORED_TEMPLATES: Vec<Template> = mirrored(&TEMPLATES);
}

fn mirrored(templates: &Vec<Template>) -> Vec<Template> {
    let mut out = templates.clone();
    out.extend(templates.iter().map(|t| t.mirrored()));
    return out;
}

pub fn get_templates() -> &'static Vec<Template> {
    &MIRRORED_TEMPLATES
}

#[cfg(test)]
pub fn get_template_by_name(name: &str) -> Option<&Template> {
    get_templates().iter().find(|&t| t.name == name)
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
            assert!(names.insert(t.name.as_str()), "Duplicate template found.");
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
            print!("{}   ... ", name);
            compare_trees(&template.ping, &template.pong, vardata, 20, eps);
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
            check_one("div_1", &[('x', -10., 10.)], 0.);
        }
        {
            // === Other templates ===
            check_one("mul_0", &[('x', -10., 10.)], 0.);
            check_one("pow_0", &[('x', -10., 10.)], 0.);
            check_one("min_expand", &[('a', -10., 10.), ('b', -10., 10.)], 1e-14);
            check_one("max_expand", &[('a', -10., 10.), ('b', -10., 10.)], 1e-14);
            check_one(
                "max_of_sub",
                &[('a', -10., 10.), ('x', -10., 10.), ('y', -10., 10.)],
                1e-14,
            );
        }
        {
            // Make sure all templates have been checked.
            let unchecked = TEMPLATES
                .iter()
                .map(|t| t.name.as_str())
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
