use lazy_static::lazy_static;

use crate::{
    mutate::TemplateCapture,
    tree::{Node::*, Tree},
};

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

#[derive(Clone)]
pub struct Template {
    name: String,
    ping: Tree,
    pong: Tree,
    dof_ping: Box<[usize]>,
    dof_pong: Box<[usize]>,
}

/// Check the capture to see if every symbol in src is bound to every
/// symbol in dst.
fn complete_capture(capture: &TemplateCapture, src: &Tree, dst: &Tree) -> bool {
    let mut lhs: Vec<_> = capture
        .bindings()
        .iter()
        .map(|(label, _index)| *label)
        .collect();
    lhs.sort();
    let mut rhs = src.symbols();
    rhs.sort();
    if lhs != rhs {
        return false;
    }
    lhs.clear();
    lhs.extend(capture.bindings().iter().filter_map(|(_label, index)| {
        match &dst.nodes()[*index] {
            Constant(_) | Unary(_, _) | Binary(_, _, _) => None,
            Symbol(l) => Some(l),
        }
    }));
    lhs.sort();
    rhs = dst.symbols();
    rhs.sort();
    if lhs != rhs {
        return false;
    }
    return true;
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

    fn valid(self) -> Option<Template> {
        // All symbols in pong must be present in ping too. Otherwise
        // the template cannot be applied to a tree.
        let symbols = self.ping().symbols();
        for label in self.pong().symbols() {
            match symbols.iter().find(|&c| *c == label) {
                Some(_) => {} // Do nothing.
                None => return None,
            }
        }
        return Some(self);
    }

    fn mirrored(&self) -> Option<Template> {
        let out = Template {
            name: {
                const REV: &str = "rev_";
                match self.name.strip_prefix(REV) {
                    Some(stripped) => stripped.to_string(),
                    None => REV.to_string() + &self.name.to_string(),
                }
            },
            ping: self.pong.clone(),
            pong: self.ping.clone(),
            dof_ping: self.dof_pong.clone(),
            dof_pong: self.dof_ping.clone(),
        }
        .valid()?;
        // Make sure the template is not symmetric. If it is,
        // mirroring will produce a redundant template. It's no harm,
        // but no use either. So in the end it is harmful because it
        // wastes resources.
        let mut capture = TemplateCapture::new();
        if capture.next_match(&out, self.ping())
            && complete_capture(&capture, out.ping(), self.ping())
        {
            return None;
        }
        return Some(out);
    }

    pub fn ping(&self) -> &Tree {
        &self.ping
    }

    pub fn pong(&self) -> &Tree {
        &self.pong
    }

    pub fn dof_ping(&self) -> &Box<[usize]> {
        return &self.dof_ping;
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
        deftemplate!(rearrange_mul_div_1
                     ping (/ (* x y) z)
                     pong (* x (/ y z))
        ),
        deftemplate!(rearrange_mul_div_2
                     ping (/ (* x y) z)
                     pong (* y (/ x z))
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
        deftemplate!(min_of_sub_1
                     ping (min (- a x) (- a y))
                     pong (- a (max x y))
        ),
        deftemplate!(min_of_sub_2
                     ping (min (- x c) (- y c))
                     pong (- (min x y) c)
        ),
        deftemplate!(min_of_add_1
                     ping (min (+ a x) (+ b x))
                     pong (+ x (min a b))
        ),
    ];

    static ref MIRRORED_TEMPLATES: Vec<Template> = mirrored(&TEMPLATES);
}

fn mirrored(templates: &Vec<Template>) -> Vec<Template> {
    let mut out = templates.clone();
    out.extend(templates.iter().filter_map(|t| t.mirrored()));
    return out;
}

pub fn get_templates() -> &'static Vec<Template> {
    &MIRRORED_TEMPLATES
}

#[cfg(test)]
pub mod test {
    use super::*;
    use std::collections::HashSet;

    #[cfg(test)]
    pub fn get_template_by_name(name: &str) -> Option<&Template> {
        get_templates().iter().find(|&t| t.name == name)
    }

    #[test]
    fn t_load_templates() {
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
    fn t_check_templates() {
        let mut checked: HashSet<&str> = HashSet::with_capacity(TEMPLATES.len());
        let mut check_one = |name: &'static str, vardata: &[(char, f64, f64)], eps: f64| {
            use crate::test::util::compare_trees;
            // Find template by name.
            let template = TEMPLATES.iter().find(|t| t.name == name).unwrap();
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
            check_one("min_of_sqrt", &[('a', 0., 10.), ('b', 0., 10.)], 1e-12);
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
            check_one(
                "rearrange_mul_div_1",
                &[('x', -10., 10.), ('y', -10., 10.), ('z', -10., 10.)],
                1e-12,
            );
            check_one(
                "rearrange_mul_div_2",
                &[('x', -10., 10.), ('y', -10., 10.), ('z', -10., 10.)],
                1e-12,
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
            check_one("square_sqrt", &[('a', 0., 10.)], 1e-12);
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
            // === Other templates ===
            check_one("min_expand", &[('a', -10., 10.), ('b', -10., 10.)], 1e-14);
            check_one("max_expand", &[('a', -10., 10.), ('b', -10., 10.)], 1e-14);
            check_one(
                "min_of_sub_1",
                &[('a', -10., 10.), ('x', -10., 10.), ('y', -10., 10.)],
                1e-14,
            );
            check_one(
                "min_of_sub_2",
                &[('x', -10., 10.), ('y', -10., 10.), ('c', -10., 10.)],
                1e-14,
            );
            check_one(
                "min_of_add_1",
                &[('a', -10., 10.), ('x', -10., 10.), ('b', -10., 10.)],
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
            assert!(
                unchecked.is_empty(),
                "\nThe following templates have not been tested:\n{}\n",
                unchecked
            );
        }
    }
}
