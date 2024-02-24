use lazy_static::lazy_static;

use crate::{
    mutate::TemplateCapture,
    tree::{Node::*, Tree},
};

/// This macro is only meant for use within this module.
macro_rules! deftemplate {
    (($($tt:tt)*)) => { // Unwrap parens.
        deftemplate!($($tt)*)
    };
    ($name: ident ping ($($ping:tt) *) pong ($($pong:tt) *)) => {
        Template::from(
            stringify!($name),
            $crate::deftree!(($($ping) *)).unwrap(),
            $crate::deftree!(($($pong) *)).unwrap()
        )
    };
}

#[derive(Clone)]
pub struct Template {
    name: String,
    ping: Tree,
    pong: Tree,
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
            Constant(_) | Unary(..) | Binary(..) | Ternary(..) => None,
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
        Template {
            name: name.to_string(),
            ping,
            pong,
        }
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

    pub fn ping_root(&self) -> usize {
        self.ping.root_indices().start
    }
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
        deftemplate!(rearrange_div_div_1
                     ping (/ (/ a b) c)
                     pong (/ a (* b c))),
        deftemplate!(rearrange_mul_1
                     ping (* x (* y z))
                     pong (* (* x z) y)),
        deftemplate!(rearrange_mul_2
                     ping (* x (* y z))
                     pong (* (* x y) z)),
        deftemplate!(rearrange_add_1
                     ping (+ x (+ y z))
                     pong (+ (+ x z) y)),
        deftemplate!(rearrange_add_2
                     ping (+ x (+ y z))
                     pong (+ (+ x y) z)),
        deftemplate!(rearrange_add_sub_1
                     ping (- (+ x y) z)
                     pong (+ (- x z) y)),
        deftemplate!(rearrange_add_sub_2
                     ping (- (+ x y) z)
                     pong (+ x (- y z))),
        deftemplate!(divide_by_self
                     ping (/ a a)
                     pong (1.0)
        ),
        deftemplate!(pow_divide_by_self
                     ping (/ (pow x a) x)
                     pong (pow x (- a 1))),
        deftemplate!(pow_divide_collapse
                     ping (/ (pow x a) (pow x b))
                     pong (pow x (- a b))),
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
                     ping (+ (/ x z) (/ y z))
                     pong (/ (+ x y) z)
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
                     ping (min (- z x) (- z y))
                     pong (- z (max x y))
        ),
        deftemplate!(min_of_sub_2
                     ping (min (- x z) (- y z))
                     pong (- (min x y) z)
        ),
        deftemplate!(min_of_add_1
                     ping (min (+ x z) (+ y z))
                     pong (+ z (min x y))
        ),

        // ======== Polynomial simplifications ========
        deftemplate!(x_plus_y_squared
                     ping (pow (+ x y) 2.)
                     pong ((+ (+ (pow x 2.) (pow y 2.)) (* 2. (* x y))))),
        deftemplate!(x_minus_y_squared
                     ping (pow (- x y) 2.)
                     pong ((- (+ (pow x 2.) (pow y 2.)) (* 2. (* x y))))),
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
            // I can't see a sensible case for simplifying matrices, that is
            // different from simplifying the elements of the matrix. So the
            // templates are resitricted to have at most 1 root.
            assert_eq!(t.ping.num_roots(), 1);
            assert_eq!(t.pong.num_roots(), 1);
            assert!(names.insert(t.name.as_str()), "Duplicate template found.");
        }
    }

    fn check_one_template(
        name: &'static str,
        vardata: &[(char, f64, f64)],
        eps: f64,
        checked: &mut HashSet<&str>,
    ) {
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
    }

    fn check_many_templates(
        names: &[&'static str],
        vardata: &[(char, f64, f64)],
        eps: f64,
        checked: &mut HashSet<&str>,
    ) {
        for name in names {
            check_one_template(*name, vardata, eps, checked);
        }
    }

    #[test]
    fn t_check_templates() {
        let mut checked: HashSet<&str> = HashSet::with_capacity(TEMPLATES.len());
        // Check each template. This is necessary they need different
        // vardata and ranges. e.g. You can't use negative values in
        // sqrt.
        {
            check_one_template(
                "distribute_mul",
                &[('k', -10., 10.), ('a', -10., 10.), ('b', -10., 10.)],
                1e-12,
                &mut checked,
            );
            check_one_template(
                "min_of_sqrt",
                &[('a', 0., 10.), ('b', 0., 10.)],
                1e-12,
                &mut checked,
            );
            check_one_template(
                "rearrange_frac",
                &[
                    ('a', -10., 10.),
                    ('b', -10., 10.),
                    ('x', -10., 10.),
                    ('y', -10., 10.),
                ],
                1e-10,
                &mut checked,
            );
            check_many_templates(
                &[
                    "add_frac",
                    "rearrange_mul_div_1",
                    "rearrange_mul_div_2",
                    "rearrange_mul_1",
                    "rearrange_mul_2",
                    "rearrange_add_1",
                    "rearrange_add_2",
                    "rearrange_add_sub_1",
                    "rearrange_add_sub_2",
                ],
                &[('x', -10., 10.), ('y', -10., 10.), ('z', -10., 10.)],
                1e-12,
                &mut checked,
            );
            check_one_template(
                "rearrange_div_div_1",
                &[('a', 0.01, 10.), ('b', 0.01, 10.), ('c', 0.01, 10.)],
                1e-12,
                &mut checked,
            );
            check_one_template("divide_by_self", &[('a', -10., 10.)], 1e-12, &mut checked);
            check_one_template(
                "pow_divide_by_self",
                &[('x', 1., 10.), ('a', -10., 10.)],
                1e-8,
                &mut checked,
            );
            check_one_template(
                "pow_divide_collapse",
                &[('x', 1., 5.), ('a', -5., 5.), ('b', -5., 5.)],
                1e-8,
                &mut checked,
            );
            check_one_template(
                "distribute_pow_div",
                &[('a', 1., 10.), ('b', 1., 10.), ('k', 0.1, 5.)],
                1e-10,
                &mut checked,
            );
            check_one_template(
                "distribute_pow_mul",
                &[('a', 1., 5.), ('b', 1., 5.), ('k', 0.5, 3.)],
                1e-10,
                &mut checked,
            );
            check_one_template("square_sqrt", &[('a', 0., 10.)], 1e-12, &mut checked);
            check_one_template("sqrt_square", &[('a', -10., 10.)], 1e-12, &mut checked);
            check_one_template("square_abs", &[('x', -10., 10.)], 0., &mut checked);
            check_many_templates(
                &["mul_exponents", "add_exponents"],
                &[('a', 1., 5.), ('x', 0.5, 3.), ('y', 0.5, 2.)],
                1e-9,
                &mut checked,
            );
        }
        {
            // === Other templates ===
            check_many_templates(
                &["min_expand", "max_expand"],
                &[('a', -10., 10.), ('b', -10., 10.)],
                1e-14,
                &mut checked,
            );
            check_many_templates(
                &["min_of_sub_1", "min_of_sub_2", "min_of_add_1"],
                &[('x', -10., 10.), ('y', -10., 10.), ('z', -10., 10.)],
                0.,
                &mut checked,
            );
        }
        {
            // === polynomial simplifications ===
            check_many_templates(
                &["x_plus_y_squared", "x_minus_y_squared"],
                &[('x', -10., 10.), ('y', -10., 10.)],
                1e-12,
                &mut checked,
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
