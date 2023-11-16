use crate::tree::{BinaryOp, BinaryOp::*, Node, Node::*, Tree, UnaryOp::*};

impl Tree {
    pub fn to_latex(&self) -> String {
        to_latex(self.root(), self.nodes())
    }
}

fn to_latex(node: &Node, nodes: &Vec<Node>) -> String {
    match node {
        Constant(val) => val.to_string(),
        Symbol(label) => label.to_string(),
        Unary(op, i) => {
            let inode = &nodes[*i];
            let ix = to_latex(inode, nodes);
            match op {
                Negate => format!("-{{{}}}", {
                    match inode {
                        // Special cases that require braces.
                        Binary(Add, ..) | Binary(Subtract, ..) => with_parens(ix),
                        Constant(_) | Symbol(_) | Unary(..) | Binary(..) => ix,
                    }
                }),
                Sqrt => format!("\\sqrt{{{}}}", ix),
                Abs => format!("\\left|{{{}}}\\right|", ix),
                Sin => format!("\\sin\\left({{{}}}\\right)", ix),
                Cos => format!("\\cos\\left({{{}}}\\right)", ix),
                Tan => format!("\\tan\\left({{{}}}\\right)", ix),
                Log => format!("\\ln\\left({{{}}}\\right)", ix),
                Exp => format!("e^{{{}}}", {
                    match inode {
                        Constant(_) | Symbol(_) | Unary(..) | Binary(Min, ..) | Binary(Max, ..) => {
                            ix
                        }
                        Binary(..) => with_parens(ix),
                    }
                }),
            }
        }
        Binary(op, lhs, rhs) => {
            let rnode = &nodes[*rhs];
            let lnode = &nodes[*lhs];
            let (lx, rx) = parens_binary(
                *op,
                lnode,
                rnode,
                to_latex(lnode, nodes),
                to_latex(rnode, nodes),
            );
            match op {
                Add => format!("{{{}}} + {{{}}}", lx, rx),
                Subtract => format!("{{{}}} - {{{}}}", lx, rx),
                Multiply => format!("{{{}}}.{{{}}}", lx, rx),
                Divide => format!("\\dfrac{{{}}}{{{}}}", lx, rx),
                Pow => format!("{{{}}}^{{{}}}", lx, rx),
                Min => format!("\\min({{{}}}, {{{}}})", lx, rx),
                Max => format!("\\max({{{}}}, {{{}}})", lx, rx),
            }
        }
    }
}

fn parens_binary(
    op: BinaryOp,
    lnode: &Node,
    rnode: &Node,
    lx: String,
    rx: String,
) -> (String, String) {
    match op {
        Add | Subtract => (parens_add_sub(lnode, lx), parens_add_sub(rnode, rx)),
        Multiply => (parens_mul(lnode, lx), parens_mul(rnode, rx)),
        Divide => (parens_div(lnode, lx), parens_div(rnode, rx)),
        Pow => (
            {
                match lnode {
                    Unary(Negate, _)
                    | Unary(Sqrt, _)
                    | Unary(Sin, _)
                    | Unary(Cos, _)
                    | Unary(Tan, _)
                    | Unary(Log, _)
                    | Unary(Exp, _)
                    | Binary(..) => with_parens(lx),
                    Constant(_) if lx.len() > 1 => with_parens(lx),
                    Constant(_) | Symbol(_) | Unary(_, _) => lx,
                }
            },
            {
                match rnode {
                    Binary(Add, ..) | Binary(Subtract, ..) => with_parens(rx),
                    Constant(_) | Symbol(_) | Unary(_, _) | Binary(_, _, _) => rx,
                }
            },
        ),
        Min | Max => (lx, rx),
    }
}

fn parens_div(node: &Node, latex: String) -> String {
    match node {
        Binary(Divide, ..) => with_parens(latex),
        _ => latex,
    }
}

fn parens_mul(node: &Node, latex: String) -> String {
    match node {
        Binary(Add, ..) | Binary(Subtract, ..) | Binary(Multiply, ..) | Unary(Negate, ..) => {
            with_parens(latex)
        }
        Binary(..) | Unary(..) | Symbol(_) | Constant(_) => latex,
    }
}

fn parens_add_sub(node: &Node, latex: String) -> String {
    match node {
        Binary(Add, ..) | Binary(Subtract, ..) | Unary(Negate, _) => with_parens(latex),
        Binary(..) | Constant(_) | Symbol(_) | Unary(..) => latex,
    }
}

fn with_parens(latex: String) -> String {
    format!("\\left({}\\right)", latex)
}

#[cfg(test)]
mod test {
    use crate::{deftree, mutate::Mutations};

    #[test]
    fn t_negate() {
        // Symbol
        assert_eq!("-{x}", deftree!(-x).to_latex());
        // Unary
        assert_eq!("-{\\sqrt{2}}", deftree!(- (sqrt 2.)).to_latex());
        assert_eq!(
            "-{\\left|{{x} + {y}}\\right|}",
            deftree!(- (abs (+ x y))).to_latex()
        );
        assert_eq!(
            "-{\\sin\\left({{2}.{x}}\\right)}",
            deftree!(- (sin (* 2 x))).to_latex()
        );
        assert_eq!(
            "-{\\cos\\left({{x}^{2}}\\right)}",
            deftree!(- (cos (pow x 2))).to_latex()
        );
        assert_eq!(
            "-{\\tan\\left({{x}^{2}}\\right)}",
            deftree!(- (tan (pow x 2))).to_latex()
        );
        assert_eq!(
            "-{\\ln\\left({{x}^{2}}\\right)}",
            deftree!(- (log (pow x 2))).to_latex()
        );
        assert_eq!(
            "-{e^{\\left({x}^{2}\\right)}}",
            deftree!(- (exp (pow x 2))).to_latex()
        );
        // Binary
        assert_eq!(
            "-{\\left({x} + {y}\\right)}",
            deftree!(- (+ x y)).to_latex()
        );
        assert_eq!(
            "-{\\left({x} - {y}\\right)}",
            deftree!(- (- x y)).to_latex()
        );
        assert_eq!("-{{x}.{y}}", deftree!(- (* x y)).to_latex());
        assert_eq!("-{\\dfrac{x}{y}}", deftree!(- (/ x y)).to_latex());
        assert_eq!("-{{x}^{y}}", deftree!(- (pow x y)).to_latex());
        assert_eq!("-{\\min({x}, {y})}", deftree!(- (min x y)).to_latex());
        assert_eq!("-{\\max({x}, {y})}", deftree!(- (max x y)).to_latex());
    }

    #[test]
    fn t_sqrt() {
        // Symbol
        assert_eq!("\\sqrt{x}", deftree!(sqrt x).to_latex());
        // Unary
        assert_eq!("\\sqrt{-{x}}", deftree!(sqrt(-x)).to_latex());
        assert_eq!(
            "\\sqrt{\\left|{{x} + {y}}\\right|}",
            deftree!(sqrt (abs (+ x y))).to_latex()
        );
        assert_eq!(
            "\\sqrt{\\sin\\left({{2}.{x}}\\right)}",
            deftree!(sqrt (sin (* 2 x))).to_latex()
        );
        assert_eq!(
            "\\sqrt{\\cos\\left({{x}^{2}}\\right)}",
            deftree!(sqrt (cos (pow x 2))).to_latex()
        );
        assert_eq!(
            "\\sqrt{\\tan\\left({{x}^{2}}\\right)}",
            deftree!(sqrt (tan (pow x 2))).to_latex()
        );
        assert_eq!(
            "\\sqrt{\\ln\\left({{x}^{2}}\\right)}",
            deftree!(sqrt(log (pow x 2))).to_latex()
        );
        assert_eq!(
            "\\sqrt{e^{\\left({x}^{2}\\right)}}",
            deftree!(sqrt(exp (pow x 2))).to_latex()
        );
        // Binary
        assert_eq!("\\sqrt{{x} + {y}}", deftree!(sqrt (+ x y)).to_latex());
        assert_eq!("\\sqrt{{x} - {y}}", deftree!(sqrt (- x y)).to_latex());
        assert_eq!("\\sqrt{{x}.{y}}", deftree!(sqrt (* x y)).to_latex());
        assert_eq!("\\sqrt{\\dfrac{x}{y}}", deftree!(sqrt (/ x y)).to_latex());
        assert_eq!("\\sqrt{{x}^{y}}", deftree!(sqrt (pow x y)).to_latex());
        assert_eq!(
            "\\sqrt{\\min({x}, {y})}",
            deftree!(sqrt (min x y)).to_latex()
        );
        assert_eq!(
            "\\sqrt{\\max({x}, {y})}",
            deftree!(sqrt (max x y)).to_latex()
        );
    }

    #[test]
    fn t_abs() {
        assert_eq!(
            "\\left|{\\dfrac{x}{y}}\\right|",
            deftree!(abs (/ x y)).to_latex()
        );
    }

    #[test]
    fn t_mutations_latex() {
        let tree = deftree!(/ (+ (* p x) (* p y)) (+ x y))
            .deduplicate()
            .unwrap();
        // let tree = deftree!(* p (/ (+ x y) (+ x y))).deduplicate().unwrap();
        println!("${}$\n", tree.to_latex());
        for m in Mutations::from(&tree) {
            match m {
                Ok(mutated) => {
                    assert_ne!(mutated, tree);
                    println!("${}$\n", mutated.to_latex());
                    // println!("{}\n", mutated.to_lisp());
                }
                Err(_) => assert!(false),
            }
        }
    }
}
