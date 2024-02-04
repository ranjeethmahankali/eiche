use crate::tree::{BinaryOp, BinaryOp::*, Node, Node::*, Tree, UnaryOp::*};

impl Tree {
    pub fn to_latex(&self) -> String {
        let roots = self.roots();
        let (rows, cols) = self.dims();
        if rows == 1 && cols == 1 {
            return to_latex(&roots[0], self.nodes());
        } else {
            let mut lx = "\\begin{bmatrix}".to_string();
            for row in 0..rows {
                for col in 0..cols {
                    lx.push('{');
                    lx.push_str(&to_latex(&roots[col * rows + row], self.nodes()));
                    lx.push('}');
                    if col < cols - 1 {
                        lx.push_str(" & ");
                    }
                }
                if row < rows - 1 {
                    lx.push_str(" \\\\ ");
                }
            }
            lx.push_str("\\end{bmatrix}");
            return lx;
        }
    }
}

fn to_latex(node: &Node, nodes: &[Node]) -> String {
    match node {
        Scalar(val) => val.to_string(),
        Symbol(label) => label.to_string(),
        Unary(op, i) => {
            let inode = &nodes[*i];
            let ix = to_latex(inode, nodes);
            match op {
                Negate => format!("-{{{}}}", {
                    match inode {
                        // Special cases that require braces.
                        Binary(Add, ..) | Binary(Subtract, ..) => with_parens(ix),
                        Scalar(_) | Symbol(_) | Unary(..) | Binary(..) => ix,
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
                        Scalar(_) | Symbol(_) | Unary(..) | Binary(Min, ..) | Binary(Max, ..) => {
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
                Min => format!("\\min\\left({{{}}}, {{{}}}\\right)", lx, rx),
                Max => format!("\\max\\left({{{}}}, {{{}}}\\right)", lx, rx),
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
                    Scalar(_) if lx.len() > 1 => with_parens(lx),
                    Scalar(_) | Symbol(_) | Unary(_, _) => lx,
                }
            },
            {
                match rnode {
                    Binary(Add, ..) | Binary(Subtract, ..) => with_parens(rx),
                    Scalar(_) | Symbol(_) | Unary(_, _) | Binary(_, _, _) => rx,
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
        Binary(..) | Unary(..) | Symbol(_) | Scalar(_) => latex,
    }
}

fn parens_add_sub(node: &Node, latex: String) -> String {
    match node {
        Binary(Add, ..) | Binary(Subtract, ..) | Unary(Negate, _) => with_parens(latex),
        Binary(..) | Scalar(_) | Symbol(_) | Unary(..) => latex,
    }
}

fn with_parens(latex: String) -> String {
    format!("\\left({}\\right)", latex)
}

#[cfg(test)]
mod test {
    use crate::deftree;

    #[test]
    fn t_negate() {
        // Symbol
        assert_eq!("-{x}", deftree!(-x).unwrap().to_latex());
        // Unary
        assert_eq!("-{\\sqrt{2}}", deftree!(- (sqrt 2.)).unwrap().to_latex());
        assert_eq!(
            "-{\\left|{{x} + {y}}\\right|}",
            deftree!(- (abs (+ x y))).unwrap().to_latex()
        );
        assert_eq!(
            "-{\\sin\\left({{2}.{x}}\\right)}",
            deftree!(- (sin (* 2 x))).unwrap().to_latex()
        );
        assert_eq!(
            "-{\\cos\\left({{x}^{2}}\\right)}",
            deftree!(- (cos (pow x 2))).unwrap().to_latex()
        );
        assert_eq!(
            "-{\\tan\\left({{x}^{2}}\\right)}",
            deftree!(- (tan (pow x 2))).unwrap().to_latex()
        );
        assert_eq!(
            "-{\\ln\\left({{x}^{2}}\\right)}",
            deftree!(- (log (pow x 2))).unwrap().to_latex()
        );
        assert_eq!(
            "-{e^{\\left({x}^{2}\\right)}}",
            deftree!(- (exp (pow x 2))).unwrap().to_latex()
        );
        // Binary
        assert_eq!(
            "-{\\left({x} + {y}\\right)}",
            deftree!(- (+ x y)).unwrap().to_latex()
        );
        assert_eq!(
            "-{\\left({x} - {y}\\right)}",
            deftree!(- (- x y)).unwrap().to_latex()
        );
        assert_eq!("-{{x}.{y}}", deftree!(- (* x y)).unwrap().to_latex());
        assert_eq!("-{\\dfrac{x}{y}}", deftree!(- (/ x y)).unwrap().to_latex());
        assert_eq!("-{{x}^{y}}", deftree!(- (pow x y)).unwrap().to_latex());
        assert_eq!(
            "-{\\min\\left({x}, {y}\\right)}",
            deftree!(- (min x y)).unwrap().to_latex()
        );
        assert_eq!(
            "-{\\max\\left({x}, {y}\\right)}",
            deftree!(- (max x y)).unwrap().to_latex()
        );
    }

    #[test]
    fn t_sqrt() {
        // Symbol
        assert_eq!("\\sqrt{x}", deftree!(sqrt x).unwrap().to_latex());
        // Unary
        assert_eq!("\\sqrt{-{x}}", deftree!(sqrt(-x)).unwrap().to_latex());
        assert_eq!(
            "\\sqrt{\\left|{{x} + {y}}\\right|}",
            deftree!(sqrt (abs (+ x y))).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{\\sin\\left({{2}.{x}}\\right)}",
            deftree!(sqrt (sin (* 2 x))).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{\\cos\\left({{x}^{2}}\\right)}",
            deftree!(sqrt (cos (pow x 2))).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{\\tan\\left({{x}^{2}}\\right)}",
            deftree!(sqrt (tan (pow x 2))).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{\\ln\\left({{x}^{2}}\\right)}",
            deftree!(sqrt(log (pow x 2))).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{e^{\\left({x}^{2}\\right)}}",
            deftree!(sqrt(exp (pow x 2))).unwrap().to_latex()
        );
        // Binary
        assert_eq!(
            "\\sqrt{{x} + {y}}",
            deftree!(sqrt (+ x y)).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{{x} - {y}}",
            deftree!(sqrt (- x y)).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{{x}.{y}}",
            deftree!(sqrt (* x y)).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{\\dfrac{x}{y}}",
            deftree!(sqrt (/ x y)).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{{x}^{y}}",
            deftree!(sqrt (pow x y)).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{\\min\\left({x}, {y}\\right)}",
            deftree!(sqrt (min x y)).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{\\max\\left({x}, {y}\\right)}",
            deftree!(sqrt (max x y)).unwrap().to_latex()
        );
    }

    #[test]
    fn t_abs() {
        assert_eq!(
            "\\left|{\\dfrac{x}{y}}\\right|",
            deftree!(abs (/ x y)).unwrap().to_latex()
        );
    }

    #[test]
    fn t_mat2x2() {
        assert_eq!(
            "\\begin{bmatrix}{a} & {c} \\\\ {b} & {d}\\end{bmatrix}",
            deftree!(concat a b c d)
                .unwrap()
                .reshape(2, 2)
                .unwrap()
                .to_latex()
        );
        assert_eq!(
            "\\begin{bmatrix}{{x} + {y}} & {{x} - {y}} \\\\ {{x}.{y}} & {\\dfrac{x}{y}}\\end{bmatrix}",
            deftree!(concat (+ x y) (* x y) (- x y) (/ x y)).unwrap()
                .reshape(2, 2)
                .unwrap()
                .to_latex()
        );
    }

    #[test]
    fn t_column_vector() {
        assert_eq!(
            "\\begin{bmatrix}{a} \\\\ {b} \\\\ {c} \\\\ {d}\\end{bmatrix}",
            deftree!(concat a b c d).unwrap().to_latex()
        );
        assert_eq!(
            "\\begin{bmatrix}{{x} + {y}} \\\\ {{x}.{y}} \\\\ {{x} - {y}} \\\\ {\\dfrac{x}{y}}\\end{bmatrix}",
            deftree!(concat (+ x y) (* x y) (- x y) (/ x y)).unwrap().to_latex()
        );
    }

    #[test]
    fn t_row_vector() {
        assert_eq!(
            "\\begin{bmatrix}{a} & {b} & {c} & {d}\\end{bmatrix}",
            deftree!(concat a b c d)
                .unwrap()
                .reshape(1, 4)
                .unwrap()
                .to_latex()
        );
        assert_eq!(
            "\\begin{bmatrix}{{x} + {y}} & {{x}.{y}} & {{x} - {y}} & {\\dfrac{x}{y}}\\end{bmatrix}",
            deftree!(concat (+ x y) (* x y) (- x y) (/ x y))
                .unwrap()
                .reshape(1, 4)
                .unwrap()
                .to_latex()
        );
    }
}
