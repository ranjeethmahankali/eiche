use crate::tree::{BinaryOp, BinaryOp::*, Node, Node::*, TernaryOp::*, Tree, UnaryOp::*};

impl Tree {
    /// Produce the latex expression for the tree.
    pub fn to_latex(&self) -> String {
        // We produce the latex for one root node at a time, and wrap them
        // inside appropriate vector / matrix brackets according to the
        // dimensions of the tree.
        let roots = self.roots();
        let (rows, cols) = self.dims();
        if rows == 1 && cols == 1 {
            to_latex(&roots[0], self.nodes())
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
            lx
        }
    }
}

/// Produce the latex expression for the subtree of a single node in a tree.
fn to_latex(node: &Node, nodes: &[Node]) -> String {
    match node {
        Constant(val) => val.to_string(),
        Symbol(label) => label.to_string(),
        Unary(op, i) => {
            let inode = &nodes[*i];
            let ix = to_latex(inode, nodes);
            match op {
                // Scalar
                Negate => format!("-{{{}}}", {
                    match inode {
                        // Special cases that require braces.
                        Binary(Add, ..) | Binary(Subtract, ..) | Ternary(Choose, ..) => {
                            with_parens(ix)
                        }
                        Constant(_) | Symbol(_) | Unary(..) | Binary(..) => ix,
                    }
                }),
                Sqrt => format!("\\sqrt{{{ix}}}"),
                Abs => format!("\\left|{{{ix}}}\\right|"),
                Sin => format!("\\sin\\left({{{ix}}}\\right)"),
                Cos => format!("\\cos\\left({{{ix}}}\\right)"),
                Tan => format!("\\tan\\left({{{ix}}}\\right)"),
                Log => format!("\\ln\\left({{{ix}}}\\right)"),
                Exp => format!("e^{{{}}}", {
                    match inode {
                        Constant(_)
                        | Symbol(_)
                        | Unary(..)
                        | Binary(Min, ..)
                        | Binary(Max, ..)
                        | Ternary(Choose, ..) => ix,
                        Binary(..) => with_parens(ix),
                    }
                }),
                Floor => format!("\\lfloor{{{ix}}}\\rfloor"),
                // Boolean
                Not => format!("\\text{{not }}{{{}}}", with_parens(ix)),
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
                // Scalar.
                Add => format!("{{{lx}}} + {{{rx}}}"),
                Subtract => format!("{{{lx}}} - {{{rx}}}"),
                Multiply => format!("{{{lx}}}\\cdot{{{rx}}}"),
                Divide => format!("\\dfrac{{{lx}}}{{{rx}}}"),
                Pow => format!("{{{lx}}}^{{{rx}}}"),
                Min => format!("\\min\\left({{{lx}}}, {{{rx}}}\\right)"),
                Max => format!("\\max\\left({{{lx}}}, {{{rx}}}\\right)"),
                Remainder => format!("{{{lx}}} \\bmod {{{rx}}}"),
                // Boolean.
                Less => format!("{{{lx}}} < {{{rx}}}"),
                LessOrEqual => format!("{{{lx}}} \\leq {{{rx}}}"),
                Equal => format!("{{{lx}}} = {{{rx}}}"),
                NotEqual => format!("{{{lx}}} \\neq {{{rx}}}"),
                Greater => format!("{{{lx}}} > {{{rx}}}"),
                GreaterOrEqual => format!("{{{lx}}} \\geq {{{rx}}}"),
                And => format!("{{{lx}}} \\text{{ and }} {{{rx}}}"),
                Or => format!("{{{lx}}} \\text{{ or }} {{{rx}}}"),
            }
        }
        Ternary(op, a, b, c) => {
            let (anode, bnode, cnode) = (&nodes[*a], &nodes[*b], &nodes[*c]);
            let (ax, bx, cx) = (
                to_latex(anode, nodes),
                to_latex(bnode, nodes),
                to_latex(cnode, nodes),
            );
            match op {
                Choose => format!(
                    "\\left\\{{ \\begin{{array}}{{lr}} {{{bx}}}, & \\text{{if }} {{{ax}}}\\\\ {{{cx}}}, \\end{{array}} \\right\\}}"
                ),
            }
        }
    }
}

/// Look at the two operands (`lnode` and `rnode`) of a binary op, and decide if
/// the latex strings of the operands (`lx` and `rx` respectively) should be
/// wrapped in parentheses. Wrap `lx` and `rx` as necessary and return them as a
/// tuple.
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
                    Constant(_) | Symbol(_) | Unary(_, _) | Ternary(Choose, ..) => lx,
                }
            },
            {
                match rnode {
                    Binary(Add, ..) | Binary(Subtract, ..) => with_parens(rx),
                    Constant(_)
                    | Symbol(_)
                    | Unary(_, _)
                    | Binary(_, _, _)
                    | Ternary(Choose, ..) => rx,
                }
            },
        ),
        Remainder => (with_parens(lx), with_parens(rx)),
        Min | Max => (lx, rx),
        Less | LessOrEqual | Equal | NotEqual | Greater | GreaterOrEqual => (lx, rx),
        And | Or => (with_parens(lx), with_parens(rx)),
    }
}

/// Given `node` that is an operand of a division, either a numerator or a
/// denominator, wrap its `latex` string in parentheses if necessary.
fn parens_div(node: &Node, latex: String) -> String {
    match node {
        Binary(Divide, ..) => with_parens(latex),
        _ => latex,
    }
}

/// Given `node` that is an operand of a multiplication and wrap its `latex`
/// string in parentheses if necessary.
fn parens_mul(node: &Node, latex: String) -> String {
    match node {
        Binary(Add, ..) | Binary(Subtract, ..) | Binary(Multiply, ..) | Unary(Negate, ..) => {
            with_parens(latex)
        }
        Binary(..) | Unary(..) | Symbol(_) | Constant(_) | Ternary(Choose, ..) => latex,
    }
}

/// Given a `node` that is an operand of an addition or subtraction, wrap its
/// `latex` string in parentheses if necessary.
fn parens_add_sub(node: &Node, latex: String) -> String {
    match node {
        Binary(Add, ..) | Binary(Subtract, ..) | Unary(Negate, _) => with_parens(latex),
        Binary(..) | Constant(_) | Symbol(_) | Unary(..) | Ternary(Choose, ..) => latex,
    }
}

fn with_parens(latex: String) -> String {
    format!("\\left({latex}\\right)")
}

#[cfg(test)]
mod test {
    use crate::deftree;

    #[test]
    fn t_negate() {
        // Symbol
        assert_eq!("-{x}", deftree!(- 'x).unwrap().to_latex());
        // Unary
        assert_eq!("-{\\sqrt{2}}", deftree!(- (sqrt 2.)).unwrap().to_latex());
        assert_eq!(
            "-{\\left|{{x} + {y}}\\right|}",
            deftree!(- (abs (+ 'x 'y))).unwrap().to_latex()
        );
        assert_eq!(
            "-{\\sin\\left({{2}\\cdot{x}}\\right)}",
            deftree!(- (sin (* 2 'x))).unwrap().to_latex()
        );
        assert_eq!(
            "-{\\cos\\left({{x}^{2}}\\right)}",
            deftree!(- (cos (pow 'x 2))).unwrap().to_latex()
        );
        assert_eq!(
            "-{\\tan\\left({{x}^{2}}\\right)}",
            deftree!(- (tan (pow 'x 2))).unwrap().to_latex()
        );
        assert_eq!(
            "-{\\ln\\left({{x}^{2}}\\right)}",
            deftree!(- (log (pow 'x 2))).unwrap().to_latex()
        );
        assert_eq!(
            "-{e^{\\left({x}^{2}\\right)}}",
            deftree!(- (exp (pow 'x 2))).unwrap().to_latex()
        );
        // Binary
        assert_eq!(
            "-{\\left({x} + {y}\\right)}",
            deftree!(- (+ 'x 'y)).unwrap().to_latex()
        );
        assert_eq!(
            "-{\\left({x} - {y}\\right)}",
            deftree!(- (- 'x 'y)).unwrap().to_latex()
        );
        assert_eq!("-{{x}\\cdot{y}}", deftree!(- (* 'x 'y)).unwrap().to_latex());
        assert_eq!(
            "-{\\dfrac{x}{y}}",
            deftree!(- (/ 'x 'y)).unwrap().to_latex()
        );
        assert_eq!("-{{x}^{y}}", deftree!(- (pow 'x 'y)).unwrap().to_latex());
        assert_eq!(
            "-{\\min\\left({x}, {y}\\right)}",
            deftree!(- (min 'x 'y)).unwrap().to_latex()
        );
        assert_eq!(
            "-{\\max\\left({x}, {y}\\right)}",
            deftree!(- (max 'x 'y)).unwrap().to_latex()
        );
    }

    #[test]
    fn t_sqrt() {
        // Symbol
        assert_eq!("\\sqrt{x}", deftree!(sqrt 'x).unwrap().to_latex());
        // Unary
        assert_eq!("\\sqrt{-{x}}", deftree!(sqrt(- 'x)).unwrap().to_latex());
        assert_eq!(
            "\\sqrt{\\left|{{x} + {y}}\\right|}",
            deftree!(sqrt (abs (+ 'x 'y))).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{\\sin\\left({{2}\\cdot{x}}\\right)}",
            deftree!(sqrt (sin (* 2 'x))).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{\\cos\\left({{x}^{2}}\\right)}",
            deftree!(sqrt (cos (pow 'x 2))).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{\\tan\\left({{x}^{2}}\\right)}",
            deftree!(sqrt (tan (pow 'x 2))).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{\\ln\\left({{x}^{2}}\\right)}",
            deftree!(sqrt(log (pow 'x 2))).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{e^{\\left({x}^{2}\\right)}}",
            deftree!(sqrt(exp (pow 'x 2))).unwrap().to_latex()
        );
        // Binary
        assert_eq!(
            "\\sqrt{{x} + {y}}",
            deftree!(sqrt (+ 'x 'y)).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{{x} - {y}}",
            deftree!(sqrt (- 'x 'y)).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{{x}\\cdot{y}}",
            deftree!(sqrt (* 'x 'y)).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{\\dfrac{x}{y}}",
            deftree!(sqrt (/ 'x 'y)).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{{x}^{y}}",
            deftree!(sqrt (pow 'x 'y)).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{\\min\\left({x}, {y}\\right)}",
            deftree!(sqrt (min 'x 'y)).unwrap().to_latex()
        );
        assert_eq!(
            "\\sqrt{\\max\\left({x}, {y}\\right)}",
            deftree!(sqrt (max 'x 'y)).unwrap().to_latex()
        );
    }

    #[test]
    fn t_abs() {
        assert_eq!(
            "\\left|{\\dfrac{x}{y}}\\right|",
            deftree!(abs (/ 'x 'y)).unwrap().to_latex()
        );
    }

    #[test]
    fn t_floor() {
        assert_eq!(
            "\\lfloor{\\dfrac{x}{y}}\\rfloor",
            deftree!(floor (/ 'x 'y)).unwrap().to_latex()
        );
    }

    #[test]
    fn t_remainder() {
        assert_eq!(
            "{\\left({x} + {y}\\right)} \\bmod {\\left({x} - {y}\\right)}",
            deftree!(rem (+ 'x 'y) (- 'x 'y)).unwrap().to_latex()
        );
    }

    #[test]
    fn t_mat2x2() {
        assert_eq!(
            "\\begin{bmatrix}{a} & {c} \\\\ {b} & {d}\\end{bmatrix}",
            deftree!(concat 'a 'b 'c 'd)
                .unwrap()
                .reshape(2, 2)
                .unwrap()
                .to_latex()
        );
        assert_eq!(
            "\\begin{bmatrix}{{x} + {y}} & {{x} - {y}} \\\\ {{x}\\cdot{y}} & {\\dfrac{x}{y}}\\end{bmatrix}",
            deftree!(concat (+ 'x 'y) (* 'x 'y) (- 'x 'y) (/ 'x 'y))
                .unwrap()
                .reshape(2, 2)
                .unwrap()
                .to_latex()
        );
    }

    #[test]
    fn t_column_vector() {
        assert_eq!(
            "\\begin{bmatrix}{a} \\\\ {b} \\\\ {c} \\\\ {d}\\end{bmatrix}",
            deftree!(concat 'a 'b 'c 'd).unwrap().to_latex()
        );
        assert_eq!(
            "\\begin{bmatrix}{{x} + {y}} \\\\ {{x}\\cdot{y}} \\\\ {{x} - {y}} \\\\ {\\dfrac{x}{y}}\\end{bmatrix}",
            deftree!(concat (+ 'x 'y) (* 'x 'y) (- 'x 'y) (/ 'x 'y))
                .unwrap()
                .to_latex()
        );
    }

    #[test]
    fn t_row_vector() {
        assert_eq!(
            "\\begin{bmatrix}{a} & {b} & {c} & {d}\\end{bmatrix}",
            deftree!(concat 'a 'b 'c 'd)
                .unwrap()
                .reshape(1, 4)
                .unwrap()
                .to_latex()
        );
        assert_eq!(
            "\\begin{bmatrix}{{x} + {y}} & {{x}\\cdot{y}} & {{x} - {y}} & {\\dfrac{x}{y}}\\end{bmatrix}",
            deftree!(concat (+ 'x 'y) (* 'x 'y) (- 'x 'y) (/ 'x 'y))
                .unwrap()
                .reshape(1, 4)
                .unwrap()
                .to_latex()
        );
    }

    #[test]
    fn t_comparison() {
        assert_eq!("{x} < {y}", deftree!(< 'x 'y).unwrap().to_latex());
        assert_eq!("{x} \\leq {y}", deftree!(<= 'x 'y).unwrap().to_latex());
        assert_eq!("{x} > {y}", deftree!(> 'x 'y).unwrap().to_latex());
        assert_eq!("{x} \\geq {y}", deftree!(>= 'x 'y).unwrap().to_latex());
        assert_eq!("{x} = {y}", deftree!(== 'x 'y).unwrap().to_latex());
        assert_eq!("{x} \\neq {y}", deftree!(!= 'x 'y).unwrap().to_latex());
    }

    #[test]
    fn t_boolean() {
        assert_eq!(
            "{\\left({x} < {0}\\right)} \\text{ and } {\\left({y} < {0}\\right)}",
            deftree!(and (< 'x 0) (< 'y 0)).unwrap().to_latex()
        );
        assert_eq!(
            "{\\left({x} < {0}\\right)} \\text{ or } {\\left({y} < {0}\\right)}",
            deftree!(or (< 'x 0) (< 'y 0)).unwrap().to_latex()
        );
        assert_eq!(
            "\\text{not }{\\left({x} < {0}\\right)}",
            deftree!(not (< 'x 0)).unwrap().to_latex()
        );
    }

    #[test]
    fn t_piecewise() {
        assert_eq!(
            "\\left\\{ \\begin{array}{lr} {-{x}}, & \\text{if } {{x} < {0}}\\\\ {x}, \\end{array} \\right\\}",
            deftree!(if (< 'x 0) (- 'x) 'x).unwrap().to_latex()
        );
    }
}
