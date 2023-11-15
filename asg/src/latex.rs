use crate::tree::{BinaryOp::*, Node, Node::*, Tree, UnaryOp::*};

fn to_latex(node: &Node, nodes: &Vec<Node>) -> String {
    match node {
        Constant(val) => val.to_string(),
        Symbol(label) => label.to_string(),
        Unary(op, i) => {
            let ix = to_latex(&nodes[*i], nodes);
            match op {
                Negate => format!("-{{{}}}", ix),
                Sqrt => format!("\\sqrt{{{}}}", ix),
                Abs => format!("|{{{}}}|", ix),
                Sin => format!("\\sin({{{}}})", ix),
                Cos => format!("\\cos({{{}}})", ix),
                Tan => format!("\\tan({{{}}})", ix),
                Log => format!("\\ln({{{}}})", ix),
                Exp => format!("e^{{{}}}", ix),
            }
        }
        Binary(op, lhs, rhs) => {
            let lx = to_latex(&nodes[*lhs], nodes);
            let rx = to_latex(&nodes[*rhs], nodes);
            match op {
                Add => format!("({{{}}} + {{{}}})", lx, rx),
                Subtract => format!("({{{}}} - {{{}}})", lx, rx),
                Multiply => format!("{{{}}}.{{{}}}", lx, rx),
                Divide => format!("\\dfrac{{{}}}{{{}}}", lx, rx),
                Pow => format!("{{{}}}^{{{}}}", lx, rx),
                Min => format!("\\min({{{}}}, {{{}}})", lx, rx),
                Max => format!("\\max({{{}}}, {{{}}})", lx, rx),
            }
        }
    }
}

impl Tree {
    pub fn to_latex(&self) -> String {
        to_latex(self.root(), self.nodes())
    }
}

#[cfg(test)]
mod test {
    use crate::deftree;

    #[test]
    fn t_negate() {
        // Symbol
        assert_eq!("-{x}", deftree!(-x).to_latex());
        // Unary
        assert_eq!("-{\\sqrt{2}}", deftree!(- (sqrt 2.)).to_latex());
        assert_eq!("-{|{({x} + {y})}|}", deftree!(- (abs (+ x y))).to_latex());
        assert_eq!("-{\\sin({{2}.{x}})}", deftree!(- (sin (* 2 x))).to_latex());
        assert_eq!(
            "-{\\cos({{x}^{2}})}",
            deftree!(- (cos (pow x 2))).to_latex()
        );
        assert_eq!(
            "-{\\tan({{x}^{2}})}",
            deftree!(- (tan (pow x 2))).to_latex()
        );
        assert_eq!("-{\\ln({{x}^{2}})}", deftree!(- (log (pow x 2))).to_latex());
        assert_eq!("-{e^{{x}^{2}}}", deftree!(- (exp (pow x 2))).to_latex());
        // Binary
        assert_eq!("-{({x} + {y})}", deftree!(- (+ x y)).to_latex());
        assert_eq!("-{({x} - {y})}", deftree!(- (- x y)).to_latex());
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
            "\\sqrt{|{({x} + {y})}|}",
            deftree!(sqrt (abs (+ x y))).to_latex()
        );
        assert_eq!(
            "\\sqrt{\\sin({{2}.{x}})}",
            deftree!(sqrt (sin (* 2 x))).to_latex()
        );
        assert_eq!(
            "\\sqrt{\\cos({{x}^{2}})}",
            deftree!(sqrt (cos (pow x 2))).to_latex()
        );
        assert_eq!(
            "\\sqrt{\\tan({{x}^{2}})}",
            deftree!(sqrt (tan (pow x 2))).to_latex()
        );
        assert_eq!(
            "\\sqrt{\\ln({{x}^{2}})}",
            deftree!(sqrt(log (pow x 2))).to_latex()
        );
        assert_eq!(
            "\\sqrt{e^{{x}^{2}}}",
            deftree!(sqrt(exp (pow x 2))).to_latex()
        );
        // Binary
        assert_eq!("\\sqrt{({x} + {y})}", deftree!(sqrt (+ x y)).to_latex());
        assert_eq!("\\sqrt{({x} - {y})}", deftree!(sqrt (- x y)).to_latex());
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
}
