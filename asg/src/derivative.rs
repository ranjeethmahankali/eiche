use crate::{
    error::Error,
    fold::fold_nodes,
    prune::Pruner,
    sort::TopoSorter,
    tree::{BinaryOp::*, MaybeTree, Node, Node::*, TernaryOp::*, Tree, UnaryOp::*, Value::*},
};

impl Tree {
    pub fn symbolic_derivative(self, params: &str) -> MaybeTree {
        let num_nodes = self.len();
        let (root_start, root_end) = {
            let root_indices = self.root_indices();
            (root_indices.start, root_indices.end)
        };
        let mut nodes = self.take_nodes();
        let mut derivs = Vec::<Node>::new();
        let mut derivmap = Vec::<Option<usize>>::new();
        let mut rootnodes = Vec::<usize>::new();
        for param in params.chars() {
            compute_symbolic_deriv(
                &nodes[0..num_nodes],
                nodes.len(),
                param,
                &mut derivs,
                &mut derivmap,
            );
            nodes.extend(derivs.drain(..));
            for ri in root_start..root_end {
                rootnodes.push(match derivmap[ri] {
                    Some(deriv) => deriv,
                    None => return Err(Error::CannotComputeSymbolicDerivative),
                });
            }
        }
        // Below operations all perform allocations. I am assuming symbolic
        // derivatives won't be computed in a hot path so it may not be a big
        // deal. But in the future, I might consider putting these resources in
        // an object that the caller can pass in. That would allow the caller to
        // reuse the resources and avoid repeated allocations.
        let mut sorter = TopoSorter::new();
        sorter.run_from_slice(&mut nodes, &mut rootnodes)?;
        let mut pruner = Pruner::new();
        pruner.run_from_slice(&mut nodes, &rootnodes);
        fold_nodes(&mut nodes)?;
        return Tree::create(nodes, (root_end - root_start, params.len()));
    }
}

fn compute_symbolic_deriv(
    nodes: &[Node],
    offset: usize,
    param: char,
    dst: &mut Vec<Node>,
    derivmap: &mut Vec<Option<usize>>,
) {
    dst.clear();
    derivmap.clear();
    derivmap.resize(nodes.len(), None);
    for ni in 0..nodes.len() {
        let deriv = match &nodes[ni] {
            Constant(_val) => Constant(Scalar(0.)),
            Symbol(label) => Constant(Scalar(if *label == param { 1. } else { 0. })),
            Unary(op, input) => {
                let inputderiv = match derivmap[*input] {
                    Some(index) => index,
                    // A unary op whose input is not differentiable is not differentiable.
                    None => continue,
                };
                match op {
                    Negate => Unary(Negate, inputderiv),
                    Sqrt => {
                        let sf = push_node(Unary(Sqrt, *input), dst) + offset;
                        let c2 = push_node(Constant(Scalar(2.)), dst) + offset;
                        let sf2 = push_node(Binary(Multiply, sf, c2), dst) + offset;
                        Binary(Divide, inputderiv, sf2)
                    }
                    Abs => {
                        // Technically the gradient should not be defined at zero
                        // exactly. But this might be a pragmatic
                        // compromise. Reconsider this decision later if it becomes
                        // a problem.
                        let zero = push_node(Constant(Scalar(0.)), dst) + offset;
                        let cond = push_node(Binary(Less, *input, zero), dst) + offset;
                        let one = push_node(Constant(Scalar(1.)), dst) + offset;
                        let neg = push_node(Constant(Scalar(-1.)), dst) + offset;
                        let df = push_node(Ternary(Choose, cond, neg, one), dst) + offset;
                        Binary(Multiply, df, inputderiv) // Chain rule.
                    }
                    Sin => Unary(Cos, *input),
                    Cos => {
                        let sin = push_node(Unary(Sin, *input), dst) + offset;
                        let negsin = push_node(Unary(Negate, sin), dst) + offset;
                        Binary(Multiply, negsin, inputderiv) // Chain rule.
                    }
                    Tan => {
                        let cos = push_node(Unary(Cos, *input), dst) + offset;
                        let one = push_node(Constant(Scalar(1.)), dst) + offset;
                        let sec = push_node(Binary(Divide, one, cos), dst) + offset;
                        let two = push_node(Constant(Scalar(2.)), dst) + offset;
                        let sec2 = push_node(Binary(Pow, sec, two), dst) + offset;
                        Binary(Multiply, sec2, inputderiv) // chain rule.
                    }
                    Log => Binary(Divide, inputderiv, *input),
                    Exp => Binary(Multiply, ni, inputderiv),
                    Not => continue,
                }
            }
            Binary(op, lhs, rhs) => {
                // Both inputs need to be differentiable, otherwise this node is not differentiable.
                let lderiv = match derivmap[*lhs] {
                    Some(val) => val,
                    None => continue,
                };
                let rderiv = match derivmap[*rhs] {
                    Some(val) => val,
                    None => continue,
                };
                match op {
                    Add => Binary(Add, lderiv, rderiv),
                    Subtract => Binary(Subtract, lderiv, rderiv),
                    Multiply => {
                        let lr = push_node(Binary(Multiply, *lhs, rderiv), dst) + offset;
                        let rl = push_node(Binary(Multiply, *rhs, lderiv), dst) + offset;
                        Binary(Add, lr, rl)
                    }
                    Divide => {
                        let lr = push_node(Binary(Multiply, lderiv, *rhs), dst) + offset;
                        let rl = push_node(Binary(Multiply, rderiv, *lhs), dst) + offset;
                        let sub = push_node(Binary(Subtract, lr, rl), dst) + offset;
                        let two = push_node(Constant(Scalar(2.)), dst) + offset;
                        let r2 = push_node(Binary(Pow, *rhs, two), dst) + offset;
                        Binary(Divide, sub, r2)
                    }
                    Pow => {
                        // https://www.physicsforums.com/threads/derivative-of-f-x-to-the-power-of-g-x-and-algebra-problem.273333/
                        let logf = push_node(Unary(Log, *lhs), dst) + offset;
                        let gderiv_logf = push_node(Binary(Multiply, rderiv, logf), dst) + offset;
                        let fderiv_over_f = push_node(Binary(Divide, lderiv, *lhs), dst) + offset;
                        let second_term =
                            push_node(Binary(Multiply, *rhs, fderiv_over_f), dst) + offset;
                        let sum = push_node(Binary(Add, gderiv_logf, second_term), dst) + offset;
                        Binary(Multiply, ni, sum)
                    }
                    Min => {
                        let cond = push_node(Binary(Less, *lhs, *rhs), dst) + offset;
                        Ternary(Choose, cond, lderiv, rderiv)
                    }
                    Max => {
                        let cond = push_node(Binary(Greater, *lhs, *rhs), dst) + offset;
                        Ternary(Choose, cond, lderiv, rderiv)
                    }
                    Less | LessOrEqual | Equal | NotEqual | Greater | GreaterOrEqual | And | Or => {
                        continue
                    }
                }
            }
            Ternary(op, a, b, c) => match op {
                Choose => {
                    let bderiv = match derivmap[*b] {
                        Some(val) => val,
                        None => continue,
                    };
                    let cderiv = match derivmap[*c] {
                        Some(val) => val,
                        None => continue,
                    };
                    Ternary(Choose, *a, bderiv, cderiv)
                }
            },
        };
        derivmap[ni] = Some(offset + dst.len());
        dst.push(deriv);
    }
}

fn push_node(node: Node, dst: &mut Vec<Node>) -> usize {
    let idx = dst.len();
    dst.push(node);
    return idx;
}

#[cfg(test)]
mod test {
    use crate::{deftree, test::util::compare_trees};

    #[test]
    fn t_polynomial() {
        assert_eq!(
            deftree!(x).unwrap().symbolic_derivative("x").unwrap(),
            deftree!(1).unwrap()
        );
        // Our symbolic derivative implementation is super general. For that
        // reason, when we differentiate something like x^2, instead of getting
        // back 2x, we'll get something that is mathematically equivalent to it,
        // but more complex. So instead of comparing trees directly, we compare
        // them numerically.
        compare_trees(
            &deftree!(pow x 2).unwrap().symbolic_derivative("x").unwrap(),
            &deftree!(* 2 x).unwrap(),
            &[('x', -10., 10.)],
            100,
            1e-14,
        );
        compare_trees(
            &deftree!(pow x 3).unwrap().symbolic_derivative("x").unwrap(),
            &deftree!(* 3 (pow x 2)).unwrap(),
            &[('x', -10., 10.)],
            100,
            1e-13,
        );
        compare_trees(
            &deftree!(+ (* 1.5 (pow x 2)) (+ (* 2.3 x) 3.46))
                .unwrap()
                .symbolic_derivative("x")
                .unwrap(),
            &deftree!(+ (* 3 x) 2.3).unwrap(),
            &[('x', -10., 10.)],
            100,
            1e-14,
        );
        compare_trees(
            &deftree!(+ (* 1.2 (pow x 3)) (+ (* 2.3 (pow x 2)) (+ (* 3.4 x) 4.5)))
                .unwrap()
                .symbolic_derivative("x")
                .unwrap(),
            &deftree!(+ (* 3.6 (pow x 2)) (+ (* 4.6 x) 3.4)).unwrap(),
            &[('x', -10., 10.)],
            100,
            1e-12,
        );
    }

    #[test]
    fn t_gradient_2d() {
        compare_trees(
            &deftree!(- (+ (pow x 2) (pow y 2)) 5)
                .unwrap()
                .symbolic_derivative("xy")
                .unwrap(),
            &deftree!(concat (* 2 x) (* 2 y)).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.)],
            20,
            1e-14,
        );
    }

    #[test]
    fn t_hessian_2d() {
        compare_trees(
            &deftree!(- (+ (pow x 3) (pow y 3)) 5)
                .unwrap()
                .symbolic_derivative("xy")
                .unwrap()
                .symbolic_derivative("xy")
                .unwrap(),
            &deftree!(concat (* 6 x) 0 0 (* 6 y)).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.)],
            20,
            1e-13,
        );
    }

    #[test]
    fn t_trigonometry() {
        compare_trees(
            &deftree!(pow (sin x) 2)
                .unwrap()
                .symbolic_derivative("x")
                .unwrap(),
            &deftree!(* 2 (* (sin x) (cos x))).unwrap(),
            &[('x', -5., 5.)],
            100,
            1e-15,
        );
        compare_trees(
            &deftree!(pow (cos x) 2)
                .unwrap()
                .symbolic_derivative("x")
                .unwrap(),
            &deftree!(* (- 2) (* (cos x) (sin x))).unwrap(),
            &[('x', -5., 5.)],
            100,
            1e-15,
        );
        compare_trees(
            &deftree!(tan x).unwrap().symbolic_derivative("x").unwrap(),
            &deftree!(pow (/ 1 (cos x)) 2).unwrap(),
            &[('x', -1.5, 1.5)],
            100,
            1e-3,
        );
    }

    #[test]
    fn t_sqrt() {
        compare_trees(
            &deftree!(sqrt x).unwrap().symbolic_derivative("x").unwrap(),
            &deftree!(* 0.5 (pow x (- 0.5))).unwrap(),
            &[('x', 0.01, 10.)],
            100,
            1e-15,
        );
        compare_trees(
            &deftree!(* x (sqrt x))
                .unwrap()
                .symbolic_derivative("x")
                .unwrap(),
            &deftree!(* 1.5 (pow x 0.5)).unwrap(),
            &[('x', 0.01, 10.)],
            100,
            1e-15,
        );
    }

    #[test]
    fn t_abs() {
        compare_trees(
            &deftree!(abs x).unwrap().symbolic_derivative("x").unwrap(),
            &deftree!(if (< x 0) (- 1.) 1.).unwrap(),
            &[('x', -10., 10.)],
            100,
            0.,
        );
    }

    #[test]
    fn t_log() {
        compare_trees(
            &deftree!(log (pow x 2))
                .unwrap()
                .symbolic_derivative("x")
                .unwrap(),
            &deftree!(/ 2 x).unwrap(),
            &[('x', 0.01, 10.)],
            100,
            1e-14,
        );
    }

    #[test]
    fn t_exp() {
        let tree = deftree!(exp x).unwrap();
        compare_trees(
            &tree.clone().symbolic_derivative("x").unwrap(),
            &tree,
            &[('x', -10., 10.)],
            100,
            0.,
        );
        compare_trees(
            &deftree!(exp (pow x 2))
                .unwrap()
                .symbolic_derivative("x")
                .unwrap(),
            &deftree!(* 2 (* x (exp (pow x 2)))).unwrap(),
            &[('x', -4., 4.)],
            100,
            1e-8,
        );
    }

    #[test]
    fn t_min_max() {
        compare_trees(
            &deftree!(min x (pow x 2))
                .unwrap()
                .symbolic_derivative("x")
                .unwrap(),
            &deftree!(if (and (> x 0) (< x 1)) (* 2 x) 1).unwrap(),
            &[('x', -3., 3.)],
            100,
            1e-14,
        );
        compare_trees(
            &deftree!(max x (pow x 2))
                .unwrap()
                .symbolic_derivative("x")
                .unwrap(),
            &deftree!(if (and (> x 0) (< x 1)) 1 (* 2 x)).unwrap(),
            &[('x', -3., 3.)],
            100,
            1e-14,
        );
    }

    #[test]
    fn t_ternary() {
        compare_trees(
            &deftree!(min x (pow x 2))
                .unwrap()
                .symbolic_derivative("x")
                .unwrap()
                .symbolic_derivative("x")
                .unwrap(),
            &deftree!(if (and (> x 0) (< x 1)) 2 0).unwrap(),
            &[('x', -3., 5.)],
            100,
            1e-15,
        );
    }
}