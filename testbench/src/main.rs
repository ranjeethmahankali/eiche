use asg::{deftree, reduce::reduce};

fn main() {
    // Reduce trees.
    let tree = deftree!(/ (+ (* k x) (* k y)) (+ x y)).unwrap();
    let max_iter = 10;
    println!("${}$\n", tree.to_latex());
    let steps = reduce(tree, max_iter).unwrap();
    for step in steps {
        println!("$= {}$\n", step.to_latex());
    }
    // Compute symbolic derivatives.
    let tree = deftree!(+ (pow x 2) (pow y 2)).unwrap();
    println!("$f(x, y) = {}$\n", tree.to_latex());
    let deriv = {
        let deriv = tree.symbolic_deriv("xy").unwrap();
        let steps = reduce(deriv, 12).unwrap();
        steps.last().unwrap().clone()
    };
    println!(
        "Derivative of f(x, y) with respect to x and y is:\n\n$${}$$\n",
        deriv.to_latex()
    );
}
