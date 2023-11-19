use asg::{deftree, reduce::reduce};

fn main() {
    let tree = deftree!(/ (+ (* k x) (* k y)) (+ x y));
    let max_iter = 10;
    println!("${}$\n", tree.to_latex());
    let steps = reduce(tree, max_iter).unwrap();
    for step in steps {
        println!("$= {}$\n", step.to_latex());
    }
}
