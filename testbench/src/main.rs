use asg::{deftree, reduce::reduce};

fn main() {
    let tree = deftree!(sqrt (+ (pow (/ x (sqrt (+ (pow x 2) (pow y 2)))) 2)
                              (pow (/ y (sqrt (+ (pow x 2) (pow y 2)))) 2)));
    let max_iter = 10;
    println!("${}$\n", tree.to_latex());
    let steps = reduce(tree, max_iter).unwrap();
    for step in steps {
        println!("${}$\n", step.to_latex());
    }
}
