use asg::deftree;

fn main() {
    let tree = deftree!(log (+ 1 (exp x)));
    println!("{}", tree);
}
