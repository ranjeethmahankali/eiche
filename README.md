# Symba

  Pet project to get acquainted with Rust, and to mess around with
  symbolic expressions, hence the name 'Symba'.

  Example:
```rust
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
            let deriv = tree.symbolic_derivative("xy").unwrap();
            let steps = reduce(deriv, 8).unwrap();
            steps.last().unwrap().clone()
        };
        println!(
            "Derivative of f(x, y) with respect to x and y is:\n${}$\n",
            deriv.to_latex()
    );
}
```

  The above program will produce the following latex output (hopefully
  rendered by GitHub):

$\dfrac{{{k}.{x}} + {{k}.{y}}}{{x} + {y}}$

$= \dfrac{{k}.{\left({x} + {y}\right)}}{{x} + {y}}$

$= {k}.{\dfrac{{x} + {y}}{{x} + {y}}}$

$= k$

$f(x, y) = {{x}^{2}} + {{y}^{2}}$

Derivative of f(x, y) with respect to x and y is:
$\begin{bmatrix}{{2}.{x}} & {\dfrac{{{y}^{2}}.{2}}{y}}\end{bmatrix}$
