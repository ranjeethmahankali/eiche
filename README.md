# Symba

  Pet project to get acquainted with Rust, and to mess around with
  symbolic expressions, hence the name 'Symba'.

  Example:
```rust
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
```

  The above program will produce the following latex output (hopefully
  rendered by GitHub):

$\dfrac{{{k}.{x}} + {{k}.{y}}}{{x} + {y}}$

$= \dfrac{{k}.{\left({x} + {y}\right)}}{{x} + {y}}$

$= {k}.{\dfrac{{x} + {y}}{{x} + {y}}}$

$= k$
