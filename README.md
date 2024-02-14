# Eiche

A library for manipulating and analyzing symbolic expression. The name means
'Oak' in German because symbolic expressions in Eiche are represented as trees.

## Usage

### Creating Expressions

The `deftree` macro is useful for defining expressions using the s-expressions
(like LISP). You can create an expression and print it out as LaTex as follows:

```rust
    let tree = deftree!(/ (+ (* k x) (* k y)) (+ x y)).unwrap();
    println!("$${}$$\n", tree.to_latex());
```

$$\dfrac{{{k}.{x}} + {{k}.{y}}}{{x} + {y}}$$

### Simplifying Expressions

You can simplify the above tree expressions using the `reduce` function as
follows:

```rust
    let steps = reduce(tree, 8).unwrap();
    for step in steps {
        println!("$$= {}$$\n", step.to_latex());
    }
```

$$= \dfrac{{k}.{\left({x} + {y}\right)}}{{x} + {y}}$$

$$= {k}.{\dfrac{{x} + {y}}{{x} + {y}}}$$

$$= k$$

Eiche can also simplify more complicated expressions. Below expressions
represents the length of a normalized vector:

$$\sqrt{{{\left(\dfrac{x}{\sqrt{{{x}^{2}} + {{y}^{2}}}}\right)}^{2}} + {{\left(\dfrac{y}{\sqrt{{{x}^{2}} + {{y}^{2}}}}\right)}^{2}}}$$

$$= \sqrt{{{\left(\dfrac{x}{\sqrt{{{x}^{2}} + {{y}^{2}}}}\right)}^{2}} + {\dfrac{{y}^{2}}{{\left(\sqrt{{{x}^{2}} + {{y}^{2}}}\right)}^{2}}}}$$

$$= \sqrt{{\dfrac{{x}^{2}}{{\left(\sqrt{{{x}^{2}} + {{y}^{2}}}\right)}^{2}}} + {\dfrac{{y}^{2}}{{\left(\sqrt{{{x}^{2}} + {{y}^{2}}}\right)}^{2}}}}$$

$$= \sqrt{\dfrac{{{x}^{2}} + {{y}^{2}}}{{\left(\sqrt{{{x}^{2}} + {{y}^{2}}}\right)}^{2}}}$$

$$= \sqrt{\dfrac{{{x}^{2}} + {{y}^{2}}}{{{x}^{2}} + {{y}^{2}}}}$$

$$= 1$$

## Differentiating

Eiche can symbolically differentiate expressions with respect to one or more
independent variables. The example below defines a function in two variables, x
and y:

```rust
    let tree = deftree!(- (+ (pow x 3) (pow y 3)) 5).unwrap();
    println!("$$f(x, y) = {}$$\n", tree.to_latex(
```

$$f(x, y) = {\left({{x}^{3}} + {{y}^{3}}\right)} - {5}$$

You can differentate above function with respect to x and y to produce a a row
vector containing the partial derivatives:

$$\begin{bmatrix}{{{x}^{3}}.{\left({3}.{\dfrac{1}{x}}\right)}} & {{{y}^{3}}.{\left({3}.{\dfrac{1}{y}}\right)}}\end{bmatrix}$$

You can differentiate the result again with respect to x and y to get the
hessian matrix:

```rust
    let hessian = deriv.symbolic_deriv("xy").unwrap();
    println!("$${}$$\n", hessian.to_latex());
```

$$\begin{bmatrix}{{{{x}^{3}}.{\left({3}.{\dfrac{-1}{{x}^{2}}}\right)}} + {{\left({3}.{\dfrac{1}{x}}\right)}.{\left({{x}^{3}}.{\left({3}.{\dfrac{1}{x}}\right)}\right)}}} & {0} \\ {0} & {{{{y}^{3}}.{\left({3}.{\dfrac{-1}{{y}^{2}}}\right)}} + {{\left({3}.{\dfrac{1}{y}}\right)}.{\left({{y}^{3}}.{\left({3}.{\dfrac{1}{y}}\right)}\right)}}}\end{bmatrix}$$

This expressions for the derivative and the hessian are unnecessarily
complicated. This is because of how Eiche applies the chain rule when
differentiating the expressions. Let's use the `reduce` function to
simplify:

```rust
    let steps = reduce(hessian, 256).unwrap();
    // The last step is the most simplified form:
    println!("$${}$$\n", steps.last().unwrap().to_latex());
```

$$\begin{bmatrix}{{x}.{6}} & {0} \\ {0} & {{y}.{6}}\end{bmatrix}$$
