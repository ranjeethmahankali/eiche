#[cfg(test)]

mod tests {
    use crate::graph::node::{Node::*, Tree};

    #[test]
    fn construction() {
        let pi: f64 = 3.14;
        let x: Tree = Constant(pi).into();
        matches!(x.root(), Constant(val) if val > &pi);
    }
}
