#[cfg(test)]
mod tests {
    use crate::tree::*;

    #[test]
    fn test_traverse() {
        let tree: Tree = pow('x'.into(), 2.0.into()) + pow('y'.into(), 2.0.into());
        let mut dt = TraverseDepth::from(&tree, true);
        println!("============================");
        {
            // A mutable reference to dt is borrowed and stored inside iter.
            let iter = dt.iter();
            for node in iter {
                println!("{:?}", node);
            }
            // iter goes out of scope.
        }
        println!("============================");
        let iter2 = dt.iter(); // But Rust won't let me borrow again here.
        for node in iter2 {
            println!("{:?}", node);
        }
        println!("============================");
    }
}
