#[cfg(test)]
mod tests {
    use crate::tree::*;

    #[test]
    fn depth_traverse() {
        let mut traverse = TraverseDepth::new();
        {
            let tree: Tree = pow('x'.into(), 2.0.into()) + pow('y'.into(), 2.0.into());
            // Make sure two successive traversal yield the same nodes.
            let a: Vec<_> = traverse
                .iter(&tree, true, false)
                .map(|(index, parent)| (index, parent))
                .collect();
            let b: Vec<_> = traverse
                .iter(&tree, true, false)
                .map(|(index, parent)| (index, parent))
                .collect();
            assert_eq!(a, b);
        }
        {
            // Make sure the same TraverseDepth can be used on multiple trees.
            let tree: Tree = pow('x'.into(), 2.0.into()) + pow('y'.into(), 2.0.into());
            let a: Vec<_> = traverse
                .iter(&tree, true, false)
                .map(|(index, parent)| (index, parent))
                .collect();
            let tree2 = tree.clone();
            let b: Vec<_> = traverse
                .iter(&tree2, true, false)
                .map(|(index, parent)| (index, parent))
                .collect();
            assert_eq!(a, b);
        }
        {
            // Check the mirrored traversal.
            let tree = {
                let x: Tree = 'x'.into();
                let y: Tree = 'y'.into();
                x + y
            };
            let normal: Vec<_> = traverse
                .iter(&tree, true, false)
                .map(|(i, p)| (i, p))
                .collect();
            let mirror: Vec<_> = traverse.iter(&tree, true, true).map(|elem| elem).collect();
            assert_eq!(normal.len(), 3);
            assert_eq!(mirror.len(), 3);
            assert_ne!(normal, mirror);
            let mirror = vec![mirror[0], mirror[2], mirror[1]];
            assert_eq!(normal, mirror);
        }
    }
}
