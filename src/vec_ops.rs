use crate::{
    BinaryOp::*,
    Error,
    Node::{self, *},
    Tree,
    tree::extend_nodes_from_slice,
};

impl Tree {
    pub fn matmul(self, other: Tree) -> Result<Tree, Error> {
        let roots_lt = self.root_indices();
        let roots_rt = other.root_indices();
        let (mut lnodes, ldims) = self.take();
        let (rnodes, rdims) = other.take();
        if ldims.1 != rdims.0 {
            return Err(Error::DimensionMismatch(ldims, rdims));
        }
        if ldims.0 == 0 || ldims.1 == 0 || rdims.0 == 0 || rdims.1 == 0 {
            return Err(Error::InvalidDimensions);
        }
        let offset = extend_nodes_from_slice(&mut lnodes, &rnodes);
        let roots_rt = (roots_rt.start + offset)..(roots_rt.end + offset);
        let (lrows, lcols) = ldims;
        let (rrows, rcols) = rdims;
        let (orows, ocols) = (lrows, rcols);
        let mut newroots: Vec<Node> = Vec::with_capacity(ocols * orows);
        for or in 0..orows {
            for oc in 0..ocols {
                let n_before = lnodes.len();
                let rcol_start = oc * rrows;
                let rcol_idx = rcol_start..(rcol_start + rrows);
                let lrow_idx = (0..lcols).map(|c| or + c * lrows);
                lnodes.extend(
                    lrow_idx
                        .zip(rcol_idx)
                        .map(|(li, ri)| Binary(Multiply, li + roots_lt.start, ri + roots_rt.start)),
                );
                let n_after = lnodes.len();
                let mut total = n_before;
                let n_prods = n_after - n_before;
                for curr in ((n_before + 1)..n_after).take(n_prods.saturating_sub(2)) {
                    let next = lnodes.len();
                    lnodes.push(Binary(Add, total, curr));
                    total = next;
                }
                newroots.push(Binary(Add, total, n_after - 1));
            }
        }
        lnodes.extend(newroots.drain(..));
        Tree::from_nodes(lnodes, (orows, ocols))
    }
}
