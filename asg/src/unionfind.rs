use std::cmp::Ordering;

/*
This implementation of union-find uses path compression, but does not
use rank. When two roots are united, if the root with the lower rank
is made a child of the root with the higher rank, it means the next
time `find` is called on one of the children of the shallower tree,
you'd have to traverse fewer nodes to reach the root. But I am
discounting this advantage at this stage, as this implementation is
only meant for internal use. Will consider it later if it becomes
important.
 */
#[derive(Debug)]
pub struct UnionFind {
    parents: Vec<usize>,
}

impl UnionFind {
    pub fn new() -> UnionFind {
        UnionFind {
            parents: Vec::new(),
        }
    }

    pub fn init(&mut self, size: usize) {
        self.parents.clear();
        self.parents.extend(0..size);
    }

    pub fn unite(&mut self, x: usize, y: usize) {
        let xroot = self.find(x);
        let yroot = self.find(y);
        match xroot.cmp(&yroot) {
            Ordering::Less | Ordering::Equal => self.parents[xroot] = yroot,
            Ordering::Greater => self.parents[yroot] = xroot,
        }
    }

    pub fn find(&mut self, id: usize) -> usize {
        if self.parents[id] != id {
            self.parents[id] = self.find(self.parents[id]);
        }
        return self.parents[id];
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn t_pair() {
        let mut ufind = UnionFind::new();
        ufind.init(2);
        assert_ne!(ufind.find(0), ufind.find(1));
        ufind.unite(0, 1);
        assert_eq!(ufind.find(0), ufind.find(1));
    }

    #[test]
    fn t_halves() {
        let mut ufind = UnionFind::new();
        const NUM: usize = 16;
        const HALF: usize = NUM / 2;
        ufind.init(NUM);
        for i in 0..NUM {
            for j in (i + 1)..NUM {
                assert_ne!(ufind.find(i), ufind.find(j));
            }
        }
        for i in 0..NUM {
            if (i / 2) % 2 == 0 {
                ufind.unite(i, i / HALF);
            } else {
                ufind.unite(i / HALF, i);
            }
        }
        for i in 0..NUM {
            assert_eq!(ufind.find(i), ufind.find(i / HALF));
        }
    }

    #[test]
    fn t_even_odd() {
        let mut ufind = UnionFind::new();
        const NUM: usize = 16;
        ufind.init(NUM);
        for i in 0..NUM {
            for j in (i + 1)..NUM {
                assert_ne!(ufind.find(i), ufind.find(j));
            }
        }
        for i in 2..NUM {
            if (i / 2) % 2 == 0 {
                ufind.unite(i, i - 2);
            } else {
                ufind.unite(i - 2, i);
            }
        }
        for i in 0..NUM {
            assert_eq!(ufind.find(i), ufind.find(i % 2));
        }
    }

    #[test]
    fn t_incremental_unite() {
        let mut ufind = UnionFind::new();
        const NUM: usize = 16;
        ufind.init(NUM);
        // First round of unions.
        let mut step = 1;
        while step <= 8 {
            for i in (0..NUM).step_by(2 * step) {
                ufind.unite(i, i + step);
            }
            step *= 2;
            for i in (0..NUM).step_by(step) {
                for j in i..i + step {
                    assert_eq!(ufind.find(i), ufind.find(j));
                }
            }
        }
    }
}
