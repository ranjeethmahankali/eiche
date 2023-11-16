#[derive(Debug)]
pub struct UnionFind {
    reps: Vec<usize>,
}

impl UnionFind {
    pub fn new() -> UnionFind {
        UnionFind { reps: Vec::new() }
    }

    pub fn reset(&mut self, size: usize) {
        self.reps.clear();
        self.reps.extend(0..size);
    }

    pub fn unite(&mut self, mut a: usize, mut b: usize) {
        if a > b {
            // Swap for deterministic union.
            (a, b) = (b, a);
        }
        while a != self.reps[a] {
            a = self.reps[a];
        }
        self.reps[b] = a;
    }

    pub fn find(&self, id: usize) -> usize {
        self.reps[id]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn t_pair() {
        let mut ufind = UnionFind::new();
        ufind.reset(2);
        assert_ne!(ufind.find(0), ufind.find(1));
        ufind.unite(0, 1);
        assert_eq!(ufind.find(0), ufind.find(1));
        assert_eq!(ufind.find(0), 0);
        assert_eq!(ufind.find(1), 0);
    }

    #[test]
    fn t_halves() {
        let mut ufind = UnionFind::new();
        const NUM: usize = 16;
        const HALF: usize = NUM / 2;
        ufind.reset(NUM);
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
        ufind.reset(NUM);
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
}
