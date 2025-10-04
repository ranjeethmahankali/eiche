use super::{JitContext, NumberType};
use crate::{Error, Tree};
use std::ffi::c_void;

type UnsafePruningFuncType = unsafe extern "C" fn(
    *const c_void, // Inputs
    *mut c_void,   // Outputs
    *const i64,    // Jump targets
    *const i8,     // Traffic signals.
);

struct JumpTable {
    targets: Box<[usize]>,
    offsets: Box<[usize]>,
    prunable: Box<[bool]>,
}

impl JumpTable {
    fn from_counts(counts: &[usize], num_roots: usize, count_threshold: usize) -> Self {
        let num_nodes = counts.len();
        let jmp_pairs = {
            // Pairs of integers representing the bounds of nodes that can be skipped by a jump.
            let mut pairs: Vec<_> = counts
                .iter()
                .take(num_nodes - num_roots) // Ignore the roots.
                .enumerate()
                .filter_map(|(i, c)| {
                    if *c >= count_threshold {
                        Some((i - c, i))
                    } else {
                        None
                    }
                })
                .collect();
            pairs.sort();
            pairs
        };
        // Now we compute the possible jump targets from each node. Because
        // there can be many targets from each node, and we'd like to avoid
        // nested vectors, we store them in a flat vector with offsets for each
        // node.
        let mut targets = Vec::with_capacity(num_nodes);
        let mut offsets = Vec::with_capacity(num_nodes);
        let mut iter = jmp_pairs.iter().peekable();
        for ni in 0..num_nodes {
            offsets.push(targets.len());
            while let Some((_, target)) = iter.next_if(|(i, _)| *i == ni) {
                targets.push(*target);
            }
        }
        // If something is a jump target, that means it's dependencies can be jumped over,
        // which in turn means that node is prunable.
        let prunable = {
            let mut prunable = vec![false; num_nodes];
            for t in targets.iter() {
                prunable[*t] = true;
            }
            prunable.into_boxed_slice()
        };
        Self {
            targets: targets.into_boxed_slice(),
            offsets: offsets.into_boxed_slice(),
            prunable,
        }
    }

    fn get_targets(&self, node: usize) -> &[usize] {
        let start = self.offsets[node];
        let stop = self
            .offsets
            .get(node + 1)
            .cloned()
            .unwrap_or(self.targets.len());
        &self.targets[start..stop]
    }

    fn num_nodes(&self) -> usize {
        self.offsets.len()
    }

    fn is_prunable(&self, node: usize) -> bool {
        self.prunable[node]
    }
}

#[derive(Debug, PartialEq, Eq)]
enum BranchType {
    None,
    Unconditional,
    Indirect(Box<[usize]>),
}

struct BlockLayout {
    block_indices: Box<[usize]>,
    branches: Box<[BranchType]>,
    jump_target_indices: Box<[usize]>,
}

impl BlockLayout {
    fn from_table(table: &JumpTable) -> Result<Self, Error> {
        #[derive(Clone, Debug)]
        enum BranchData<'a> {
            None,
            Unconditional,
            Indirect(&'a [usize]),
        }
        let branches: Box<[BranchData<'_>]> = (0..table.num_nodes())
            .scan(false, |need_branch, ni| {
                let targets = table.get_targets(ni);
                Some(
                    match (
                        targets.is_empty(),
                        std::mem::replace(need_branch, table.is_prunable(ni)),
                    ) {
                        (false, _) => BranchData::Indirect(targets),
                        (_, true) => BranchData::Unconditional,
                        _ => BranchData::None,
                    },
                )
            })
            .collect();
        // Find the indices of each block.
        let block_indices: Box<[usize]> = branches
            .iter()
            .scan(0usize, |index, branch| {
                *index += match branch {
                    BranchData::None => 0,
                    BranchData::Unconditional | BranchData::Indirect(_) => 1,
                };
                Some(*index)
            })
            .collect();
        debug_assert!(
            branches.iter().all(|branch| match branch {
                BranchData::None | BranchData::Unconditional => true,
                BranchData::Indirect(targets) => targets.iter().all(|ti| match branches[*ti + 1] {
                    BranchData::None => false,
                    BranchData::Unconditional | BranchData::Indirect(_) => true,
                }),
            }),
            "A new block must begin right after each target of an indirect branch. Invalid block layout. This should never happen."
        );
        let jump_target_indices = branches
            .iter()
            .scan(0usize, |index, branch| {
                let prev = *index;
                *index += match branch {
                    BranchData::None | BranchData::Unconditional => 0,
                    BranchData::Indirect(_) => 1,
                };
                Some(prev)
            })
            .collect();
        let branches = branches
            .iter()
            .map(|branch| match branch {
                BranchData::None => BranchType::None,
                BranchData::Unconditional => BranchType::Unconditional,
                BranchData::Indirect(targets) => {
                    BranchType::Indirect(targets.iter().map(|ti| block_indices[*ti + 1]).collect())
                }
            })
            .collect();
        Ok(Self {
            block_indices,
            branches,
            jump_target_indices,
        })
    }

    fn num_blocks(&self) -> usize {
        1 + self.block_indices.last().cloned().unwrap_or(0usize)
    }
}

impl Tree {
    pub fn jit_compile_with_pruning<'ctx, T>(
        &'ctx self,
        _context: &'ctx JitContext,
        prune_threshold: usize,
    ) -> Result<(), Error>
    where
        T: NumberType,
    {
        let (tree, counts) = self.control_dependence_sorted()?;
        let num_roots = tree.num_roots();
        let jtable = JumpTable::from_counts(&counts, num_roots, prune_threshold);
        let signal_offsets: Vec<_> = (0..tree.len())
            .map(|i| jtable.is_prunable(i))
            .scan(0usize, |scan, current| {
                let prev = *scan;
                *scan += if current { 1 } else { 0 };
                Some(prev)
            })
            .collect();
        let num_signals = signal_offsets.last().cloned().ok_or(Error::EmptyTree)?;
        debug_assert_eq!(
            (0..tree.len()).filter(|i| jtable.is_prunable(*i)).count(),
            num_signals,
            "The number of prunable nodes should be equal to the number of signals computed via scan."
        );
        todo!("Not Implemented");
    }
}

#[cfg(test)]
mod test {
    use super::{BlockLayout, JumpTable};
    use crate::{deftree, llvm_jit::pruning_single::BranchType};

    #[test]
    fn t_block_layout_small_tree() {
        let (tree, counts) = deftree!(max (+ (+ 'x 2.) (+ 'y 2.)) (+ 'x 'y))
            .unwrap()
            .compacted()
            .unwrap()
            .control_dependence_sorted()
            .unwrap();
        assert_eq!(&counts, &[0usize, 0, 0, 0, 0, 3, 0, 7]);
        let num_roots = tree.num_roots();
        let jtable = JumpTable::from_counts(&counts, num_roots, 1);
        let layout = BlockLayout::from_table(&jtable).unwrap();
        assert_eq!(
            layout.branches.as_ref(),
            &[
                BranchType::None,
                BranchType::None,
                BranchType::Indirect([2].into()),
                BranchType::None,
                BranchType::None,
                BranchType::None,
                BranchType::Unconditional,
                BranchType::None
            ]
        );
        assert_eq!(layout.num_blocks(), 3);
    }

    #[test]
    fn t_block_layout_medium_tree() {
        let (tree, counts) = deftree!(max
                            (+ (pow 'x 2.) (pow 'y 2.))
                            (+ (pow (- 'x 2.5) 2.) (pow (- 'y 2.5) 2.)))
        .unwrap()
        .compacted()
        .unwrap()
        .control_dependence_sorted()
        .unwrap();
        assert_eq!(&counts, &[0usize, 0, 0, 0, 0, 1, 0, 1, 5, 0, 0, 2, 12]);
        let num_roots = tree.num_roots();
        let jtable = JumpTable::from_counts(&counts, num_roots, 1);
        let layout = BlockLayout::from_table(&jtable).unwrap();
        assert_eq!(layout.num_blocks(), 7);
        assert_eq!(
            layout.branches.as_ref(),
            &[
                BranchType::None,
                BranchType::None,
                BranchType::None,
                BranchType::Indirect([5].into()),
                BranchType::Indirect([3].into()),
                BranchType::None,
                BranchType::Indirect([4].into()),
                BranchType::None,
                BranchType::Unconditional,
                BranchType::Indirect([6].into()),
                BranchType::None,
                BranchType::None,
                BranchType::Unconditional
            ]
        );
    }
}
