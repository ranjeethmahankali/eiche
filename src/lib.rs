mod analyze;
mod compile;
mod dedup;
mod derivative;
mod dual;
mod error;
mod eval;
mod fold;
mod hash;
mod interval;
mod io;
mod latex;
mod macros;
mod matrix_ops;
mod mutate;
mod prune;
mod reduce;
mod substitute;
mod template;
mod tree;
mod walk;

#[cfg(feature = "llvm-jit")]
pub mod llvm_jit;

#[cfg(feature = "llvm-jit")]
pub use llvm_jit::{
    JitContext,
    interval::{JitIntervalFn, JitIntervalFnSync},
    pruning::{
        JitPruner, JitPrunerSync, JitPruningFn, JitPruningFnSync, JitPruningIntervalFn,
        JitPruningIntervalFnSync, JitPruningSimdFn, JitPruningSimdFnSync,
    },
    simd_array::{JitSimdFn, NativeSimdFunc, SimdVec, Wide},
    single::{JitFn, JitFnSync},
};

pub mod test_util;

pub use analyze::DependencyTable;
pub use dedup::{Deduplicater, equivalent_trees};
pub use derivative::{numerical_deriv, symbolic_deriv};
pub use dual::{Dual, DualEvaluator};
pub use error::Error;
pub use eval::ValueEvaluator;
pub use fold::fold;
pub use interval::{
    Interval, IntervalEvaluator,
    pruning_eval::{PruningError, PruningState, ValuePruningEvaluator},
};
pub use prune::Pruner;
pub use reduce::reduce;
pub use tree::{
    BinaryOp, Node, TernaryOp, Tree, UnaryOp, Value, abs, add, and, cos, div, dot, equals, exp,
    extract, floor, geq, greater, is_node_scalar, l2norm, leq, less, log, matmul, max, min, mul,
    negate, neq, normalize, not, or, pow, rem, reshape, sin, sqrt, sub, tan, transpose,
};
pub use walk::{DepthIterator, DepthWalker, NodeOrdering};
