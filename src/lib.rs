mod compile;
mod dedup;
mod derivative;
mod dominator;
mod dual;
mod error;
mod eval;
mod fold;
mod hash;
mod interval;
mod io;
mod latex;
mod macros;
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
    simd_array::{JitSimdFn, NativeSimdFunc, SimdVec, Wfloat},
    single::{JitFn, JitFnSync},
};

#[cfg(test)]
mod test;

pub use dedup::{Deduplicater, equivalent_trees};
pub use derivative::{numerical_deriv, symbolic_deriv};
pub use dual::{Dual, DualEvaluator};
pub use error::Error;
pub use eval::ValueEvaluator;
pub use interval::{
    Interval, IntervalEvaluator,
    pruning_eval::{PruningError, PruningState, ValuePruningEvaluator},
};
pub use prune::Pruner;
pub use reduce::reduce;
pub use tree::{
    BinaryOp, Node, TernaryOp, Tree, UnaryOp, Value, abs, add, and, cos, div, equals, exp, floor,
    geq, greater, leq, less, log, max, min, mul, negate, neq, not, or, pow, rem, reshape, sin,
    sqrt, sub, tan,
};
pub use walk::{DepthIterator, DepthWalker};
