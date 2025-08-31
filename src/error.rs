use crate::interval::pruning_eval::PruningError;

#[derive(Debug, Clone)]
pub enum Error {
    /// Nodes are not in a valid topological order.
    WrongNodeOrder,
    /// Nodes have cyclic dependencies.
    CyclicGraph,
    /// A constant node contains NaN.
    ContainsNaN,
    /// Tree conains no nodes.
    EmptyTree,
    /// Root nodes depend on each other. They must be isolated from each other
    /// in a valid tree.
    DependentRootNodes,
    /// The roots of the tree are invalid. They must be at the end of the tree.
    InvalidRoots,
    /// A mismatch between two dimensions, for example, during a reshape operation.
    DimensionMismatch((usize, usize), (usize, usize)),
    InvalidDimensions,
    /// The datatype encountered is not what was expected. For example, if a
    /// computation was expecting a scalar input and finds a boolean value.
    TypeMismatch,
    /// Something went wrong when trying to do interval airthmetic.
    InvalidInterval,
    /// Index out of bounds,
    IndexOutOfBounds(usize, usize),

    // Evaluation related errors
    /// A symbol was not assigned a value before evaluating.
    VariableNotFound(char),

    // Mutations and templates.
    InvalidTemplateCapture,
    UnboundTemplateSymbol,

    // Derivatives.
    CannotComputeSymbolicDerivative,
    CannotComputeNumericDerivative,

    // Jit
    InputSizeMismatch(usize, usize),
    OutputSizeMismatch(usize, usize),
    CannotCreateJitModule,
    CannotCompileIntrinsic(&'static str),
    JitCompilationError(String),

    // Pruning.
    Pruning(PruningError),
}
