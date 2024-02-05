#[derive(Debug)]
pub enum Error {
    /// Nodes are not in a valid topological order.
    WrongNodeOrder,
    /// A constant node contains NaN.
    ContainsNaN,
    /// Tree conains no nodes.
    EmptyTree,
    /// A mismatch between two dimensions, for example, during a reshape operation.
    DimensionMismatch((usize, usize), (usize, usize)),
    TypeMismatch,

    // Evaluation related errors
    /// A symbol was not assigned a value before evaluating.
    VariableNotFound(char),
    /// A register with uninitialized value was encountered during
    /// evaluation. This could mean the topology of the tree is
    /// broken.
    UninitializedValueRead,

    // Topological errors
    /// The datatype encountered is not what was expected. For example, if a
    /// computation was expecting a scalar input and finds a boolean value.
    CyclicGraph,
    InvalidTopology,

    // Mutations and templates.
    InvalidTemplateCapture,
    UnboundTemplateSymbol,
}