use std::fmt::Debug;

#[derive(Clone, PartialEq, Eq)]
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

    // Serialization.
    IOError(String),

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
}

impl Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Error::*;
        match self {
            WrongNodeOrder => write!(f, "WrongNodeOrder"),
            CyclicGraph => write!(f, "CyclicGraph"),
            ContainsNaN => write!(f, "ContainsNaN"),
            EmptyTree => write!(f, "EmptyTree"),
            DependentRootNodes => write!(f, "DependentRootNodes"),
            InvalidRoots => write!(f, "InvalidRoots"),
            DimensionMismatch(a, b) => f
                .debug_tuple("DimensionMismatch")
                .field(a)
                .field(b)
                .finish(),
            InvalidDimensions => write!(f, "InvalidDimensions"),
            TypeMismatch => write!(f, "TypeMismatch"),
            InvalidInterval => write!(f, "InvalidInterval"),
            IndexOutOfBounds(a, b) => f.debug_tuple("IndexOutOfBounds").field(a).field(b).finish(),
            IOError(msg) => f.debug_tuple("IOError").field(msg).finish(),
            VariableNotFound(label) => f.debug_tuple("VariableNotFound").field(label).finish(),
            InvalidTemplateCapture => write!(f, "InvalidTemplateCapture"),
            UnboundTemplateSymbol => write!(f, "UnboundTemplateSymbol"),
            CannotComputeSymbolicDerivative => write!(f, "CannotComputeSymbolicDerivative"),
            CannotComputeNumericDerivative => write!(f, "CannotComputeNumericDerivative"),
            InputSizeMismatch(actual, expected) => f
                .debug_tuple("InputSizeMismatch")
                .field(actual)
                .field(expected)
                .finish(),
            OutputSizeMismatch(actual, expected) => f
                .debug_tuple("OutputSizeMismatch")
                .field(actual)
                .field(expected)
                .finish(),
            CannotCreateJitModule => write!(f, "CannotCreateJitModule"),
            CannotCompileIntrinsic(name) => {
                f.debug_tuple("CannotCompileIntrinsic").field(name).finish()
            }
            JitCompilationError(message) => {
                write!(f, "JitCompilationError: \n{message}")
            }
        }
    }
}
