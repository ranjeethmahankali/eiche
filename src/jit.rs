use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::{ExecutionEngine, JitFunction, UnsafeFunctionPointer},
    intrinsics::Intrinsic,
    module::Module,
    types::{BasicTypeEnum, FloatType},
    values::{BasicMetadataValueEnum, BasicValueEnum},
    FloatPredicate, OptimizationLevel,
};

use crate::{
    error::Error,
    tree::{BinaryOp::*, Node::*, TernaryOp::*, Tree, UnaryOp::*, Value::*},
};

impl Tree {
    pub fn jit_compile<'ctx, F: UnsafeFunctionPointer>(
        &'ctx self,
        context: &'ctx JitContext,
    ) -> Result<JitFunction<F>, Error> {
        assert!(self.dims() == (1, 1));
        let context = &context.inner;
        let compiler = JitCompiler::new(&context)?;
        let builder = &compiler.builder;
        let f64_type = context.f64_type();
        let bool_type = context.bool_type();
        let num_symbols = self.symbols().len();
        let fn_type = f64_type.fn_type(&vec![f64_type.into(); num_symbols], false);
        let function = compiler.module.add_function("symba-tree-fn", fn_type, None);
        let basic_block = context.append_basic_block(function, "entry");
        builder.position_at_end(basic_block);
        let mut regs: Vec<BasicValueEnum> = Vec::with_capacity(self.len());
        let mut symbol_regs: Vec<(char, BasicValueEnum)> = Vec::new();
        for (ni, node) in self.nodes().iter().enumerate() {
            let reg = match node {
                Constant(val) => match val {
                    Bool(val) => BasicValueEnum::IntValue(
                        bool_type.const_int(if *val { 1 } else { 0 }, false),
                    ),
                    Scalar(val) => BasicValueEnum::FloatValue(f64_type.const_float(*val)),
                },
                Symbol(label) => match symbol_regs.iter().find(|(l, _r)| l == label) {
                    Some((_l, r)) => *r,
                    None => {
                        let r = BasicValueEnum::FloatValue(
                            function
                                .get_nth_param(symbol_regs.len() as u32)
                                .ok_or(Error::CannotAllocateFunctionArgument)?
                                .into_float_value(),
                        );
                        symbol_regs.push((*label, r));
                        r
                    }
                },
                Unary(op, input) => match op {
                    Negate => BasicValueEnum::FloatValue(
                        builder
                            .build_float_neg(
                                regs[*input].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|_| Error::JitCompilationError)?,
                    ),
                    Sqrt => build_float_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.sqrt.*",
                        "sqrt_call",
                        regs[*input],
                        f64_type,
                    )?,
                    Abs => build_float_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.fabs.*",
                        "abs_call",
                        regs[*input],
                        f64_type,
                    )?,
                    Sin => build_float_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.sin.*",
                        "sin_call",
                        regs[*input],
                        f64_type,
                    )?,
                    Cos => build_float_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.cos.*",
                        "cos_call",
                        regs[*input],
                        f64_type,
                    )?,
                    Tan => {
                        let sin = build_float_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.sin.*",
                            "sin_call",
                            regs[*input],
                            f64_type,
                        )?;
                        let cos = build_float_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.cos.*",
                            "cos_call",
                            regs[*input],
                            f64_type,
                        )?;
                        BasicValueEnum::FloatValue(
                            builder
                                .build_float_div(
                                    sin.into_float_value(),
                                    cos.into_float_value(),
                                    &format!("reg_{}", ni),
                                )
                                .map_err(|_| Error::JitCompilationError)?,
                        )
                    }
                    Log => build_float_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.log.*",
                        "log_call",
                        regs[*input],
                        f64_type,
                    )?,
                    Exp => build_float_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.exp.*",
                        "exp_call",
                        regs[*input],
                        f64_type,
                    )?,
                    Not => BasicValueEnum::IntValue(
                        builder
                            .build_not(regs[*input].into_int_value(), &format!("reg_{}", ni))
                            .map_err(|_| Error::JitCompilationError)?,
                    ),
                },
                Binary(op, lhs, rhs) => match op {
                    Add => BasicValueEnum::FloatValue(
                        builder
                            .build_float_add(
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|_| Error::JitCompilationError)?,
                    ),
                    Subtract => BasicValueEnum::FloatValue(
                        builder
                            .build_float_sub(
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|_| Error::JitCompilationError)?,
                    ),
                    Multiply => BasicValueEnum::FloatValue(
                        builder
                            .build_float_mul(
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|_| Error::JitCompilationError)?,
                    ),
                    Divide => BasicValueEnum::FloatValue(
                        builder
                            .build_float_div(
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|_| Error::JitCompilationError)?,
                    ),
                    Pow => build_float_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.pow.*",
                        "pow_call",
                        regs[*lhs],
                        regs[*rhs],
                        f64_type,
                    )?,
                    Min => build_float_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.minnum.*",
                        "min_call",
                        regs[*lhs],
                        regs[*rhs],
                        f64_type,
                    )?,
                    Max => build_float_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.maxnum.*",
                        "max_call",
                        regs[*lhs],
                        regs[*rhs],
                        f64_type,
                    )?,
                    Less => BasicValueEnum::IntValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::ULT,
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|_| Error::JitCompilationError)?,
                    ),
                    LessOrEqual => BasicValueEnum::IntValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::ULE,
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|_| Error::JitCompilationError)?,
                    ),
                    Equal => BasicValueEnum::IntValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::UEQ,
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|_| Error::JitCompilationError)?,
                    ),
                    NotEqual => BasicValueEnum::IntValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::UNE,
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|_| Error::JitCompilationError)?,
                    ),
                    Greater => BasicValueEnum::IntValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::UGT,
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|_| Error::JitCompilationError)?,
                    ),
                    GreaterOrEqual => BasicValueEnum::IntValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::UGE,
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|_| Error::JitCompilationError)?,
                    ),
                    And => BasicValueEnum::IntValue(
                        builder
                            .build_and(
                                regs[*lhs].into_int_value(),
                                regs[*rhs].into_int_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|_| Error::JitCompilationError)?,
                    ),
                    Or => BasicValueEnum::IntValue(
                        builder
                            .build_or(
                                regs[*lhs].into_int_value(),
                                regs[*rhs].into_int_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|_| Error::JitCompilationError)?,
                    ),
                },
                Ternary(op, a, b, c) => match op {
                    Choose => builder
                        .build_select(
                            regs[*a].into_int_value(),
                            regs[*b].into_float_value(),
                            regs[*c].into_float_value(),
                            &format!("reg_{}", ni),
                        )
                        .map_err(|_| Error::JitCompilationError)?,
                },
            };
            regs.push(reg);
        }
        builder
            .build_return(Some(regs.last().ok_or(Error::JitCompilationError)?))
            .map_err(|_| Error::JitCompilationError)?;
        return unsafe {
            compiler
                .engine
                .get_function("symba-tree-fn")
                .map_err(|_| Error::JitCompilationError)
        };
    }
}

pub fn build_float_unary_intrinsic<'ctx>(
    builder: &'ctx Builder,
    module: &'ctx Module,
    name: &'static str,
    call_name: &'static str,
    input: BasicValueEnum<'ctx>,
    f64_type: FloatType<'ctx>,
) -> Result<BasicValueEnum<'ctx>, Error> {
    let intrinsic = Intrinsic::find(name).ok_or(Error::CannotCompileIntrinsic(name))?;
    let intrinsic_fn = intrinsic
        .get_declaration(module, &[BasicTypeEnum::FloatType(f64_type)])
        .ok_or(Error::CannotCompileIntrinsic(name))?;
    builder
        .build_call(
            intrinsic_fn,
            &[BasicMetadataValueEnum::FloatValue(input.into_float_value())],
            call_name,
        )
        .map_err(|_| Error::CannotCompileIntrinsic(name))?
        .try_as_basic_value()
        .left()
        .ok_or(Error::CannotCompileIntrinsic(name))
}

pub fn build_float_binary_intrinsic<'ctx>(
    builder: &'ctx Builder,
    module: &'ctx Module,
    name: &'static str,
    call_name: &'static str,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    f64_type: FloatType<'ctx>,
) -> Result<BasicValueEnum<'ctx>, Error> {
    let intrinsic = Intrinsic::find(name).ok_or(Error::CannotCompileIntrinsic(name))?;
    let intrinsic_fn = intrinsic
        .get_declaration(
            module,
            &[
                BasicTypeEnum::FloatType(f64_type),
                BasicTypeEnum::FloatType(f64_type),
            ],
        )
        .ok_or(Error::CannotCompileIntrinsic(name))?;
    builder
        .build_call(
            intrinsic_fn,
            &[
                BasicMetadataValueEnum::FloatValue(lhs.into_float_value()),
                BasicMetadataValueEnum::FloatValue(rhs.into_float_value()),
            ],
            call_name,
        )
        .map_err(|_| Error::CannotCompileIntrinsic(name))?
        .try_as_basic_value()
        .left()
        .ok_or(Error::CannotCompileIntrinsic(name))
}

pub struct JitContext {
    inner: Context,
}

struct JitCompiler<'ctx> {
    engine: ExecutionEngine<'ctx>,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
}

impl<'ctx> JitCompiler<'ctx> {
    pub fn new(context: &'ctx Context) -> Result<JitCompiler<'ctx>, Error> {
        let module = context.create_module("symba-tree");
        let engine = module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .map_err(|_| Error::CannotCreateJitModule)?;
        let module = context.create_module("symba-tree");
        Ok(JitCompiler {
            engine,
            module,
            builder: context.create_builder(),
        })
    }
}
