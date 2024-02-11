use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::{ExecutionEngine, JitFunction},
    module::Module,
    values::BasicValueEnum,
    OptimizationLevel,
};

use crate::{
    error::Error,
    tree::{BinaryOp::*, Node::*, TernaryOp::*, Tree, UnaryOp::*, Value::*},
};

impl Tree {
    pub fn jit_compile<'ctx, F>(&self, context: &'ctx JitContext) -> Result<JitFunction<F>, Error> {
        assert!(self.dims() == (1, 1));
        let context = &context.inner;
        let compiler = JitCompiler::new(&context)?;
        let builder = &compiler.builder;
        let f64_type = context.f64_type();
        let bool_type = context.bool_type();
        let num_symbols = self.symbols().len();
        let fn_type = f64_type.fn_type(&vec![f64_type.into(); num_symbols], false);
        let function = compiler.module.add_function("symba-tree-fn", fn_type, None);
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
                    Sqrt => todo!(),
                    Abs => todo!(),
                    Sin => todo!(),
                    Cos => todo!(),
                    Tan => todo!(),
                    Log => todo!(),
                    Exp => todo!(),
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
                    Pow => todo!(),
                    Min => todo!(),
                    Max => todo!(),
                    Less => todo!(),
                    LessOrEqual => todo!(),
                    Equal => todo!(),
                    NotEqual => todo!(),
                    Greater => todo!(),
                    GreaterOrEqual => todo!(),
                    And => todo!(),
                    Or => todo!(),
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
        todo!();
    }
}

pub struct JitContext {
    inner: Context,
}

struct JitCompiler<'ctx> {
    _engine: ExecutionEngine<'ctx>,
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
            _engine: engine,
            module,
            builder: context.create_builder(),
        })
    }
}
