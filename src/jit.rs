use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::{ExecutionEngine, JitFunction},
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

pub struct JitEvaluator<'ctx, const N_INPUTS: usize, const N_OUTPUTS: usize> {
    func: JitFunction<'ctx, unsafe extern "C" fn([f64; N_INPUTS]) -> [f64; N_OUTPUTS]>,
}

impl Tree {
    pub fn jit_compile<'ctx, const N_INPUTS: usize, const N_OUTPUTS: usize>(
        &'ctx self,
        context: &'ctx JitContext,
    ) -> Result<JitEvaluator<N_INPUTS, N_OUTPUTS>, Error> {
        const FUNC_NAME: &str = "symba_func";
        let num_roots = self.num_roots();
        if num_roots != N_OUTPUTS {
            return Err(Error::OutputSizeMismatch(num_roots, N_OUTPUTS));
        }
        let symbols = self.symbols();
        if symbols.len() != N_INPUTS {
            return Err(Error::InputSizeMismatch(symbols.len(), N_INPUTS));
        }
        let context = &context.inner;
        let compiler = JitCompiler::new(&context)?;
        let builder = &compiler.builder;
        let f64_type = context.f64_type();
        let return_type = f64_type.array_type(num_roots as u32);
        let bool_type = context.bool_type();
        let arg_type = f64_type.array_type(symbols.len() as u32);
        let fn_type = return_type.fn_type(&[arg_type.into()], false);
        let function = compiler.module.add_function(FUNC_NAME, fn_type, None);
        let basic_block = context.append_basic_block(function, "entry");
        builder.position_at_end(basic_block);
        let mut regs: Vec<BasicValueEnum> = Vec::with_capacity(self.len());
        for (ni, node) in self.nodes().iter().enumerate() {
            let reg = match node {
                Constant(val) => match val {
                    Bool(val) => BasicValueEnum::IntValue(
                        bool_type.const_int(if *val { 1 } else { 0 }, false),
                    ),
                    Scalar(val) => BasicValueEnum::FloatValue(f64_type.const_float(*val)),
                },
                Symbol(label) => {
                    let args = function
                        .get_first_param()
                        .ok_or(Error::CannotReadInput(*label))?
                        .into_array_value();
                    builder
                        .build_extract_value(
                            args,
                            symbols
                                .iter()
                                .position(|c| c == label)
                                .ok_or(Error::CannotReadInput(*label))?
                                as u32,
                            &format!("reg_{}", label),
                        )
                        .map_err(|_| Error::CannotReadInput(*label))?
                }
                Unary(op, input) => match op {
                    Negate => BasicValueEnum::FloatValue(
                        builder
                            .build_float_neg(
                                regs[*input].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(format!("{e:?}")))?,
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
                                .map_err(|e| Error::JitCompilationError(format!("{e:?}")))?,
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
                            .map_err(|e| Error::JitCompilationError(format!("{e:?}")))?,
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
                            .map_err(|e| Error::JitCompilationError(format!("{e:?}")))?,
                    ),
                    Subtract => BasicValueEnum::FloatValue(
                        builder
                            .build_float_sub(
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(format!("{e:?}")))?,
                    ),
                    Multiply => BasicValueEnum::FloatValue(
                        builder
                            .build_float_mul(
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(format!("{e:?}")))?,
                    ),
                    Divide => BasicValueEnum::FloatValue(
                        builder
                            .build_float_div(
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(format!("{e:?}")))?,
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
                            .map_err(|e| Error::JitCompilationError(format!("{e:?}")))?,
                    ),
                    LessOrEqual => BasicValueEnum::IntValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::ULE,
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(format!("{e:?}")))?,
                    ),
                    Equal => BasicValueEnum::IntValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::UEQ,
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(format!("{e:?}")))?,
                    ),
                    NotEqual => BasicValueEnum::IntValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::UNE,
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(format!("{e:?}")))?,
                    ),
                    Greater => BasicValueEnum::IntValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::UGT,
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(format!("{e:?}")))?,
                    ),
                    GreaterOrEqual => BasicValueEnum::IntValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::UGE,
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(format!("{e:?}")))?,
                    ),
                    And => BasicValueEnum::IntValue(
                        builder
                            .build_and(
                                regs[*lhs].into_int_value(),
                                regs[*rhs].into_int_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(format!("{e:?}")))?,
                    ),
                    Or => BasicValueEnum::IntValue(
                        builder
                            .build_or(
                                regs[*lhs].into_int_value(),
                                regs[*rhs].into_int_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(format!("{e:?}")))?,
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
                        .map_err(|e| Error::JitCompilationError(format!("{e:?}")))?,
                },
            };
            regs.push(reg);
        }
        builder
            .build_aggregate_return(&regs[(self.len() - num_roots)..])
            .map_err(|e| Error::JitCompilationError(format!("{e:?}")))?;
        return unsafe {
            Ok(JitEvaluator {
                func: compiler
                    .engine
                    .get_function(FUNC_NAME)
                    .map_err(|e| Error::JitCompilationError(format!("{e:?}")))?,
            })
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

impl JitContext {
    pub fn new() -> JitContext {
        JitContext {
            inner: Context::create(),
        }
    }
}

struct JitCompiler<'ctx> {
    engine: ExecutionEngine<'ctx>,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
}

impl<'ctx> JitCompiler<'ctx> {
    pub fn new(context: &'ctx Context) -> Result<JitCompiler<'ctx>, Error> {
        let module = context.create_module("symba_module");
        let engine = module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .map_err(|_| Error::CannotCreateJitModule)?;
        Ok(JitCompiler {
            engine,
            module,
            builder: context.create_builder(),
        })
    }
}

impl<'ctx, const N_INPUTS: usize, const N_OUTPUTS: usize> JitEvaluator<'ctx, N_INPUTS, N_OUTPUTS> {
    pub fn run(&self, inputs: [f64; N_INPUTS]) -> [f64; N_OUTPUTS] {
        unsafe { self.func.call(inputs) }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{deftree, test::util::check_tree_eval};

    fn check_jit_eval<const N_INPUTS: usize, const N_OUTPUTS: usize>(
        tree: &Tree,
        vardata: &[(char, f64, f64)],
        samples_per_var: usize,
        eps: f64,
    ) {
        let context = JitContext::new();
        let jiteval = tree.jit_compile::<N_INPUTS, N_OUTPUTS>(&context).unwrap();
        check_tree_eval(
            tree.clone(),
            |inputs: &[f64], outputs: &mut [f64]| {
                assert_eq!(inputs.len(), N_INPUTS);
                assert_eq!(outputs.len(), N_OUTPUTS);
                let inputs: [f64; N_INPUTS] = inputs.try_into().unwrap();
                let results = jiteval.run(inputs);
                outputs.copy_from_slice(&results);
            },
            vardata,
            samples_per_var,
            eps,
        );
    }

    #[test]
    fn t_prod_sum() {
        let tree = deftree!(concat (+ x y) (* x y)).unwrap();
        check_jit_eval::<2, 2>(&tree, &[('x', -10., 10.), ('y', -10., 10.)], 100, 0.);
    }

    #[test]
    fn t_sub_div() {
        let tree = deftree!(concat (- x y) (/ x y)).unwrap();
        check_jit_eval::<2, 2>(&tree, &[('x', -10., 10.), ('y', -10., 10.)], 20, 0.);
    }

    #[test]
    fn t_pow() {
        let tree = deftree!(pow x 2).unwrap();
        check_jit_eval::<1, 1>(&tree, &[('x', -10., -10.)], 100, 0.);
    }

    #[test]
    fn t_sqrt() {
        let tree = deftree!(sqrt x).unwrap();
        check_jit_eval::<1, 1>(&tree, &[('x', 0.01, 10.)], 100, 0.);
    }
}
