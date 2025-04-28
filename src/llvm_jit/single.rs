use super::{JitCompiler, JitContext};
use crate::{
    error::Error,
    tree::{BinaryOp::*, Node::*, TernaryOp::*, Tree, UnaryOp::*, Value::*},
};
use inkwell::{
    AddressSpace, FloatPredicate, OptimizationLevel,
    builder::Builder,
    context::Context,
    execution_engine::JitFunction,
    intrinsics::Intrinsic,
    module::Module,
    types::{BasicTypeEnum, FloatType},
    values::{BasicMetadataValueEnum, BasicValueEnum},
};
use std::{
    ffi::c_void,
    ops::{Add, AddAssign, Div, DivAssign, MulAssign, Neg, Sub, SubAssign},
};

type UnsafeFuncType = unsafe extern "C" fn(*const c_void, *mut c_void);

pub trait NumberType:
    Copy
    + PartialOrd
    + Neg
    + Div
    + DivAssign
    + Add
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + MulAssign
{
    fn nan() -> Self;

    fn jit_type(context: &Context) -> FloatType<'_>;

    fn from_f64(val: f64) -> Self;

    fn min(a: Self, b: Self) -> Self;

    fn max(a: Self, b: Self) -> Self;
}

impl NumberType for f32 {
    fn nan() -> Self {
        f32::NAN
    }

    fn jit_type(context: &Context) -> FloatType<'_> {
        context.f32_type()
    }

    fn from_f64(val: f64) -> Self {
        val as f32
    }

    fn min(a: Self, b: Self) -> Self {
        f32::min(a, b)
    }

    fn max(a: Self, b: Self) -> Self {
        f32::max(a, b)
    }
}

impl NumberType for f64 {
    fn nan() -> Self {
        f64::NAN
    }

    fn jit_type(context: &Context) -> FloatType<'_> {
        context.f64_type()
    }

    fn from_f64(val: f64) -> Self {
        val
    }

    fn min(a: Self, b: Self) -> Self {
        f64::min(a, b)
    }

    fn max(a: Self, b: Self) -> Self {
        f64::max(a, b)
    }
}

/// This represents a JIT compiled tree. This is a wrapper around the JIT compiled native function.
pub struct JitFn<'ctx, T>
where
    T: NumberType,
{
    func: JitFunction<'ctx, UnsafeFuncType>,
    num_inputs: usize,
    outputs: Vec<T>,
}

impl<'ctx, T> JitFn<'ctx, T>
where
    T: NumberType,
{
    pub fn create(
        func: JitFunction<'ctx, UnsafeFuncType>,
        num_inputs: usize,
        num_outputs: usize,
    ) -> JitFn<'ctx, T> {
        // Construct evaluator
        JitFn {
            func,
            num_inputs,
            outputs: vec![T::nan(); num_outputs],
        }
    }
}

impl Tree {
    /// JIT compile a tree and return a native evaluator.
    pub fn jit_compile<'ctx, T>(
        &'ctx self,
        context: &'ctx JitContext,
    ) -> Result<JitFn<'ctx, T>, Error>
    where
        T: NumberType,
    {
        const FUNC_NAME: &str = "eiche_func";
        let num_roots = self.num_roots();
        let symbols = self.symbols();
        let context = &context.inner;
        let compiler = JitCompiler::new(context)?;
        let builder = &compiler.builder;
        let float_type = T::jit_type(context);
        let float_ptr_type = context.ptr_type(AddressSpace::default());
        let bool_type = context.bool_type();
        let fn_type = context
            .void_type()
            .fn_type(&[float_ptr_type.into(), float_ptr_type.into()], false);
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
                    Scalar(val) => BasicValueEnum::FloatValue(float_type.const_float(*val)),
                },
                Symbol(label) => {
                    let inputs = function
                        .get_first_param()
                        .ok_or(Error::JitCompilationError("Cannot read inputs".to_string()))?
                        .into_pointer_value();
                    let ptr = unsafe {
                        builder.build_gep(
                            float_type,
                            inputs,
                            &[context.i64_type().const_int(
                                symbols.iter().position(|c| c == label).ok_or(
                                    Error::JitCompilationError("Cannot find symbol".to_string()),
                                )? as u64,
                                false,
                            )],
                            &format!("arg_{}", *label),
                        )?
                    };
                    builder.build_load(float_type, ptr, &format!("reg_{}", *label))?
                }
                Unary(op, input) => match op {
                    Negate => BasicValueEnum::FloatValue(builder.build_float_neg(
                        regs[*input].into_float_value(),
                        &format!("reg_{}", ni),
                    )?),
                    Sqrt => build_float_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.sqrt.*",
                        "sqrt_call",
                        regs[*input],
                        float_type,
                    )?,
                    Abs => build_float_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.fabs.*",
                        "abs_call",
                        regs[*input],
                        float_type,
                    )?,
                    Sin => build_float_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.sin.*",
                        "sin_call",
                        regs[*input],
                        float_type,
                    )?,
                    Cos => build_float_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.cos.*",
                        "cos_call",
                        regs[*input],
                        float_type,
                    )?,
                    Tan => {
                        let sin = build_float_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.sin.*",
                            "sin_call",
                            regs[*input],
                            float_type,
                        )?;
                        let cos = build_float_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.cos.*",
                            "cos_call",
                            regs[*input],
                            float_type,
                        )?;
                        BasicValueEnum::FloatValue(builder.build_float_div(
                            sin.into_float_value(),
                            cos.into_float_value(),
                            &format!("reg_{}", ni),
                        )?)
                    }
                    Log => build_float_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.log.*",
                        "log_call",
                        regs[*input],
                        float_type,
                    )?,
                    Exp => build_float_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.exp.*",
                        "exp_call",
                        regs[*input],
                        float_type,
                    )?,
                    Floor => build_float_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.floor.*",
                        "floor_call",
                        regs[*input],
                        float_type,
                    )?,
                    Not => BasicValueEnum::IntValue(
                        builder.build_not(regs[*input].into_int_value(), &format!("reg_{}", ni))?,
                    ),
                },
                Binary(op, lhs, rhs) => match op {
                    Add => BasicValueEnum::FloatValue(builder.build_float_add(
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("reg_{}", ni),
                    )?),
                    Subtract => BasicValueEnum::FloatValue(builder.build_float_sub(
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("reg_{}", ni),
                    )?),
                    Multiply => BasicValueEnum::FloatValue(builder.build_float_mul(
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("reg_{}", ni),
                    )?),
                    Divide => BasicValueEnum::FloatValue(builder.build_float_div(
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("reg_{}", ni),
                    )?),
                    Pow => build_float_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.pow.*",
                        "pow_call",
                        regs[*lhs],
                        regs[*rhs],
                        float_type,
                    )?,
                    Min => build_float_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.minnum.*",
                        "min_call",
                        regs[*lhs],
                        regs[*rhs],
                        float_type,
                    )?,
                    Max => build_float_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.maxnum.*",
                        "max_call",
                        regs[*lhs],
                        regs[*rhs],
                        float_type,
                    )?,
                    Remainder => BasicValueEnum::FloatValue(builder.build_float_rem(
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("reg_{}", ni),
                    )?),
                    Less => BasicValueEnum::IntValue(builder.build_float_compare(
                        FloatPredicate::ULT,
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("reg_{}", ni),
                    )?),
                    LessOrEqual => BasicValueEnum::IntValue(builder.build_float_compare(
                        FloatPredicate::ULE,
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("reg_{}", ni),
                    )?),
                    Equal => BasicValueEnum::IntValue(builder.build_float_compare(
                        FloatPredicate::UEQ,
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("reg_{}", ni),
                    )?),
                    NotEqual => BasicValueEnum::IntValue(builder.build_float_compare(
                        FloatPredicate::UNE,
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("reg_{}", ni),
                    )?),
                    Greater => BasicValueEnum::IntValue(builder.build_float_compare(
                        FloatPredicate::UGT,
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("reg_{}", ni),
                    )?),
                    GreaterOrEqual => BasicValueEnum::IntValue(builder.build_float_compare(
                        FloatPredicate::UGE,
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("reg_{}", ni),
                    )?),
                    And => BasicValueEnum::IntValue(builder.build_and(
                        regs[*lhs].into_int_value(),
                        regs[*rhs].into_int_value(),
                        &format!("reg_{}", ni),
                    )?),
                    Or => BasicValueEnum::IntValue(builder.build_or(
                        regs[*lhs].into_int_value(),
                        regs[*rhs].into_int_value(),
                        &format!("reg_{}", ni),
                    )?),
                },
                Ternary(op, a, b, c) => match op {
                    Choose => builder.build_select(
                        regs[*a].into_int_value(),
                        regs[*b].into_float_value(),
                        regs[*c].into_float_value(),
                        &format!("reg_{}", ni),
                    )?,
                },
            };
            regs.push(reg);
        }
        // Copy the outputs.
        let outputs = function
            .get_last_param()
            .ok_or(Error::JitCompilationError(
                "Cannot write to outputs".to_string(),
            ))?
            .into_pointer_value();
        for (i, reg) in regs[(self.len() - num_roots)..].iter().enumerate() {
            let dst = unsafe {
                builder.build_gep(
                    float_type,
                    outputs,
                    &[context.i64_type().const_int(i as u64, false)],
                    &format!("output_{}", i),
                )?
            };
            builder.build_store(dst, *reg)?;
        }
        builder.build_return(None)?;
        compiler.run_passes();
        let engine = compiler
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .map_err(|_| Error::CannotCreateJitModule)?;
        let func = unsafe { engine.get_function(FUNC_NAME)? };
        compiler.write_asm(&std::path::Path::new(
            "/mnt/d/dev/linux/fidgetmark/hex_llvm.obj",
        )); // temporary code.
        compiler.write_llvm_ir(&std::path::Path::new(
            "/mnt/d/dev/linux/fidgetmark/hex_llvm.ir",
        )); // temporary code.
        Ok(JitFn::create(func, symbols.len(), num_roots))
    }
}

fn build_float_unary_intrinsic<'ctx>(
    builder: &'ctx Builder,
    module: &'ctx Module,
    name: &'static str,
    call_name: &'static str,
    input: BasicValueEnum<'ctx>,
    float_type: FloatType<'ctx>,
) -> Result<BasicValueEnum<'ctx>, Error> {
    let intrinsic = Intrinsic::find(name).ok_or(Error::CannotCompileIntrinsic(name))?;
    let intrinsic_fn = intrinsic
        .get_declaration(module, &[BasicTypeEnum::FloatType(float_type)])
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

fn build_float_binary_intrinsic<'ctx>(
    builder: &'ctx Builder,
    module: &'ctx Module,
    name: &'static str,
    call_name: &'static str,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    float_type: FloatType<'ctx>,
) -> Result<BasicValueEnum<'ctx>, Error> {
    let intrinsic = Intrinsic::find(name).ok_or(Error::CannotCompileIntrinsic(name))?;
    let intrinsic_fn = intrinsic
        .get_declaration(
            module,
            &[
                BasicTypeEnum::FloatType(float_type),
                BasicTypeEnum::FloatType(float_type),
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

impl<T> JitFn<'_, T>
where
    T: NumberType,
{
    /// Run the JIT evaluator with the given input values. The number of input
    /// values is expected to be the same as the number of variables in the
    /// tree. The variables are substituted with the input values in the same
    /// order as returned by calling `tree.symbols()` which was compiled to
    /// produce this evaluator.
    pub fn run(&mut self, inputs: &[T]) -> Result<&[T], Error> {
        if inputs.len() != self.num_inputs {
            return Err(Error::InputSizeMismatch(inputs.len(), self.num_inputs));
        }
        Ok(self.run_unchecked(inputs))
    }

    /// Same as `run` except it doesn't check to make sure the `inputs` slice is
    /// of the correct length. This can be used when the caller is sure the
    /// inputs are correct, and this check can be omitted.
    pub fn run_unchecked(&mut self, inputs: &[T]) -> &[T] {
        unsafe {
            self.func
                .call(inputs.as_ptr().cast(), self.outputs.as_mut_ptr().cast())
        };
        &self.outputs
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{deftree, test::check_value_eval};

    fn check_jit_eval(
        tree: &Tree,
        vardata: &[(char, f64, f64)],
        samples_per_var: usize,
        eps64: f64,
        eps32: f32,
    ) {
        let context = JitContext::default();
        {
            // f64.
            let mut jiteval = tree.jit_compile(&context).unwrap();
            check_value_eval(
                tree.clone(),
                |inputs: &[f64], outputs: &mut [f64]| {
                    let results = jiteval.run(inputs).unwrap();
                    outputs.copy_from_slice(results);
                },
                vardata,
                samples_per_var,
                eps64,
            );
        }
        {
            // f32
            let mut jiteval = tree.jit_compile(&context).unwrap();
            let mut inpf32 = Vec::new();
            let mut outf64 = Vec::new();
            check_value_eval(
                tree.clone(),
                |inputs: &[f64], outputs: &mut [f64]| {
                    inpf32.clear();
                    inpf32.extend(inputs.iter().map(|i| *i as f32));
                    let results = jiteval.run(&inpf32).unwrap();
                    outf64.clear();
                    outf64.extend(results.iter().map(|r| *r as f64));
                    outputs.copy_from_slice(&outf64);
                },
                vardata,
                samples_per_var,
                eps32 as f64,
            );
        }
    }

    #[test]
    fn t_sum() {
        check_jit_eval(
            &deftree!(+ x y).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.)],
            100,
            0.,
            1e-5,
        );
    }

    #[test]
    fn t_prod_sum() {
        check_jit_eval(
            &deftree!(concat (+ x y) (* x y)).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.)],
            100,
            0.,
            1e-4,
        );
    }

    #[test]
    fn t_sub_div() {
        check_jit_eval(
            &deftree!(concat (- x y) (/ x y)).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.)],
            20,
            0.,
            1e-4,
        );
    }

    #[test]
    fn t_pow() {
        check_jit_eval(
            &deftree!(pow x 2).unwrap(),
            &[('x', -10., -10.)],
            100,
            0.,
            0.,
        );
    }

    #[test]
    fn t_sqrt() {
        check_jit_eval(
            &deftree!(sqrt x).unwrap(),
            &[('x', 0.01, 10.)],
            100,
            0.,
            1e-6,
        );
    }

    #[test]
    fn t_circle() {
        check_jit_eval(
            &deftree!(- (sqrt (+ (pow x 2) (pow y 2))) 3).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.)],
            20,
            0.,
            1e-6,
        );
    }

    #[test]
    fn t_sum_3() {
        check_jit_eval(
            &deftree!(+ (+ x 3) (+ y z)).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            5,
            0.,
            1e-5,
        );
    }

    #[test]
    fn t_sphere() {
        check_jit_eval(
            &deftree!(- (sqrt (+ (pow x 2) (+ (pow y 2) (pow z 2)))) 3).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            0.,
            1e-6,
        )
    }

    #[test]
    fn t_negate() {
        check_jit_eval(
            &deftree!(* (- x) (+ y z)).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            0.,
            1e-5,
        );
    }

    #[test]
    fn t_abs() {
        check_jit_eval(
            &deftree!(* (abs x) (+ (abs y) (abs z))).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            0.,
            1e-5,
        );
    }

    #[test]
    fn t_trigonometry() {
        check_jit_eval(
            &deftree!(/ (+ (sin x) (cos y)) (+ 0.27 (pow (tan z) 2))).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            1e-14,
            1e-5,
        );
    }

    #[test]
    fn t_log_exp() {
        check_jit_eval(
            &deftree!(/ (+ 1 (log x)) (+ 1 (exp y))).unwrap(),
            &[('x', 0.1, 5.), ('y', 0.1, 5.)],
            10,
            0.,
            1e-6,
        );
    }

    #[test]
    fn t_min_max() {
        check_jit_eval(
            &deftree!(
                (max (min
                      (- (sqrt (+ (+ (pow (- x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 2.75)
                      (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 4.))
                 (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (+ y 3.) 2.)) (pow (- z 4.) 2.))) 5.25))
            )
            .unwrap(),
            &[('x', -10., 10.), ('y', -9., 10.), ('z', -11., 12.)],
            20,
            0.,
            1e-5,
        );
    }

    #[test]
    fn t_floor() {
        check_jit_eval(
            &deftree!(floor (+ (pow x 2) (sin x))).unwrap(),
            &[('x', -5., 5.)],
            100,
            0.,
            0.,
        );
    }

    #[test]
    fn t_remainder() {
        check_jit_eval(
            &deftree!(rem (pow x 2) (+ 2 (sin x))).unwrap(),
            &[('x', 1., 5.)],
            100,
            1e-15,
            1e-5,
        );
    }

    #[test]
    fn t_choose() {
        check_jit_eval(
            &deftree!(if (> x 0) x (-x)).unwrap(),
            &[('x', -10., 10.)],
            100,
            0.,
            1e-6,
        );
        check_jit_eval(
            &deftree!(if (< x 0) (- x) x).unwrap(),
            &[('x', -10., 10.)],
            100,
            0.,
            1e-6,
        );
    }

    #[test]
    fn t_or_and() {
        check_jit_eval(
            &deftree!(if (and (> x 0) (< x 1)) (* 2 x) 1).unwrap(),
            &[('x', -3., 3.)],
            100,
            0.,
            1e-6,
        );
    }

    #[test]
    fn t_not() {
        check_jit_eval(
            &deftree!(if (not (> x 0)) (- (pow x 3) (pow y 3)) (+ (pow x 2) (pow y 2))).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.)],
            100,
            1e-14,
            1e-4,
        );
    }
}

#[cfg(test)]
mod sphere_test {
    use super::*;
    use crate::{
        assert_float_eq,
        dedup::Deduplicater,
        deftree,
        eval::ValueEvaluator,
        prune::Pruner,
        tree::{Tree, min},
    };
    use rand::{SeedableRng, rngs::StdRng};

    fn sample_range(range: (f64, f64), rng: &mut StdRng) -> f64 {
        use rand::Rng;
        range.0 + rng.random::<f64>() * (range.1 - range.0)
    }

    const RADIUS_RANGE: (f64, f64) = (0.2, 2.);
    const X_RANGE: (f64, f64) = (0., 100.);
    const Y_RANGE: (f64, f64) = (0., 100.);
    const Z_RANGE: (f64, f64) = (0., 100.);
    const N_SPHERES: usize = 500;
    const N_QUERIES: usize = 500;

    fn sphere_union() -> Tree {
        let mut rng = StdRng::seed_from_u64(42);
        let mut make_sphere = || -> Result<Tree, Error> {
            deftree!(- (sqrt (+ (+
                                 (pow (- x (const sample_range(X_RANGE, &mut rng))) 2)
                                 (pow (- y (const sample_range(Y_RANGE, &mut rng))) 2))
                              (pow (- z (const sample_range(Z_RANGE, &mut rng))) 2)))
                     (const sample_range(RADIUS_RANGE, &mut rng)))
        };
        let mut tree = make_sphere();
        for _ in 1..N_SPHERES {
            tree = min(tree, make_sphere());
        }
        let tree = tree.unwrap();
        assert_eq!(tree.dims(), (1, 1));
        tree
    }

    #[test]
    fn t_compare_jit_eval() {
        let mut rng = StdRng::seed_from_u64(234);
        let queries: Vec<[f64; 3]> = (0..N_QUERIES)
            .map(|_| {
                [
                    sample_range(X_RANGE, &mut rng),
                    sample_range(Y_RANGE, &mut rng),
                    sample_range(Z_RANGE, &mut rng),
                ]
            })
            .collect();
        let tree = {
            let mut dedup = Deduplicater::new();
            let mut pruner = Pruner::new();
            sphere_union()
                .fold()
                .unwrap()
                .deduplicate(&mut dedup)
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
        };
        // Run the evaluator.
        let mut val_eval: Vec<f64> = Vec::with_capacity(N_QUERIES);
        {
            let mut eval = ValueEvaluator::new(&tree);
            val_eval.extend(queries.iter().map(|coords| {
                eval.set_value('x', coords[0].into());
                eval.set_value('y', coords[1].into());
                eval.set_value('z', coords[2].into());
                let results = eval.run().unwrap();
                results[0].scalar().unwrap()
            }));
        }
        // Now run the jit evaluation.
        let mut val_jit: Vec<f64> = Vec::with_capacity(N_QUERIES);
        {
            let context = JitContext::default();
            let mut eval = tree.jit_compile(&context).unwrap();
            val_jit.extend(queries.iter().map(|coords| {
                let results = eval.run(coords).unwrap();
                results[0]
            }));
        }
        assert_eq!(val_eval.len(), val_jit.len());
        for (l, r) in val_eval.iter().zip(val_jit.iter()) {
            assert_float_eq!(l, r, 1e-15);
        }
    }
}
