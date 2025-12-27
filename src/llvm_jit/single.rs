use super::{
    JitCompiler, JitContext, NumberType, build_float_binary_intrinsic, build_float_unary_intrinsic,
    fast_math,
};
use crate::{
    BinaryOp::*,
    Error,
    Node::*,
    TernaryOp::*,
    Tree,
    UnaryOp::*,
    Value::{self, *},
};
use inkwell::{
    AddressSpace, FloatPredicate, OptimizationLevel,
    execution_engine::JitFunction,
    values::{BasicValue, BasicValueEnum},
};
use std::{ffi::c_void, marker::PhantomData};

type NativeFunc = unsafe extern "C" fn(
    *const c_void, // Inputs
    *mut c_void,   // Outputs
);

/// This represents a JIT compiled tree. This is a wrapper around the JIT compiled native function.
#[derive(Clone)]
pub struct JitFn<'ctx, T>
where
    T: NumberType,
{
    func: JitFunction<'ctx, NativeFunc>,
    num_inputs: usize,
    num_outputs: usize,
    _phantom: PhantomData<T>,
}

/**
`JitFn` is not thread safe, because it contains the executable memory where the
JIT machine code resides, somewhere inside the Execution Engine. LLVM doesn't
implement the `Send` trait for this block of memory, because it doesn't know
what's in the JIT machine code, it doesn't know if that code itself is thread
safe, or has side effects. This `JitFnSync` can be pulled out of a `JitFn`, via
the `.as_async()` function, and is thread safe. It implements the `Send`
trait. This is OK, because we know the machine code represents a mathematical
expression without any side effects. So we pull out the function pointer and
wrap it in this struct, that can be shared across threads. Still the execution
engine held inside the original `JitSmdFn` needs to outlive this sync wrapper,
because it owns the block of executable memory. To guarantee that, this structs
pseudo borrows (via a phantom) from the `JitFn`. It has to be done via a phantom
othwerwise we can't implement The Sync trait on this.
*/
pub struct JitFnSync<'ctx, T>
where
    T: NumberType,
{
    func: NativeFunc,
    num_inputs: usize,
    num_outputs: usize,
    _phantom: PhantomData<&'ctx JitFn<'ctx, T>>,
}

unsafe impl<'ctx, T> Sync for JitFnSync<'ctx, T> where T: NumberType {}

impl Tree {
    /// JIT compile a tree and return a native evaluator.
    pub fn jit_compile<'ctx, T>(
        &'ctx self,
        context: &'ctx JitContext,
        params: &str,
    ) -> Result<JitFn<'ctx, T>, Error>
    where
        T: NumberType,
    {
        if !self.is_scalar() {
            // Only support scalar output trees.
            return Err(Error::TypeMismatch);
        }
        let num_roots = self.num_roots();
        let func_name = context.new_func_name::<T>(None);
        let context = &context.inner;
        let compiler = JitCompiler::new(context)?;
        let builder = &compiler.builder;
        let float_type = T::jit_type(context);
        let float_ptr_type = context.ptr_type(AddressSpace::default());
        let bool_type = context.bool_type();
        let fn_type = context
            .void_type()
            .fn_type(&[float_ptr_type.into(), float_ptr_type.into()], false);
        let function = compiler.module.add_function(&func_name, fn_type, None);
        compiler.set_attributes(function, context)?;
        builder.position_at_end(context.append_basic_block(function, "entry"));
        let mut regs: Vec<BasicValueEnum> = Vec::with_capacity(self.len());
        for (ni, node) in self.nodes().iter().enumerate() {
            let reg = match node {
                Constant(val) => match val {
                    Bool(val) => bool_type
                        .const_int(if *val { 1 } else { 0 }, false)
                        .as_basic_value_enum(),
                    Scalar(val) => float_type.const_float(*val).as_basic_value_enum(),
                },
                Symbol(label) => {
                    let inputs = function
                        .get_first_param()
                        .ok_or(Error::JitCompilationError("Cannot read inputs".to_string()))?
                        .into_pointer_value();
                    // SAFETY: GEP can segfault if the index is out of
                    // bounds. The offset calculation looks pretty solid, and is
                    // thoroughly tested.
                    let ptr = unsafe {
                        builder.build_gep(
                            float_type,
                            inputs,
                            &[context.i64_type().const_int(
                                params.chars().position(|c| c == *label).ok_or(
                                    Error::JitCompilationError("Cannot find symbol".to_string()),
                                )? as u64,
                                false,
                            )],
                            &format!("arg_{}", *label),
                        )?
                    };
                    builder.build_load(float_type, ptr, &format!("val_{}", *label))?
                }
                Unary(op, input) => {
                    match op {
                        Negate => fast_math(builder.build_float_neg(
                            regs[*input].into_float_value(),
                            &format!("val_{ni}"),
                        )?)
                        .as_basic_value_enum(),
                        Sqrt => fast_math(build_float_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.sqrt.*",
                            "sqrt_call",
                            regs[*input].into_float_value(),
                        )?),
                        Abs => fast_math(build_float_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.fabs.*",
                            "abs_call",
                            regs[*input].into_float_value(),
                        )?),
                        Sin => fast_math(build_float_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.sin.*",
                            "sin_call",
                            regs[*input].into_float_value(),
                        )?),
                        Cos => fast_math(build_float_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.cos.*",
                            "cos_call",
                            regs[*input].into_float_value(),
                        )?),
                        Tan => {
                            let sin = fast_math(build_float_unary_intrinsic(
                                builder,
                                &compiler.module,
                                "llvm.sin.*",
                                "sin_call",
                                regs[*input].into_float_value(),
                            )?);
                            let cos = fast_math(build_float_unary_intrinsic(
                                builder,
                                &compiler.module,
                                "llvm.cos.*",
                                "cos_call",
                                regs[*input].into_float_value(),
                            )?);
                            fast_math(builder.build_float_div(
                                sin.into_float_value(),
                                cos.into_float_value(),
                                &format!("val_{ni}"),
                            )?)
                            .as_basic_value_enum()
                        }
                        Log => fast_math(build_float_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.log.*",
                            "log_call",
                            regs[*input].into_float_value(),
                        )?),
                        Exp => fast_math(build_float_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.exp.*",
                            "exp_call",
                            regs[*input].into_float_value(),
                        )?),
                        Floor => fast_math(build_float_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.floor.*",
                            "floor_call",
                            regs[*input].into_float_value(),
                        )?),
                        Not => builder
                            .build_not(regs[*input].into_int_value(), &format!("val_{ni}"))?
                            .as_basic_value_enum(),
                    }
                }
                Binary(op, lhs, rhs) => match op {
                    Add => fast_math(builder.build_float_add(
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("val_{ni}"),
                    )?)
                    .as_basic_value_enum(),
                    Subtract => fast_math(builder.build_float_sub(
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("val_{ni}"),
                    )?)
                    .as_basic_value_enum(),
                    Multiply => fast_math(builder.build_float_mul(
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("val_{ni}"),
                    )?)
                    .as_basic_value_enum(),
                    Divide => fast_math(builder.build_float_div(
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("val_{ni}"),
                    )?)
                    .as_basic_value_enum(),
                    Pow if matches!(self.node(*rhs), Constant(Value::Scalar(2.0))) => {
                        let input = regs[*lhs].into_float_value();
                        fast_math(builder.build_float_mul(input, input, &format!("val_{ni}"))?)
                            .as_basic_value_enum()
                    }
                    Pow => fast_math(build_float_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.pow.*",
                        "pow_call",
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                    )?),
                    Min => fast_math(build_float_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.minnum.*",
                        "min_call",
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                    )?),
                    Max => fast_math(build_float_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.maxnum.*",
                        "max_call",
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                    )?),
                    Remainder => fast_math(builder.build_float_rem(
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("val_{ni}"),
                    )?)
                    .as_basic_value_enum(),
                    Less => fast_math(builder.build_float_compare(
                        FloatPredicate::ULT,
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("val_{ni}"),
                    )?)
                    .as_basic_value_enum(),
                    LessOrEqual => fast_math(builder.build_float_compare(
                        FloatPredicate::ULE,
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("val_{ni}"),
                    )?)
                    .as_basic_value_enum(),
                    Equal => fast_math(builder.build_float_compare(
                        FloatPredicate::UEQ,
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("val_{ni}"),
                    )?)
                    .as_basic_value_enum(),
                    NotEqual => fast_math(builder.build_float_compare(
                        FloatPredicate::UNE,
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("val_{ni}"),
                    )?)
                    .as_basic_value_enum(),
                    Greater => fast_math(builder.build_float_compare(
                        FloatPredicate::UGT,
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("val_{ni}"),
                    )?)
                    .as_basic_value_enum(),
                    GreaterOrEqual => fast_math(builder.build_float_compare(
                        FloatPredicate::UGE,
                        regs[*lhs].into_float_value(),
                        regs[*rhs].into_float_value(),
                        &format!("val_{ni}"),
                    )?)
                    .as_basic_value_enum(),
                    And => builder
                        .build_and(
                            regs[*lhs].into_int_value(),
                            regs[*rhs].into_int_value(),
                            &format!("val_{ni}"),
                        )?
                        .as_basic_value_enum(),
                    Or => builder
                        .build_or(
                            regs[*lhs].into_int_value(),
                            regs[*rhs].into_int_value(),
                            &format!("val_{ni}"),
                        )?
                        .as_basic_value_enum(),
                },
                Ternary(op, a, b, c) => match op {
                    Choose => builder.build_select(
                        regs[*a].into_int_value(),
                        regs[*b].into_float_value(),
                        regs[*c].into_float_value(),
                        &format!("val_{ni}"),
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
            // SAFETY: GEP can segfault if the index is out of bounds. The
            // offset calculation looks pretty solid, and is thoroughly tested.
            let dst = unsafe {
                builder.build_gep(
                    float_type,
                    outputs,
                    &[context.i64_type().const_int(i as u64, false)],
                    &format!("output_{i}"),
                )?
            };
            builder.build_store(dst, *reg)?;
        }
        builder.build_return(None)?;
        compiler.run_passes("mem2reg,instcombine,reassociate,gvn,instcombine,slp-vectorizer,instcombine,simplifycfg,adce")?;
        let engine = compiler
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .map_err(|_| Error::CannotCreateJitModule)?;
        // SAFETY: The signature is correct, and well tested. The function
        // pointer should never be invalidated, because we allocated a dedicated
        // execution engine, with it's own block of executable memory, that will
        // live as long as the function wrapper lives.
        let func = unsafe { engine.get_function(&func_name)? };
        Ok(JitFn {
            func,
            num_inputs: params.len(),
            num_outputs: num_roots,
            _phantom: PhantomData,
        })
    }
}

impl<'ctx, T> JitFn<'ctx, T>
where
    T: NumberType,
{
    /// Run the JIT evaluator with the given input values. The number of input
    /// values is expected to be the same as the number of variables in the
    /// tree. The variables are substituted with the input values in the same
    /// order as returned by calling `tree.symbols()` which was compiled to
    /// produce this evaluator.
    pub fn run(&self, inputs: &[T], outputs: &mut [T]) -> Result<(), Error> {
        if inputs.len() != self.num_inputs {
            return Err(Error::InputSizeMismatch(inputs.len(), self.num_inputs));
        } else if outputs.len() != self.num_outputs {
            return Err(Error::OutputSizeMismatch(outputs.len(), self.num_outputs));
        }
        // SAFETY: We just checked above.
        unsafe { self.run_unchecked(inputs, outputs) };
        Ok(())
    }

    /**
    Same as `run` except it doesn't check to make sure the `inputs` slice is of
    the correct length. This can be used when the caller is sure the inputs are
    correct, and this check can be omitted.

    # Safety

    The caller has to make sure the number of inputs matches the
    number of symbols of the tree, and the number of outputs match the number
    of roots of the tree from which this JIT function was created.
     */
    pub unsafe fn run_unchecked(&self, inputs: &[T], outputs: &mut [T]) {
        // SAFETY: We told the caller it is their responsiblity.
        unsafe {
            self.func
                .call(inputs.as_ptr().cast(), outputs.as_mut_ptr().cast());
        }
    }

    pub fn as_sync(&'ctx self) -> JitFnSync<'ctx, T> {
        JitFnSync {
            // SAFETY: Accessing the raw function pointer. This is ok, because
            // this borrows from Self, which owns an Rc reference to the
            // execution engine that owns the block of executable memory to
            // which the function pointer points.
            func: unsafe { self.func.as_raw() },
            num_inputs: self.num_inputs,
            num_outputs: self.num_outputs,
            _phantom: PhantomData,
        }
    }
}

impl<'ctx, T> JitFnSync<'ctx, T>
where
    T: NumberType,
{
    pub fn run(&self, inputs: &[T], outputs: &mut [T]) -> Result<(), Error> {
        if (inputs.len() != self.num_inputs) || (outputs.len() != self.num_outputs) {
            return Err(Error::InputSizeMismatch(inputs.len(), self.num_inputs));
        }
        // SAFETY: We just checked above.
        let _: () = unsafe { self.run_unchecked(inputs, outputs) };
        Ok(())
    }

    /**
    Same as [`run`], without the bounds checking.

    # Safety

    The caller has to make sure the number of inputs matches the number of
    symbols of the tree, and the number of outputs match the number of roots of
    the tree from which this JIT function was created.
     */
    pub unsafe fn run_unchecked(&self, inputs: &[T], outputs: &mut [T]) {
        // SAFETY: We told the caller it is their responsiblity.
        unsafe {
            (self.func)(inputs.as_ptr().cast(), outputs.as_mut_ptr().cast());
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{deftree, test_util::check_value_eval};

    fn check_jit_eval(
        tree: &Tree,
        vardata: &[(char, f64, f64)],
        samples_per_var: usize,
        eps64: f64,
        eps32: f32,
    ) {
        let context = JitContext::default();
        let params: String = vardata.iter().map(|(c, ..)| *c).collect();
        {
            // f64.
            let jiteval = tree.jit_compile(&context, &params).unwrap();
            check_value_eval(
                tree.clone(),
                |inputs: &[f64], outputs: &mut [f64]| {
                    jiteval.run(inputs, outputs).unwrap();
                },
                vardata,
                samples_per_var,
                eps64,
            );
        }
        {
            // f32
            let jiteval = tree.jit_compile(&context, &params).unwrap();
            let mut inpf32 = Vec::new();
            let mut outf32 = vec![0.; tree.num_roots()];
            let mut outf64 = Vec::new();
            check_value_eval(
                tree.clone(),
                |inputs: &[f64], outputs: &mut [f64]| {
                    inpf32.clear();
                    inpf32.extend(inputs.iter().map(|i| *i as f32));
                    jiteval.run(&inpf32, &mut outf32).unwrap();
                    outf64.clear();
                    outf64.extend(outf32.iter().map(|v| *v as f64));
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
            &deftree!(+ 'x 'y).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.)],
            100,
            0.,
            1e-5,
        );
    }

    #[test]
    fn t_prod_sum() {
        check_jit_eval(
            &deftree!(concat (+ 'x 'y) (* 'x 'y)).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.)],
            100,
            0.,
            1e-4,
        );
    }

    #[test]
    fn t_sub_div() {
        check_jit_eval(
            &deftree!(concat (- 'x 'y) (/ 'x 'y)).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.)],
            20,
            0.,
            1e-4,
        );
    }

    #[test]
    fn t_pow() {
        check_jit_eval(
            &deftree!(pow 'x 2).unwrap(),
            &[('x', -10., -10.)],
            100,
            0.,
            0.,
        );
    }

    #[test]
    fn t_sqrt() {
        check_jit_eval(
            &deftree!(sqrt 'x).unwrap(),
            &[('x', 0.01, 10.)],
            100,
            0.,
            1e-6,
        );
    }

    #[test]
    fn t_circle() {
        check_jit_eval(
            &deftree!(- (sqrt (+ (pow 'x 2) (pow 'y 2))) 3).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.)],
            20,
            1e-14,
            1e-6,
        );
    }

    #[test]
    fn t_sum_3() {
        check_jit_eval(
            &deftree!(+ (+ 'x 3) (+ 'y 'z)).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            5,
            1e-14,
            1e-5,
        );
    }

    #[test]
    fn t_sphere() {
        check_jit_eval(
            &deftree!(- (sqrt (+ (pow 'x 2) (+ (pow 'y 2) (pow 'z 2)))) 3).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            1e-14,
            1e-6,
        )
    }

    #[test]
    fn t_negate() {
        check_jit_eval(
            &deftree!(* (- 'x) (+ 'y 'z)).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            0.,
            1e-5,
        );
    }

    #[test]
    fn t_abs() {
        check_jit_eval(
            &deftree!(* (abs 'x) (+ (abs 'y) (abs 'z))).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            0.,
            1e-5,
        );
    }

    #[test]
    fn t_trigonometry() {
        check_jit_eval(
            &deftree!(/ (+ (sin 'x) (cos 'y)) (+ 0.27 (pow (tan 'z) 2))).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            1e-14,
            1e-5,
        );
    }

    #[test]
    fn t_log_exp() {
        check_jit_eval(
            &deftree!(/ (+ 1 (log 'x)) (+ 1 (exp 'y))).unwrap(),
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
                      (- (sqrt (+ (+ (pow (- 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 2.75)
                      (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 4.))
                 (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (+ 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 5.25))
            )
            .unwrap(),
            &[('x', -10., 10.), ('y', -9., 10.), ('z', -11., 12.)],
            20,
            1e-14,
            1e-5,
        );
    }

    #[test]
    fn t_floor() {
        check_jit_eval(
            &deftree!(floor (+ (pow 'x 2) (sin 'x))).unwrap(),
            &[('x', -5., 5.)],
            100,
            0.,
            0.,
        );
    }

    #[test]
    fn t_remainder() {
        check_jit_eval(
            &deftree!(rem (pow 'x 2) (+ 2 (sin 'x))).unwrap(),
            &[('x', 1., 5.)],
            100,
            1e-15,
            1e-5,
        );
    }

    #[test]
    fn t_choose() {
        check_jit_eval(
            &deftree!(if (> 'x 0) 'x (- 'x)).unwrap(),
            &[('x', -10., 10.)],
            100,
            0.,
            1e-6,
        );
        check_jit_eval(
            &deftree!(if (< 'x 0) (- 'x) 'x).unwrap(),
            &[('x', -10., 10.)],
            100,
            0.,
            1e-6,
        );
    }

    #[test]
    fn t_or_and() {
        check_jit_eval(
            &deftree!(if (and (> 'x 0) (< 'x 1)) (* 2 'x) 1).unwrap(),
            &[('x', -3., 3.)],
            100,
            0.,
            1e-6,
        );
    }

    #[test]
    fn t_not() {
        check_jit_eval(
            &deftree!(if (not (> 'x 0)) (- (pow 'x 3) (pow 'y 3)) (+ (pow 'x 2) (pow 'y 2)))
                .unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.)],
            100,
            1e-13,
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
                                 (pow (- 'x (const sample_range(X_RANGE, &mut rng))) 2)
                                 (pow (- 'y (const sample_range(Y_RANGE, &mut rng))) 2))
                              (pow (- 'z (const sample_range(Z_RANGE, &mut rng))) 2)))
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
            let eval = tree.jit_compile(&context, "xyz").unwrap();
            val_jit.extend(queries.iter().map(|coords| {
                let mut output = [0.];
                eval.run(coords, &mut output).unwrap();
                output[0]
            }));
        }
        assert_eq!(val_eval.len(), val_jit.len());
        for (l, r) in val_eval.iter().zip(val_jit.iter()) {
            assert_float_eq!(l, r, 1e-14);
        }
    }
}
