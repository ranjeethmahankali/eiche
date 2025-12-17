use super::{JitContext, NumberType, build_vec_unary_intrinsic};
use crate::{
    BinaryOp::*, Error, Node::*, TernaryOp::*, Tree, UnaryOp::*, Value, llvm_jit::JitCompiler,
};
use inkwell::{
    AddressSpace, FloatPredicate, OptimizationLevel,
    builder::Builder,
    context::Context,
    execution_engine::JitFunction,
    types::VectorType,
    values::{BasicValue, BasicValueEnum, VectorValue},
};
use std::{f64::consts::FRAC_PI_2, ffi::c_void, marker::PhantomData};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub type IntervalType64 = __m128d;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub type IntervalType32 = __m128;

pub type NativeIntervalFunc = unsafe extern "C" fn(*const c_void, *mut c_void);

#[derive(Clone)]
pub struct JitIntervalFn<'ctx, T>
where
    T: NumberType,
{
    func: JitFunction<'ctx, NativeIntervalFunc>,
    num_inputs: usize,
    num_outputs: usize,
    _phantom: PhantomData<T>,
}

/**
`JitIntervalFn` is not thread safe, because it contains the executable memory
where the JIT machine code resides, somewhere inside the Execution Engine. LLVM
doesn't implement the `Send` trait for this block of memory, because it doesn't
know what's in the JIT machine code, it doesn't know if that code itself is
thread safe, or has side effects. This `JitIntervalFnSync` can be pulled out of
a `JitIntervalFn`, via the `.as_async()` function, and is thread safe. It
implements the `Send` trait. This is OK, because we know the machine code
represents a mathematical expression without any side effects. So we pull out
the function pointer and wrap it in this struct, that can be shared across
threads. Still the execution engine held inside the original `JitSmdFn` needs to
outlive this sync wrapper, because it owns the block of executable memory. To
guarantee that, this structs pseudo borrows (via a phantom) from the
`JitIntervalFn`. It has to be done via a phantom othwerwise we can't implement
The Sync trait on this.
*/
pub struct JitIntervalFnSync<'ctx, T>
where
    T: NumberType,
{
    func: NativeIntervalFunc,
    num_inputs: usize,
    num_outputs: usize,
    _phantom: PhantomData<&'ctx JitIntervalFn<'ctx, T>>,
}

unsafe impl<'ctx, T> Sync for JitIntervalFnSync<'ctx, T> where T: NumberType {}

impl Tree {
    /// JIT compile the tree for interval evaluations.
    pub fn jit_compile_interval<'ctx, T>(
        &'ctx self,
        context: &'ctx JitContext,
        params: &str,
    ) -> Result<JitIntervalFn<'ctx, T>, Error>
    where
        T: NumberType,
    {
        if !self.is_scalar() {
            // Only support scalar output trees.
            return Err(Error::TypeMismatch);
        }
        let num_roots = self.num_roots();
        let func_name = context.new_func_name::<T>(Some("interval"));
        let context = &context.inner;
        let compiler = JitCompiler::new(context)?;
        let builder = &compiler.builder;
        let float_type = T::jit_type(context);
        let interval_type = float_type.vec_type(2);
        let iptr_type = context.ptr_type(AddressSpace::default());
        let bool_type = context.bool_type();
        let fn_type = context
            .void_type()
            .fn_type(&[iptr_type.into(), iptr_type.into()], false);
        let function = compiler.module.add_function(&func_name, fn_type, None);
        builder.position_at_end(context.append_basic_block(function, "entry"));
        let mut regs = Vec::<BasicValueEnum>::with_capacity(self.len());
        for (ni, node) in self.nodes().iter().enumerate() {
            let reg = match node {
                Constant(value) => match value {
                    Value::Bool(flag) => VectorType::const_vector(
                        &[bool_type.const_int(if *flag { 1 } else { 0 }, false); 2],
                    )
                    .as_basic_value_enum(),
                    Value::Scalar(value) => VectorType::const_vector(&[
                        float_type.const_float(*value),
                        float_type.const_float(*value),
                    ])
                    .as_basic_value_enum(),
                },
                Symbol(label) => {
                    let inputs = function
                        .get_first_param()
                        .ok_or(Error::JitCompilationError("Cannot read inputs".to_string()))?
                        .into_pointer_value();
                    let ptr = unsafe {
                        builder.build_gep(
                            interval_type,
                            inputs,
                            &[context.i64_type().const_int(
                                params.chars().position(|c| c == *label).ok_or(
                                    Error::JitCompilationError("Cannot find symbol".to_string()),
                                )? as u64,
                                false,
                            )],
                            &format!("arg_ptr_{}", *label),
                        )?
                    };
                    builder.build_load(interval_type, ptr, &format!("arg_{}", *label))?
                }
                Unary(op, input) => match op {
                    // For negate all we need to do is swap the vector lanes.
                    Negate => build_interval_negate(
                        regs[*input].into_vector_value(),
                        context,
                        builder,
                        ni,
                        &format!("reg_{ni}"),
                    )?
                    .as_basic_value_enum(),
                    Sqrt => {
                        let ireg = regs[*input].into_vector_value();
                        builder.build_select(
                            // Check each lane for NaN, then reduce to check if this interval is empty.
                            build_vec_unary_intrinsic(
                                builder,
                                &compiler.module,
                                "llvm.vector.reduce.and.*",
                                &format!("reduce_call_{ni}"),
                                builder.build_float_compare(
                                    FloatPredicate::UNO,
                                    ireg,
                                    ireg,
                                    &format!("check_empty_{ni}"),
                                )?,
                            )?
                            .into_int_value(),
                            // The interval is empty, so return an emtpy (NaN) interval.
                            VectorType::const_vector(&[
                                float_type.const_float(f64::NAN),
                                float_type.const_float(f64::NAN),
                            ])
                            .as_basic_value_enum(),
                            {
                                // Interval is not empty.
                                let lt_zero = builder.build_float_compare(
                                    FloatPredicate::ULT,
                                    ireg,
                                    VectorType::const_vector(&[
                                        float_type.const_float(0.),
                                        float_type.const_float(0.),
                                    ]),
                                    &format!("lt_zero_{ni}"),
                                )?;
                                let sqrt = build_vec_unary_intrinsic(
                                    builder,
                                    &compiler.module,
                                    "llvm.sqrt.*",
                                    &format!("sqrt_call_{ni}"),
                                    build_vec_unary_intrinsic(
                                        builder,
                                        &compiler.module,
                                        "llvm.fabs.*",
                                        &format!("fabs_call_{ni}"),
                                        ireg,
                                    )?
                                    .into_vector_value(),
                                )?
                                .into_vector_value();
                                /* This a nested if. First we check the sign of
                                 * the lower bound, then we check the sign of
                                 * the upper bound in the nested select
                                 * statement. Then we return different things.
                                 */
                                builder.build_select(
                                    // Check first element of vec.
                                    builder
                                        .build_extract_element(
                                            lt_zero,
                                            context.i32_type().const_int(0, false),
                                            &format!("first_lt_zero_{ni}"),
                                        )?
                                        .into_int_value(),
                                    builder.build_select(
                                        builder
                                            .build_extract_element(
                                                lt_zero,
                                                context.i32_type().const_int(1, false),
                                                &format!("second_lt_zero_{ni}"),
                                            )?
                                            .into_int_value(),
                                        VectorType::const_vector(&[
                                            float_type.const_float(f64::NAN),
                                            float_type.const_float(f64::NAN),
                                        ]),
                                        builder.build_float_mul(
                                            sqrt,
                                            VectorType::const_vector(&[
                                                float_type.const_float(0.0),
                                                float_type.const_float(1.0),
                                            ]),
                                            &format!("sqrt_domain_clipping_{ni}"),
                                        )?,
                                        &format!("sqrt_edge_case_{ni}"),
                                    )?,
                                    sqrt.as_basic_value_enum(),
                                    &format!("sqrt_branching_{ni}"),
                                )?
                            },
                            &format!("reg_{ni}"),
                        )?
                    }
                    Abs => {
                        let ireg = regs[*input].into_vector_value();
                        let lt_zero = builder.build_float_compare(
                            FloatPredicate::ULT,
                            ireg,
                            VectorType::const_vector(&[
                                float_type.const_float(0.),
                                float_type.const_float(0.),
                            ]),
                            &format!("lt_zero_{ni}"),
                        )?;
                        builder.build_select(
                            builder
                                .build_extract_element(
                                    lt_zero,
                                    context.i32_type().const_int(1, false),
                                    &format!("first_lt_zero_{ni}"),
                                )?
                                .into_int_value(),
                            // (-hi, -lo)
                            build_interval_negate(
                                ireg,
                                context,
                                builder,
                                ni,
                                &format!("intermediate_1_{ni}"),
                            )?,
                            builder
                                .build_select(
                                    builder
                                        .build_extract_element(
                                            lt_zero,
                                            context.i32_type().const_int(0, false),
                                            &format!("first_lt_zero_{ni}"),
                                        )?
                                        .into_int_value(),
                                    // (0.0, max(abs(lo), abs(hi)))
                                    builder.build_insert_element(
                                        interval_type.const_zero(),
                                        build_vec_unary_intrinsic(
                                            builder,
                                            &compiler.module,
                                            "llvm.vector.reduce.fmax.*",
                                            &format!("fmax_reduce_call_{ni}"),
                                            build_vec_unary_intrinsic(
                                                builder,
                                                &compiler.module,
                                                "llvm.fabs.*",
                                                &format!("abs_call_{ni}"),
                                                ireg,
                                            )?
                                            .into_vector_value(),
                                        )?
                                        .into_float_value(),
                                        context.i32_type().const_int(1, false),
                                        &format!("intermediate_2_{ni}"),
                                    )?,
                                    // (lo, hi),
                                    ireg,
                                    &format!("intermediate_3_{ni}"),
                                )?
                                .into_vector_value(),
                            &format!("reg_{ni}"),
                        )?
                    }
                    Sin => {
                        let ireg = regs[*input].into_vector_value();
                        let qinterval = build_vec_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.floor.*",
                            &format!("intermediate_floor_{ni}"),
                            builder.build_float_div(
                                ireg,
                                VectorType::const_vector(&[
                                    float_type.const_float(FRAC_PI_2),
                                    float_type.const_float(FRAC_PI_2),
                                ]),
                                &format!("div_pi_{ni}"),
                            )?,
                        )?
                        .into_vector_value();
                        let (lo, hi) = (
                            builder
                                .build_extract_element(
                                    ireg,
                                    context.i32_type().const_int(0, false),
                                    &format!("extract_lo_{ni}"),
                                )?
                                .into_float_value(),
                            builder
                                .build_extract_element(
                                    ireg,
                                    context.i32_type().const_int(1, false),
                                    &format!("extract_lo_{ni}"),
                                )?
                                .into_float_value(),
                        );
                        let qlo = builder
                            .build_extract_element(
                                qinterval,
                                context.i32_type().const_int(0, false),
                                &format!("q_extract_1_{ni}"),
                            )?
                            .into_float_value();
                        let nval = builder
                            .build_select(
                                builder.build_float_compare(
                                    FloatPredicate::UEQ,
                                    lo,
                                    hi,
                                    &format!("lo_hi_compare_{ni}"),
                                )?,
                                float_type.const_float(0.0),
                                builder.build_float_sub(
                                    builder
                                        .build_extract_element(
                                            qinterval,
                                            context.i32_type().const_int(1, false),
                                            &format!("q_extract_0_{ni}"),
                                        )?
                                        .into_float_value(),
                                    qlo,
                                    &format!("nval_sub_{ni}"),
                                )?,
                                &format!("nval_{ni}"),
                            )?
                            .into_float_value();
                        let qval = builder.build_float_rem(
                            qlo,
                            float_type.const_float(4.0),
                            &format!("q_rem_val_{ni}"),
                        )?;
                        let qval = builder
                            .build_select(
                                builder.build_float_compare(
                                    FloatPredicate::ULT,
                                    qval,
                                    float_type.const_zero(),
                                    &format!("rem_euclid_compare_{ni}"),
                                )?,
                                builder.build_float_add(
                                    qval,
                                    float_type.const_float(4.0),
                                    &format!("rem_euclid_correction_{ni}"),
                                )?,
                                qval,
                                &format!("q_rem_euclid_val_{ni}"),
                            )?
                            .into_float_value();
                        let sin_base = build_vec_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.sin.*",
                            &format!("sin_base_{ni}"),
                            ireg,
                        )?
                        .into_vector_value();
                        let full_range = VectorType::const_vector(&[
                            float_type.const_float(-1.0),
                            float_type.const_float(1.0),
                        ]);
                        // Below part matches the long if/else chain in the
                        // plain interval implementation. Go through the pairs
                        // in reverse and accumulate a nested ternary
                        // expression.
                        const QN_COND_PAIRS: [[(f64, f64); 2]; 4] = [
                            [(0.0, 1.0), (3.0, 2.0)],
                            [(1.0, 2.0), (2.0, 1.0)],
                            [(0.0, 3.0), (3.0, 4.0)],
                            [(1.0, 4.0), (2.0, 3.0)],
                        ];
                        let out_vals = [
                            sin_base,
                            builder.build_shuffle_vector(
                                sin_base,
                                sin_base.get_type().get_undef(),
                                VectorType::const_vector(&[
                                    context.i32_type().const_int(1, false),
                                    context.i32_type().const_int(0, false),
                                ]),
                                &format!("out_val_case_2_{ni}"),
                            )?,
                            builder.build_insert_element(
                                full_range,
                                build_vec_unary_intrinsic(
                                    builder,
                                    &compiler.module,
                                    "llvm.vector.reduce.fmin.*",
                                    &format!("case_3_min_reduce_{ni}"),
                                    sin_base,
                                )?
                                .into_float_value(),
                                context.i32_type().const_int(0, false),
                                &format!("out_val_case_3_{ni}"),
                            )?,
                            builder.build_insert_element(
                                full_range,
                                build_vec_unary_intrinsic(
                                    builder,
                                    &compiler.module,
                                    "llvm.vector.reduce.fmax.*",
                                    &format!("case_3_min_reduce_{ni}"),
                                    sin_base,
                                )?
                                .into_float_value(),
                                context.i32_type().const_int(1, false),
                                &format!("out_val_case_3_{ni}"),
                            )?,
                        ];
                        let out = QN_COND_PAIRS
                            .iter()
                            .zip(out_vals.iter())
                            .enumerate()
                            .try_rfold(
                                full_range,
                                |acc, (i, (pairs, out))| -> Result<VectorValue<'_>, Error> {
                                    let mut conds =
                                        [bool_type.get_poison(), bool_type.get_poison()];
                                    for ((q, n), dst) in pairs.iter().zip(conds.iter_mut()) {
                                        *dst = builder.build_and(
                                            builder.build_float_compare(
                                                FloatPredicate::UEQ,
                                                qval,
                                                float_type.const_float(*q),
                                                &format!("q_compare_{q}_{ni}"),
                                            )?,
                                            builder.build_float_compare(
                                                FloatPredicate::ULT,
                                                nval,
                                                float_type.const_float(*n),
                                                &format!("n_compare_{n}_{ni}"),
                                            )?,
                                            &format!("and_q_n_{ni}"),
                                        )?;
                                    }
                                    Ok(builder
                                        .build_select(
                                            builder.build_or(
                                                conds[0],
                                                conds[1],
                                                &format!("and_q_n_cond_{ni}"),
                                            )?,
                                            *out,
                                            acc,
                                            &format!("case_compare_{i}_{ni}"),
                                        )?
                                        .into_vector_value())
                                },
                            )?
                            .as_basic_value_enum();
                        builder.build_select(
                            build_vec_unary_intrinsic(
                                builder,
                                &compiler.module,
                                "llvm.vector.reduce.and.*",
                                &format!("reduce_call_{ni}"),
                                builder.build_float_compare(
                                    FloatPredicate::UNO,
                                    ireg,
                                    ireg,
                                    &format!("check_empty_{ni}"),
                                )?,
                            )?
                            .into_int_value(),
                            VectorType::const_vector(&[
                                float_type.const_float(f64::NAN),
                                float_type.const_float(f64::NAN),
                            ])
                            .as_basic_value_enum(),
                            out,
                            &format!("reg_{ni}"),
                        )?
                    }
                    Cos => todo!(),
                    Tan => todo!(),
                    Log => todo!(),
                    Exp => todo!(),
                    Floor => todo!(),
                    Not => todo!(),
                },
                Binary(op, _lhs, _rhs) => match op {
                    Add => todo!(),
                    Subtract => todo!(),
                    Multiply => todo!(),
                    Divide => todo!(),
                    Pow => todo!(),
                    Min => todo!(),
                    Max => todo!(),
                    Remainder => todo!(),
                    Less => todo!(),
                    LessOrEqual => todo!(),
                    Equal => todo!(),
                    NotEqual => todo!(),
                    Greater => todo!(),
                    GreaterOrEqual => todo!(),
                    And => todo!(),
                    Or => todo!(),
                },
                Ternary(op, _a, _b, _c) => match op {
                    Choose => todo!(),
                },
            };
            regs.push(reg);
        }
        // Compile instructions to copy the outputs to the out argument.
        let outputs = function
            .get_nth_param(1)
            .ok_or(Error::JitCompilationError(
                "Cannot read output address".to_string(),
            ))?
            .into_pointer_value();
        for (i, reg) in regs[(self.len() - num_roots)..].iter().enumerate() {
            let dst = unsafe {
                builder.build_gep(
                    interval_type,
                    outputs,
                    &[context.i64_type().const_int(i as u64, false)],
                    &format!("output_ptr_{i}"),
                )?
            };
            builder.build_store(dst, reg.into_vector_value())?;
        }
        builder.build_return(None)?;
        compiler.run_passes();
        let engine = compiler
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .map_err(|_| Error::CannotCreateJitModule)?;
        // SAFETY: The signature is correct, and well tested. The function
        // pointer should never be invalidated, because we allocated a dedicated
        // execution engine, with it's own block of executable memory, that will
        // live as long as the function wrapper lives.
        let func = unsafe { engine.get_function(&func_name)? };
        Ok(JitIntervalFn {
            func,
            num_inputs: params.len(),
            num_outputs: num_roots,
            _phantom: PhantomData,
        })
    }
}

fn build_interval_negate<'ctx>(
    input: VectorValue<'ctx>,
    context: &'ctx Context,
    builder: &'ctx Builder,
    index: usize,
    name: &str,
) -> Result<VectorValue<'ctx>, Error> {
    Ok(builder.build_shuffle_vector(
        builder.build_float_neg(input, &format!("negate_{index}"))?,
        input.get_type().get_undef(),
        VectorType::const_vector(&[
            context.i32_type().const_int(1, false),
            context.i32_type().const_int(0, false),
        ]),
        name,
    )?)
}

impl<'ctx, T: NumberType> JitIntervalFn<'ctx, T> {
    pub fn run(&self, inputs: &[[T; 2]], outputs: &mut [[T; 2]]) -> Result<(), Error> {
        if inputs.len() != self.num_inputs {
            return Err(Error::InputSizeMismatch(inputs.len(), self.num_inputs));
        } else if outputs.len() != self.num_outputs {
            return Err(Error::OutputSizeMismatch(outputs.len(), self.num_outputs));
        }
        // SAFETY: We just checked the size of the slices above.
        unsafe { self.run_unchecked(inputs, outputs) };
        Ok(())
    }

    pub unsafe fn run_unchecked(&self, inputs: &[[T; 2]], outputs: &mut [[T; 2]]) {
        // SAFETY: we told the caller it is their responsiblity.
        unsafe {
            self.func
                .call(inputs.as_ptr().cast(), outputs.as_mut_ptr().cast())
        }
    }

    pub fn as_sync(&'ctx self) -> JitIntervalFnSync<'ctx, T> {
        JitIntervalFnSync {
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

impl<'ctx, T: NumberType> JitIntervalFnSync<'ctx, T> {
    pub fn run(&self, inputs: &[[T; 2]], outputs: &mut [[T; 2]]) -> Result<(), Error> {
        if inputs.len() != self.num_inputs {
            return Err(Error::InputSizeMismatch(inputs.len(), self.num_inputs));
        } else if outputs.len() != self.num_outputs {
            return Err(Error::OutputSizeMismatch(outputs.len(), self.num_outputs));
        }
        // SAFETY: We just checked the size of the slices above.
        unsafe { self.run_unchecked(inputs, outputs) };
        Ok(())
    }

    pub unsafe fn run_unchecked(&self, inputs: &[[T; 2]], outputs: &mut [[T; 2]]) {
        // SAFETY: we told the caller it is their responsiblity.
        unsafe { (self.func)(inputs.as_ptr().cast(), outputs.as_mut_ptr().cast()) }
    }
}

#[cfg(test)]
mod test {
    use crate::{Error, JitContext, deftree};

    #[test]
    fn t_jit_interval_negate_f32() {
        let tree = deftree!(- 'x).unwrap();
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<f32>(&context, "x").unwrap();
        // All positive.
        let mut outputs = [[f32::NAN, f32::NAN]];
        eval.run(&[[2.0, 3.0]], &mut outputs)
            .expect("Failed to run the jit function");
        assert_eq!(outputs[0], [-3.0, -2.0]);
        // All negative.
        eval.run(&[[-5.245, -3.123]], &mut outputs)
            .expect("Failed to run the jit function");
        assert_eq!(outputs[0], [3.123, 5.245]);
        // Spanning across zero.
        eval.run(&[[-2.3345, 5.23445]], &mut outputs)
            .expect("Failed to run the jit function");
        assert_eq!(outputs[0], [-5.23445, 2.3345]);
        // Wrong number of inputs / outputs.
        matches!(
            eval.run(&[[-5.245, -3.123], [-2.3345, 5.23445]], &mut outputs),
            Err(Error::InputSizeMismatch(2, 1))
        );
        let mut outputs = [[f32::NAN, f32::NAN], [f32::NAN, f32::NAN]];
        matches!(
            eval.run(&[[-5.245, -3.123]], &mut outputs),
            Err(Error::OutputSizeMismatch(2, 1))
        );
    }

    #[test]
    fn t_jit_interval_negate_f64() {
        let tree = deftree!(- 'x).unwrap();
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<f64>(&context, "x").unwrap();
        // All positive.
        let mut outputs = [[f64::NAN, f64::NAN]];
        eval.run(&[[2.0, 3.0]], &mut outputs)
            .expect("Failed to run the jit function");
        assert_eq!(outputs[0], [-3.0, -2.0]);
        // Test with heap allocation to check alignment
        let inputs = vec![[2.0, 3.0]];
        let mut outputs = vec![[f64::NAN, f64::NAN]];
        eval.run(&inputs, &mut outputs)
            .expect("Failed with heap allocation");
        assert_eq!(outputs[0], [-3.0, -2.0]);
        // All negative.
        eval.run(&[[-5.245, -3.123]], &mut outputs)
            .expect("Failed to run the jit function");
        assert_eq!(outputs[0], [3.123, 5.245]);
        // Spanning across zero.
        eval.run(&[[-2.3345, 5.23445]], &mut outputs)
            .expect("Failed to run the jit function");
        assert_eq!(outputs[0], [-5.23445, 2.3345]);
        // Wrong number of inputs / outputs.
        matches!(
            eval.run(&[[-5.245, -3.123], [-2.3345, 5.23445]], &mut outputs),
            Err(Error::InputSizeMismatch(2, 1))
        );
        let mut outputs = [[f64::NAN, f64::NAN], [f64::NAN, f64::NAN]];
        matches!(
            eval.run(&[[-5.245, -3.123]], &mut outputs),
            Err(Error::OutputSizeMismatch(2, 1))
        );
    }

    #[test]
    fn t_jit_interval_sqrt_f32() {
        let tree = deftree!(sqrt 'x).unwrap();
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<f32>(&context, "x").unwrap();
        let mut outputs = [[f32::NAN, f32::NAN]];
        eval.run(&[[f32::NAN, f32::NAN]], &mut outputs)
            .expect("Failed to run the jit function");
        assert!(outputs[0].iter().all(|v| v.is_nan()));
        {
            let interval = [2.0, 3.0];
            eval.run(&[interval], &mut outputs)
                .expect("Failed to run the jit function");
            assert_eq!(outputs[0], interval.map(|v| v.sqrt()));
        }
        {
            let interval = [-2.0, 3.0];
            eval.run(&[interval], &mut outputs)
                .expect("Failed to run the jit function");
            assert_eq!(outputs[0], [0.0, 3.0f32.sqrt()]);
        }
        {
            let interval = [-3.0, -2.0];
            eval.run(&[interval], &mut outputs)
                .expect("Failed to run the jit function");
            assert!(outputs[0].iter().all(|v| v.is_nan()));
        }
    }

    #[test]
    fn t_jit_interval_sqrt_f64() {
        let tree = deftree!(sqrt 'x).unwrap();
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<f64>(&context, "x").unwrap();
        let mut outputs = [[f64::NAN, f64::NAN]];
        eval.run(&[[f64::NAN, f64::NAN]], &mut outputs)
            .expect("Failed to run the jit function");
        assert!(outputs[0].iter().all(|v| v.is_nan()));
        {
            let interval = [2.0, 3.0];
            eval.run(&[interval], &mut outputs)
                .expect("Failed to run the jit function");
            assert_eq!(outputs[0], interval.map(|v| v.sqrt()));
        }
        {
            let interval = [-2.0, 3.0];
            eval.run(&[interval], &mut outputs)
                .expect("Failed to run the jit function");
            assert_eq!(outputs[0], [0.0, 3.0f64.sqrt()]);
        }
        {
            let interval = [-3.0, -2.0];
            eval.run(&[interval], &mut outputs)
                .expect("Failed to run the jit function");
            assert!(outputs[0].iter().all(|v| v.is_nan()));
        }
    }

    #[test]
    fn t_jit_interval_abs_f32() {
        let tree = deftree!(abs 'x).unwrap();
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<f32>(&context, "x").unwrap();
        let mut outputs = [[f32::NAN, f32::NAN]];
        eval.run(&[[f32::NAN, f32::NAN]], &mut outputs)
            .expect("Failed to run the jit function");
        assert!(outputs[0].iter().all(|v| v.is_nan()));
        {
            let interval = [2.0, 3.0];
            eval.run(&[interval], &mut outputs)
                .expect("Failed to run the jit function");
            assert_eq!(outputs[0], interval);
        }
        {
            let interval = [-2.0, 3.0];
            eval.run(&[interval], &mut outputs)
                .expect("Failed to run the jit function");
            assert_eq!(outputs[0], [0.0, 3.0]);
        }
        {
            let interval = [-3.0, -2.0];
            eval.run(&[interval], &mut outputs)
                .expect("Failed to run the jit function");
            assert_eq!(outputs[0], [2.0, 3.0]);
        }
    }

    #[test]
    fn t_jit_interval_abs_f64() {
        let tree = deftree!(abs 'x).unwrap();
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<f64>(&context, "x").unwrap();
        let mut outputs = [[f64::NAN, f64::NAN]];
        eval.run(&[[f64::NAN, f64::NAN]], &mut outputs)
            .expect("Failed to run the jit function");
        assert!(outputs[0].iter().all(|v| v.is_nan()));
        {
            let interval = [2.0, 3.0];
            eval.run(&[interval], &mut outputs)
                .expect("Failed to run the jit function");
            assert_eq!(outputs[0], interval);
        }
        {
            let interval = [-2.0, 3.0];
            eval.run(&[interval], &mut outputs)
                .expect("Failed to run the jit function");
            assert_eq!(outputs[0], [0.0, 3.0]);
        }
        {
            let interval = [-3.0, -2.0];
            eval.run(&[interval], &mut outputs)
                .expect("Failed to run the jit function");
            assert_eq!(outputs[0], [2.0, 3.0]);
        }
    }

    #[test]
    fn t_jit_interval_sin_f64() {
        use std::f64::consts::{FRAC_PI_2, PI};

        let tree = deftree!(sin 'x).unwrap();
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<f64>(&context, "x").unwrap();
        let mut outputs = [[f64::NAN, f64::NAN]];
        // Test 1: NaN interval should return NaN
        eval.run(&[[f64::NAN, f64::NAN]], &mut outputs)
            .expect("Failed to run the jit function");
        assert!(
            outputs[0][0].is_nan() && outputs[0][1].is_nan(),
            "NaN test failed"
        );
        // Test 2: Point interval (lo == hi)
        eval.run(&[[0.0, 0.0]], &mut outputs).unwrap();
        assert_eq!(outputs[0], [0.0, 0.0], "Point at 0 failed");
        eval.run(&[[FRAC_PI_2, FRAC_PI_2]], &mut outputs).unwrap();
        let expected = FRAC_PI_2.sin();
        assert!(
            (outputs[0][0] - expected).abs() < 1e-10 && (outputs[0][1] - expected).abs() < 1e-10,
            "Point at π/2 failed"
        );
        // Test 3: Small interval in Q0 [0, π/2) - monotonically increasing
        // sin is increasing here, so result should be [sin(lo), sin(hi)]
        let interval = [0.1, 0.4];
        eval.run(&[interval], &mut outputs).unwrap();
        assert_eq!(
            outputs[0],
            [interval[0].sin(), interval[1].sin()],
            "Q0 monotonic increasing failed"
        );
        // Test 4: Small interval in Q1 [π/2, π) - monotonically decreasing
        // sin is decreasing here, so result should be [sin(hi), sin(lo)]
        let interval = [FRAC_PI_2 + 0.1, FRAC_PI_2 + 0.4];
        eval.run(&[interval], &mut outputs).unwrap();
        assert_eq!(
            outputs[0],
            [interval[1].sin(), interval[0].sin()],
            "Q1 monotonic decreasing failed"
        );
        // Test 5: Small interval in Q2 [π, 3π/2) - monotonically decreasing
        let interval = [PI + 0.1, PI + 0.4];
        eval.run(&[interval], &mut outputs).unwrap();
        assert_eq!(
            outputs[0],
            [interval[1].sin(), interval[0].sin()],
            "Q2 monotonic decreasing failed"
        );
        // Test 6: Small interval in Q3 [3π/2, 2π) - monotonically increasing
        let interval = [3.0 * FRAC_PI_2 + 0.1, 3.0 * FRAC_PI_2 + 0.4];
        eval.run(&[interval], &mut outputs).unwrap();
        assert_eq!(
            outputs[0],
            [interval[0].sin(), interval[1].sin()],
            "Q3 monotonic increasing failed"
        );
        // Test 7: Interval crossing π/2 (includes maximum)
        // Should return [min(sin(lo), sin(hi)), 1.0]
        let interval = [0.5, 2.0]; // crosses π/2 ≈ 1.57
        eval.run(&[interval], &mut outputs).unwrap();
        let min_endpoint = interval[0].sin().min(interval[1].sin());
        assert_eq!(
            outputs[0],
            [min_endpoint, 1.0],
            "Interval crossing π/2 (max) failed"
        );
        // Test 8: Interval crossing 3π/2 (includes minimum)
        // Should return [-1.0, max(sin(lo), sin(hi))]
        let interval = [4.0, 5.5]; // crosses 3π/2 ≈ 4.71
        eval.run(&[interval], &mut outputs).unwrap();
        let max_endpoint = interval[0].sin().max(interval[1].sin());
        assert_eq!(
            outputs[0],
            [-1.0, max_endpoint],
            "Interval crossing 3π/2 (min) failed"
        );
        // Test 9: Interval spanning both max and min
        // Should return [-1.0, 1.0]
        let interval = [0.0, 3.0 * FRAC_PI_2 + 0.1]; // Goes past 3π/2 to hit both extrema
        eval.run(&[interval], &mut outputs).unwrap();
        assert_eq!(
            outputs[0],
            [-1.0, 1.0],
            "Interval spanning both extrema failed"
        );
        // Test 10: Interval spanning full period or more
        let interval = [0.0, 2.0 * PI];
        eval.run(&[interval], &mut outputs).unwrap();
        assert_eq!(outputs[0], [-1.0, 1.0], "Full period interval failed");
        let interval = [0.0, 3.0 * PI];
        eval.run(&[interval], &mut outputs).unwrap();
        assert_eq!(outputs[0], [-1.0, 1.0], "Multiple period interval failed");
        // Test 11: Negative intervals - small interval in negative Q0
        let interval = [-0.5, -0.1];
        eval.run(&[interval], &mut outputs).unwrap();
        assert_eq!(
            outputs[0],
            [interval[0].sin(), interval[1].sin()],
            "Negative Q0 failed"
        );
        // Test 12: Negative interval crossing -π/2 (includes minimum at -π/2)
        let interval = [-2.0, -1.0]; // crosses -π/2 ≈ -1.57
        eval.run(&[interval], &mut outputs).unwrap();
        let max_endpoint = interval[0].sin().max(interval[1].sin());
        assert_eq!(
            outputs[0],
            [-1.0, max_endpoint],
            "Negative interval crossing -π/2 failed"
        );
        // Test 13: Symmetric interval around zero
        let interval = [-0.5, 0.5];
        eval.run(&[interval], &mut outputs).unwrap();
        assert_eq!(
            outputs[0],
            [interval[0].sin(), interval[1].sin()],
            "Symmetric around zero failed"
        );
        // Test 14: Large positive values
        let interval = [100.0, 100.5];
        eval.run(&[interval], &mut outputs).unwrap();
        // The exact result depends on quadrant, just check it's valid
        assert!(
            outputs[0][0] >= -1.0 && outputs[0][0] <= 1.0,
            "Large positive lo out of range"
        );
        assert!(
            outputs[0][1] >= -1.0 && outputs[0][1] <= 1.0,
            "Large positive hi out of range"
        );
        assert!(
            outputs[0][0] <= outputs[0][1],
            "Large positive interval not ordered"
        );
        // Test 15: Large negative values
        let interval = [-100.5, -100.0];
        eval.run(&[interval], &mut outputs).unwrap();
        assert!(
            outputs[0][0] >= -1.0 && outputs[0][0] <= 1.0,
            "Large negative lo out of range"
        );
        assert!(
            outputs[0][1] >= -1.0 && outputs[0][1] <= 1.0,
            "Large negative hi out of range"
        );
        assert!(
            outputs[0][0] <= outputs[0][1],
            "Large negative interval not ordered"
        );
        // Test 16: Interval exactly [0, π/2]
        let interval = [0.0, FRAC_PI_2];
        eval.run(&[interval], &mut outputs).unwrap();
        assert_eq!(outputs[0], [0.0, 1.0], "Exact [0, π/2] failed");
        // Test 17: Interval exactly [π/2, π]
        let interval = [FRAC_PI_2, PI];
        eval.run(&[interval], &mut outputs).unwrap();
        assert!(
            (outputs[0][0] - 0.0).abs() < 1e-10,
            "Exact [π/2, π] lower bound failed"
        );
        assert!(
            (outputs[0][1] - 1.0).abs() < 1e-10,
            "Exact [π/2, π] upper bound failed"
        );
        // Test 18: Very small interval (numerical precision test)
        let interval = [1.0, 1.0 + 1e-10];
        eval.run(&[interval], &mut outputs).unwrap();
        assert!(
            outputs[0][0] <= outputs[0][1],
            "Very small interval not ordered"
        );
        assert!(
            (outputs[0][1] - outputs[0][0]).abs() < 1e-9,
            "Very small interval too wide"
        );
        // Test 19: Infinity inputs (should handle gracefully)
        eval.run(&[[f64::INFINITY, f64::INFINITY]], &mut outputs)
            .unwrap();
        // Behavior with infinity is implementation-defined, just check no crash
        eval.run(&[[f64::NEG_INFINITY, f64::NEG_INFINITY]], &mut outputs)
            .unwrap();
        // Behavior with infinity is implementation-defined, just check no crash
        // Test 20: Mixed finite and special values
        eval.run(&[[0.0, f64::INFINITY]], &mut outputs).unwrap();
        // Should likely return [-1, 1] as it spans everything
    }
}
