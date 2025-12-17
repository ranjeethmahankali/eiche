use super::{JitContext, NumberType, build_vec_unary_intrinsic};
use crate::{
    BinaryOp::*, Error, Node::*, TernaryOp::*, Tree, UnaryOp::*, Value, llvm_jit::JitCompiler,
};
use inkwell::{
    AddressSpace, FloatPredicate, OptimizationLevel,
    builder::Builder,
    context::Context,
    execution_engine::JitFunction,
    module::Module,
    types::{IntType, VectorType},
    values::{BasicValueEnum, VectorValue},
};
use std::{ffi::c_void, marker::PhantomData};

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
                    Value::Bool(flag) => BasicValueEnum::VectorValue(VectorType::const_vector(
                        &[bool_type.const_int(if *flag { 1 } else { 0 }, false); 2],
                    )),
                    Value::Scalar(value) => {
                        BasicValueEnum::VectorValue(VectorType::const_vector(&[
                            float_type.const_float(*value),
                            float_type.const_float(*value),
                        ]))
                    }
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
                    Negate => BasicValueEnum::VectorValue(builder.build_shuffle_vector(
                        builder.build_float_neg(
                            regs[*input].into_vector_value(),
                            &format!("negate_{ni}"),
                        )?,
                        interval_type.get_undef(),
                        VectorType::const_vector(&[
                            context.i32_type().const_int(1, false),
                            context.i32_type().const_int(0, false),
                        ]),
                        &format!("reg_{ni}"),
                    )?),
                    Sqrt => {
                        let ireg = regs[*input].into_vector_value();
                        builder.build_select(
                            // Check each lane for NaN, then reduce to check if this interval is empty.
                            build_vec_unary_intrinsic(
                                builder,
                                &compiler.module,
                                "llvm.vector.reduce.mul.*",
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
                            BasicValueEnum::VectorValue(VectorType::const_vector(&[
                                float_type.const_float(f64::NAN),
                                float_type.const_float(f64::NAN),
                            ])),
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
                                    BasicValueEnum::VectorValue(sqrt),
                                    &format!("sqrt_branching_{ni}"),
                                )?
                            },
                            &format!("reg_{ni}"),
                        )?
                    }
                    Abs => todo!(),
                    Sin => todo!(),
                    Cos => todo!(),
                    Tan => todo!(),
                    Log => todo!(),
                    Exp => todo!(),
                    Floor => todo!(),
                    Not => todo!(),
                },
                Binary(op, lhs, rhs) => match op {
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
                Ternary(op, a, b, c) => match op {
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
        unsafe {
            self.func
                .call(inputs.as_ptr().cast(), outputs.as_mut_ptr().cast())
        }
    }
}

fn build_widen_interval<'ctx, T: NumberType>(
    input: VectorValue<'ctx>,
    context: &'ctx Context,
    builder: &'ctx Builder,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    let itype = T::jit_int_type(context);
    let ftype = T::jit_type(context);
    let bits = builder
        .build_bit_cast(
            input,
            itype.vec_type(2),
            &format!("float_to_int_cast_{index}"),
        )?
        .into_vector_value();
    let shifted = builder.build_int_add(
        bits,
        VectorType::const_vector(&[itype.const_int(u64::MAX, true), itype.const_int(1, false)]),
        &format!("shifted_bits_{index}"),
    )?;
    // NaN or +inf check
    let keep = builder.build_or(
        builder.build_float_compare(
            FloatPredicate::UNO,
            input,
            input,
            &format!("is_nan_{index}"),
        )?,
        builder.build_float_compare(
            FloatPredicate::OEQ,
            input,
            VectorType::const_vector(&[
                ftype.const_float(f64::INFINITY),
                ftype.const_float(f64::INFINITY),
            ]),
            &format!("is_inf_{index}"),
        )?,
        &format!("bit_shift_mask_{index}"),
    )?;
    Ok(builder
        .build_bit_cast(
            builder
                .build_select(keep, bits, shifted, "final_bits")?
                .into_vector_value(),
            ftype.vec_type(2),
            "rounded",
        )?
        .into_vector_value())
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
    fn t_jit_interval_sqrt() {
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
}
