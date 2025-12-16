use super::{JitContext, NumberType};
use crate::{
    BinaryOp::*, Error, Node::*, TernaryOp::*, Tree, UnaryOp::*, Value, llvm_jit::JitCompiler,
};
use inkwell::{
    AddressSpace, OptimizationLevel, execution_engine::JitFunction, types::VectorType,
    values::BasicValueEnum,
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

#[cfg(target_arch = "aarch64")]
type IntervalType64 = float64x2_t;
#[cfg(target_arch = "aarch64")]
type IntervalType32 = float32x2_t;

pub type NativeIntervalFunc = unsafe extern "C" fn(*const c_void, *mut c_void);

#[derive(Clone)]
pub struct JitIntervalFn<'ctx, T>
where
    T: NumberType,
{
    func: JitFunction<'ctx, NativeIntervalFunc>,
    inputs: Box<[[T; 2]]>,
    outputs: Box<[[T; 2]]>,
    _phantom: PhantomData<T>,
}

pub struct JitIntervalFnSync<'ctx, T>
where
    T: NumberType,
{
    func: NativeIntervalFunc,
    inputs: Box<[[T; 2]]>,
    outputs: Box<[[T; 2]]>,
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
        let mut regs = Vec::<BasicValueEnum>::with_capacity(self.len());
        for (ni, node) in self.nodes().iter().enumerate() {
            let reg = match node {
                Constant(value) => match value {
                    Value::Bool(flag) => BasicValueEnum::VectorValue(VectorType::const_vector(
                        &[bool_type.const_int(if *flag { 1 } else { 0 }, false); 2],
                    )),
                    Value::Scalar(value) => {
                        BasicValueEnum::VectorValue(VectorType::const_vector(&[
                            float_type.const_float(-value),
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
                    BasicValueEnum::VectorValue(
                        builder.build_float_mul(
                            VectorType::const_vector(&[
                                float_type.const_float(-1.0),
                                float_type.const_float(1.0),
                            ]),
                            builder
                                .build_load(interval_type, ptr, &format!("val_{}", *label))?
                                .into_vector_value(),
                            &format!("arg_{}", *label),
                        )?,
                    )
                }
                Unary(op, input) => match op {
                    // For negate all we need to do is swap the vector lanes.
                    Negate => BasicValueEnum::VectorValue(builder.build_shuffle_vector(
                        regs[*input].into_vector_value(),
                        interval_type.get_undef(),
                        VectorType::const_vector(&[
                            context.i32_type().const_int(1, false),
                            context.i32_type().const_int(0, false),
                        ]),
                        &format!("reg_{ni}"),
                    )?),
                    Sqrt => regs[*input],
                    Abs => regs[*input],
                    Sin => regs[*input],
                    Cos => regs[*input],
                    Tan => regs[*input],
                    Log => regs[*input],
                    Exp => regs[*input],
                    Floor => regs[*input],
                    Not => regs[*input],
                },
                Binary(op, lhs, rhs) => match op {
                    Add => regs[*lhs],
                    Subtract => regs[*lhs],
                    Multiply => regs[*lhs],
                    Divide => regs[*rhs],
                    Pow => regs[*rhs],
                    Min => regs[*rhs],
                    Max => regs[*rhs],
                    Remainder => regs[*rhs],
                    Less => regs[*lhs],
                    LessOrEqual => regs[*lhs],
                    Equal => regs[*lhs],
                    NotEqual => regs[*rhs],
                    Greater => regs[*rhs],
                    GreaterOrEqual => regs[*rhs],
                    And => regs[*lhs],
                    Or => regs[*lhs],
                },
                Ternary(op, a, b, c) => match op {
                    Choose => regs[*a],
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
            builder.build_store(
                dst,
                builder.build_float_mul(
                    VectorType::const_vector(&[
                        float_type.const_float(-1.0),
                        float_type.const_float(1.0),
                    ]),
                    reg.into_vector_value(),
                    &format!("output_value_{i}"),
                )?,
            )?;
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
            inputs: vec![[T::nan(); 2]; params.len()].into_boxed_slice(),
            outputs: vec![[T::nan(); 2]; num_roots].into_boxed_slice(),
            _phantom: PhantomData,
        })
    }
}
