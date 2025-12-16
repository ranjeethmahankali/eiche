use super::{JitContext, NumberType};
use crate::{
    BinaryOp::*, Error, Node::*, TernaryOp::*, Tree, UnaryOp::*, Value, llvm_jit::JitCompiler,
};
use inkwell::{
    AddressSpace, execution_engine::JitFunction, types::VectorType, values::BasicValueEnum,
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

trait IntervalNumType: NumberType {
    type IntervalT;
}

impl IntervalNumType for f32 {
    type IntervalT = IntervalType32;
}

impl IntervalNumType for f64 {
    type IntervalT = IntervalType64;
}

#[derive(Clone)]
pub struct JitIntervalFn<'ctx, T>
where
    T: IntervalNumType,
{
    func: JitFunction<'ctx, NativeIntervalFunc>,
    inputs: Box<[T::IntervalT]>,
    outputs: Box<[T::IntervalT]>,
    _phantom: PhantomData<T>,
}

pub struct JitIntervalFnSync<'ctx, T>
where
    T: IntervalNumType,
{
    func: NativeIntervalFunc,
    inputs: Box<[T::IntervalT]>,
    outputs: Box<[T::IntervalT]>,
    _phantom: PhantomData<&'ctx JitIntervalFn<'ctx, T>>,
}

unsafe impl<'ctx, T> Sync for JitIntervalFnSync<'ctx, T> where T: IntervalNumType {}

impl Tree {
    /// JIT compile the tree for interval evaluations.
    pub fn jit_compile_interval<'ctx, T>(
        &'ctx self,
        context: &'ctx JitContext,
        params: &str,
    ) -> Result<JitIntervalFn<'ctx, T>, Error>
    where
        T: IntervalNumType,
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
                    Value::Scalar(value) => BasicValueEnum::VectorValue(VectorType::const_vector(
                        &[float_type.const_float(*value); 2],
                    )),
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
                            &format!("arg_{}", *label),
                        )?
                    };
                    builder.build_load(interval_type, ptr, &format!("val_{}", *label))?
                }
                Unary(op, input) => match op {
                    Negate => todo!(),
                    Sqrt => todo!(),
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
        todo!("Not Implemented");
    }
}
