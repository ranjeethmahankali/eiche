use super::{JitContext, NumberType};
use crate::{Error, Tree, llvm_jit::JitCompiler};
use inkwell::execution_engine::JitFunction;
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
    ) -> Result<JitIntervalFnSync<'ctx, T>, Error>
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
        todo!("Not Implemented");
    }
}
