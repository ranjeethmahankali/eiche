use inkwell::{
    AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel,
    builder::Builder,
    context::Context,
    execution_engine::JitFunction,
    intrinsics::Intrinsic,
    module::Module,
    types::{BasicTypeEnum, FloatType, VectorType},
    values::{BasicMetadataValueEnum, BasicValueEnum},
};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{JitCompiler, JitContext, NumberType};
use crate::{
    error::Error,
    tree::{BinaryOp::*, Node::*, TernaryOp::*, Tree, UnaryOp::*, Value::*},
};
use std::{marker::PhantomData, mem::size_of};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
type SimdType = __m256d;

#[cfg(target_arch = "aarch64")]
type SimdType = float64x1x4_t;

const SIMD_F32_SIZE: usize = size_of::<SimdType>() / size_of::<f32>();
const SIMD_F64_SIZE: usize = size_of::<SimdType>() / size_of::<f64>();

/// Thin wrapper around a simd floating point value. The union makes it easier
/// to access the individual floating point numbers.
#[repr(C)]
#[derive(Copy, Clone)]
pub union Wfloat {
    valsf32: [f32; SIMD_F32_SIZE],
    valsf64: [f64; SIMD_F64_SIZE],
    reg: SimdType,
}

/// This trait exists to allow reuse of code between f32 and f64 types with
/// generics. i.e. this enables sharing the code to compile and run the compiled
/// tree for both f32 and f64. This could represent a simd vector of f64 values,
/// or that of twice as many f32 values.
pub trait SimdVec<T>
where
    T: Copy,
{
    /// The number of values of type T in the wide simd type.
    const SIMD_VEC_SIZE: usize;

    /// Get a simd vector filled with NaNs.
    fn nan() -> Wfloat;

    /// Set the entry at `idx` to value `val`.
    fn set(&mut self, val: T, idx: usize);

    /// Get the value at index `idx`.
    fn get(&self, idx: usize) -> T;

    /// Get the type of float, either f32 or f64.
    fn float_type(context: &Context) -> FloatType<'_>;

    /// Get a constant value with all entries in the simd vector populated with `val`.
    fn const_float(val: f64, context: &Context) -> BasicValueEnum<'_>;

    /// Get a constant value with all entries in the simd vector populated with `val`.
    fn const_bool(val: bool, context: &Context) -> BasicValueEnum<'_>;
}

impl SimdVec<f32> for Wfloat {
    const SIMD_VEC_SIZE: usize = SIMD_F32_SIZE;

    fn nan() -> Wfloat {
        Wfloat {
            valsf32: [f32::NAN; <Self as SimdVec<f32>>::SIMD_VEC_SIZE],
        }
    }

    fn set(&mut self, val: f32, idx: usize) {
        unsafe { self.valsf32[idx] = val }
    }

    fn get(&self, idx: usize) -> f32 {
        unsafe { self.valsf32[idx] }
    }

    fn float_type(context: &Context) -> FloatType<'_> {
        context.f32_type()
    }

    fn const_float(val: f64, context: &Context) -> BasicValueEnum<'_> {
        BasicValueEnum::VectorValue(VectorType::const_vector(
            &[<Self as SimdVec<f32>>::float_type(context).const_float(val);
                <Self as SimdVec<f32>>::SIMD_VEC_SIZE],
        ))
    }

    fn const_bool(val: bool, context: &Context) -> BasicValueEnum<'_> {
        BasicValueEnum::VectorValue(VectorType::const_vector(
            &[context
                .bool_type()
                .const_int(if val { 1 } else { 0 }, false);
                <Self as SimdVec<f32>>::SIMD_VEC_SIZE],
        ))
    }
}

impl SimdVec<f64> for Wfloat {
    const SIMD_VEC_SIZE: usize = SIMD_F64_SIZE;

    fn nan() -> Wfloat {
        Wfloat {
            valsf64: [f64::NAN; <Self as SimdVec<f64>>::SIMD_VEC_SIZE],
        }
    }

    fn set(&mut self, val: f64, idx: usize) {
        unsafe { self.valsf64[idx] = val }
    }

    fn get(&self, idx: usize) -> f64 {
        unsafe { self.valsf64[idx] }
    }

    fn float_type(context: &Context) -> FloatType<'_> {
        context.f64_type()
    }

    fn const_float(val: f64, context: &Context) -> BasicValueEnum<'_> {
        BasicValueEnum::VectorValue(VectorType::const_vector(
            &[<Self as SimdVec<f64>>::float_type(context).const_float(val);
                <Self as SimdVec<f64>>::SIMD_VEC_SIZE],
        ))
    }

    fn const_bool(val: bool, context: &Context) -> BasicValueEnum<'_> {
        BasicValueEnum::VectorValue(VectorType::const_vector(
            &[context
                .bool_type()
                .const_int(if val { 1 } else { 0 }, false);
                <Self as SimdVec<f64>>::SIMD_VEC_SIZE],
        ))
    }
}

type UnsafeFuncType = unsafe extern "C" fn(*const SimdType, *mut SimdType, u64);

/// Thin wrapper around the compiled native JIT function to do simd evaluations.
pub struct JitSimdFn<'ctx, T>
where
    Wfloat: SimdVec<T>,
    T: NumberType,
{
    func: JitFunction<'ctx, UnsafeFuncType>,
    phantom: PhantomData<T>, // This only exists to specialize the type for type T.
}

pub struct JitSimdFnSync<'ctx, T>
where
    Wfloat: SimdVec<T>,
    T: NumberType,
{
    func: UnsafeFuncType,
    phantom: PhantomData<&'ctx JitSimdFn<'ctx, T>>,
}

unsafe impl<'ctx, T> Sync for JitSimdFnSync<'ctx, T>
where
    Wfloat: SimdVec<T>,
    T: NumberType,
{
}

pub struct JitSimdBuffers<T>
where
    Wfloat: SimdVec<T>,
    T: NumberType,
{
    num_samples: usize,
    num_inputs: usize,
    num_outputs: usize,
    inputs: Vec<Wfloat>,
    outputs: Vec<Wfloat>,
    phantom: PhantomData<T>, // This only exists to specialize the type for type T.
}

impl<'ctx, T> JitSimdFn<'ctx, T>
where
    Wfloat: SimdVec<T>,
    T: NumberType,
{
    pub fn run(&self, buf: &mut JitSimdBuffers<T>) {
        unsafe {
            self.func.call(
                buf.inputs.as_ptr() as *const SimdType,
                buf.outputs.as_mut_ptr() as *mut SimdType,
                buf.num_simd_iters() as u64,
            );
        }
    }

    pub fn as_async(&'ctx self) -> JitSimdFnSync<'ctx, T> {
        JitSimdFnSync {
            func: unsafe { self.func.as_raw() },
            phantom: PhantomData,
        }
    }
}

impl<'ctx, T> JitSimdFnSync<'ctx, T>
where
    Wfloat: SimdVec<T>,
    T: NumberType,
{
    pub fn run(&self, buf: &mut JitSimdBuffers<T>) {
        unsafe {
            (self.func)(
                buf.inputs.as_ptr() as *const SimdType,
                buf.outputs.as_mut_ptr() as *mut SimdType,
                buf.num_simd_iters() as u64,
            );
        }
    }
}

impl<T> JitSimdBuffers<T>
where
    Wfloat: SimdVec<T>,
    T: NumberType,
{
    const SIMD_VEC_SIZE: usize = <Wfloat as SimdVec<T>>::SIMD_VEC_SIZE;

    pub fn new(tree: &Tree) -> Self {
        Self {
            num_samples: 0,
            num_inputs: tree.symbols().len(),
            num_outputs: tree.num_roots(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            phantom: PhantomData,
        }
    }

    pub fn reset_for_tree(&mut self, tree: &Tree) {
        self.num_samples = 0;
        self.num_inputs = tree.symbols().len();
        self.num_outputs = tree.num_roots();
        self.inputs.clear();
        self.outputs.clear();
    }

    fn num_simd_iters(&self) -> usize {
        (self.num_samples / Self::SIMD_VEC_SIZE)
            + if self.num_samples % Self::SIMD_VEC_SIZE > 0 {
                1
            } else {
                0
            }
    }

    /// Push a new set of input values. The length of `sample` is expected to be
    /// the same as the number of symbols in the tree that was compiled to
    /// produce this JIT evaluator. The values are substituted into the
    /// variables in the same order as they are returned by calling
    /// `tree.symbols` on the tree that produced this JIT evaluator.
    pub fn pack(&mut self, sample: &[T]) -> Result<(), Error> {
        if sample.len() != self.num_inputs {
            return Err(Error::InputSizeMismatch(sample.len(), self.num_inputs));
        }
        let index = self.num_samples % Self::SIMD_VEC_SIZE;
        if index == 0 {
            self.inputs.extend(std::iter::repeat_n(
                <Wfloat as SimdVec<T>>::nan(),
                self.num_inputs,
            ));
            self.outputs.extend(std::iter::repeat_n(
                <Wfloat as SimdVec<T>>::nan(),
                self.num_outputs,
            ));
        }
        let inpsize = self.inputs.len();
        for (reg, val) in self.inputs[(inpsize - self.num_inputs)..]
            .iter_mut()
            .zip(sample.iter())
        {
            <Wfloat as SimdVec<T>>::set(reg, *val, index);
        }
        self.num_samples += 1;
        Ok(())
    }

    /// Clear all inputs and outputs.
    pub fn clear(&mut self) {
        self.inputs.clear();
        self.clear_outputs();
    }

    pub fn clear_outputs(&mut self) {
        self.outputs.clear();
        self.num_samples = 0;
    }

    pub fn unpack_outputs(&self) -> impl Iterator<Item = T> {
        debug_assert_eq!(self.outputs.len() % self.num_outputs, 0);
        dbg!(self.num_samples);
        self.outputs
            .chunks_exact(self.num_outputs)
            .flat_map(|chunk| {
                (0..Self::SIMD_VEC_SIZE).flat_map(|lane| {
                    chunk
                        .iter()
                        .map(move |simd| <Wfloat as SimdVec<T>>::get(simd, lane))
                })
            })
            .take(self.num_samples * self.num_outputs)
    }
}

impl Tree {
    /// Compile the tree for doing native simd calculations.
    pub fn jit_compile_array<'ctx, T>(
        &'ctx self,
        context: &'ctx JitContext,
    ) -> Result<JitSimdFn<'ctx, T>, Error>
    where
        Wfloat: SimdVec<T>,
        T: NumberType,
    {
        let num_roots = self.num_roots();
        let symbols = self.symbols();
        let func_name = context.new_func_name::<T, true>();
        let context = &context.inner;
        let compiler = JitCompiler::new(context)?;
        let builder = &compiler.builder;
        let float_type = <Wfloat as SimdVec<T>>::float_type(context);
        let i64_type = context.i64_type();
        let fvec_type = float_type.vec_type(<Wfloat as SimdVec<T>>::SIMD_VEC_SIZE as u32);
        let fptr_type = context.ptr_type(AddressSpace::default());
        let fn_type = context.void_type().fn_type(
            &[fptr_type.into(), fptr_type.into(), i64_type.into()],
            false,
        );
        let function = compiler.module.add_function(&func_name, fn_type, None);
        let start_block = context.append_basic_block(function, "entry");
        let loop_block = context.append_basic_block(function, "loop");
        let end_block = context.append_basic_block(function, "end");
        // Extract the function args.
        builder.position_at_end(start_block);
        let inputs = function
            .get_nth_param(0)
            .ok_or(Error::JitCompilationError("Cannot read inputs".to_string()))?
            .into_pointer_value();
        let eval_len = function
            .get_nth_param(2)
            .ok_or(Error::JitCompilationError(
                "Cannot read number of evaluations".to_string(),
            ))?
            .into_int_value();
        builder.build_unconditional_branch(loop_block)?;
        // Start the loop
        builder.position_at_end(loop_block);
        let phi = builder.build_phi(i64_type, "counter_phi")?;
        phi.add_incoming(&[(&i64_type.const_int(0, false), start_block)]);
        let index = phi.as_basic_value().into_int_value();
        let mut regs: Vec<BasicValueEnum> = Vec::with_capacity(self.len());
        for (ni, node) in self.nodes().iter().enumerate() {
            let reg = match node {
                Constant(val) => match val {
                    Bool(val) => <Wfloat as SimdVec<T>>::const_bool(*val, context),
                    Scalar(val) => <Wfloat as SimdVec<T>>::const_float(*val, context),
                },
                Symbol(label) => {
                    let offset = builder.build_int_add(
                        builder.build_int_mul(
                            index,
                            i64_type.const_int(symbols.len() as u64, false),
                            &format!("input_offset_mul_{label}"),
                        )?,
                        i64_type.const_int(
                            symbols.iter().position(|c| c == label).ok_or(
                                Error::JitCompilationError("Cannot find symbol".to_string()),
                            )? as u64,
                            false,
                        ),
                        &format!("input_offset_add_{label}"),
                    )?;
                    builder.build_load(
                        fvec_type,
                        unsafe {
                            builder.build_gep(fvec_type, inputs, &[offset], &format!("arg_{label}"))
                        }?,
                        &format!("arg_{label}"),
                    )?
                }
                Unary(op, input) => match op {
                    Negate => {
                        BasicValueEnum::VectorValue(builder.build_float_neg(
                            regs[*input].into_vector_value(),
                            &format!("reg_{ni}"),
                        )?)
                    }
                    Sqrt => build_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.sqrt.*",
                        "sqrt_call",
                        regs[*input],
                        fvec_type,
                    )?,
                    Abs => build_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.fabs.*",
                        "abs_call",
                        regs[*input],
                        fvec_type,
                    )?,
                    Sin => build_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.sin.*",
                        "sin_call",
                        regs[*input],
                        fvec_type,
                    )?,
                    Cos => build_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.cos.*",
                        "cos_call",
                        regs[*input],
                        fvec_type,
                    )?,
                    Tan => {
                        let sin = build_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.sin.*",
                            "sin_call",
                            regs[*input],
                            fvec_type,
                        )?;
                        let cos = build_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.cos.*",
                            "cos_call",
                            regs[*input],
                            fvec_type,
                        )?;
                        BasicValueEnum::VectorValue(builder.build_float_div(
                            sin.into_vector_value(),
                            cos.into_vector_value(),
                            &format!("reg_{ni}"),
                        )?)
                    }
                    Log => build_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.log.*",
                        "log_call",
                        regs[*input],
                        fvec_type,
                    )?,
                    Exp => build_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.exp.*",
                        "exp_call",
                        regs[*input],
                        fvec_type,
                    )?,
                    Floor => build_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.floor.*",
                        "floor_call",
                        regs[*input],
                        fvec_type,
                    )?,
                    Not => BasicValueEnum::VectorValue(
                        builder
                            .build_not(regs[*input].into_vector_value(), &format!("reg_{ni}"))?,
                    ),
                },
                Binary(op, lhs, rhs) => match op {
                    Add => BasicValueEnum::VectorValue(builder.build_float_add(
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        &format!("reg_{ni}"),
                    )?),
                    Subtract => BasicValueEnum::VectorValue(builder.build_float_sub(
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        &format!("reg_{ni}"),
                    )?),
                    Multiply => BasicValueEnum::VectorValue(builder.build_float_mul(
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        &format!("reg_{ni}"),
                    )?),
                    Divide => BasicValueEnum::VectorValue(builder.build_float_div(
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        &format!("reg_{ni}"),
                    )?),
                    Pow => build_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.pow.*",
                        "pow_call",
                        regs[*lhs],
                        regs[*rhs],
                        fvec_type,
                    )?,
                    Min => build_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.minnum.*",
                        "min_call",
                        regs[*lhs],
                        regs[*rhs],
                        fvec_type,
                    )?,
                    Max => build_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.maxnum.*",
                        "max_call",
                        regs[*lhs],
                        regs[*rhs],
                        fvec_type,
                    )?,
                    Remainder => BasicValueEnum::VectorValue(builder.build_float_rem(
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        &format!("reg_{ni}"),
                    )?),
                    Less => BasicValueEnum::VectorValue(builder.build_float_compare(
                        FloatPredicate::ULT,
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        &format!("reg_{ni}"),
                    )?),
                    LessOrEqual => BasicValueEnum::VectorValue(builder.build_float_compare(
                        FloatPredicate::ULE,
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        &format!("reg_{ni}"),
                    )?),
                    Equal => BasicValueEnum::VectorValue(builder.build_float_compare(
                        FloatPredicate::UEQ,
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        &format!("reg_{ni}"),
                    )?),
                    NotEqual => BasicValueEnum::VectorValue(builder.build_float_compare(
                        FloatPredicate::UNE,
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        &format!("reg_{ni}"),
                    )?),
                    Greater => BasicValueEnum::VectorValue(builder.build_float_compare(
                        FloatPredicate::UGT,
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        &format!("reg_{ni}"),
                    )?),
                    GreaterOrEqual => BasicValueEnum::VectorValue(builder.build_float_compare(
                        FloatPredicate::UGE,
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        &format!("reg_{ni}"),
                    )?),
                    And => BasicValueEnum::VectorValue(builder.build_and(
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        &format!("reg_{ni}"),
                    )?),
                    Or => BasicValueEnum::VectorValue(builder.build_or(
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        &format!("reg_{ni}"),
                    )?),
                },
                Ternary(op, a, b, c) => match op {
                    Choose => builder.build_select(
                        regs[*a].into_vector_value(),
                        regs[*b].into_vector_value(),
                        regs[*c].into_vector_value(),
                        &format!("reg_{ni}"),
                    )?,
                },
            };
            regs.push(reg);
        }
        // Copy the outputs.
        let outputs = function
            .get_nth_param(1)
            .ok_or(Error::JitCompilationError(
                "Cannot read output address".to_string(),
            ))?
            .into_pointer_value();
        for (i, reg) in regs[(self.len() - num_roots)..].iter().enumerate() {
            let offset = builder.build_int_add(
                builder.build_int_mul(
                    index,
                    i64_type.const_int(num_roots as u64, false),
                    "offset_mul",
                )?,
                i64_type.const_int(i as u64, false),
                "offset_add",
            )?;
            let dst = unsafe {
                builder.build_gep(fvec_type, outputs, &[offset], &format!("output_{i}"))?
            };
            builder.build_store(dst, *reg)?;
        }
        // Check to see if the loop should go on.
        let next = builder.build_int_add(index, i64_type.const_int(1, false), "increment")?;
        phi.add_incoming(&[(&next, loop_block)]);
        let cmp = builder.build_int_compare(IntPredicate::ULT, next, eval_len, "loop-check")?;
        builder.build_conditional_branch(cmp, loop_block, end_block)?;
        // End loop and return.
        builder.position_at_end(end_block);
        builder.build_return(None)?;
        compiler.run_passes();
        let engine = compiler
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .map_err(|_| Error::CannotCreateJitModule)?;
        let func = unsafe { engine.get_function(&func_name)? };
        Ok(JitSimdFn::<T> {
            func,
            phantom: PhantomData,
        })
    }
}

fn build_unary_intrinsic<'ctx>(
    builder: &'ctx Builder,
    module: &'ctx Module,
    name: &'static str,
    call_name: &'static str,
    input: BasicValueEnum<'ctx>,
    vec_type: VectorType<'ctx>,
) -> Result<BasicValueEnum<'ctx>, Error> {
    let intrinsic = Intrinsic::find(name).ok_or(Error::CannotCompileIntrinsic(name))?;
    let intrinsic_fn = intrinsic
        .get_declaration(module, &[BasicTypeEnum::VectorType(vec_type)])
        .ok_or(Error::CannotCompileIntrinsic(name))?;
    builder
        .build_call(
            intrinsic_fn,
            &[BasicMetadataValueEnum::VectorValue(
                input.into_vector_value(),
            )],
            call_name,
        )
        .map_err(|_| Error::CannotCompileIntrinsic(name))?
        .try_as_basic_value()
        .left()
        .ok_or(Error::CannotCompileIntrinsic(name))
}

fn build_binary_intrinsic<'ctx>(
    builder: &'ctx Builder,
    module: &'ctx Module,
    name: &'static str,
    call_name: &'static str,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    vec_type: VectorType<'ctx>,
) -> Result<BasicValueEnum<'ctx>, Error> {
    let intrinsic = Intrinsic::find(name).ok_or(Error::CannotCompileIntrinsic(name))?;
    let intrinsic_fn = intrinsic
        .get_declaration(
            module,
            &[
                BasicTypeEnum::VectorType(vec_type),
                BasicTypeEnum::VectorType(vec_type),
            ],
        )
        .ok_or(Error::CannotCompileIntrinsic(name))?;
    builder
        .build_call(
            intrinsic_fn,
            &[
                BasicMetadataValueEnum::VectorValue(lhs.into_vector_value()),
                BasicMetadataValueEnum::VectorValue(rhs.into_vector_value()),
            ],
            call_name,
        )
        .map_err(|_| Error::CannotCompileIntrinsic(name))?
        .try_as_basic_value()
        .left()
        .ok_or(Error::CannotCompileIntrinsic(name))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{assert_float_eq, deftree, eval::ValueEvaluator, test::Sampler};

    fn check_jit_eval(
        tree: &Tree,
        vardata: &[(char, f64, f64)],
        samples_per_var: usize,
        eps64: f64,
        eps32: f32,
    ) {
        let context = JitContext::default();
        let eval64 = tree.jit_compile_array::<f64>(&context).unwrap();
        let eval32 = tree.jit_compile_array::<f32>(&context).unwrap();
        let mut buf64 = JitSimdBuffers::<f64>::new(tree);
        let mut buf32 = JitSimdBuffers::<f32>::new(tree);
        let mut eval = ValueEvaluator::new(tree);
        let mut sampler = Sampler::new(vardata, samples_per_var, 42);
        let mut expected = Vec::with_capacity(
            tree.num_roots() * usize::pow(samples_per_var, vardata.len() as u32),
        );
        let mut sample32 = Vec::new(); // Temporary storage.
        while let Some(sample) = sampler.next() {
            for (label, value) in vardata.iter().map(|(label, ..)| *label).zip(sample.iter()) {
                eval.set_value(label, (*value).into());
            }
            expected.extend(
                eval.run()
                    .unwrap()
                    .iter()
                    .map(|value| value.scalar().unwrap()),
            );
            buf64.pack(sample).unwrap();
            {
                // f32
                sample32.clear();
                sample32.extend(sample.iter().map(|s| *s as f32));
                buf32.pack(&sample32).unwrap();
            }
        }
        {
            // Run and check f64.
            eval64.run(&mut buf64);
            let actual: Vec<_> = buf64.unpack_outputs().collect();
            assert_eq!(actual.len(), expected.len());
            for (l, r) in actual.iter().zip(expected.iter()) {
                assert_float_eq!(l, r, eps64);
            }
        }
        {
            // Run and check f32.
            eval32.run(&mut buf32);
            let actual: Vec<_> = buf32.unpack_outputs().collect();
            assert_eq!(actual.len(), expected.len());
            for (l, r) in actual.iter().zip(expected.iter()) {
                assert_float_eq!(*l as f64, r, eps32 as f64);
            }
        }
    }

    #[test]
    fn t_mul() {
        check_jit_eval(
            &deftree!(* 'x 'y).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.)],
            20,
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
            1e-6,
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
            0.,
            1e-6,
        );
    }

    #[test]
    fn t_sum_3() {
        check_jit_eval(
            &deftree!(+ (+ 'x 3) (+ 'y 'z)).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            5,
            0.,
            1e-5,
        );
    }

    #[test]
    fn t_sphere() {
        check_jit_eval(
            &deftree!(- (sqrt (+ (pow 'x 2) (+ (pow 'y 2) (pow 'z 2)))) 3).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            0.,
            1e-6,
        );
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
            0.,
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
            1e-6,
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
    fn t_compare_jit_simd() {
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
        let val_jit: Vec<_> = {
            let context = JitContext::default();
            let eval = tree.jit_compile_array(&context).unwrap();
            let mut buf = JitSimdBuffers::new(&tree);
            for q in queries {
                buf.pack(&q).unwrap();
            }
            eval.run(&mut buf);
            buf.unpack_outputs().collect()
        };
        assert_eq!(val_eval.len(), val_jit.len());
        for (l, r) in val_eval.iter().zip(val_jit.iter()) {
            assert_float_eq!(l, r, 1e-15);
        }
    }
}
