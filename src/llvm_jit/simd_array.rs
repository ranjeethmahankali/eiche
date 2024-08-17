use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::JitFunction,
    intrinsics::Intrinsic,
    module::Module,
    types::{BasicTypeEnum, FloatType, VectorType},
    values::{BasicMetadataValueEnum, BasicValueEnum},
    AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel,
};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::{marker::PhantomData, mem::size_of};

use super::{JitCompiler, JitContext};
use crate::{
    error::Error,
    tree::{BinaryOp::*, Node::*, TernaryOp::*, Tree, UnaryOp::*, Value::*},
};

type SimdType = __m256d;
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

impl Wfloat {
    fn ptr(&self) -> *const SimdType {
        unsafe { &self.reg as *const SimdType }
    }

    fn ptr_mut(&mut self) -> *mut SimdType {
        unsafe { &mut self.reg as *mut SimdType }
    }
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
    fn float_type<'ctx>(context: &'ctx Context) -> FloatType<'ctx>;

    /// Get a constant value with all entries in the simd vector populated with `val`.
    fn const_float<'ctx>(val: f64, context: &'ctx Context) -> BasicValueEnum<'ctx>;

    /// Get a constant value with all entries in the simd vector populated with `val`.
    fn const_bool<'ctx>(val: bool, context: &'ctx Context) -> BasicValueEnum<'ctx>;
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

    fn float_type<'ctx>(context: &'ctx Context) -> FloatType<'ctx> {
        context.f32_type()
    }

    fn const_float<'ctx>(val: f64, context: &'ctx Context) -> BasicValueEnum<'ctx> {
        BasicValueEnum::VectorValue(VectorType::const_vector(
            &[<Self as SimdVec<f32>>::float_type(context).const_float(val as f64);
                <Self as SimdVec<f32>>::SIMD_VEC_SIZE],
        ))
    }

    fn const_bool<'ctx>(val: bool, context: &'ctx Context) -> BasicValueEnum<'ctx> {
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

    fn float_type<'ctx>(context: &'ctx Context) -> FloatType<'ctx> {
        context.f64_type()
    }

    fn const_float<'ctx>(val: f64, context: &'ctx Context) -> BasicValueEnum<'ctx> {
        BasicValueEnum::VectorValue(VectorType::const_vector(
            &[<Self as SimdVec<f64>>::float_type(context).const_float(val);
                <Self as SimdVec<f64>>::SIMD_VEC_SIZE],
        ))
    }

    fn const_bool<'ctx>(val: bool, context: &'ctx Context) -> BasicValueEnum<'ctx> {
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
    T: Copy,
{
    func: JitFunction<'ctx, UnsafeFuncType>,
    num_inputs: usize,
    num_outputs: usize,
    num_eval: usize,
    inputs: Vec<Wfloat>,
    outputs: Vec<Wfloat>,
    phantom: PhantomData<T>, // This only exists to specialize the type for type T, as T is not used in anything else.
}

impl<'ctx, T> JitSimdFn<'ctx, T>
where
    Wfloat: SimdVec<T>,
    T: Copy,
{
    const SIMD_VEC_SIZE: usize = <Wfloat as SimdVec<T>>::SIMD_VEC_SIZE;

    /// Create a new instance from a compiled native function.
    pub fn create(
        func: JitFunction<'ctx, UnsafeFuncType>,
        num_inputs: usize,
        num_outputs: usize,
    ) -> JitSimdFn<'ctx, T> {
        JitSimdFn::<T> {
            func,
            num_inputs,
            num_outputs,
            num_eval: 0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            phantom: PhantomData::default(),
        }
    }

    /// Push a new set of input values. The length of `sample` is expected to be
    /// the same as the number of symbols in the tree that was compiled to
    /// produce this JIT evaluator. The values are substituted into the
    /// variables in the same order as they are returned by calling
    /// `tree.symbols` on the tree that produced this JIT evaluator.
    pub fn push(&mut self, sample: &[T]) -> Result<(), Error> {
        if sample.len() != self.num_inputs {
            return Err(Error::InputSizeMismatch(sample.len(), self.num_inputs));
        }
        let index = self.num_eval % Self::SIMD_VEC_SIZE;
        if index == 0 {
            self.inputs
                .extend(std::iter::repeat(<Wfloat as SimdVec<T>>::nan()).take(self.num_inputs));
            self.outputs
                .extend(std::iter::repeat(<Wfloat as SimdVec<T>>::nan()).take(self.num_outputs));
        }
        let inpsize = self.inputs.len();
        for (reg, val) in self.inputs[(inpsize - self.num_inputs)..]
            .iter_mut()
            .zip(sample.iter())
        {
            <Wfloat as SimdVec<T>>::set(reg, *val, index);
        }
        self.num_eval += 1;
        return Ok(());
    }

    /// Clear all inputs and outputs.
    pub fn clear(&mut self) {
        self.inputs.clear();
        self.outputs.clear();
        self.num_eval = 0;
    }

    fn num_regs(&self) -> usize {
        (self.num_eval / Self::SIMD_VEC_SIZE)
            + if self.num_eval % Self::SIMD_VEC_SIZE > 0 {
                1
            } else {
                0
            }
    }

    /// Run the evaluator with all the input values that are pushed into it, and
    /// write the output values into the `dst` vector. Any values previously in
    /// `dst` are erased.
    pub fn run(&mut self, dst: &mut Vec<T>) {
        unsafe {
            self.func.call(
                self.inputs[0].ptr(),
                self.outputs[0].ptr_mut(),
                self.num_regs() as u64,
            );
        }
        dst.clear();
        dst.reserve(self.num_outputs * self.num_eval);
        let mut offset = 0;
        let mut num_vals = 0;
        while offset < self.outputs.len() && num_vals < self.num_eval {
            for i in 0..Self::SIMD_VEC_SIZE {
                for wf in &self.outputs[offset..(offset + self.num_outputs)] {
                    dst.push(<Wfloat as SimdVec<T>>::get(&wf, i));
                }
                num_vals += 1;
                if num_vals >= self.num_eval {
                    break;
                }
            }
            offset += self.num_outputs;
        }
    }
}

impl Tree {
    /// Compile the tree for doing native simd calculations.
    pub fn jit_compile_array<'ctx, T>(
        &'ctx self,
        context: &'ctx JitContext,
    ) -> Result<JitSimdFn<T>, Error>
    where
        Wfloat: SimdVec<T>,
        T: Copy,
    {
        const FUNC_NAME: &str = "eiche_func";
        let num_roots = self.num_roots();
        let symbols = self.symbols();
        let context = &context.inner;
        let compiler = JitCompiler::new(context)?;
        let builder = &compiler.builder;
        let float_type = <Wfloat as SimdVec<T>>::float_type(context);
        let i64_type = context.i64_type();
        let fvec_type = float_type.vec_type(<Wfloat as SimdVec<T>>::SIMD_VEC_SIZE as u32);
        let fptr_type = fvec_type.ptr_type(AddressSpace::default());
        let fn_type = context.void_type().fn_type(
            &[fptr_type.into(), fptr_type.into(), i64_type.into()],
            false,
        );
        let function = compiler.module.add_function(FUNC_NAME, fn_type, None);
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
        builder
            .build_unconditional_branch(loop_block)
            .map_err(|e| Error::JitCompilationError(e.to_string()))?;
        // Start the loop
        builder.position_at_end(loop_block);
        let phi = builder
            .build_phi(i64_type, "counter_phi")
            .map_err(|e| Error::JitCompilationError(e.to_string()))?;
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
                    let offset = builder
                        .build_int_add(
                            builder
                                .build_int_mul(
                                    index,
                                    i64_type.const_int(symbols.len() as u64, false),
                                    &format!("input_offset_mul_{}", label),
                                )
                                .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                            i64_type.const_int(
                                symbols.iter().position(|c| c == label).ok_or(
                                    Error::JitCompilationError("Cannot find symbol".to_string()),
                                )? as u64,
                                false,
                            ),
                            &format!("input_offset_add_{}", label),
                        )
                        .map_err(|e| Error::JitCompilationError(e.to_string()))?;
                    builder
                        .build_load(
                            unsafe {
                                builder.build_gep(inputs, &[offset], &format!("arg_{}", label))
                            }
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                            &format!("arg_{}", label),
                        )
                        .map_err(|e| Error::JitCompilationError(e.to_string()))?
                }
                Unary(op, input) => match op {
                    Negate => BasicValueEnum::VectorValue(
                        builder
                            .build_float_neg(
                                regs[*input].into_vector_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
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
                        BasicValueEnum::VectorValue(
                            builder
                                .build_float_div(
                                    sin.into_vector_value(),
                                    cos.into_vector_value(),
                                    &format!("reg_{}", ni),
                                )
                                .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                        )
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
                    Not => BasicValueEnum::VectorValue(
                        builder
                            .build_not(regs[*input].into_vector_value(), &format!("reg_{}", ni))
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                },
                Binary(op, lhs, rhs) => match op {
                    Add => BasicValueEnum::VectorValue(
                        builder
                            .build_float_add(
                                regs[*lhs].into_vector_value(),
                                regs[*rhs].into_vector_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    Subtract => BasicValueEnum::VectorValue(
                        builder
                            .build_float_sub(
                                regs[*lhs].into_vector_value(),
                                regs[*rhs].into_vector_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    Multiply => BasicValueEnum::VectorValue(
                        builder
                            .build_float_mul(
                                regs[*lhs].into_vector_value(),
                                regs[*rhs].into_vector_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    Divide => BasicValueEnum::VectorValue(
                        builder
                            .build_float_div(
                                regs[*lhs].into_vector_value(),
                                regs[*rhs].into_vector_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
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
                    Less => BasicValueEnum::VectorValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::ULT,
                                regs[*lhs].into_vector_value(),
                                regs[*rhs].into_vector_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    LessOrEqual => BasicValueEnum::VectorValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::ULE,
                                regs[*lhs].into_vector_value(),
                                regs[*rhs].into_vector_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    Equal => BasicValueEnum::VectorValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::UEQ,
                                regs[*lhs].into_vector_value(),
                                regs[*rhs].into_vector_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    NotEqual => BasicValueEnum::VectorValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::UNE,
                                regs[*lhs].into_vector_value(),
                                regs[*rhs].into_vector_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    Greater => BasicValueEnum::VectorValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::UGT,
                                regs[*lhs].into_vector_value(),
                                regs[*rhs].into_vector_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    GreaterOrEqual => BasicValueEnum::VectorValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::UGE,
                                regs[*lhs].into_vector_value(),
                                regs[*rhs].into_vector_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    And => BasicValueEnum::VectorValue(
                        builder
                            .build_and(
                                regs[*lhs].into_vector_value(),
                                regs[*rhs].into_vector_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    Or => BasicValueEnum::VectorValue(
                        builder
                            .build_or(
                                regs[*lhs].into_vector_value(),
                                regs[*rhs].into_vector_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                },
                Ternary(op, a, b, c) => match op {
                    Choose => builder
                        .build_select(
                            regs[*a].into_vector_value(),
                            regs[*b].into_vector_value(),
                            regs[*c].into_vector_value(),
                            &format!("reg_{}", ni),
                        )
                        .map_err(|e| Error::JitCompilationError(e.to_string()))?,
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
            let offset = builder
                .build_int_add(
                    builder
                        .build_int_mul(
                            index,
                            i64_type.const_int(num_roots as u64, false),
                            "offset_mul",
                        )
                        .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    i64_type.const_int(i as u64, false),
                    "offset_add",
                )
                .map_err(|e| Error::JitCompilationError(e.to_string()))?;
            let dst = unsafe {
                builder
                    .build_gep(outputs, &[offset], &format!("output_{}", i))
                    .map_err(|e| Error::JitCompilationError(e.to_string()))?
            };
            builder
                .build_store(dst, *reg)
                .map_err(|e| Error::JitCompilationError(e.to_string()))?;
        }
        // Check to see if the loop should go on.
        let next = builder
            .build_int_add(index, i64_type.const_int(1, false), "increment")
            .map_err(|e| Error::JitCompilationError(e.to_string()))?;
        phi.add_incoming(&[(&next, loop_block)]);
        let cmp = builder
            .build_int_compare(IntPredicate::ULT, next, eval_len, "loop-check")
            .map_err(|e| Error::JitCompilationError(e.to_string()))?;
        builder
            .build_conditional_branch(cmp, loop_block, end_block)
            .map_err(|e| Error::JitCompilationError(e.to_string()))?;
        // End loop and return.
        builder.position_at_end(end_block);
        builder
            .build_return(None)
            .map_err(|e| Error::JitCompilationError(e.to_string()))?;
        compiler.run_passes();
        let engine = compiler
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .map_err(|_| Error::CannotCreateJitModule)?;
        let func = unsafe {
            engine
                .get_function(FUNC_NAME)
                .map_err(|e| Error::JitCompilationError(e.to_string()))?
        };
        Ok(JitSimdFn::<T>::create(func, symbols.len(), num_roots))
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
    use crate::{
        deftree,
        eval::ValueEvaluator,
        test::util::{assert_float_eq, Sampler},
    };

    use super::*;

    fn check_jit_eval(tree: &Tree, vardata: &[(char, f64, f64)], samples_per_var: usize, eps: f64) {
        let context = JitContext::default();
        let mut jiteval = tree.jit_compile_array(&context).unwrap();
        let mut eval = ValueEvaluator::new(&tree);
        let mut sampler = Sampler::new(vardata, samples_per_var, 42);
        let mut expected = Vec::with_capacity(
            tree.num_roots() * usize::pow(samples_per_var, vardata.len() as u32),
        );
        let mut actual = Vec::with_capacity(expected.capacity());
        while let Some(sample) = sampler.next() {
            for (label, value) in vardata.iter().map(|(label, ..)| *label).zip(sample.iter()) {
                eval.set_scalar(label, *value);
            }
            expected.extend(
                eval.run()
                    .unwrap()
                    .iter()
                    .map(|value| value.scalar().unwrap()),
            );
            jiteval.push(sample).unwrap();
        }
        jiteval.run(&mut actual);
        assert_eq!(actual.len(), expected.len());
        for (l, r) in actual.iter().zip(expected.iter()) {
            assert_float_eq!(l, r, eps);
        }
    }

    #[test]
    fn t_mul() {
        check_jit_eval(
            &deftree!(* x y).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.)],
            20,
            0.,
        );
    }

    #[test]
    fn t_prod_sum() {
        check_jit_eval(
            &deftree!(concat (+ x y) (* x y)).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.)],
            100,
            0.,
        );
    }

    #[test]
    fn t_sub_div() {
        check_jit_eval(
            &deftree!(concat (- x y) (/ x y)).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.)],
            20,
            0.,
        );
    }

    #[test]
    fn t_pow() {
        check_jit_eval(&deftree!(pow x 2).unwrap(), &[('x', -10., -10.)], 100, 0.);
    }

    #[test]
    fn t_sqrt() {
        check_jit_eval(&deftree!(sqrt x).unwrap(), &[('x', 0.01, 10.)], 100, 0.);
    }

    #[test]
    fn t_circle() {
        check_jit_eval(
            &deftree!(- (sqrt (+ (pow x 2) (pow y 2))) 3).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.)],
            20,
            0.,
        );
    }

    #[test]
    fn t_sum_3() {
        check_jit_eval(
            &deftree!(+ (+ x 3) (+ y z)).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            5,
            0.,
        );
    }

    #[test]
    fn t_sphere() {
        check_jit_eval(
            &deftree!(- (sqrt (+ (pow x 2) (+ (pow y 2) (pow z 2)))) 3).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            0.,
        )
    }

    #[test]
    fn t_negate() {
        check_jit_eval(
            &deftree!(* (- x) (+ y z)).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            0.,
        );
    }

    #[test]
    fn t_abs() {
        check_jit_eval(
            &deftree!(* (abs x) (+ (abs y) (abs z))).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            0.,
        );
    }

    #[test]
    fn t_trigonometry() {
        check_jit_eval(
            &deftree!(/ (+ (sin x) (cos y)) (+ 0.27 (pow (tan z) 2))).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            1e-14,
        );
    }

    #[test]
    fn t_log_exp() {
        check_jit_eval(
            &deftree!(/ (+ 1 (log x)) (+ 1 (exp y))).unwrap(),
            &[('x', 0.1, 5.), ('y', 0.1, 5.)],
            10,
            0.,
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
        );
    }

    #[test]
    fn t_choose() {
        check_jit_eval(
            &deftree!(if (> x 0) x (-x)).unwrap(),
            &[('x', -10., 10.)],
            100,
            0.,
        );
        check_jit_eval(
            &deftree!(if (< x 0) (- x) x).unwrap(),
            &[('x', -10., 10.)],
            100,
            0.,
        );
    }

    #[test]
    fn t_or_and() {
        check_jit_eval(
            &deftree!(if (and (> x 0) (< x 1)) (* 2 x) 1).unwrap(),
            &[('x', -3., 3.)],
            100,
            0.,
        );
    }

    #[test]
    fn t_not() {
        check_jit_eval(
            &deftree!(if (not (> x 0)) (- (pow x 3) (pow y 3)) (+ (pow x 2) (pow y 2))).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.)],
            100,
            1e-14,
        );
    }
}

#[cfg(test)]
mod perft {
    use super::*;
    use crate::{
        dedup::Deduplicater,
        deftree,
        eval::ValueEvaluator,
        prune::Pruner,
        test::util::assert_float_eq,
        // test::util::assert_float_eq,
        tree::{min, MaybeTree, Tree},
    };
    use rand::{rngs::StdRng, SeedableRng};
    use std::time::{Duration, Instant};

    fn _sample_range(range: (f64, f64), rng: &mut StdRng) -> f64 {
        use rand::Rng;
        range.0 + rng.gen::<f64>() * (range.1 - range.0)
    }
    const _RADIUS_RANGE: (f64, f64) = (0.2, 2.);
    const _X_RANGE: (f64, f64) = (0., 100.);
    const _Y_RANGE: (f64, f64) = (0., 100.);
    const _Z_RANGE: (f64, f64) = (0., 100.);
    const _N_SPHERES: usize = 5000;
    const _N_QUERIES: usize = 5000;

    fn _sphere_union() -> Tree {
        let mut rng = StdRng::seed_from_u64(42);
        let mut make_sphere = || -> MaybeTree {
            deftree!(- (sqrt (+ (+
                                 (pow (- x (const _sample_range(_X_RANGE, &mut rng))) 2)
                                 (pow (- y (const _sample_range(_Y_RANGE, &mut rng))) 2))
                              (pow (- z (const _sample_range(_Z_RANGE, &mut rng))) 2)))
                     (const _sample_range(_RADIUS_RANGE, &mut rng)))
        };
        let mut tree = make_sphere();
        for _ in 1.._N_SPHERES {
            tree = min(tree, make_sphere());
        }
        let tree = tree.unwrap();
        assert_eq!(tree.dims(), (1, 1));
        tree
    }

    fn _benchmark_eval(
        values: &mut Vec<f64>,
        queries: &[[f64; 3]],
        eval: &mut ValueEvaluator,
    ) -> Duration {
        let before = Instant::now();
        values.extend(queries.iter().map(|coords| {
            eval.set_scalar('x', coords[0]);
            eval.set_scalar('y', coords[1]);
            eval.set_scalar('z', coords[2]);
            let results = eval.run().unwrap();
            results[0].scalar().unwrap()
        }));
        Instant::now() - before
    }

    fn _benchmark_jit(
        values: &mut Vec<f64>,
        queries: &[[f64; 3]],
        eval: &mut JitSimdFn<f64>,
    ) -> Duration {
        for q in queries {
            eval.push(q).unwrap();
        }
        let before = Instant::now();
        eval.run(values);
        Instant::now() - before
    }

    // Run this function to bench mark the performance
    fn _t_perft() {
        let mut rng = StdRng::seed_from_u64(234);
        let queries: Vec<[f64; 3]> = (0.._N_QUERIES)
            .map(|_| {
                [
                    _sample_range(_X_RANGE, &mut rng),
                    _sample_range(_Y_RANGE, &mut rng),
                    _sample_range(_Z_RANGE, &mut rng),
                ]
            })
            .collect();
        let before = Instant::now();
        let tree = {
            let mut dedup = Deduplicater::new();
            let mut pruner = Pruner::new();
            _sphere_union()
                .fold()
                .unwrap()
                .deduplicate(&mut dedup)
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
        };
        println!(
            "Tree creation time: {}ms",
            (Instant::now() - before).as_millis()
        );
        let mut values1: Vec<f64> = Vec::with_capacity(_N_QUERIES);
        let mut eval = ValueEvaluator::new(&tree);
        let evaltime = _benchmark_eval(&mut values1, &queries, &mut eval);
        println!("ValueEvaluator time: {}ms", evaltime.as_millis());
        let mut values2: Vec<f64> = Vec::with_capacity(_N_QUERIES);
        let context = JitContext::default();
        let mut jiteval = {
            let before = Instant::now();
            let jiteval = tree.jit_compile_array(&context).unwrap();
            println!(
                "Compilation time: {}ms",
                (Instant::now() - before).as_millis()
            );
            jiteval
        };
        let jittime = _benchmark_jit(&mut values2, &queries, &mut jiteval);
        println!("Jit time: {}ms", jittime.as_millis());
        let ratio = evaltime.as_millis() as f64 / jittime.as_millis() as f64;
        println!("Ratio: {}", ratio);
        assert_eq!(values1.len(), values2.len());
        for (l, r) in values1.iter().zip(values2.iter()) {
            assert_float_eq!(l, r, 1e-15);
        }
    }
}
