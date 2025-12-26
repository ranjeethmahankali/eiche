use super::{
    JitContext, NumberType, build_float_unary_intrinsic, build_vec_binary_intrinsic,
    build_vec_unary_intrinsic,
};
use crate::{
    BinaryOp::*,
    Error, Interval,
    Node::*,
    TernaryOp::*,
    Tree,
    UnaryOp::*,
    Value,
    eval::ValueType,
    interval::IntervalClass,
    llvm_jit::{JitCompiler, build_float_binary_intrinsic},
};
use inkwell::{
    AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel,
    builder::Builder,
    context::Context,
    execution_engine::JitFunction,
    intrinsics::Intrinsic,
    module::Module,
    types::{BasicTypeEnum, VectorType},
    values::{
        BasicMetadataValueEnum, BasicValue, BasicValueEnum, FloatValue, FunctionValue, IntValue,
        VectorValue,
    },
};
use std::{
    f64::consts::{FRAC_PI_2, PI},
    ffi::c_void,
    marker::PhantomData,
};

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

struct Constants<'ctx> {
    // Integers.
    i32_zero: IntValue<'ctx>,
    i32_one: IntValue<'ctx>,
    i32_two: IntValue<'ctx>,
    i32_three: IntValue<'ctx>,
    i32_four: IntValue<'ctx>,
    i32_five: IntValue<'ctx>,
    i32_six: IntValue<'ctx>,
    i32_seven: IntValue<'ctx>,
    // Floats.
    flt_zero: FloatValue<'ctx>,
    flt_one: FloatValue<'ctx>,
    flt_neg_inf: FloatValue<'ctx>,
    flt_inf: FloatValue<'ctx>,
    flt_pi: FloatValue<'ctx>,
    flt_pi_over_2: FloatValue<'ctx>,
    flt_two: FloatValue<'ctx>,
    flt_half: FloatValue<'ctx>,
    // Bools
    bool_false: IntValue<'ctx>,
    bool_true: IntValue<'ctx>,
    bool_poison: IntValue<'ctx>,
    // Intervals.
    interval_zero: VectorValue<'ctx>,
    interval_false_false: VectorValue<'ctx>,
    interval_false_true: VectorValue<'ctx>,
    interval_true_true: VectorValue<'ctx>,
    interval_empty: VectorValue<'ctx>,
    interval_entire: VectorValue<'ctx>,
    interval_neg_one_to_one: VectorValue<'ctx>,
    // Integer vectors.
    ivec_count_to_3: VectorValue<'ctx>,
}

impl<'ctx> Constants<'ctx> {
    fn create<T: NumberType>(context: &'ctx Context) -> Self {
        let i32_type = context.i32_type();
        let flt_type = T::jit_type(context);
        let interval_type = flt_type.vec_type(2);
        let bool_type = context.bool_type();
        Self {
            // Integers.
            i32_zero: i32_type.const_int(0, false),
            i32_one: i32_type.const_int(1, false),
            i32_two: i32_type.const_int(2, false),
            i32_three: i32_type.const_int(3, false),
            i32_four: i32_type.const_int(4, false),
            i32_five: i32_type.const_int(5, false),
            i32_six: i32_type.const_int(6, false),
            i32_seven: i32_type.const_int(7, false),
            // Floats.
            flt_zero: flt_type.const_float(0.0),
            flt_one: flt_type.const_float(1.0),
            flt_neg_inf: flt_type.const_float(f64::NEG_INFINITY),
            flt_inf: flt_type.const_float(f64::INFINITY),
            flt_pi: flt_type.const_float(PI),
            flt_pi_over_2: flt_type.const_float(FRAC_PI_2),
            flt_two: flt_type.const_float(2.0),
            flt_half: flt_type.const_float(0.5),
            // Bools
            bool_false: bool_type.const_int(0, false),
            bool_true: bool_type.const_int(1, false),
            bool_poison: bool_type.get_poison(),
            // Intervals
            interval_zero: interval_type.const_zero(),
            interval_false_false: VectorType::const_vector(&[
                bool_type.const_int(0, false),
                bool_type.const_int(0, false),
            ]),
            interval_false_true: VectorType::const_vector(&[
                bool_type.const_int(0, false),
                bool_type.const_int(1, false),
            ]),
            interval_true_true: VectorType::const_vector(&[
                bool_type.const_int(1, false),
                bool_type.const_int(1, false),
            ]),
            interval_empty: VectorType::const_vector(&[
                flt_type.const_float(f64::NAN),
                flt_type.const_float(f64::NAN),
            ]),
            interval_entire: VectorType::const_vector(&[
                flt_type.const_float(f64::NEG_INFINITY),
                flt_type.const_float(f64::INFINITY),
            ]),
            interval_neg_one_to_one: VectorType::const_vector(&[
                flt_type.const_float(-1.0),
                flt_type.const_float(1.0),
            ]),
            // Integer vectors.
            ivec_count_to_3: VectorType::const_vector(&[
                i32_type.const_int(0, false),
                i32_type.const_int(1, false),
                i32_type.const_int(2, false),
                i32_type.const_int(3, false),
            ]),
        }
    }
}

fn compute_ranges(tree: &Tree) -> Result<Box<[Interval]>, Error> {
    let mut ranges = Vec::with_capacity(tree.len());
    for node in tree.nodes().iter() {
        let out = match node {
            Constant(value) => Interval::from_value(*value)?,
            Symbol(_) => Interval::default(),
            Unary(op, input) => Interval::unary_op(*op, ranges[*input])?,
            Binary(op, lhs, rhs) => Interval::binary_op(*op, ranges[*lhs], ranges[*rhs])?,
            Ternary(op, a, b, c) => Interval::ternary_op(*op, ranges[*a], ranges[*b], ranges[*c])?,
        };
        ranges.push(out);
    }
    assert_eq!(
        ranges.len(),
        tree.len(),
        "Number of nodes and ranges must be equal. This is an assert not an error because this should never happen."
    );
    Ok(ranges.into_boxed_slice())
}

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
        let ranges = compute_ranges(self)?;
        let func_name = context.new_func_name::<T>(Some("interval"));
        let context = &context.inner;
        let compiler = JitCompiler::new(context)?;
        let builder = &compiler.builder;
        let flt_type = T::jit_type(context);
        let interval_type = flt_type.vec_type(2);
        let iptr_type = context.ptr_type(AddressSpace::default());
        let constants = Constants::create::<T>(context);
        let fn_type = context
            .void_type()
            .fn_type(&[iptr_type.into(), iptr_type.into()], false);
        let function = compiler.module.add_function(&func_name, fn_type, None);
        builder.position_at_end(context.append_basic_block(function, "entry"));
        let mut regs = Vec::<BasicValueEnum>::with_capacity(self.len());
        for (index, node) in self.nodes().iter().enumerate() {
            let reg = match node {
                Constant(value) => match value {
                    Value::Bool(flag) => VectorType::const_vector(
                        &[if *flag {
                            constants.bool_true
                        } else {
                            constants.bool_false
                        }; 2],
                    )
                    .as_basic_value_enum(),
                    Value::Scalar(value) => {
                        let val = flt_type.const_float(*value);
                        VectorType::const_vector(&[val, val]).as_basic_value_enum()
                    }
                },
                Symbol(label) => {
                    let inputs = function
                        .get_first_param()
                        .ok_or(Error::JitCompilationError("Cannot read inputs".to_string()))?
                        .into_pointer_value();
                    // # SAFETY: This is unit tested a lot. If this goes wrong we get seg-fault.
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
                    let out = builder.build_load(interval_type, ptr, &format!("arg_{}", *label))?;
                    if let Some(inst) = out.as_instruction_value() {
                        /*
                        Rust arrays only guarantee alignment with the size of T,
                        where as LLVM load / store instructions expect alignment
                        with the vector size (double the size of T). This
                        mismatch can cause a segfault. So we manually set the
                        alignment for this load instruction.
                        */
                        inst.set_alignment(std::mem::size_of::<T>() as u32)
                            .map_err(|msg| {
                                Error::JitCompilationError(format!(
                                    "Cannot set alignment when loading value: {msg}"
                                ))
                            })?;
                    }
                    out
                }
                Unary(op, input) => match op {
                    // For negate all we need to do is swap the vector lanes.
                    Negate => build_interval_negate(
                        regs[*input].into_vector_value(),
                        builder,
                        &constants,
                        index,
                        &format!("reg_{index}"),
                    )?
                    .as_basic_value_enum(),
                    Sqrt => build_interval_sqrt(
                        regs[*input].into_vector_value(),
                        ranges[*input].scalar()?,
                        builder,
                        &compiler.module,
                        &constants,
                        index,
                    )?
                    .as_basic_value_enum(),
                    Abs => build_interval_abs(
                        regs[*input].into_vector_value(),
                        ranges[*input].scalar()?,
                        builder,
                        &compiler.module,
                        &constants,
                        index,
                    )?
                    .as_basic_value_enum(),
                    Sin => build_interval_sin(
                        regs[*input].into_vector_value(),
                        builder,
                        &compiler.module,
                        &constants,
                        index,
                    )?
                    .as_basic_value_enum(),
                    Cos => build_interval_cos(
                        regs[*input].into_vector_value(),
                        builder,
                        &compiler.module,
                        &constants,
                        index,
                    )?
                    .as_basic_value_enum(),
                    Tan => build_interval_tan(
                        regs[*input].into_vector_value(),
                        builder,
                        &compiler.module,
                        &constants,
                        index,
                    )?
                    .as_basic_value_enum(),
                    Log => build_interval_log(
                        regs[*input].into_vector_value(),
                        ranges[*input].scalar()?,
                        builder,
                        &compiler.module,
                        &constants,
                        index,
                    )?
                    .as_basic_value_enum(),
                    Exp => build_vec_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.exp.*",
                        &format!("exp_call_{index}"),
                        regs[*input].into_vector_value(),
                    )?,
                    Floor => build_vec_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.floor.*",
                        &format!("floor_call_{index}"),
                        regs[*input].into_vector_value(),
                    )?,
                    Not => build_interval_not(
                        regs[*input].into_vector_value(),
                        ranges[*input].boolean()?,
                        builder,
                        &compiler.module,
                        &constants,
                        index,
                    )?
                    .as_basic_value_enum(),
                },
                Binary(op, lhs, rhs) => match op {
                    Add => builder
                        .build_float_add(
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                            &format!("reg_{index}"),
                        )?
                        .as_basic_value_enum(),
                    Subtract => builder
                        .build_float_sub(
                            regs[*lhs].into_vector_value(),
                            build_interval_flip(
                                regs[*rhs].into_vector_value(),
                                builder,
                                &constants,
                                index,
                            )?,
                            &format!("reg_{index}"),
                        )?
                        .as_basic_value_enum(),
                    Multiply => build_interval_mul(
                        (
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                        ),
                        (ranges[*lhs].scalar()?, ranges[*rhs].scalar()?),
                        builder,
                        &compiler.module,
                        &constants,
                        index,
                    )?
                    .as_basic_value_enum(),
                    Divide => build_interval_div(
                        (
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                        ),
                        builder,
                        &compiler.module,
                        &constants,
                        index,
                    )?
                    .as_basic_value_enum(),
                    Pow if matches!(self.node(*rhs), Constant(Value::Scalar(2.0))) => {
                        build_interval_square(
                            regs[*lhs].into_vector_value(),
                            ranges[*lhs].scalar()?,
                            builder,
                            &compiler.module,
                            &constants,
                            index,
                        )?
                        .as_basic_value_enum()
                    }
                    Pow => build_interval_pow(
                        (
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                        ),
                        (ranges[*lhs].scalar()?, ranges[*rhs].scalar()?),
                        builder,
                        &compiler.module,
                        index,
                        function,
                        &constants,
                    )?
                    .as_basic_value_enum(),
                    Min => build_vec_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.minnum.*",
                        &format!("min_call_{index}"),
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                    )?,
                    Max => build_vec_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.maxnum.*",
                        &format!("max_call_{index}"),
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                    )?,
                    Remainder => build_interval_remainder(
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        builder,
                        &compiler.module,
                        &constants,
                        index,
                    )?
                    .as_basic_value_enum(),
                    Less => build_interval_less(
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        builder,
                        &compiler.module,
                        &constants,
                        index,
                    )?
                    .as_basic_value_enum(),
                    LessOrEqual => build_interval_less_equal(
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        builder,
                        &compiler.module,
                        &constants,
                        index,
                    )?
                    .as_basic_value_enum(),
                    Equal => build_interval_equal(
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        builder,
                        &compiler.module,
                        &constants,
                        index,
                    )?
                    .as_basic_value_enum(),
                    NotEqual => build_interval_not_equal(
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        builder,
                        &compiler.module,
                        &constants,
                        index,
                    )?
                    .as_basic_value_enum(),
                    Greater => build_interval_less(
                        regs[*rhs].into_vector_value(),
                        regs[*lhs].into_vector_value(),
                        builder,
                        &compiler.module,
                        &constants,
                        index,
                    )?
                    .as_basic_value_enum(),
                    GreaterOrEqual => build_interval_less_equal(
                        regs[*rhs].into_vector_value(),
                        regs[*lhs].into_vector_value(),
                        builder,
                        &compiler.module,
                        &constants,
                        index,
                    )?
                    .as_basic_value_enum(),
                    And => build_interval_and(
                        (
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                        ),
                        (ranges[*lhs].boolean()?, ranges[*rhs].boolean()?),
                        builder,
                        &compiler.module,
                        &constants,
                        index,
                    )?
                    .as_basic_value_enum(),
                    Or => build_interval_or(
                        (
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                        ),
                        (ranges[*lhs].boolean()?, ranges[*rhs].boolean()?),
                        builder,
                        &compiler.module,
                        &constants,
                        index,
                    )?
                    .as_basic_value_enum(),
                },
                Ternary(op, a, b, c) => match op {
                    Choose => build_interval_choose(
                        regs[*a].into_vector_value(),
                        regs[*b].into_vector_value(),
                        regs[*c].into_vector_value(),
                        builder,
                        &compiler.module,
                        &constants,
                        index,
                    )?
                    .as_basic_value_enum(),
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
            // # SAFETY: This is unit tested a lot. If this fails, we segfault.
            let dst = unsafe {
                builder.build_gep(
                    interval_type,
                    outputs,
                    &[context.i64_type().const_int(i as u64, false)],
                    &format!("output_ptr_{i}"),
                )?
            };
            let store_inst = builder.build_store(dst, reg.into_vector_value())?;
            /*
            Rust arrays only guarantee alignment with the size of T, where as
            LLVM load / store instructions expect alignment with the vector size
            (double the size of T). This mismatch can cause a segfault. So we
            manually set the alignment for this store instruction.
            */
            store_inst
                .set_alignment(std::mem::size_of::<T>() as u32)
                .map_err(|e| {
                    Error::JitCompilationError(format!(
                        "Cannot set alignment when storing output: {e}"
                    ))
                })?;
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

fn build_interval_not<'ctx>(
    input: VectorValue<'ctx>,
    range: (bool, bool),
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    match range {
        (true, true) => Ok(constants.interval_false_false),
        (false, false) => Ok(constants.interval_true_true),
        (true, false) | (false, true) => {
            let all_true = build_vec_unary_intrinsic(
                builder,
                module,
                "llvm.vector.reduce.and.*",
                &format!("not_all_true_reduce_{index}"),
                input,
            )?
            .into_int_value();
            let mixed = build_vec_unary_intrinsic(
                builder,
                module,
                "llvm.vector.reduce.xor.*",
                &format!("not_all_true_reduce_{index}"),
                input,
            )?
            .into_int_value();
            Ok(builder
                .build_select(
                    all_true,
                    constants.interval_false_false,
                    builder
                        .build_select(
                            mixed,
                            constants.interval_false_true,
                            constants.interval_true_true,
                            &format!("not_mixed_choice_{index}"),
                        )?
                        .into_vector_value(),
                    &format!("not_{index}"),
                )?
                .into_vector_value())
        }
    }
}

fn build_interval_choose<'ctx>(
    cond: VectorValue<'ctx>,
    iftrue: VectorValue<'ctx>,
    iffalse: VectorValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    let cond_all_true = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.vector.reduce.and.*",
        &format!("choose_cond_all_true_reduce_{index}"),
        cond,
    )?
    .into_int_value();
    let cond_mixed = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.vector.reduce.xor.*",
        &format!("choose_cond_all_true_reduce_{index}"),
        cond,
    )?
    .into_int_value();
    Ok(builder
        .build_select(
            cond_all_true,
            iftrue,
            builder
                .build_select(
                    cond_mixed,
                    match (
                        iftrue.get_type().get_element_type(),
                        iffalse.get_type().get_element_type(),
                    ) {
                        (BasicTypeEnum::FloatType(_), BasicTypeEnum::FloatType(_)) => builder
                            .build_float_mul(
                                constants.interval_neg_one_to_one,
                                build_vec_binary_intrinsic(
                                    builder,
                                    module,
                                    "llvm.maxnum.*",
                                    &format!("choose_max_call_{index}"),
                                    builder.build_float_mul(
                                        constants.interval_neg_one_to_one,
                                        iftrue,
                                        &format!("choose_true_branch_sign_change_{index}"),
                                    )?,
                                    builder.build_float_mul(
                                        constants.interval_neg_one_to_one,
                                        iffalse,
                                        &format!("choose_false_branch_sign_change_{index}"),
                                    )?,
                                )?
                                .into_vector_value(),
                                &format!("choose_sign_revert_{index}"),
                            )?,
                        (BasicTypeEnum::IntType(_), BasicTypeEnum::IntType(_)) => {
                            let combined = builder.build_shuffle_vector(
                                iftrue,
                                iffalse,
                                constants.ivec_count_to_3,
                                &format!("choose_boolean_combine_{index}"),
                            )?;
                            let all_same = builder.build_int_compare(
                                IntPredicate::EQ,
                                build_vec_unary_intrinsic(
                                    builder,
                                    module,
                                    "llvm.vector.reduce.and.*",
                                    &format!("choose_boolean_and_reduce_{index}"),
                                    combined,
                                )?
                                .into_int_value(),
                                build_vec_unary_intrinsic(
                                    builder,
                                    module,
                                    "llvm.vector.reduce.or.*",
                                    &format!("choose_boolean_or_reduce_{index}"),
                                    combined,
                                )?
                                .into_int_value(),
                                &format!("choose_boolean_all_same_check_{index}"),
                            )?;
                            builder
                                .build_select(
                                    all_same,
                                    iftrue,
                                    constants.interval_false_true,
                                    &format!("choose_boolean_out_{index}"),
                                )?
                                .into_vector_value()
                        }
                        _ => return Err(Error::TypeMismatch),
                    },
                    iffalse,
                    &format!("choose_mixed_false_choice_{index}"),
                )?
                .into_vector_value(),
            &format!("choose_{index}"),
        )?
        .into_vector_value())
}

fn build_interval_or<'ctx>(
    inputs: (VectorValue<'ctx>, VectorValue<'ctx>),
    ranges: ((bool, bool), (bool, bool)),
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    let (lhs, rhs) = inputs;
    let ((llo, lhi), (rlo, rhi)) = ranges;
    if (llo && lhi) || (rlo && rhi) {
        return Ok(constants.interval_true_true);
    } else if !llo && !lhi && !rlo && !rhi {
        return Ok(constants.interval_false_false);
    }
    let all_false = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.vector.reduce.and.*",
        &format!("or_all_true_reduce_{index}"),
        builder.build_not(
            builder.build_shuffle_vector(
                lhs,
                rhs,
                constants.ivec_count_to_3,
                &format!("or_all_true_check_{index}"),
            )?,
            &format!("or_all_false_flip_{index}"),
        )?,
    )?
    .into_int_value();
    let one_side_true = builder.build_or(
        build_vec_unary_intrinsic(
            builder,
            module,
            "llvm.vector.reduce.and.*",
            &format!("or_lhs_negate_reduce_{index}"),
            lhs,
        )?
        .into_int_value(),
        build_vec_unary_intrinsic(
            builder,
            module,
            "llvm.vector.reduce.and.*",
            &format!("or_rhs_negate_reduce_{index}"),
            rhs,
        )?
        .into_int_value(),
        &format!("or_one_side_all_false_{index}"),
    )?;
    Ok(builder
        .build_select(
            all_false,
            constants.interval_false_false,
            builder
                .build_select(
                    one_side_true,
                    constants.interval_true_true,
                    constants.interval_false_true,
                    &format!("or_one_side_false_choice_{index}"),
                )?
                .into_vector_value(),
            &format!("or_{index}"),
        )?
        .into_vector_value())
}

fn build_interval_and<'ctx>(
    inputs: (VectorValue<'ctx>, VectorValue<'ctx>),
    ranges: ((bool, bool), (bool, bool)),
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    let (lhs, rhs) = inputs;
    let ((llo, lhi), (rlo, rhi)) = ranges;
    if (!llo && !lhi) || (!rlo && !rhi) {
        return Ok(constants.interval_false_false);
    } else if llo && lhi && rlo && rhi {
        return Ok(constants.interval_true_true);
    }
    let all_true = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.vector.reduce.and.*",
        &format!("and_all_true_reduce_{index}"),
        builder.build_shuffle_vector(
            lhs,
            rhs,
            constants.ivec_count_to_3,
            &format!("and_all_true_check_{index}"),
        )?,
    )?
    .into_int_value();
    let one_side_false = builder.build_or(
        build_vec_unary_intrinsic(
            builder,
            module,
            "llvm.vector.reduce.and.*",
            &format!("and_lhs_negate_reduce_{index}"),
            builder.build_not(lhs, &format!("and_lhs_negate_{index}"))?,
        )?
        .into_int_value(),
        build_vec_unary_intrinsic(
            builder,
            module,
            "llvm.vector.reduce.and.*",
            &format!("and_rhs_negate_reduce_{index}"),
            builder.build_not(rhs, &format!("and_rhs_negate_{index}"))?,
        )?
        .into_int_value(),
        &format!("and_one_side_all_false_{index}"),
    )?;
    Ok(builder
        .build_select(
            all_true,
            constants.interval_true_true,
            builder
                .build_select(
                    one_side_false,
                    constants.interval_false_false,
                    constants.interval_false_true,
                    &format!("and_one_side_false_choice_{index}"),
                )?
                .into_vector_value(),
            &format!("and_{index}"),
        )?
        .into_vector_value())
}

struct InequalityFlags<'ctx> {
    either_empty: IntValue<'ctx>,
    strictly_before: IntValue<'ctx>,
    strictly_after: IntValue<'ctx>,
    touching: VectorValue<'ctx>,
}

fn build_interval_inequality_flags<'ctx>(
    lhs: VectorValue<'ctx>,
    rhs: VectorValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<InequalityFlags<'ctx>, Error> {
    let either_empty = builder.build_or(
        build_check_interval_empty(lhs, builder, module, index)?,
        build_check_interval_empty(rhs, builder, module, index)?,
        &format!("less_equal_either_empty_check_{index}"),
    )?;
    // Compare (-a, b) with (-d, c).
    let masked_lhs = builder.build_float_mul(
        constants.interval_neg_one_to_one,
        lhs,
        &format!("less_sign_adjust_lhs_{index}"),
    )?;
    let masked_rhs = builder.build_float_mul(
        constants.interval_neg_one_to_one,
        build_interval_flip(rhs, builder, constants, index)?,
        &format!("less_equal_sign_adjust_rhs_{index}"),
    )?;
    let cross_compare = builder.build_float_compare(
        FloatPredicate::ULT,
        masked_lhs,
        masked_rhs,
        &format!("less_equal_cross_compare_{index}"),
    )?;
    let strictly_after = builder
        .build_extract_element(
            cross_compare,
            constants.i32_zero,
            &format!("less_equal_a_gt_d_check_{index}"),
        )?
        .into_int_value();
    let strictly_before = builder
        .build_extract_element(
            cross_compare,
            constants.i32_one,
            &format!("less_equal_b_lt_c_check_{index}"),
        )?
        .into_int_value();
    let touching = builder.build_float_compare(
        FloatPredicate::UEQ,
        masked_lhs,
        masked_rhs,
        &format!("less_equal_eq_comp_{index}"),
    )?;
    Ok(InequalityFlags {
        either_empty,
        strictly_before,
        strictly_after,
        touching,
    })
}

fn build_interval_less<'ctx>(
    lhs: VectorValue<'ctx>,
    rhs: VectorValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    if lhs == rhs {
        return Ok(constants.interval_false_false);
    }
    let InequalityFlags {
        either_empty,
        strictly_before,
        strictly_after,
        touching: _touching,
    } = build_interval_inequality_flags(lhs, rhs, builder, module, constants, index)?;
    Ok(builder
        .build_select(
            builder.build_or(
                either_empty,
                strictly_before,
                &format!("less_empty_or_before_check_{index}"),
            )?,
            constants.interval_true_true,
            builder
                .build_select(
                    strictly_after,
                    constants.interval_false_false,
                    constants.interval_false_true,
                    &format!("less_a_gt_d_choice_{index}"),
                )?
                .into_vector_value(),
            &format!("less_{index}"),
        )?
        .into_vector_value())
}

fn build_interval_less_equal<'ctx>(
    lhs: VectorValue<'ctx>,
    rhs: VectorValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    let InequalityFlags {
        either_empty,
        strictly_before,
        strictly_after,
        touching,
    } = build_interval_inequality_flags(lhs, rhs, builder, module, constants, index)?;
    let touching_left = builder
        .build_extract_element(
            touching,
            constants.i32_one,
            &format!("less_equal_touching_left_{index}"),
        )?
        .into_int_value();
    Ok(builder
        .build_select(
            builder.build_or(
                either_empty,
                builder.build_or(
                    strictly_before,
                    touching_left,
                    &format!("less_equal_before_or_touching_{index}"),
                )?,
                &format!("less_equal_empty_or_before_{index}"),
            )?,
            constants.interval_true_true,
            builder
                .build_select(
                    strictly_after,
                    constants.interval_false_false,
                    constants.interval_false_true,
                    &format!("less_a_gt_d_chocie_{index}"),
                )?
                .into_vector_value(),
            &format!("less_b_lt_c_choice_{index}"),
        )?
        .into_vector_value())
}

fn build_interval_equality_flags<'ctx>(
    lhs: VectorValue<'ctx>,
    rhs: VectorValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<(IntValue<'ctx>, IntValue<'ctx>), Error> {
    let either_empty = builder.build_or(
        build_check_interval_empty(lhs, builder, module, index)?,
        build_check_interval_empty(rhs, builder, module, index)?,
        &format!("less_equal_either_empty_check_{index}"),
    )?;
    // Compare (-a, b) with (-d, c).
    let masked_lhs = builder.build_float_mul(
        constants.interval_neg_one_to_one,
        lhs,
        &format!("less_sign_adjust_lhs_{index}"),
    )?;
    let masked_rhs = builder.build_float_mul(
        constants.interval_neg_one_to_one,
        build_interval_flip(rhs, builder, constants, index)?,
        &format!("less_equal_sign_adjust_rhs_{index}"),
    )?;
    let no_overlap = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.vector.reduce.or.*",
        &format!("equal_no_overlap_check_{index}"),
        builder.build_float_compare(
            FloatPredicate::ULT,
            masked_lhs,
            masked_rhs,
            &format!("less_equal_cross_compare_{index}"),
        )?,
    )?
    .into_int_value();
    // To determine if the interval is a singleton, we're checking the masked
    // intervals lanewise. But since we swapped the bounds of rhs, if they're
    // equal lane wise, they must be singletons as long as b >= a and d >= c
    // hold.
    let matching_singleton = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.vector.reduce.and.*",
        &format!("equal_exact_singleton_check_{index}"),
        builder.build_float_compare(
            FloatPredicate::UEQ,
            masked_lhs,
            masked_rhs,
            &format!("equal_exact_singleton_lanewise_check_{index}"),
        )?,
    )?
    .into_int_value();
    let no_overlap = builder.build_or(
        either_empty,
        no_overlap,
        &format!("equal_empty_or_no_overlap_{index}"),
    )?;
    Ok((no_overlap, matching_singleton))
}

fn build_interval_equal<'ctx>(
    lhs: VectorValue<'ctx>,
    rhs: VectorValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    if lhs == rhs {
        return Ok(constants.interval_true_true);
    }
    let (no_overlap, matching_singleton) =
        build_interval_equality_flags(lhs, rhs, builder, module, constants, index)?;
    Ok(builder
        .build_select(
            no_overlap,
            constants.interval_false_false,
            builder
                .build_select(
                    matching_singleton,
                    constants.interval_true_true,
                    constants.interval_false_true,
                    &format!("equal_matching_singleton_select_{index}"),
                )?
                .into_vector_value(),
            &format!("equal_no_overlap_select_{index}"),
        )?
        .into_vector_value())
}

fn build_interval_not_equal<'ctx>(
    lhs: VectorValue<'ctx>,
    rhs: VectorValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    if lhs == rhs {
        return Ok(constants.interval_false_false);
    }
    let (no_overlap, matching_singleton) =
        build_interval_equality_flags(lhs, rhs, builder, module, constants, index)?;
    Ok(builder
        .build_select(
            no_overlap,
            constants.interval_true_true,
            builder
                .build_select(
                    matching_singleton,
                    constants.interval_false_false,
                    constants.interval_false_true,
                    &format!("equal_matching_singleton_select_{index}"),
                )?
                .into_vector_value(),
            &format!("equal_no_overlap_select_{index}"),
        )?
        .into_vector_value())
}

fn build_interval_remainder<'ctx>(
    lhs: VectorValue<'ctx>,
    rhs: VectorValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    let div_result = build_interval_div((lhs, rhs), builder, module, constants, index)?;
    let mul_result = builder.build_float_mul(
        build_vec_unary_intrinsic(
            builder,
            module,
            "llvm.floor.*",
            &format!("remainder_floor_call_{index}"),
            div_result,
        )?
        .into_vector_value(),
        rhs,
        &format!("remainder_floor_mul_{index}"),
    )?;
    Ok(builder.build_float_sub(
        lhs,
        build_interval_flip(mul_result, builder, constants, index)?,
        &format!("remainder_final_sub_{index}"),
    )?)
}

fn build_interval_log<'ctx>(
    input: VectorValue<'ctx>,
    range: (f64, f64),
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    if range.0 > 0.0 && range.1 > 0.0 {
        return Ok(
            build_vec_unary_intrinsic(builder, module, "llvm.log.*", "log_call", input)?
                .into_vector_value(),
        );
    } else if range.0 < 0.0 && range.1 < 0.0 {
        return Ok(constants.interval_empty);
    }
    let is_neg = builder.build_float_compare(
        FloatPredicate::ULE,
        input,
        constants.interval_zero,
        &format!("log_neg_compare_{index}"),
    )?;
    let log_base = build_vec_unary_intrinsic(builder, module, "llvm.log.*", "log_call", input)?
        .into_vector_value();
    Ok(builder
        .build_select(
            builder
                .build_extract_element(
                    is_neg,
                    constants.i32_one,
                    &format!("log_hi_neg_check_{index}"),
                )?
                .into_int_value(),
            constants.interval_empty,
            builder
                .build_select(
                    builder
                        .build_extract_element(
                            is_neg,
                            constants.i32_zero,
                            &format!("log_hi_neg_check_{index}"),
                        )?
                        .into_int_value(),
                    builder.build_insert_element(
                        log_base,
                        constants.flt_neg_inf,
                        constants.i32_zero,
                        &format!("log_range_across_zero_{index}"),
                    )?,
                    log_base,
                    &format!("log_simple_case_{index}"),
                )?
                .into_vector_value(),
            &format!("reg_{index}"),
        )?
        .into_vector_value())
}

fn build_interval_tan<'ctx>(
    input: VectorValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    let lo = builder
        .build_extract_element(input, constants.i32_zero, &format!("tan_width_rhs_{index}"))?
        .into_float_value();
    let width = builder.build_float_sub(
        builder
            .build_extract_element(input, constants.i32_one, &format!("tan_width_lhs_{index}"))?
            .into_float_value(),
        lo,
        &format!("tan_width_{index}"),
    )?;
    let out = builder.build_select(
        builder.build_float_compare(
            FloatPredicate::UGE,
            width,
            constants.flt_pi,
            &format!("tan_pi_compare_{index}"),
        )?,
        constants.interval_entire,
        {
            // Shift lo to an equivalent value in -pi/2 to pi/2.
            let lo = builder.build_float_sub(
                build_float_rem_euclid(
                    builder.build_float_add(
                        lo,
                        constants.flt_pi_over_2,
                        &format!("tan_pi_shift_add_{index}"),
                    )?,
                    constants.flt_pi,
                    builder,
                    constants,
                    &format!("tan_rem_euclid_{index}"),
                    index,
                )?,
                constants.flt_pi_over_2,
                &format!("tan_shifted_lo_{index}"),
            )?;
            let hi = builder.build_float_add(lo, width, &format!("tan_shifted_hi_{index}"))?;
            builder
                .build_select(
                    builder.build_float_compare(
                        FloatPredicate::UGE,
                        hi,
                        constants.flt_pi_over_2,
                        &format!("tan_second_compare_{index}"),
                    )?,
                    constants.interval_entire,
                    {
                        let sin = build_vec_unary_intrinsic(
                            builder,
                            module,
                            "llvm.sin.*",
                            "sin_call",
                            input,
                        )?;
                        let cos = build_vec_unary_intrinsic(
                            builder,
                            module,
                            "llvm.cos.*",
                            "cos_call",
                            input,
                        )?;
                        builder.build_float_div(
                            sin.into_vector_value(),
                            cos.into_vector_value(),
                            &format!("reg_{index}"),
                        )?
                    },
                    &format!("tan_regular_tan_{index}"),
                )?
                .into_vector_value()
        },
        &format!("reg_{index}"),
    )?;
    Ok(builder
        .build_select(
            build_check_interval_empty(input, builder, module, index)?,
            constants.interval_empty.as_basic_value_enum(),
            out,
            &format!("reg_{index}"),
        )?
        .into_vector_value())
}

fn build_interval_cos<'ctx>(
    input: VectorValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    let qinterval = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.floor.*",
        &format!("intermediate_floor_{index}"),
        builder.build_float_div(
            input,
            VectorType::const_vector(&[constants.flt_pi, constants.flt_pi]),
            &format!("div_pi_{index}"),
        )?,
    )?
    .into_vector_value();
    let (lo, hi) = (
        builder
            .build_extract_element(input, constants.i32_zero, &format!("extract_lo_{index}"))?
            .into_float_value(),
        builder
            .build_extract_element(input, constants.i32_one, &format!("extract_lo_{index}"))?
            .into_float_value(),
    );
    let qlo = builder
        .build_extract_element(
            qinterval,
            constants.i32_zero,
            &format!("q_extract_1_{index}"),
        )?
        .into_float_value();
    let nval = builder
        .build_select(
            builder.build_float_compare(
                FloatPredicate::UEQ,
                lo,
                hi,
                &format!("lo_hi_compare_{index}"),
            )?,
            constants.flt_zero,
            builder.build_float_sub(
                builder
                    .build_extract_element(
                        qinterval,
                        constants.i32_one,
                        &format!("q_extract_0_{index}"),
                    )?
                    .into_float_value(),
                qlo,
                &format!("nval_sub_{index}"),
            )?,
            &format!("nval_{index}"),
        )?
        .into_float_value();
    let qval = builder
        .build_select(
            builder.build_float_compare(
                FloatPredicate::UEQ,
                qlo,
                builder.build_float_mul(
                    constants.flt_two,
                    build_float_unary_intrinsic(
                        builder,
                        module,
                        "llvm.floor.*",
                        &format!("intermediate_qval_floor_{index}"),
                        builder.build_float_mul(
                            qlo,
                            constants.flt_half,
                            &format!("qval_half_mul_{index}"),
                        )?,
                    )?
                    .into_float_value(),
                    &format!("qval_doubling_{index}"),
                )?,
                &format!("qval_comparison_{index}"),
            )?,
            constants.flt_zero,
            constants.flt_one,
            &format!("qval_{index}"),
        )?
        .into_float_value();
    let q_zero = builder.build_float_compare(
        FloatPredicate::UEQ,
        qval,
        constants.flt_zero,
        &format!("qval_is_zero_{index}"),
    )?;
    let cos_base = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.cos.*",
        &format!("sin_base_{index}"),
        input,
    )?
    .into_vector_value();
    let out = builder.build_select(
        builder.build_float_compare(
            FloatPredicate::UEQ,
            nval,
            constants.flt_zero,
            &format!("nval_zero_compare_{index}"),
        )?,
        builder
            .build_select(
                q_zero,
                build_interval_flip(cos_base, builder, constants, index)?,
                cos_base,
                &format!("edge_case_1_{index}"),
            )?
            .into_vector_value(),
        builder
            .build_select(
                builder.build_float_compare(
                    FloatPredicate::ULE,
                    nval,
                    constants.flt_one,
                    &format!("nval_one_compare_{index}"),
                )?,
                builder
                    .build_select(
                        q_zero,
                        builder.build_insert_element(
                            constants.interval_neg_one_to_one,
                            build_vec_unary_intrinsic(
                                builder,
                                module,
                                "llvm.vector.reduce.fmax.*",
                                &format!("case_3_max_reduce_{index}"),
                                cos_base,
                            )?
                            .into_float_value(),
                            constants.i32_one,
                            &format!("out_val_case_2_{index}"),
                        )?,
                        builder.build_insert_element(
                            constants.interval_neg_one_to_one,
                            build_vec_unary_intrinsic(
                                builder,
                                module,
                                "llvm.vector.reduce.fmin.*",
                                &format!("case_3_min_reduce_{index}"),
                                cos_base,
                            )?
                            .into_float_value(),
                            constants.i32_zero,
                            &format!("out_val_case_3_{index}"),
                        )?,
                        &format!("nval_cases_{index}"),
                    )?
                    .into_vector_value(),
                constants.interval_neg_one_to_one,
                &format!("out_val_edge_case_0_{index}"),
            )?
            .into_vector_value(),
        &format!("out_val_{index}"),
    )?;
    Ok(builder
        .build_select(
            build_check_interval_empty(input, builder, module, index)?,
            constants.interval_empty.as_basic_value_enum(),
            out,
            &format!("reg_{index}"),
        )?
        .into_vector_value())
}

fn build_interval_sin<'ctx>(
    input: VectorValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    let flt_type = input.get_type().get_element_type().into_float_type();
    let qinterval = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.floor.*",
        &format!("intermediate_floor_{index}"),
        builder.build_float_div(
            input,
            VectorType::const_vector(&[constants.flt_pi_over_2, constants.flt_pi_over_2]),
            &format!("div_pi_{index}"),
        )?,
    )?
    .into_vector_value();
    let (lo, hi) = (
        builder
            .build_extract_element(input, constants.i32_zero, &format!("extract_lo_{index}"))?
            .into_float_value(),
        builder
            .build_extract_element(input, constants.i32_one, &format!("extract_lo_{index}"))?
            .into_float_value(),
    );
    let qlo = builder
        .build_extract_element(
            qinterval,
            constants.i32_zero,
            &format!("q_extract_1_{index}"),
        )?
        .into_float_value();
    let nval = builder
        .build_select(
            builder.build_float_compare(
                FloatPredicate::UEQ,
                lo,
                hi,
                &format!("lo_hi_compare_{index}"),
            )?,
            constants.flt_zero,
            builder.build_float_sub(
                builder
                    .build_extract_element(
                        qinterval,
                        constants.i32_one,
                        &format!("q_extract_0_{index}"),
                    )?
                    .into_float_value(),
                qlo,
                &format!("nval_sub_{index}"),
            )?,
            &format!("nval_{index}"),
        )?
        .into_float_value();
    let qval = build_float_rem_euclid(
        qlo,
        flt_type.const_float(4.0),
        builder,
        constants,
        &format!("q_rem_euclid_val_{index}"),
        index,
    )?;
    let sin_base = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.sin.*",
        &format!("sin_base_{index}"),
        input,
    )?
    .into_vector_value();
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
        build_interval_flip(sin_base, builder, constants, index)?,
        builder.build_insert_element(
            constants.interval_neg_one_to_one,
            build_vec_unary_intrinsic(
                builder,
                module,
                "llvm.vector.reduce.fmin.*",
                &format!("case_3_min_reduce_{index}"),
                sin_base,
            )?
            .into_float_value(),
            constants.i32_zero,
            &format!("out_val_case_3_{index}"),
        )?,
        builder.build_insert_element(
            constants.interval_neg_one_to_one,
            build_vec_unary_intrinsic(
                builder,
                module,
                "llvm.vector.reduce.fmax.*",
                &format!("case_3_max_reduce_{index}"),
                sin_base,
            )?
            .into_float_value(),
            constants.i32_one,
            &format!("out_val_case_3_{index}"),
        )?,
    ];
    let out = QN_COND_PAIRS
        .iter()
        .zip(out_vals.iter())
        .enumerate()
        .try_rfold(
            constants.interval_neg_one_to_one,
            |acc, (i, (pairs, out))| -> Result<VectorValue<'_>, Error> {
                let mut conds = [constants.bool_poison, constants.bool_poison];
                for ((q, n), dst) in pairs.iter().zip(conds.iter_mut()) {
                    *dst = builder.build_and(
                        builder.build_float_compare(
                            FloatPredicate::UEQ,
                            qval,
                            flt_type.const_float(*q),
                            &format!("q_compare_{q}_{index}"),
                        )?,
                        builder.build_float_compare(
                            FloatPredicate::ULT,
                            nval,
                            flt_type.const_float(*n),
                            &format!("n_compare_{n}_{index}"),
                        )?,
                        &format!("and_q_n_{index}"),
                    )?;
                }
                Ok(builder
                    .build_select(
                        builder.build_or(conds[0], conds[1], &format!("and_q_n_cond_{index}"))?,
                        *out,
                        acc,
                        &format!("case_compare_{i}_{index}"),
                    )?
                    .into_vector_value())
            },
        )?
        .as_basic_value_enum();
    Ok(builder
        .build_select(
            build_check_interval_empty(input, builder, module, index)?,
            constants.interval_empty.as_basic_value_enum(),
            out,
            &format!("reg_{index}"),
        )?
        .into_vector_value())
}

fn build_float_vec_powi<'ctx>(
    base: VectorValue<'ctx>,
    exp: IntValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    call_name: &str,
) -> Result<VectorValue<'ctx>, Error> {
    const NAME: &str = "llvm.powi.*";
    let intrinsic = Intrinsic::find(NAME).ok_or(Error::CannotCompileIntrinsic(NAME))?;
    let intrinsic_fn = intrinsic
        .get_declaration(
            module,
            &[
                BasicTypeEnum::VectorType(base.get_type()),
                BasicTypeEnum::IntType(exp.get_type()),
            ],
        )
        .ok_or(Error::CannotCompileIntrinsic(NAME))?;
    builder
        .build_call(
            intrinsic_fn,
            &[
                BasicMetadataValueEnum::VectorValue(base),
                BasicMetadataValueEnum::IntValue(exp),
            ],
            call_name,
        )
        .map_err(|_| Error::CannotCompileIntrinsic(NAME))?
        .try_as_basic_value()
        .left()
        .ok_or(Error::CannotCompileIntrinsic(NAME))
        .map(|v| v.into_vector_value())
}

fn build_interval_square<'ctx>(
    input: VectorValue<'ctx>,
    range: (f64, f64),
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    let sqbase = builder.build_float_mul(input, input, &format!("pow_lhs_square_base_{index}"))?;
    if range.0 >= 0.0 && range.1 >= 0.0 {
        return Ok(sqbase);
    } else if range.0 < 0.0 && range.1 < 0.0 {
        return build_interval_flip(sqbase, builder, constants, index);
    }
    let is_neg = builder.build_float_compare(
        FloatPredicate::ULT,
        input,
        constants.interval_zero,
        &format!("pow_square_case_zero_spanning_check_{index}"),
    )?;
    let is_spanning_zero = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.vector.reduce.xor.*",
        &format!("pow_square_case_zero_spanning_xor_{index}"),
        is_neg,
    )?
    .into_int_value();
    let all_neg = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.vector.reduce.and.*",
        &format!("pow_square_case_zero_spanning_xor_{index}"),
        is_neg,
    )?
    .into_int_value();
    Ok(builder
        .build_select(
            is_spanning_zero,
            builder.build_insert_element(
                constants.interval_zero,
                build_vec_unary_intrinsic(
                    builder,
                    module,
                    "llvm.vector.reduce.fmax.*",
                    &format!("pow_square_case_zero_spanning_max_{index}"),
                    sqbase,
                )?
                .into_float_value(),
                constants.i32_one,
                &format!("pow_square_insert_elem_{index}"),
            )?,
            builder
                .build_select(
                    all_neg,
                    build_interval_flip(sqbase, builder, constants, index)?,
                    sqbase,
                    &format!("pow_square_case_base_neg_cases_{index}"),
                )?
                .into_vector_value(),
            &format!("pow_square_case_{index}"),
        )?
        .into_vector_value())
}

fn build_interval_pow<'ctx>(
    (lhs, rhs): (VectorValue<'ctx>, VectorValue<'ctx>),
    (range_left, _range_right): ((f64, f64), (f64, f64)),
    builder: &'ctx Builder,
    module: &'ctx Module,
    index: usize,
    function: FunctionValue<'ctx>,
    constants: &Constants<'ctx>,
) -> Result<VectorValue<'ctx>, Error> {
    let context = module.get_context();
    let is_any_nan = builder.build_or(
        build_vec_unary_intrinsic(
            builder,
            module,
            "llvm.vector.reduce.or.*",
            &format!("pow_nan_check_{index}"),
            builder.build_float_compare(
                FloatPredicate::UNO,
                lhs,
                lhs,
                &format!("pow_lane_wise_nan_check_{index}"),
            )?,
        )?
        .into_int_value(),
        build_vec_unary_intrinsic(
            builder,
            module,
            "llvm.vector.reduce.or.*",
            &format!("pow_nan_check_{index}"),
            builder.build_float_compare(
                FloatPredicate::UNO,
                rhs,
                rhs,
                &format!("pow_lane_wise_nan_check_{index}"),
            )?,
        )?
        .into_int_value(),
        &format!("pow_any_nan_check_{index}"),
    )?;
    let is_exponent_zero = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.vector.reduce.and.*",
        &format!("pow_zero_check_reduce_{index}"),
        builder.build_float_compare(
            FloatPredicate::UEQ,
            rhs,
            constants.interval_zero,
            &format!("pow_zero_check_{index}"),
        )?,
    )?
    .into_int_value();
    let rhs_floor = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.floor.*",
        &format!("pow_integer_check_floor_call_{index}"),
        rhs,
    )?
    .into_vector_value();
    // Create all the blocks up front before doing conditional branching.
    let simple_bb = context.append_basic_block(function, &format!("pow_simple_case_bb_{index}"));
    let test_square_bb =
        context.append_basic_block(function, &format!("pow_square_test_bb_{index}"));
    let square_bb = context.append_basic_block(function, &format!("pow_square_case_bb{index}"));
    let test_integer_bb =
        context.append_basic_block(function, &format!("pow_integer_case_test_bb_{index}"));
    let integer_bb = context.append_basic_block(function, &format!("pow_integer_case_bb_{index}"));
    let general_bb = context.append_basic_block(function, &format!("pow_general_exponent_{index}"));
    let merge_bb =
        context.append_basic_block(function, &format!("pow_merge_outer_cases_bb_{index}"));
    builder.build_conditional_branch(
        builder.build_or(
            is_any_nan,
            is_exponent_zero,
            &format!("pow_simple_check_or_{index}"),
        )?,
        simple_bb,
        test_square_bb,
    )?;
    let simple_out = {
        builder.position_at_end(simple_bb);
        let out = builder
            .build_select(
                is_any_nan,
                constants.interval_empty,
                VectorType::const_vector(&[constants.flt_one, constants.flt_one]),
                &format!("pow_simple_cases_{index}"),
            )?
            .into_vector_value();
        builder.build_unconditional_branch(merge_bb)?;
        builder.position_at_end(test_square_bb);
        out
    };
    let is_square = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.vector.reduce.and.*",
        &format!("pow_square_check_reduce_{index}"),
        builder.build_float_compare(
            FloatPredicate::UEQ,
            rhs,
            VectorType::const_vector(&[constants.flt_two, constants.flt_two]),
            &format!("pow_square_check_{index}"),
        )?,
    )?
    .into_int_value();
    builder.build_conditional_branch(is_square, square_bb, test_integer_bb)?;
    let square_out = {
        builder.position_at_end(square_bb);
        let out = build_interval_square(lhs, range_left, builder, module, constants, index)?;
        builder.build_unconditional_branch(merge_bb)?;
        builder.position_at_end(test_integer_bb);
        out
    };
    let is_exponent_singleton_integer = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.vector.reduce.and.*",
        &format!("pow_integer_check_reduce_{index}"),
        builder.build_float_compare(
            FloatPredicate::UEQ,
            rhs_floor,
            build_interval_flip(rhs, builder, constants, index)?,
            &format!("pow_integer_check_compare_{index}"),
        )?,
    )?
    .into_int_value();
    builder.build_conditional_branch(is_exponent_singleton_integer, integer_bb, general_bb)?;
    // We now go inside the integer case, and return the last inner block and
    // let it shadow the integer case outside afterwards. Because LLVM wants to
    // know the last inner most block of a phi. So shadowing is helpful.
    let (integer_out, integer_bb) = {
        builder.position_at_end(integer_bb);
        let exponent = builder.build_float_to_signed_int(
            builder
                .build_extract_element(
                    rhs_floor,
                    constants.i32_zero,
                    &format!("pow_extract_floor_{index}"),
                )?
                .into_float_value(),
            context.i32_type(),
            &format!("pow_exponent_to_integer_convert_{index}"),
        )?;
        let is_odd = builder.build_and(
            exponent,
            constants.i32_one,
            &format!("pow_integer_exp_odd_check_{index}"),
        )?;
        let is_even = builder.build_int_compare(
            IntPredicate::EQ,
            is_odd,
            constants.i32_zero,
            &format!("pow_integer_even_check_{index}"),
        )?;
        let is_neg = builder.build_int_compare(
            IntPredicate::SLT,
            exponent,
            constants.i32_zero,
            &format!("pow_integer_neg_check_{index}"),
        )?;
        let is_base_zero = build_vec_unary_intrinsic(
            builder,
            module,
            "llvm.vector.reduce.and.*",
            &format!("pow_integer_base_zero_check_reduce_{index}"),
            builder.build_float_compare(
                FloatPredicate::UEQ,
                lhs,
                constants.interval_zero,
                &format!("pow_integer_base_zero_check_{index}"),
            )?,
        )?
        .into_int_value();
        let even_bb = context.append_basic_block(function, &format!("pow_integer_even_bb_{index}"));
        let odd_bb = context.append_basic_block(function, &format!("pow_integer_else_bb_{index}"));
        let integer_merge_bb =
            context.append_basic_block(function, &format!("pow_odd_even_merge_{index}"));
        builder.build_conditional_branch(is_even, even_bb, odd_bb)?;
        let even_case = {
            builder.position_at_end(even_bb);
            let abs_powi = build_float_vec_powi(
                build_interval_abs(lhs, range_left, builder, module, constants, index)?,
                exponent,
                builder,
                module,
                &format!("pow_even_integer_powi_call_{index}"),
            )?;
            let out = builder
                .build_select(
                    is_neg,
                    builder
                        .build_select(
                            is_base_zero,
                            constants.interval_empty,
                            build_interval_flip(abs_powi, builder, constants, index)?,
                            &format!("pow_even_integer_zero_base_check_{index}"),
                        )?
                        .into_vector_value(),
                    abs_powi,
                    &format!("pow_even_integer_case_{index}"),
                )?
                .into_vector_value();
            builder.build_unconditional_branch(integer_merge_bb)?;
            out
        };
        let odd_case = {
            builder.position_at_end(odd_bb);
            let powi_base = build_float_vec_powi(
                lhs,
                exponent,
                builder,
                module,
                &format!("pow_odd_exp_base_powi_call_{index}"),
            )?;
            let out = builder
                .build_select(
                    is_neg,
                    builder
                        .build_select(
                            is_base_zero,
                            constants.interval_empty,
                            builder
                                .build_select(
                                    builder.build_and(
                                        builder.build_float_compare(
                                            FloatPredicate::ULT,
                                            builder
                                                .build_extract_element(
                                                    lhs,
                                                    constants.i32_zero,
                                                    &format!("pow_odd_exp_extract_lower_{index}"),
                                                )?
                                                .into_float_value(),
                                            constants.flt_zero,
                                            &format!("pow_odd_exp_lower_neg_check_{index}"),
                                        )?,
                                        builder.build_float_compare(
                                            FloatPredicate::UGT,
                                            builder
                                                .build_extract_element(
                                                    lhs,
                                                    constants.i32_one,
                                                    &format!("pow_odd_exp_extract_upper_{index}"),
                                                )?
                                                .into_float_value(),
                                            constants.flt_zero,
                                            &format!("pow_odd_exp_upper_positive_check_{index}"),
                                        )?,
                                        &format!("pow_odd_exp_base_zero_spanning_check_{index}"),
                                    )?,
                                    constants.interval_entire,
                                    build_interval_flip(powi_base, builder, constants, index)?,
                                    &format!("pow_odd_exp_case_{index}"),
                                )?
                                .into_vector_value(),
                            &format!("pow_odd_exp_base_zero_check_{index}"),
                        )?
                        .into_vector_value(),
                    powi_base,
                    &format!("pow_odd_exp_neg_check_{index}"),
                )?
                .into_vector_value();
            builder.build_unconditional_branch(integer_merge_bb)?;
            out
        };
        builder.position_at_end(integer_merge_bb);
        let phi = builder.build_phi(lhs.get_type(), &format!("pow_integer_case_output_{index}"))?;
        phi.add_incoming(&[(&even_case, even_bb), (&odd_case, odd_bb)]);
        builder.build_unconditional_branch(merge_bb)?;
        (phi.as_basic_value().into_vector_value(), integer_merge_bb)
    };
    let general_out: VectorValue<'ctx> = {
        builder.position_at_end(general_bb);
        let lhs = build_vec_binary_intrinsic(
            builder,
            module,
            "llvm.minnum.*",
            &format!("pow_general_domain_adjust_min_call_{index}"),
            build_vec_binary_intrinsic(
                builder,
                module,
                "llvm.maxnum.*",
                &format!("pow_general_domain_adjust_max_call_{index}"),
                lhs,
                constants.interval_zero,
            )?
            .into_vector_value(),
            VectorType::const_vector(&[constants.flt_inf, constants.flt_inf]),
        )?
        .into_vector_value();
        let (a, b) = build_interval_unpack(lhs, builder, constants, "pow_general_case_", index)?;
        let (c, d) = build_interval_unpack(rhs, builder, constants, "pow_general_case_", index)?;
        let ac = build_float_binary_intrinsic(
            builder,
            module,
            "llvm.pow.*",
            &format!("pow_general_pow_ac_{index}"),
            a,
            c,
        )?
        .into_float_value();
        let ad = build_float_binary_intrinsic(
            builder,
            module,
            "llvm.pow.*",
            &format!("pow_general_pow_ad_{index}"),
            a,
            d,
        )?
        .into_float_value();
        let bc = build_float_binary_intrinsic(
            builder,
            module,
            "llvm.pow.*",
            &format!("pow_general_pow_bc_{index}"),
            b,
            c,
        )?
        .into_float_value();
        let bd = build_float_binary_intrinsic(
            builder,
            module,
            "llvm.pow.*",
            &format!("pow_general_pow_bd_{index}"),
            b,
            d,
        )?
        .into_float_value();
        // Extract values from vector for ergonomic use later.
        let rhi_is_neg = builder.build_float_compare(
            FloatPredicate::ULE,
            d,
            constants.flt_zero,
            &format!("pow_general_rhi_neg_check_{index}"),
        )?;
        let lhi_is_zero = builder.build_float_compare(
            FloatPredicate::UEQ,
            b,
            constants.flt_zero,
            &format!("pow_general_lhi_zero_check_{index}"),
        )?;
        let lhi_lt_one = builder.build_float_compare(
            FloatPredicate::ULT,
            b,
            constants.flt_one,
            &format!("pow_general_lhi_lt_one_check_{index}"),
        )?;
        let llo_gt_one = builder.build_float_compare(
            FloatPredicate::UGT,
            a,
            constants.flt_one,
            &format!("pow_general_llo_gt_one_check_{index}"),
        )?;
        let rlo_gt_zero = builder.build_float_compare(
            FloatPredicate::UGT,
            c,
            constants.flt_zero,
            &format!("pow_general_rlo_gt_zero_{index}"),
        )?;
        let out = builder
            .build_select(
                rhi_is_neg,
                builder
                    .build_select(
                        lhi_is_zero,
                        constants.interval_empty,
                        builder
                            .build_select(
                                lhi_lt_one,
                                build_interval_compose(
                                    bd,
                                    ac,
                                    builder,
                                    constants,
                                    "pow_general_case",
                                    index,
                                )?,
                                builder
                                    .build_select(
                                        llo_gt_one,
                                        build_interval_compose(
                                            bc,
                                            ad,
                                            builder,
                                            constants,
                                            "pow_general_case",
                                            index,
                                        )?,
                                        build_interval_compose(
                                            bc,
                                            ac,
                                            builder,
                                            constants,
                                            "pow_general_case",
                                            index,
                                        )?,
                                        &format!("pow_general_mask_choice_llo_gt_one_{index}"),
                                    )?
                                    .into_vector_value(),
                                &format!("pow_general_mask_choice_lhi_lt_one_{index}"),
                            )?
                            .into_vector_value(),
                        &format!("pow_general_mask_choice_lhi_is_zero_{index}"),
                    )?
                    .into_vector_value(),
                builder
                    .build_select(
                        rlo_gt_zero,
                        builder
                            .build_select(
                                lhi_lt_one,
                                build_interval_compose(
                                    ad,
                                    bc,
                                    builder,
                                    constants,
                                    "pow_general_case",
                                    index,
                                )?,
                                builder
                                    .build_select(
                                        llo_gt_one,
                                        build_interval_compose(
                                            ac,
                                            bd,
                                            builder,
                                            constants,
                                            "pow_general_case",
                                            index,
                                        )?,
                                        build_interval_compose(
                                            ad,
                                            bd,
                                            builder,
                                            constants,
                                            "pow_general_case",
                                            index,
                                        )?,
                                        &format!("pow_general_mask_choice_llo_gt_one_{index}"),
                                    )?
                                    .into_vector_value(),
                                &format!("pow_general_mask_choice_rlo_gt_zero_{index}"),
                            )?
                            .into_vector_value(),
                        build_interval_compose(
                            build_float_binary_intrinsic(
                                builder,
                                module,
                                "llvm.minnum.*",
                                &format!("pow_general_case_last_case_adbc_{index}"),
                                ad,
                                bc,
                            )?
                            .into_float_value(),
                            build_float_binary_intrinsic(
                                builder,
                                module,
                                "llvm.maxnum.*",
                                &format!("pow_general_case_last_case_acbd_{index}"),
                                ac,
                                bd,
                            )?
                            .into_float_value(),
                            builder,
                            constants,
                            "pow_general_case",
                            index,
                        )?,
                        &format!("pow_general_mask_choice_rlo_gt_zero_{index}"),
                    )?
                    .into_vector_value(),
                &format!("pow_general_mask_choice_rhi_is_neg_{index}"),
            )?
            .into_vector_value();
        builder.build_unconditional_branch(merge_bb)?;
        out
    };
    builder.position_at_end(merge_bb);
    let phi = builder.build_phi(lhs.get_type(), &format!("outer_branch_phi_{index}"))?;
    phi.add_incoming(&[
        (&simple_out, simple_bb),
        (&square_out, square_bb),
        (&integer_out, integer_bb),
        (&general_out, general_bb),
    ]);
    Ok(phi.as_basic_value().into_vector_value())
}

fn build_interval_abs<'ctx>(
    input: VectorValue<'ctx>,
    range: (f64, f64),
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    if range.0 < 0.0 && range.1 < 0.0 {
        return build_interval_negate(
            input,
            builder,
            constants,
            index,
            &format!("intermediate_1_{index}"),
        );
    } else if range.0 >= 0.0 && range.1 >= 0.0 {
        return Ok(input);
    }
    let lt_zero = builder.build_float_compare(
        FloatPredicate::ULT,
        input,
        constants.interval_zero,
        &format!("lt_zero_{index}"),
    )?;
    Ok(builder
        .build_select(
            builder
                .build_extract_element(
                    lt_zero,
                    constants.i32_one,
                    &format!("first_lt_zero_{index}"),
                )?
                .into_int_value(),
            // (-hi, -lo)
            build_interval_negate(
                input,
                builder,
                constants,
                index,
                &format!("intermediate_1_{index}"),
            )?,
            builder
                .build_select(
                    builder
                        .build_extract_element(
                            lt_zero,
                            constants.i32_zero,
                            &format!("first_lt_zero_{index}"),
                        )?
                        .into_int_value(),
                    // (0.0, max(abs(lo), abs(hi)))
                    builder.build_insert_element(
                        constants.interval_zero,
                        build_vec_unary_intrinsic(
                            builder,
                            module,
                            "llvm.vector.reduce.fmax.*",
                            &format!("fmax_reduce_call_{index}"),
                            build_vec_unary_intrinsic(
                                builder,
                                module,
                                "llvm.fabs.*",
                                &format!("abs_call_{index}"),
                                input,
                            )?
                            .into_vector_value(),
                        )?
                        .into_float_value(),
                        constants.i32_one,
                        &format!("intermediate_2_{index}"),
                    )?,
                    // (lo, hi),
                    input,
                    &format!("intermediate_3_{index}"),
                )?
                .into_vector_value(),
            &format!("reg_{index}"),
        )?
        .into_vector_value())
}

fn build_interval_sqrt<'ctx>(
    input: VectorValue<'ctx>,
    range: (f64, f64),
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    let sqrt = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.sqrt.*",
        &format!("sqrt_call_{index}"),
        input,
    )?
    .into_vector_value();
    if range.1 < 0.0 && range.0 < 0.0 {
        Ok(constants.interval_empty)
    } else if range.1 >= 0.0 && range.0 >= 0.0 {
        Ok(sqrt)
    } else {
        let is_neg = builder.build_float_compare(
            FloatPredicate::ULT,
            input,
            constants.interval_zero,
            &format!("lt_zero_{index}"),
        )?;
        let all_neg = build_vec_unary_intrinsic(
            builder,
            module,
            "llvm.vector.reduce.and.*",
            &format!("sqrt_all_neg_check_{index}"),
            is_neg,
        )?
        .into_int_value();
        let spanning_zero = build_vec_unary_intrinsic(
            builder,
            module,
            "llvm.vector.reduce.xor.*",
            &format!("sqrt_all_neg_check_{index}"),
            is_neg,
        )?
        .into_int_value();
        Ok(builder
            .build_select(
                all_neg,
                constants.interval_empty.as_basic_value_enum(),
                builder.build_select(
                    spanning_zero,
                    builder.build_insert_element(
                        sqrt,
                        constants.flt_zero,
                        constants.i32_zero,
                        &format!("sqrt_domain_clipping_{index}"),
                    )?,
                    sqrt,
                    &format!("sqrt_branching_{index}"),
                )?,
                &format!("reg_{index}"),
            )?
            .into_vector_value())
    }
}

fn build_interval_flip<'ctx>(
    input: VectorValue<'ctx>,
    builder: &'ctx Builder,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    Ok(builder.build_shuffle_vector(
        input,
        input.get_type().get_undef(),
        VectorType::const_vector(&[constants.i32_one, constants.i32_zero]),
        &format!("out_val_case_2_{index}"),
    )?)
}

fn build_interval_compose<'ctx>(
    lo: FloatValue<'ctx>,
    hi: FloatValue<'ctx>,
    builder: &'ctx Builder,
    constants: &Constants<'ctx>,
    suffix: &str,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    Ok(builder.build_insert_element(
        builder.build_insert_element(
            constants.interval_zero,
            lo,
            constants.i32_zero,
            &format!("interval_compose_{suffix}_lo_{index}"),
        )?,
        hi,
        constants.i32_one,
        &format!("interval_compose_{suffix}_hi_{index}"),
    )?)
}

fn build_interval_unpack<'ctx>(
    input: VectorValue<'ctx>,
    builder: &'ctx Builder,
    constants: &Constants<'ctx>,
    prefix: &str,
    index: usize,
) -> Result<(FloatValue<'ctx>, FloatValue<'ctx>), Error> {
    let lo = builder
        .build_extract_element(
            input,
            constants.i32_zero,
            &format!("{prefix}_interval_unpack_left_{index}"),
        )?
        .into_float_value();
    let hi = builder
        .build_extract_element(
            input,
            constants.i32_one,
            &format!("{prefix}_interval_unpack_right_{index}"),
        )?
        .into_float_value();
    Ok((lo, hi))
}

fn build_interval_div<'ctx>(
    inputs: (VectorValue<'ctx>, VectorValue<'ctx>),
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    use crate::interval::IntervalClass::*;
    let i32_type = module.get_context().i32_type();
    let (lhs, rhs) = inputs;
    let mask = builder.build_int_add(
        builder.build_int_mul(
            build_interval_classify(lhs, builder, module, constants, index)?,
            constants.i32_seven,
            &format!("interval_div_mask_imul_{index}"),
        )?,
        build_interval_classify(rhs, builder, module, constants, index)?,
        &format!("interval_div_mask_{index}"),
    )?;
    let straight = builder.build_float_div(lhs, rhs, &format!("interval_div_straight_{index}"))?;
    let cross = builder.build_float_div(
        lhs,
        build_interval_flip(rhs, builder, constants, index)?,
        &format!("interval_div_cross_{index}"),
    )?;
    let combos = builder.build_shuffle_vector(
        straight,
        cross,
        constants.ivec_count_to_3,
        &format!("interval_div_concat_cases_{index}"),
    )?;
    const CASES: [&[(IntervalClass, IntervalClass)]; 12] = [
        // Spanning zero, so output includes everything.
        &[
            (Spanning, Spanning),
            (Spanning, NegativeZero),
            (Spanning, ZeroPositive),
            (NegativeZero, Spanning),
            (Negative, Spanning),
            (ZeroPositive, Spanning),
            (Positive, Spanning),
        ],
        // Just zero no matter what.
        &[
            (SingletonZero, Spanning),
            (SingletonZero, NegativeZero),
            (SingletonZero, Negative),
            (SingletonZero, ZeroPositive),
            (SingletonZero, Positive),
        ],
        &[(Spanning, Negative)],
        &[(Spanning, Positive)],
        &[(NegativeZero, NegativeZero), (Negative, NegativeZero)],
        &[(NegativeZero, Negative), (Negative, Negative)],
        &[(NegativeZero, ZeroPositive), (Negative, ZeroPositive)],
        &[(NegativeZero, Positive), (Negative, Positive)],
        &[(ZeroPositive, NegativeZero), (Positive, NegativeZero)],
        &[(ZeroPositive, Negative), (Positive, Negative)],
        &[(ZeroPositive, ZeroPositive), (Positive, ZeroPositive)],
        &[(ZeroPositive, Positive), (Positive, Positive)],
    ];
    let outputs: [VectorValue<'ctx>; 12] = [
        constants.interval_entire,
        constants.interval_zero,
        builder.build_shuffle_vector(
            combos,
            combos.get_type().get_undef(),
            VectorType::const_vector(&[constants.i32_one, constants.i32_two]),
            &format!("interval_div_case_spanning_negative_{index}"),
        )?,
        builder.build_shuffle_vector(
            combos,
            combos.get_type().get_undef(),
            VectorType::const_vector(&[constants.i32_zero, constants.i32_three]),
            &format!("interval_div_case_spanning_positive_{index}"),
        )?,
        build_interval_compose(
            builder
                .build_extract_element(
                    combos,
                    constants.i32_three,
                    &format!("interval_div_case_neg_neg_zero_intermediate_0_{index}"),
                )?
                .into_float_value(),
            constants.flt_inf,
            builder,
            constants,
            "case_neg_neg_zero",
            index,
        )?,
        builder.build_shuffle_vector(
            combos,
            combos.get_type().get_undef(),
            VectorType::const_vector(&[constants.i32_three, constants.i32_two]),
            &format!("interval_div_case_neg_zero_neg_{index}"),
        )?,
        build_interval_compose(
            constants.flt_neg_inf,
            builder
                .build_extract_element(
                    combos,
                    constants.i32_one,
                    &format!("interval_div_case_neg_zero_positive_upper_{index}"),
                )?
                .into_float_value(),
            builder,
            constants,
            "interval_div_case_neg_zero_positive",
            index,
        )?,
        straight,
        build_interval_compose(
            constants.flt_neg_inf,
            builder
                .build_extract_element(
                    combos,
                    constants.i32_zero,
                    &format!("interval_div_case_zero_positive_neg_{index}"),
                )?
                .into_float_value(),
            builder,
            constants,
            "interval_div_case_zero_positive_neg",
            index,
        )?,
        build_interval_flip(straight, builder, constants, index)?,
        build_interval_compose(
            builder
                .build_extract_element(
                    combos,
                    constants.i32_two,
                    &format!("interval_div_case_zero_positive_extract_{index}"),
                )?
                .into_float_value(),
            constants.flt_inf,
            builder,
            constants,
            "interval_div_case_zero_positive",
            index,
        )?,
        cross,
    ];
    CASES
        .iter()
        .rev()
        .zip(outputs.into_iter().rev())
        .enumerate()
        .try_fold(constants.interval_empty, |acc, (i, (cases, out))| {
            let (_, cond) = cases
                .iter()
                .enumerate()
                .map(|(j, (lcase, rcase))| -> Result<IntValue<'ctx>, Error> {
                    Ok(builder.build_int_compare(
                        IntPredicate::EQ,
                        mask,
                        i32_type.const_int((*lcase as u64) * 7 + (*rcase as u64), false),
                        &format!("interval_div_case_{i}_subcase_{j}_{index}"),
                    )?)
                })
                .enumerate()
                .reduce(|(_, acc), (j, current)| match (acc, current) {
                    (Ok(acc), Ok(current)) => match builder.build_or(
                        acc,
                        current,
                        &format!("interval_div_case_{i}_combine_{j}_{index}"),
                    ) {
                        Ok(result) => (j, Ok(result)),
                        Err(e) => (j, Err(e.into())),
                    },
                    (Ok(_), Err(e)) | (Err(e), Ok(_)) | (Err(e), Err(_)) => (j, Err(e)),
                })
                .expect("Unable to combine all the cases when compiling interval division");
            let cond = cond?;
            Ok(builder
                .build_select(cond, out, acc, &format!("interval_div_case_{i}_choice"))?
                .into_vector_value())
        })
}

/**
Classify an interval based on it's relation ship to zero:

Empty = 0,
Negative = 1,
NegativeZero = 2,
SingletonZero = 3,
Spanning = 4,
ZeroPositive = 5,
Positive = 6,
 */
fn build_interval_classify<'ctx>(
    input: VectorValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<IntValue<'ctx>, Error> {
    let (is_empty, is_neg, is_eq) = (
        build_check_interval_empty(input, builder, module, index)?,
        builder.build_float_compare(
            FloatPredicate::ULT,
            input,
            constants.interval_zero,
            &format!("interval_classify_neg_check_{index}"),
        )?,
        builder.build_float_compare(
            FloatPredicate::UEQ,
            input,
            constants.interval_zero,
            &format!("interval_classify_zero_check_{index}"),
        )?,
    );
    let (lneg, rneg, leq, req) = (
        builder
            .build_extract_element(
                is_neg,
                constants.i32_zero,
                &format!("interval_classify_left_neg_{index}"),
            )?
            .into_int_value(),
        builder
            .build_extract_element(
                is_neg,
                constants.i32_one,
                &format!("interval_classify_right_neg_{index}"),
            )?
            .into_int_value(),
        builder
            .build_extract_element(
                is_eq,
                constants.i32_zero,
                &format!("interval_classify_left_eq_{index}"),
            )?
            .into_int_value(),
        builder
            .build_extract_element(
                is_eq,
                constants.i32_one,
                &format!("interval_classify_right_eq_{index}"),
            )?
            .into_int_value(),
    );
    Ok(builder
        .build_select(
            is_empty,
            constants.i32_zero,
            builder
                .build_select(
                    lneg,
                    builder
                        .build_select(
                            rneg,
                            constants.i32_one,
                            builder
                                .build_select(
                                    req,
                                    constants.i32_two,
                                    constants.i32_four,
                                    &format!("interval_classify_lneg_not_rneg_cases_{index}"),
                                )?
                                .into_int_value(),
                            &format!("interval_classify_lneg_cases_{index}"),
                        )?
                        .into_int_value(),
                    builder
                        .build_select(
                            leq,
                            builder
                                .build_select(
                                    req,
                                    constants.i32_three,
                                    constants.i32_five,
                                    &format!("interval_classify_leq_req_cases_{index}"),
                                )?
                                .into_int_value(),
                            constants.i32_six,
                            &format!("interval_classify_leq_cases_{index}"),
                        )?
                        .into_int_value(),
                    &format!("interval_classify_non_empty_cases_{index}"),
                )?
                .into_int_value(),
            &format!("interval_classify_{index}"),
        )?
        .into_int_value())
}

fn build_interval_mul<'ctx>(
    inputs: (VectorValue<'ctx>, VectorValue<'ctx>),
    ranges: ((f64, f64), (f64, f64)),
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &Constants<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    let (lhs, rhs) = inputs;
    let (range_left, range_right) = ranges;
    if lhs == rhs {
        build_interval_square(lhs, range_left, builder, module, constants, index)
    } else if (range_left.0 == 0.0 && range_left.1 == 0.0)
        || (range_right.0 == 0.0 && range_right.1 == 0.0)
    {
        Ok(constants.interval_zero)
    } else if range_left.0 == 1.0 && range_left.1 == 1.0 {
        Ok(rhs)
    } else if range_right.0 == 1.0 && range_right.1 == 1.0 {
        Ok(lhs)
    } else {
        let straight = builder.build_float_mul(lhs, rhs, &format!("mul_straight_{index}"))?;
        let cross = builder.build_float_mul(
            lhs,
            build_interval_flip(rhs, builder, constants, index)?,
            &format!("mul_cross_{index}"),
        )?;
        let concat = builder.build_shuffle_vector(
            straight,
            cross,
            constants.ivec_count_to_3,
            &format!("mul_concat_candidates_{index}"),
        )?;
        build_interval_compose(
            build_vec_unary_intrinsic(
                builder,
                module,
                "llvm.vector.reduce.fmin.*",
                &format!("mul_fmin_reduce_call_{index}"),
                concat,
            )?
            .into_float_value(),
            build_vec_unary_intrinsic(
                builder,
                module,
                "llvm.vector.reduce.fmax.*",
                &format!("mul_fmax_reduce_call_{index}"),
                concat,
            )?
            .into_float_value(),
            builder,
            constants,
            "interval_mul_compose",
            index,
        )
    }
}

fn build_interval_negate<'ctx>(
    input: VectorValue<'ctx>,
    builder: &'ctx Builder,
    constants: &Constants<'ctx>,
    index: usize,
    name: &str,
) -> Result<VectorValue<'ctx>, Error> {
    Ok(builder.build_shuffle_vector(
        builder.build_float_neg(input, &format!("negate_{index}"))?,
        input.get_type().get_undef(),
        VectorType::const_vector(&[constants.i32_one, constants.i32_zero]),
        name,
    )?)
}

fn build_check_interval_empty<'ctx>(
    input: VectorValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    index: usize,
) -> Result<IntValue<'ctx>, Error> {
    Ok(build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.vector.reduce.and.*",
        &format!("reduce_call_{index}"),
        builder.build_float_compare(
            FloatPredicate::UNO,
            input,
            input,
            &format!("check_empty_{index}"),
        )?,
    )?
    .into_int_value())
}

fn build_float_rem_euclid<'ctx>(
    lhs: FloatValue<'ctx>,
    rhs: FloatValue<'ctx>,
    builder: &Builder<'ctx>,
    constants: &Constants<'ctx>,
    name: &str,
    index: usize,
) -> Result<FloatValue<'ctx>, Error> {
    let qval = builder.build_float_rem(lhs, rhs, &format!("q_rem_val_{index}"))?;
    Ok(builder
        .build_select(
            builder.build_float_compare(
                FloatPredicate::ULT,
                qval,
                constants.flt_zero,
                &format!("rem_euclid_compare_{index}"),
            )?,
            builder.build_float_add(qval, rhs, &format!("rem_euclid_correction_{index}"))?,
            qval,
            name,
        )?
        .into_float_value())
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

    /// Same as above, but without bounds checking.
    ///
    /// # SAFETY
    ///
    /// The user is responsible for making sure the length of the `inputs` and
    /// `outputs` slices are correct.
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

    /// Same as above, but without bounds checking.
    ///
    /// # SAFETY
    ///
    /// The user is responsible for making sure the length of the `inputs` and
    /// `outputs` slices are correct.
    pub unsafe fn run_unchecked(&self, inputs: &[[T; 2]], outputs: &mut [[T; 2]]) {
        // SAFETY: we told the caller it is their responsiblity.
        unsafe { (self.func)(inputs.as_ptr().cast(), outputs.as_mut_ptr().cast()) }
    }
}

#[cfg(test)]
mod test {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use crate::{
        Error, Interval, IntervalEvaluator, JitContext, Tree, assert_float_eq, deftree,
        llvm_jit::NumberType,
        test_util::{self, Sampler},
    };

    const EPS: f64 = f64::EPSILON * 2.0;

    fn is_common(lo: f64, hi: f64) -> bool {
        lo.is_finite() && hi.is_finite() && lo <= hi
    }

    fn is_empty(lo: f64, hi: f64) -> bool {
        lo.is_nan() && hi.is_nan()
    }

    fn is_entire(lo: f64, hi: f64) -> bool {
        lo == f64::NEG_INFINITY && hi == f64::INFINITY
    }

    fn is_subset_of(a: &[f64; 2], b: &[f64; 2]) -> bool {
        (a[0] + EPS) >= b[0] && a[1] <= (b[1] + EPS)
    }

    fn contains((lo, hi): &(f64, f64), val: f64) -> bool {
        val.is_finite() && (val + EPS) >= *lo && val <= (*hi + EPS)
    }

    /**
    Helper function to check interval evaluations by evaluating the given
    tree. `vardata` is expected to contain a list of variables and the lower and
    upper bounds defining the range in which those variables can be sampled during
    testing. In essence, `vardata` defines one large interval in which to sample the
    tree.

    This function will sample many sub intervals within this large interval and
    ensure that the output intervals of the sub-intervals are subsets of the output
    interval of the large interval. This function samples many values in this
    interval and ensure the values are contained in the output interval of the
    sub-interval that contains the sample. All this is just to ensure the accuracy
    of the interval evaluations.

    `samples_per_var` defines the number of values to be sampled per variable. So
    the tree will be evaluated a total of `pow(samples_per_var, vardata.len())`
    times. `intervals_per_var` defines the number of sub intervals to sample per
    variable. So the treee will be evaluated for a total of `pow(intervals_per_var,
    vardata.len())` number of sub intervals.
    */
    pub fn check_interval_eval(
        tree: Tree,
        vardata: &[(char, f64, f64)],
        samples_per_var: usize,
        intervals_per_var: usize,
    ) {
        let num_roots = tree.num_roots();
        let context = JitContext::default();
        let params: String = vardata.iter().map(|(c, _, _)| *c).collect();
        let eval = tree.jit_compile::<f64>(&context, &params).unwrap();
        let ieval = tree.jit_compile_interval(&context, &params).unwrap();
        // Evaluate the full interval and get the range of output values of the tree.
        let total_range: Box<[[f64; 2]]> = {
            let inputs: Box<[_]> = vardata
                .iter()
                .map(|(_, lower, upper)| [*lower, *upper])
                .collect();
            let mut outputs = vec![[f64::NAN, f64::NAN]; num_roots].into_boxed_slice();
            ieval.run(&inputs, &mut outputs).unwrap();
            for [lo, hi] in outputs.iter() {
                assert!(is_common(*lo, *hi));
            }
            outputs
        };
        assert_eq!(total_range.len(), num_roots);
        let mut sampler = Sampler::new(vardata, samples_per_var, 42);
        // When we compute a sub-interval, we will cache the results here.
        let mut computed_intervals = vec![
            (f64::INFINITY, f64::NEG_INFINITY);
            intervals_per_var.pow(vardata.len() as u32) * num_roots
        ];
        // Flags for whether or not a sub-interval is already computed and cached.
        let mut computed = vec![false; intervals_per_var.pow(vardata.len() as u32)];
        // Steps that define the sub intervals on a per variable basis.
        let steps: Vec<_> = vardata
            .iter()
            .map(|(_label, lower, upper)| (upper - lower) / intervals_per_var as f64)
            .collect();
        /*
        Sample values, evaluate them and ensure they're within the output
        interval of the sub-interval that contains the sample.
        */
        while let Some(sample) = sampler.next() {
            assert_eq!(sample.len(), vardata.len());
            /*
            Find the index of the interval that the sample belongs in, and also get
            the sub-interval that contains the sample. The index here is a flattened
            index, similar to `x + y * X_SIZE + z * X_SIZE * Y_SIZE`, but
            generalized for arbitrary dimensions. The dimensions are equal to the
            number of variables.
            */
            let (index, isample, _) = sample.iter().zip(vardata.iter()).zip(steps.iter()).fold(
                (0usize, Vec::new(), 1usize),
                |(mut idx, mut intervals, mut multiplier),
                 ((value, (_label, lower, _upper)), step)| {
                    let local_idx = f64::floor((value - lower) / step);
                    idx += (local_idx as usize) * multiplier;
                    let inf = lower + local_idx * step;
                    intervals.push((inf, inf + step));
                    multiplier *= intervals_per_var;
                    (idx, intervals, multiplier)
                },
            );
            assert!(index < computed.len());
            // Get the interval that is expected to contain the values output by this sample.
            let expected_range = {
                let offset = index * num_roots;
                if !computed[index] {
                    // Evaluate the interval and cache it.
                    let inputs: Box<[_]> = isample.iter().map(|(lo, hi)| [*lo, *hi]).collect();
                    let mut iresults = vec![[f64::NAN, f64::NAN]; num_roots].into_boxed_slice();
                    ieval.run(&inputs, &mut iresults).unwrap();
                    for (i, [lo, hi]) in iresults.iter().enumerate() {
                        assert!(!is_empty(*lo, *hi));
                        assert!(!is_entire(*lo, *hi));
                        assert!(is_common(*lo, *hi));
                        assert!(is_subset_of(&[*lo, *hi], &total_range[i]));
                        computed_intervals[offset + i] = (*lo, *hi);
                    }
                    computed[index] = true;
                }
                &computed_intervals[offset..(offset + num_roots)]
            };
            // Evaluate the sample and ensure the output is within the interval.
            let mut results = vec![f64::NAN; num_roots].into_boxed_slice();
            eval.run(sample, &mut results).unwrap();
            assert_eq!(num_roots, results.len());
            assert_eq!(results.len(), expected_range.len());
            for (range, value) in expected_range.iter().zip(results.iter()) {
                assert!(contains(range, *value));
            }
        }
    }

    #[test]
    fn t_interval_sum() {
        check_interval_eval(
            deftree!(+ 'x 'y).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.)],
            10,
            5,
        );
    }

    #[test]
    fn t_interval_tree_1() {
        check_interval_eval(
            deftree!(/ (pow (log (+ (sin 'x) 2.)) 3.) (+ (cos 'x) 2.)).unwrap(),
            &[('x', -2.5, 2.5)],
            100,
            10,
        );
    }

    #[test]
    fn t_interval_squaring() {
        check_interval_eval(deftree!(pow 'x 2.).unwrap(), &[('x', -10., 10.)], 20, 5);
        check_interval_eval(
            deftree!(pow (- 'x 1.) 2.).unwrap(),
            &[('x', -10., 10.)],
            20,
            5,
        );
    }

    #[test]
    fn t_interval_cube() {
        check_interval_eval(
            deftree!(pow 'x 3.).unwrap(),
            &[('x', -10., 10.), ('y', -9., 10.)],
            20,
            5,
        );
    }

    #[test]
    fn t_interval_rational_pow() {
        check_interval_eval(deftree!(pow 'x 3.15).unwrap(), &[('x', 0., 10.)], 20, 5);
        check_interval_eval(
            deftree!(pow 'x 2.4556634543).unwrap(),
            &[('x', 2.212, 8.199)],
            20,
            5,
        );
        check_interval_eval(
            deftree!(pow 'x 45.23).unwrap(),
            &[('x', 2.222222, 11.112342)],
            20,
            5,
        );
    }

    #[test]
    fn t_interval_distance_to_point() {
        check_interval_eval(
            deftree!(sqrt (+ (pow (- 'x 2.) 2.) (pow (- 'y 3.) 2.))).unwrap(),
            &[('x', -10., 10.), ('y', -9., 10.)],
            20,
            5,
        );
    }

    #[test]
    fn t_interval_tree_2() {
        check_interval_eval(
            deftree!(
                (max (min
                      (- (sqrt (+ (+ (pow (- 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 2.75)
                      (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 4.))
                 (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (+ 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 5.25))
            )
            .unwrap(),
            &[('x', -10., 10.), ('y', -9., 10.), ('z', -11., 12.)],
            20,
            5,
        );
    }

    #[test]
    fn t_interval_trees_concat_0() {
        check_interval_eval(
            deftree!(concat
                            (log 'x)
                            (+ 'x (pow 'y 2.)))
            .unwrap(),
            &[('x', 1., 10.), ('y', 1., 10.)],
            20,
            5,
        );
    }

    #[test]
    fn t_interval_trees_concat_1() {
        check_interval_eval(
            deftree!(concat
                     (/ (pow (log (+ (sin 'x) 2.)) 3.) (+ (cos 'x) 2.))
                     (+ 'x 'y)
                     ((max (min
                            (- (sqrt (+ (+ (pow (- 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 2.75)
                            (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 4.))
                       (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (+ 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 5.25))
            )).unwrap(),
            &[('x', -10., 10.), ('y', -9., 10.), ('z', -11., 12.)],
            20,
            5
        );
    }

    #[test]
    fn t_interval_choose() {
        check_interval_eval(
            deftree!(if (> 'x 0) 'x (- 'x)).unwrap(),
            &[('x', -10., 10.)],
            100,
            10,
        );
        check_interval_eval(
            deftree!(if (< 'x 0) (- 'x) 'x).unwrap(),
            &[('x', -10., 10.)],
            100,
            10,
        );
    }

    #[test]
    fn t_floor() {
        check_interval_eval(
            deftree!(floor (/ (pow 'x 2) (+ 2 (sin 'x)))).unwrap(),
            &[('x', 1., 5.)],
            100,
            10,
        );
    }

    #[test]
    fn t_remainder() {
        check_interval_eval(
            deftree!(rem (pow 'x 2) (+ 2 (sin 'x))).unwrap(),
            &[('x', 1., 5.)],
            100,
            10,
        );
    }

    #[test]
    fn t_integer_exponents_negative() {
        // Negative even exponent
        check_interval_eval(deftree!(pow 'x (- 2.)).unwrap(), &[('x', 2., 5.)], 20, 5);
        // Negative odd exponent
        check_interval_eval(deftree!(pow 'x (- 3.)).unwrap(), &[('x', 1., 3.)], 20, 5);
        // Negative exponent with interval entirely below 1
        check_interval_eval(deftree!(pow 'x (- 2.)).unwrap(), &[('x', 0.2, 0.8)], 20, 5);
    }

    #[test]
    fn t_pow_around_one() {
        // Intervals straddling 1.0 with various exponents
        check_interval_eval(deftree!(pow 'x 2.5).unwrap(), &[('x', 0.5, 1.5)], 20, 5);
        check_interval_eval(deftree!(pow 'x (- 2.5)).unwrap(), &[('x', 0.5, 1.5)], 20, 5);
    }

    #[test]
    fn t_pow_exponent_crossing_zero() {
        // Exponent interval straddling 0
        check_interval_eval(
            deftree!(pow 'x 'y).unwrap(),
            &[('x', 2., 3.), ('y', -1., 1.)],
            15,
            4,
        );
    }

    #[test]
    fn t_exp() {
        check_interval_eval(deftree!(exp 'x).unwrap(), &[('x', -2., 2.)], 50, 10);
    }

    #[test]
    fn t_negate() {
        check_interval_eval(deftree!(- 'x).unwrap(), &[('x', -5., 5.)], 20, 5);
    }

    #[test]
    fn t_interval_abs() {
        // check_interval_eval(deftree!(abs 'x).unwrap(), &[('x', -5., 5.)], 20, 5);
        check_interval_eval(deftree!(abs 'x).unwrap(), &[('x', -3., -1.)], 20, 5);
    }

    #[test]
    fn t_min_max_direct() {
        check_interval_eval(
            deftree!(min 'x 'y).unwrap(),
            &[('x', -5., 5.), ('y', -3., 7.)],
            20,
            5,
        );
        check_interval_eval(
            deftree!(max 'x 'y).unwrap(),
            &[('x', -5., 5.), ('y', -3., 7.)],
            20,
            5,
        );
    }

    #[test]
    fn t_jit_interval_comparisons() {
        check_interval_eval(
            deftree!(if (== 'x 'y) (+ 'x 2.5) (- 'y 1.523)).unwrap(),
            &[('x', 0., 5.), ('y', 3., 8.)],
            20,
            5,
        );
        check_interval_eval(
            deftree!(if (!= 'x 'y) (+ 'x 2.5) (- 'y 1.523)).unwrap(),
            &[('x', 0., 5.), ('y', 3., 8.)],
            20,
            5,
        );
        check_interval_eval(
            deftree!(if (< 'x 'y) (+ 'x 2.5) (- 'y 1.523)).unwrap(),
            &[('x', 0., 5.), ('y', 3., 8.)],
            20,
            5,
        );
        check_interval_eval(
            deftree!(if (<= 'x 'y) (+ 'x 2.5) (- 'y 1.523)).unwrap(),
            &[('x', 0., 5.), ('y', 3., 8.)],
            20,
            5,
        );
    }

    #[test]
    fn t_jit_interval_boolean_ops() {
        check_interval_eval(
            deftree!(if (and (> 'x 0) (< 'y 5)) (- 'x 2.) (+ 'y 1.5)).unwrap(),
            &[('x', -2., 3.), ('y', 2., 7.)],
            20,
            5,
        );
        check_interval_eval(
            deftree!(if (or (> 'x 5) (< 'y 0)) (- 'x 2.) (+ 'y 1.5)).unwrap(),
            &[('x', -2., 3.), ('y', 2., 7.)],
            20,
            5,
        );
        check_interval_eval(
            deftree!(if (not (> 'x 0)) (- 'x 2.) 1.5).unwrap(),
            &[('x', -5., 5.)],
            20,
            5,
        );
    }

    fn test_jit_interval_negate<T: NumberType>() {
        let tree = deftree!(- 'x).unwrap();
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<T>(&context, "x").unwrap();
        // All positive.
        let mut outputs = [[T::nan(), T::nan()]];
        eval.run(&[[T::from_f64(2.0), T::from_f64(3.0)]], &mut outputs)
            .expect("Failed to run the jit function");
        assert_eq!(outputs[0], [T::from_f64(-3.0), T::from_f64(-2.0)]);
        // Test with heap allocation to check alignment
        let inputs = vec![[T::from_f64(2.0), T::from_f64(3.0)]];
        let mut outputs = vec![[T::nan(), T::nan()]];
        eval.run(&inputs, &mut outputs)
            .expect("Failed with heap allocation");
        assert_eq!(outputs[0], [T::from_f64(-3.0), T::from_f64(-2.0)]);
        // All negative.
        eval.run(&[[T::from_f64(-5.245), T::from_f64(-3.123)]], &mut outputs)
            .expect("Failed to run the jit function");
        assert_eq!(outputs[0], [T::from_f64(3.123), T::from_f64(5.245)]);
        // Spanning across zero.
        eval.run(
            &[[T::from_f64(-2.3345), T::from_f64(5.23445)]],
            &mut outputs,
        )
        .expect("Failed to run the jit function");
        assert_eq!(outputs[0], [T::from_f64(-5.23445), T::from_f64(2.3345)]);
        // Wrong number of inputs / outputs.
        matches!(
            eval.run(
                &[
                    [T::from_f64(-5.245), T::from_f64(-3.123)],
                    [T::from_f64(-2.3345), T::from_f64(5.23445)]
                ],
                &mut outputs
            ),
            Err(Error::InputSizeMismatch(2, 1))
        );
        let mut outputs = [[T::nan(), T::nan()], [T::nan(), T::nan()]];
        matches!(
            eval.run(&[[T::from_f64(-5.245), T::from_f64(-3.123)]], &mut outputs),
            Err(Error::OutputSizeMismatch(2, 1))
        );
    }

    #[test]
    fn t_jit_interval_negate_f64() {
        test_jit_interval_negate::<f32>();
        test_jit_interval_negate::<f64>();
    }

    fn test_jit_interval_sqrt<T: NumberType>() {
        let tree = deftree!(sqrt 'x).unwrap();
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<T>(&context, "x").unwrap();
        let mut outputs = [[T::nan(), T::nan()]];
        eval.run(&[[T::nan(), T::nan()]], &mut outputs)
            .expect("Failed to run the jit function");
        assert!(outputs[0].iter().all(|v| v.is_nan()));
        {
            let interval = [2.0, 3.0];
            eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
                .expect("Failed to run the jit function");
            assert_eq!(outputs[0], interval.map(|v| T::from_f64(v.sqrt())));
        }
        {
            let interval = [-2.0, 3.0];
            eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
                .expect("Failed to run the jit function");
            assert_eq!(outputs[0], [0.0, 3.0f64.sqrt()].map(|v| T::from_f64(v)));
        }
        {
            let interval = [-3.0, -2.0];
            eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
                .expect("Failed to run the jit function");
            assert!(outputs[0].iter().all(|v| v.is_nan()));
        }
    }

    #[test]
    fn t_jit_interval_sqrt_f64() {
        test_jit_interval_sqrt::<f32>();
        test_jit_interval_sqrt::<f64>();
    }

    fn test_jit_interval_abs<T: NumberType>() {
        let tree = deftree!(abs 'x).unwrap();
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<T>(&context, "x").unwrap();
        let mut outputs = [[T::nan(), T::nan()]];
        eval.run(&[[T::nan(), T::nan()]], &mut outputs)
            .expect("Failed to run the jit function");
        assert!(outputs[0].iter().all(|v| v.is_nan()));
        {
            let interval = [2.0, 3.0].map(|v| T::from_f64(v));
            eval.run(&[interval], &mut outputs)
                .expect("Failed to run the jit function");
            assert_eq!(outputs[0], interval);
        }
        {
            let interval = [-2.0, 3.0].map(|v| T::from_f64(v));
            eval.run(&[interval], &mut outputs)
                .expect("Failed to run the jit function");
            assert_eq!(outputs[0], [0.0, 3.0].map(|v| T::from_f64(v)));
        }
        {
            let interval = [-3.0, -2.0].map(|v| T::from_f64(v));
            eval.run(&[interval], &mut outputs)
                .expect("Failed to run the jit function");
            assert_eq!(outputs[0], [2.0, 3.0].map(|v| T::from_f64(v)));
        }
    }

    #[test]
    fn t_jit_interval_abs() {
        test_jit_interval_abs::<f32>();
        test_jit_interval_abs::<f64>();
    }

    fn test_jit_interval_sin<T: NumberType>() {
        use std::f64::consts::{FRAC_PI_2, PI};
        let tree = deftree!(sin 'x).unwrap();
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<T>(&context, "x").unwrap();
        let mut outputs = [[T::nan(), T::nan()]];
        // Test 1: NaN interval should return NaN
        eval.run(&[[T::nan(), T::nan()]], &mut outputs)
            .expect("Failed to run the jit function");
        assert!(
            outputs[0][0].is_nan() && outputs[0][1].is_nan(),
            "NaN test failed"
        );
        // Test 2: Point interval (lo == hi)
        eval.run(&[[0.0, 0.0].map(|v| T::from_f64(v))], &mut outputs)
            .unwrap();
        assert_eq!(
            outputs[0],
            [0.0, 0.0].map(|v| T::from_f64(v)),
            "Point at 0 failed"
        );
        eval.run(
            &[[FRAC_PI_2, FRAC_PI_2].map(|v| T::from_f64(v))],
            &mut outputs,
        )
        .unwrap();
        let expected = FRAC_PI_2.sin();
        const EPS: f64 = 1e-7;
        assert_float_eq!(outputs[0][0].to_f64(), expected, EPS);
        assert_float_eq!(outputs[0][1].to_f64(), expected, EPS);
        // Test 3: Small interval in Q0 [0, π/2) - monotonically increasing
        // sin is increasing here, so result should be [sin(lo), sin(hi)]
        let interval = [0.1, 0.4];
        eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
            .unwrap();
        assert_float_eq!(outputs[0][0].to_f64(), interval[0].sin().to_f64(), 1e-8);
        assert_float_eq!(outputs[0][1].to_f64(), interval[1].sin().to_f64(), 1e-8);
        // Test 4: Small interval in Q1 [π/2, π) - monotonically decreasing
        // sin is decreasing here, so result should be [sin(hi), sin(lo)]
        let interval = [FRAC_PI_2 + 0.1, FRAC_PI_2 + 0.4];
        eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
            .unwrap();
        assert_float_eq!(outputs[0][0].to_f64(), interval[1].sin().to_f64(), 1e-7);
        assert_float_eq!(outputs[0][1].to_f64(), interval[0].sin().to_f64(), 1e-7);
        // Test 5: Small interval in Q2 [π, 3π/2) - monotonically decreasing
        let interval = [PI + 0.1, PI + 0.4];
        eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
            .unwrap();
        assert_float_eq!(outputs[0][0].to_f64(), interval[1].sin().to_f64(), 1e-7);
        assert_float_eq!(outputs[0][1].to_f64(), interval[0].sin().to_f64(), 1e-7);
        // Test 6: Small interval in Q3 [3π/2, 2π) - monotonically increasing
        let interval = [3.0 * FRAC_PI_2 + 0.1, 3.0 * FRAC_PI_2 + 0.4];
        eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
            .unwrap();
        assert_float_eq!(outputs[0][0].to_f64(), interval[0].sin().to_f64(), 1e-7);
        assert_float_eq!(outputs[0][1].to_f64(), interval[1].sin().to_f64(), 1e-7);
        // Test 7: Interval crossing π/2 (includes maximum)
        // Should return [min(sin(lo), sin(hi)), 1.0]
        let interval = [0.5, 2.0]; // crosses π/2 ≈ 1.57
        eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
            .unwrap();
        let min_endpoint = interval[0].sin().min(interval[1].sin());
        assert_float_eq!(outputs[0][0].to_f64(), min_endpoint, EPS);
        assert_float_eq!(outputs[0][1].to_f64(), 1.0, EPS);
        // Test 8: Interval crossing 3π/2 (includes minimum)
        // Should return [-1.0, max(sin(lo), sin(hi))]
        let interval = [4.0, 5.5]; // crosses 3π/2 ≈ 4.71
        eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
            .unwrap();
        let max_endpoint = interval[0].sin().max(interval[1].sin());
        assert_float_eq!(outputs[0][0].to_f64(), -1.0, EPS);
        assert_float_eq!(outputs[0][1].to_f64(), max_endpoint, EPS);
        // Test 9: Interval spanning both max and min
        // Should return [-1.0, 1.0]
        let interval = [0.0, 3.0 * FRAC_PI_2 + 0.1]; // Goes past 3π/2 to hit both extrema
        eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
            .unwrap();
        assert_float_eq!(outputs[0][0].to_f64(), -1.0, EPS);
        assert_float_eq!(outputs[0][1].to_f64(), 1.0, EPS);
        // Test 10: Interval spanning full period or more
        let interval = [0.0, 2.0 * PI];
        eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
            .unwrap();
        assert_float_eq!(outputs[0][0].to_f64(), -1.0, EPS);
        assert_float_eq!(outputs[0][1].to_f64(), 1.0, EPS);
        let interval = [0.0, 3.0 * PI];
        eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
            .unwrap();
        assert_float_eq!(outputs[0][0].to_f64(), -1.0, EPS);
        assert_float_eq!(outputs[0][1].to_f64(), 1.0, EPS);
        // Test 11: Negative intervals - small interval in negative Q0
        let interval = [-0.5, -0.1];
        eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
            .unwrap();
        assert_float_eq!(outputs[0][0].to_f64(), interval[0].sin(), EPS);
        assert_float_eq!(outputs[0][1].to_f64(), interval[1].sin(), EPS);
        // Test 12: Negative interval crossing -π/2 (includes minimum at -π/2)
        let interval = [-2.0, -1.0]; // crosses -π/2 ≈ -1.57
        eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
            .unwrap();
        let max_endpoint = interval[0].sin().max(interval[1].sin());
        assert_float_eq!(outputs[0][0].to_f64(), -1.0, EPS);
        assert_float_eq!(outputs[0][1].to_f64(), max_endpoint, EPS);
        // Test 13: Symmetric interval around zero
        let interval = [-0.5, 0.5];
        eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
            .unwrap();
        assert_float_eq!(outputs[0][0].to_f64(), interval[0].sin(), EPS);
        assert_float_eq!(outputs[0][1].to_f64(), interval[1].sin(), EPS);
        // Test 14: Large positive values
        let interval = [100.0, 100.5];
        eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
            .unwrap();
        // The exact result depends on quadrant, just check it's valid
        assert!(
            outputs[0][0].to_f64() >= -1.0 && outputs[0][0].to_f64() <= 1.0,
            "Large positive lo out of range"
        );
        assert!(
            outputs[0][1].to_f64() >= -1.0 && outputs[0][1].to_f64() <= 1.0,
            "Large positive hi out of range"
        );
        assert!(
            outputs[0][0].to_f64() <= outputs[0][1].to_f64(),
            "Large positive interval not ordered"
        );
        // Test 15: Large negative values
        let interval = [-100.5, -100.0];
        eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
            .unwrap();
        assert!(
            outputs[0][0].to_f64() >= -1.0 && outputs[0][0].to_f64() <= 1.0,
            "Large negative lo out of range"
        );
        assert!(
            outputs[0][1].to_f64() >= -1.0 && outputs[0][1].to_f64() <= 1.0,
            "Large negative hi out of range"
        );
        assert!(
            outputs[0][0] <= outputs[0][1],
            "Large negative interval not ordered"
        );
        // Test 16: Interval exactly [0, π/2]
        let interval = [0.0, FRAC_PI_2];
        eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
            .unwrap();
        assert_eq!(
            outputs[0].map(|v| v.to_f64()),
            [0.0, 1.0],
            "Exact [0, π/2] failed"
        );
        // Test 17: Interval exactly [π/2, π]
        let interval = [FRAC_PI_2, PI];
        eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
            .unwrap();
        assert!(
            (outputs[0][0].to_f64() - 0.0).abs() < EPS,
            "Exact [π/2, π] lower bound failed"
        );
        assert!(
            (outputs[0][1].to_f64() - 1.0).abs() < EPS,
            "Exact [π/2, π] upper bound failed"
        );
        // Test 18: Very small interval (numerical precision test)
        let interval = [1.0, 1.0 + 1e-10];
        eval.run(&[interval.map(|v| T::from_f64(v))], &mut outputs)
            .unwrap();
        assert!(
            outputs[0][0] <= outputs[0][1],
            "Very small interval not ordered"
        );
        assert!(
            (outputs[0][1].to_f64() - outputs[0][0].to_f64()).abs() < 1e-9,
            "Very small interval too wide"
        );
        // Test 19: Infinity inputs (should handle gracefully)
        eval.run(
            &[[f64::INFINITY, f64::INFINITY].map(|v| T::from_f64(v))],
            &mut outputs,
        )
        .unwrap();
        // Behavior with infinity is implementation-defined, just check no crash
        eval.run(
            &[[f64::NEG_INFINITY, f64::NEG_INFINITY].map(|v| T::from_f64(v))],
            &mut outputs,
        )
        .unwrap();
        // Behavior with infinity is implementation-defined, just check no crash
        // Test 20: Mixed finite and special values
        eval.run(
            &[[0.0, f64::INFINITY].map(|v| T::from_f64(v))],
            &mut outputs,
        )
        .unwrap();
        // Should likely return [-1, 1] as it spans everything
    }

    #[test]
    fn t_jit_interval_sin() {
        test_jit_interval_sin::<f32>();
        test_jit_interval_sin::<f64>();
    }

    #[test]
    fn t_jit_interval_cos_f64() {
        use std::f64::consts::{FRAC_PI_2, PI};
        let tree = deftree!(cos 'x).unwrap();
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
        assert_eq!(outputs[0], [1.0, 1.0], "Point at 0 failed");
        eval.run(&[[PI, PI]], &mut outputs).unwrap();
        let expected = PI.cos();
        assert!(
            (outputs[0][0] - expected).abs() < 1e-10 && (outputs[0][1] - expected).abs() < 1e-10,
            "Point at π failed: got [{}, {}], expected [{}, {}]",
            outputs[0][0],
            outputs[0][1],
            expected,
            expected
        );
        // Test 3: Small interval in Q0 [0, π/2) - monotonically decreasing
        // cos is decreasing here, so result should be [cos(hi), cos(lo)]
        let interval = [0.1, 0.4];
        eval.run(&[interval], &mut outputs).unwrap();
        assert_eq!(
            outputs[0].map(|v| v.to_f64()),
            [interval[1].cos(), interval[0].cos()],
            "Q0 monotonic decreasing failed"
        );
        // Test 4: Small interval in Q1 [π/2, π) - monotonically decreasing
        // cos is decreasing here, so result should be [cos(hi), cos(lo)]
        let interval = [FRAC_PI_2 + 0.1, FRAC_PI_2 + 0.4];
        eval.run(&[interval], &mut outputs).unwrap();
        assert_eq!(
            outputs[0],
            [interval[1].cos(), interval[0].cos()],
            "Q1 monotonic decreasing failed"
        );
        // Test 5: Small interval in Q2 [π, 3π/2) - monotonically increasing
        let interval = [PI + 0.1, PI + 0.4];
        eval.run(&[interval], &mut outputs).unwrap();
        assert_eq!(
            outputs[0],
            [interval[0].cos(), interval[1].cos()],
            "Q2 monotonic increasing failed"
        );
        // Test 6: Small interval in Q3 [3π/2, 2π) - monotonically increasing
        let interval = [3.0 * FRAC_PI_2 + 0.1, 3.0 * FRAC_PI_2 + 0.4];
        eval.run(&[interval], &mut outputs).unwrap();
        assert_eq!(
            outputs[0],
            [interval[0].cos(), interval[1].cos()],
            "Q3 monotonic increasing failed"
        );
        // Test 7: Interval crossing 0/2π (includes maximum at 1)
        // cos has maximum at 0, 2π, 4π, etc
        let interval = [2.0 * PI - 0.5, 2.0 * PI + 0.5]; // crosses 2π
        eval.run(&[interval], &mut outputs).unwrap();
        let min_endpoint = interval[0].cos().min(interval[1].cos());
        assert!(
            (outputs[0][0] - min_endpoint).abs() < 1e-10,
            "Interval crossing 2π (max): lower bound failed, got {}, expected {}",
            outputs[0][0],
            min_endpoint
        );
        assert!(
            (outputs[0][1] - 1.0).abs() < 1e-10,
            "Interval crossing 2π (max): upper bound failed, got {}, expected 1.0",
            outputs[0][1]
        );
        // Test 8: Interval crossing π (includes minimum at -1)
        // cos has minimum at π, 3π, 5π, etc
        let interval = [2.5, 3.5]; // crosses π ≈ 3.14
        eval.run(&[interval], &mut outputs).unwrap();
        let max_endpoint = interval[0].cos().max(interval[1].cos());
        assert!(
            (outputs[0][0] - (-1.0)).abs() < 1e-10,
            "Interval crossing π (min): lower bound failed, got {}, expected -1.0",
            outputs[0][0]
        );
        assert!(
            (outputs[0][1] - max_endpoint).abs() < 1e-10,
            "Interval crossing π (min): upper bound failed, got {}, expected {}",
            outputs[0][1],
            max_endpoint
        );
        // Test 9: Interval starting at 0 and crossing π/2
        let interval = [0.0, FRAC_PI_2 + 0.5];
        eval.run(&[interval], &mut outputs).unwrap();
        let min_val = interval[1].cos();
        assert!(
            (outputs[0][0] - min_val).abs() < 1e-10,
            "Interval [0, π/2+ε]: lower bound failed"
        );
        assert!(
            (outputs[0][1] - 1.0).abs() < 1e-10,
            "Interval [0, π/2+ε]: upper bound failed"
        );
        // Test 10: Interval spanning both max and min
        let interval = [0.0, PI + 0.5]; // Includes 0 (max=1) and π (min=-1)
        eval.run(&[interval], &mut outputs).unwrap();
        assert_eq!(
            outputs[0],
            [-1.0, 1.0],
            "Interval spanning both extrema failed"
        );
        // Test 11: Interval spanning full period or more
        let interval = [0.0, 2.0 * PI];
        eval.run(&[interval], &mut outputs).unwrap();
        assert_eq!(outputs[0], [-1.0, 1.0], "Full period interval failed");
        let interval = [0.0, 3.0 * PI];
        eval.run(&[interval], &mut outputs).unwrap();
        assert_eq!(outputs[0], [-1.0, 1.0], "Multiple period interval failed");
        // Test 12: Negative intervals - small interval in negative Q0
        let interval = [-0.5, -0.1];
        eval.run(&[interval], &mut outputs).unwrap();
        // cos is even, so cos(-x) = cos(x), and decreasing away from 0
        assert_eq!(
            outputs[0],
            [interval[0].cos(), interval[1].cos()],
            "Negative Q0 failed: got [{}, {}], expected [{}, {}]",
            outputs[0][0],
            outputs[0][1],
            interval[1].cos(),
            interval[0].cos()
        );
        // Test 13: Negative interval crossing 0 (includes maximum)
        let interval = [-0.5, 0.5]; // crosses 0 where cos = 1
        eval.run(&[interval], &mut outputs).unwrap();
        let min_val = interval[0].cos().min(interval[1].cos());
        assert!(
            (outputs[0][0] - min_val).abs() < 1e-10,
            "Interval crossing 0: lower bound failed"
        );
        assert!(
            (outputs[0][1] - 1.0).abs() < 1e-10,
            "Interval crossing 0: upper bound failed"
        );
        // Test 14: Negative interval crossing -π (includes minimum)
        let interval = [-3.5, -2.5]; // crosses -π ≈ -3.14
        eval.run(&[interval], &mut outputs).unwrap();
        let max_val = interval[0].cos().max(interval[1].cos());
        assert!(
            (outputs[0][0] - (-1.0)).abs() < 1e-10,
            "Interval crossing -π: lower bound failed"
        );
        assert!(
            (outputs[0][1] - max_val).abs() < 1e-10,
            "Interval crossing -π: upper bound failed"
        );
        // Test 15: Symmetric interval around π/2
        let interval = [FRAC_PI_2 - 0.3, FRAC_PI_2 + 0.3];
        eval.run(&[interval], &mut outputs).unwrap();
        let expected_min = (FRAC_PI_2 + 0.3).cos();
        let expected_max = (FRAC_PI_2 - 0.3).cos();
        assert!(
            (outputs[0][0] - expected_min).abs() < 1e-10,
            "Symmetric around π/2 failed: lower bound"
        );
        assert!(
            (outputs[0][1] - expected_max).abs() < 1e-10,
            "Symmetric around π/2 failed: upper bound"
        );
        // Test 16: Large positive values
        let interval = [100.0, 100.5];
        eval.run(&[interval], &mut outputs).unwrap();
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
        // Test 17: Large negative values
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
        // Test 18: Interval exactly [0, π/2]
        let interval = [0.0, FRAC_PI_2];
        eval.run(&[interval], &mut outputs).unwrap();
        assert!(
            (outputs[0][0] - 0.0).abs() < 1e-10,
            "Exact [0, π/2] lower bound failed"
        );
        assert!(
            (outputs[0][1] - 1.0).abs() < 1e-10,
            "Exact [0, π/2] upper bound failed"
        );
        // Test 19: Interval exactly [π/2, π]
        let interval = [FRAC_PI_2, PI];
        eval.run(&[interval], &mut outputs).unwrap();
        assert!(
            (outputs[0][0] - (-1.0)).abs() < 1e-10,
            "Exact [π/2, π] lower bound failed"
        );
        assert!(
            (outputs[0][1] - 0.0).abs() < 1e-10,
            "Exact [π/2, π] upper bound failed"
        );
        // Test 20: Very small interval (numerical precision test)
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
        // Test 21: Infinity inputs (should handle gracefully)
        eval.run(&[[f64::INFINITY, f64::INFINITY]], &mut outputs)
            .unwrap();
        // Behavior with infinity is implementation-defined, just check no crash
        eval.run(&[[f64::NEG_INFINITY, f64::NEG_INFINITY]], &mut outputs)
            .unwrap();
        // Behavior with infinity is implementation-defined, just check no crash
        // Test 22: Mixed finite and special values
        eval.run(&[[0.0, f64::INFINITY]], &mut outputs).unwrap();
        // Should likely return [-1, 1] as it spans everything
    }

    #[test]
    fn t_jit_interval_tan_f64() {
        use std::f64::consts::{FRAC_PI_2, PI};
        let tree = deftree!(tan 'x).unwrap();
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<f64>(&context, "x").unwrap();
        let mut outputs = [[f64::NAN, f64::NAN]];
        // NaN interval
        eval.run(&[[f64::NAN, f64::NAN]], &mut outputs).unwrap();
        assert!(outputs[0][0].is_nan() && outputs[0][1].is_nan());
        // Point intervals
        eval.run(&[[0.0, 0.0]], &mut outputs).unwrap();
        assert_eq!(outputs[0], [0.0, 0.0]);
        eval.run(&[[PI, PI]], &mut outputs).unwrap();
        assert!((outputs[0][0] - PI.tan()).abs() < 1e-10);
        // Small monotonic intervals (no discontinuity crossing)
        for interval in [[0.1, 0.4], [-0.4, -0.1], [-0.5, 0.5]] {
            eval.run(&[interval], &mut outputs).unwrap();
            assert!((outputs[0][0] - interval[0].tan()).abs() < 1e-10);
            assert!((outputs[0][1] - interval[1].tan()).abs() < 1e-10);
        }
        // Intervals crossing π/2 discontinuities (should return ENTIRE)
        for interval in [
            [1.0, 2.0],                           // crossing π/2
            [-2.0, -1.0],                         // crossing -π/2
            [4.0, 5.0],                           // crossing 3π/2
            [FRAC_PI_2 - 0.01, FRAC_PI_2 + 0.01], // small crossing
        ] {
            eval.run(&[interval], &mut outputs).unwrap();
            assert_eq!(outputs[0], [f64::NEG_INFINITY, f64::INFINITY]);
        }
        // Intervals with width >= π (should return ENTIRE)
        for interval in [
            [0.0, PI],
            [-2.0, 2.0],
            [0.0, 10.0],
            [-10.0, 0.0],
            [0.0, f64::INFINITY],
        ] {
            eval.run(&[interval], &mut outputs).unwrap();
            assert_eq!(outputs[0], [f64::NEG_INFINITY, f64::INFINITY]);
        }
        // Intervals just before discontinuity
        eval.run(&[[1.0, FRAC_PI_2 - 0.01]], &mut outputs).unwrap();
        assert!((outputs[0][0] - 1.0f64.tan()).abs() < 1e-10);
        assert!((outputs[0][1] - (FRAC_PI_2 - 0.01).tan()).abs() < 1e-9);
        eval.run(&[[-FRAC_PI_2 + 0.01, -1.0]], &mut outputs)
            .unwrap();
        assert!((outputs[0][0] - (-FRAC_PI_2 + 0.01).tan()).abs() < 1e-9);
        assert!((outputs[0][1] - (-1.0f64).tan()).abs() < 1e-10);
        // Different periods
        for interval in [
            [PI + 0.1, PI + 0.4],
            [-PI + 0.1, -PI + 0.4],
            [2.0 * PI, 2.0 * PI + FRAC_PI_2 / 2.0],
        ] {
            eval.run(&[interval], &mut outputs).unwrap();
            assert!((outputs[0][0] - interval[0].tan()).abs() < 1e-9);
            assert!((outputs[0][1] - interval[1].tan()).abs() < 1e-9);
        }
        // Maximal interval without discontinuity
        let eps = 0.001;
        eval.run(&[[-FRAC_PI_2 + eps, FRAC_PI_2 - eps]], &mut outputs)
            .unwrap();
        assert!((outputs[0][0] - (-FRAC_PI_2 + eps).tan()).abs() < 1e-9);
        assert!((outputs[0][1] - (FRAC_PI_2 - eps).tan()).abs() < 1e-9);
        // Very large value near discontinuity
        eval.run(&[[0.0, FRAC_PI_2 - 1e-8]], &mut outputs).unwrap();
        assert!((outputs[0][0] - 0.0).abs() < 1e-10);
        assert!(outputs[0][1] > 1e6);
        // Very small interval (numerical precision)
        eval.run(&[[0.5, 0.5 + 1e-10]], &mut outputs).unwrap();
        assert!(outputs[0][0] <= outputs[0][1]);
        assert!((outputs[0][1] - outputs[0][0]).abs() < 1e-8);
        // Infinity inputs (should not crash)
        eval.run(&[[f64::INFINITY, f64::INFINITY]], &mut outputs)
            .unwrap();
        eval.run(&[[f64::NEG_INFINITY, f64::NEG_INFINITY]], &mut outputs)
            .unwrap();
    }

    #[test]
    fn t_jit_interval_log_f64() {
        let tree = deftree!(log 'x).unwrap();
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<f64>(&context, "x").unwrap();
        let mut outputs = [[f64::NAN, f64::NAN]];
        // NaN interval
        eval.run(&[[f64::NAN, f64::NAN]], &mut outputs).unwrap();
        assert!(outputs[0][0].is_nan() && outputs[0][1].is_nan());
        // Both bounds negative (should return NaN)
        for interval in [[-5.0, -1.0], [-10.0, -0.001]] {
            eval.run(&[interval], &mut outputs).unwrap();
            assert!(outputs[0][0].is_nan() && outputs[0][1].is_nan());
        }
        // Upper bound at or below 0 (should return NaN)
        for interval in [[1.0, 0.0], [1.0, -1.0], [0.0, 0.0]] {
            eval.run(&[interval], &mut outputs).unwrap();
            assert!(outputs[0][0].is_nan() && outputs[0][1].is_nan());
        }
        // Lower bound negative/zero, upper positive (should return [NEG_INFINITY, ln(hi)])
        for (interval, expected_hi) in [
            ([-2.0, 3.0], 3.0f64.ln()),
            ([-0.5, 2.0], 2.0f64.ln()),
            ([0.0, 5.0], 5.0f64.ln()),
        ] {
            eval.run(&[interval], &mut outputs).unwrap();
            assert_eq!(outputs[0][0], f64::NEG_INFINITY);
            assert!((outputs[0][1] - expected_hi).abs() < 1e-10);
        }
        // Both bounds positive (normal case)
        for interval in [[0.5, 2.0], [1.0, 10.0], [2.0, 8.0], [0.001, 0.1]] {
            eval.run(&[interval], &mut outputs).unwrap();
            assert!((outputs[0][0] - interval[0].ln()).abs() < 1e-10);
            assert!((outputs[0][1] - interval[1].ln()).abs() < 1e-10);
        }
        // Point intervals
        eval.run(&[[1.0, 1.0]], &mut outputs).unwrap();
        assert_eq!(outputs[0], [0.0, 0.0]);
        eval.run(&[[std::f64::consts::E, std::f64::consts::E]], &mut outputs)
            .unwrap();
        assert!((outputs[0][0] - 1.0).abs() < 1e-10);
        // Interval containing 1 (ln(1) = 0)
        eval.run(&[[0.5, 2.0]], &mut outputs).unwrap();
        assert!(outputs[0][0] < 0.0 && outputs[0][1] > 0.0);
        // Very small positive values (large negative results)
        eval.run(&[[1e-10, 1e-8]], &mut outputs).unwrap();
        assert!(outputs[0][0] < -18.0 && outputs[0][1] < -16.0);
        // Very large positive values
        eval.run(&[[1e8, 1e10]], &mut outputs).unwrap();
        assert!(outputs[0][0] > 18.0 && outputs[0][1] > 23.0);
        // Interval very close to 0
        eval.run(&[[1e-100, 1e-50]], &mut outputs).unwrap();
        assert!(outputs[0][0] < -230.0 && outputs[0][1] < -115.0);
        // Infinity upper bound
        eval.run(&[[1.0, f64::INFINITY]], &mut outputs).unwrap();
        assert_eq!(outputs[0][0], 0.0);
        assert_eq!(outputs[0][1], f64::INFINITY);
    }

    #[test]
    fn t_jit_interval_floor_f64() {
        let tree = deftree!(floor 'x).unwrap();
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<f64>(&context, "x").unwrap();
        let mut outputs = [[f64::NAN, f64::NAN]];
        // NaN interval
        eval.run(&[[f64::NAN, f64::NAN]], &mut outputs).unwrap();
        assert!(outputs[0][0].is_nan() && outputs[0][1].is_nan());
        // Positive, negative, and spanning zero intervals
        for interval in [
            [0.1, 0.9],
            [1.5, 3.7],
            [5.0, 5.0],
            [-0.9, -0.1],
            [-3.7, -1.5],
            [-2.5, 3.7],
            [-0.5, 0.5],
        ] {
            eval.run(&[interval], &mut outputs).unwrap();
            assert_eq!(outputs[0], [interval[0].floor(), interval[1].floor()]);
        }
        // Already integer values
        for interval in [[1.0, 5.0], [-3.0, 2.0], [0.0, 0.0]] {
            eval.run(&[interval], &mut outputs).unwrap();
            assert_eq!(outputs[0], [interval[0], interval[1]]);
        }
        // Infinities
        eval.run(&[[f64::NEG_INFINITY, f64::INFINITY]], &mut outputs)
            .unwrap();
        assert_eq!(outputs[0], [f64::NEG_INFINITY, f64::INFINITY]);
        eval.run(&[[f64::NEG_INFINITY, 5.5]], &mut outputs).unwrap();
        assert_eq!(outputs[0], [f64::NEG_INFINITY, 5.0]);
        eval.run(&[[5.5, f64::INFINITY]], &mut outputs).unwrap();
        assert_eq!(outputs[0], [5.0, f64::INFINITY]);
    }

    #[test]
    fn t_jit_interval_add_f64() {
        let tree = deftree!(+ 'x 'y).unwrap();
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<f64>(&context, "xy").unwrap();
        let mut outputs = [[f64::NAN, f64::NAN]];
        // NaN intervals
        eval.run(&[[f64::NAN, f64::NAN], [1.0, 2.0]], &mut outputs)
            .unwrap();
        assert!(outputs[0][0].is_nan() && outputs[0][1].is_nan());
        eval.run(&[[1.0, 2.0], [f64::NAN, f64::NAN]], &mut outputs)
            .unwrap();
        assert!(outputs[0][0].is_nan() && outputs[0][1].is_nan());
        // Both positive, both negative, and mixed signs
        for (x, y) in [
            ([1.0, 2.0], [3.0, 4.0]),
            ([-2.0, -1.0], [-4.0, -3.0]),
            ([-2.0, 3.0], [-1.0, 4.0]),
            ([-5.0, -1.0], [2.0, 6.0]),
            ([1.0, 5.0], [-3.0, -1.0]),
        ] {
            eval.run(&[x, y], &mut outputs).unwrap();
            assert_eq!(outputs[0], [x[0] + y[0], x[1] + y[1]]);
        }
        // With zeros
        eval.run(&[[0.0, 0.0], [0.0, 0.0]], &mut outputs).unwrap();
        assert_eq!(outputs[0], [0.0, 0.0]);
        eval.run(&[[-5.0, 5.0], [0.0, 0.0]], &mut outputs).unwrap();
        assert_eq!(outputs[0], [-5.0, 5.0]);
        // With infinities
        eval.run(
            &[[f64::NEG_INFINITY, f64::INFINITY], [1.0, 2.0]],
            &mut outputs,
        )
        .unwrap();
        assert_eq!(outputs[0], [f64::NEG_INFINITY, f64::INFINITY]);
        eval.run(
            &[[1.0, 2.0], [f64::NEG_INFINITY, f64::INFINITY]],
            &mut outputs,
        )
        .unwrap();
        assert_eq!(outputs[0], [f64::NEG_INFINITY, f64::INFINITY]);
    }

    #[test]
    fn t_jit_interval_subtract_f64() {
        let tree = deftree!(- 'x 'y).unwrap();
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<f64>(&context, "xy").unwrap();
        let mut outputs = [[f64::NAN, f64::NAN]];
        // NaN intervals
        eval.run(&[[f64::NAN, f64::NAN], [1.0, 2.0]], &mut outputs)
            .unwrap();
        assert!(outputs[0][0].is_nan() && outputs[0][1].is_nan());
        eval.run(&[[1.0, 2.0], [f64::NAN, f64::NAN]], &mut outputs)
            .unwrap();
        assert!(outputs[0][0].is_nan() && outputs[0][1].is_nan());
        // Both positive, both negative, and mixed signs
        for (x, y) in [
            ([5.0, 7.0], [1.0, 2.0]),
            ([1.0, 2.0], [5.0, 7.0]),
            ([-2.0, -1.0], [-5.0, -3.0]),
            ([-2.0, 3.0], [-1.0, 4.0]),
            ([-5.0, -1.0], [2.0, 6.0]),
            ([1.0, 5.0], [-3.0, -1.0]),
        ] {
            eval.run(&[x, y], &mut outputs).unwrap();
            assert_eq!(outputs[0], [x[0] - y[1], x[1] - y[0]]);
        }
        // With zeros
        eval.run(&[[0.0, 0.0], [0.0, 0.0]], &mut outputs).unwrap();
        assert_eq!(outputs[0], [0.0, 0.0]);
        eval.run(&[[5.0, 10.0], [0.0, 0.0]], &mut outputs).unwrap();
        assert_eq!(outputs[0], [5.0, 10.0]);
        eval.run(&[[0.0, 0.0], [5.0, 10.0]], &mut outputs).unwrap();
        assert_eq!(outputs[0], [-10.0, -5.0]);
        // With infinities
        eval.run(
            &[[f64::NEG_INFINITY, f64::INFINITY], [1.0, 2.0]],
            &mut outputs,
        )
        .unwrap();
        assert_eq!(outputs[0], [f64::NEG_INFINITY, f64::INFINITY]);
        eval.run(
            &[[1.0, 2.0], [f64::NEG_INFINITY, f64::INFINITY]],
            &mut outputs,
        )
        .unwrap();
        assert_eq!(outputs[0], [f64::NEG_INFINITY, f64::INFINITY]);
    }

    #[test]
    fn t_jit_interval_multiply_f64() {
        let tree = deftree!(* 'x 'y).unwrap();
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<f64>(&context, "xy").unwrap();
        let mut outputs = [[f64::NAN, f64::NAN]];
        // Helper to compute expected result
        let mul = |x: [f64; 2], y: [f64; 2]| -> [f64; 2] {
            let products = [x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1]];
            [
                products.iter().copied().fold(f64::INFINITY, f64::min),
                products.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ]
        };
        // NaN intervals
        eval.run(&[[f64::NAN, f64::NAN], [1.0, 2.0]], &mut outputs)
            .unwrap();
        assert!(outputs[0][0].is_nan() && outputs[0][1].is_nan());
        eval.run(&[[1.0, 2.0], [f64::NAN, f64::NAN]], &mut outputs)
            .unwrap();
        assert!(outputs[0][0].is_nan() && outputs[0][1].is_nan());
        // Both positive, both negative, and mixed signs
        for (x, y) in [
            ([2.0, 3.0], [4.0, 5.0]),
            ([0.5, 1.5], [2.0, 3.0]),
            ([-3.0, -2.0], [-5.0, -4.0]),
            ([2.0, 3.0], [-5.0, -4.0]),
            ([-3.0, -2.0], [4.0, 5.0]),
        ] {
            eval.run(&[x, y], &mut outputs).unwrap();
            assert_eq!(outputs[0], mul(x, y));
        }
        // Intervals containing zero
        for (x, y) in [
            ([-2.0, 3.0], [4.0, 5.0]),
            ([2.0, 3.0], [-5.0, 4.0]),
            ([-2.0, 3.0], [-5.0, 4.0]),
            ([0.0, 3.0], [2.0, 5.0]),
            ([-3.0, 0.0], [2.0, 5.0]),
        ] {
            eval.run(&[x, y], &mut outputs).unwrap();
            assert_eq!(outputs[0], mul(x, y));
        }
        // Zero intervals
        eval.run(&[[0.0, 0.0], [0.0, 0.0]], &mut outputs).unwrap();
        assert_eq!(outputs[0], [0.0, 0.0]);
        eval.run(&[[0.0, 0.0], [5.0, 10.0]], &mut outputs).unwrap();
        assert_eq!(outputs[0], [0.0, 0.0]);
        eval.run(&[[5.0, 10.0], [0.0, 0.0]], &mut outputs).unwrap();
        assert_eq!(outputs[0], [0.0, 0.0]);
        // With infinities
        eval.run(
            &[[f64::NEG_INFINITY, f64::INFINITY], [2.0, 3.0]],
            &mut outputs,
        )
        .unwrap();
        assert_eq!(outputs[0], [f64::NEG_INFINITY, f64::INFINITY]);
        eval.run(
            &[[2.0, 3.0], [f64::NEG_INFINITY, f64::INFINITY]],
            &mut outputs,
        )
        .unwrap();
        assert_eq!(outputs[0], [f64::NEG_INFINITY, f64::INFINITY]);
        eval.run(&[[f64::INFINITY, f64::INFINITY], [2.0, 3.0]], &mut outputs)
            .unwrap();
        assert_eq!(outputs[0], [f64::INFINITY, f64::INFINITY]);
        eval.run(
            &[[f64::NEG_INFINITY, f64::NEG_INFINITY], [2.0, 3.0]],
            &mut outputs,
        )
        .unwrap();
        assert_eq!(outputs[0], [f64::NEG_INFINITY, f64::NEG_INFINITY]);
    }

    #[test]
    fn t_jit_interval_div_f64() {
        let tree = deftree!(/ 'x 'y).unwrap();
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<f64>(&context, "xy").unwrap();
        let mut outputs = [[f64::NAN, f64::NAN]];
        eval.run(&[[4.0, 12.0], [2.0, 3.0]], &mut outputs).unwrap(); // Divisor entirely positive
        assert_eq!(outputs[0], [4.0 / 3.0, 12.0 / 2.0]);
        eval.run(&[[4.0, 12.0], [-3.0, -2.0]], &mut outputs) // Divisor entirely negative
            .unwrap();
        assert_eq!(outputs[0], [12.0 / -2.0, 4.0 / -3.0]);
        eval.run(&[[-12.0, -4.0], [2.0, 3.0]], &mut outputs) // Dividend negative, divisor positive
            .unwrap();
        assert_eq!(outputs[0], [-12.0 / 2.0, -4.0 / 3.0]);
        eval.run(&[[-4.0, 12.0], [2.0, 3.0]], &mut outputs).unwrap(); // Dividend crossing zero, divisor positive
        assert_eq!(outputs[0], [-4.0 / 2.0, 12.0 / 2.0]);
        eval.run(&[[-6.0, 0.0], [-3.0, 0.0]], &mut outputs).unwrap(); // Divisor [negative, 0], dividend non-positive
        assert_eq!(outputs[0], [0.0, f64::INFINITY]);
        eval.run(&[[0.0, 6.0], [0.0, 3.0]], &mut outputs).unwrap(); // Divisor [0, positive], dividend non-negative
        assert_eq!(outputs[0], [0.0, f64::INFINITY]);
        eval.run(&[[0.0, 6.0], [-3.0, 0.0]], &mut outputs).unwrap(); // Divisor [negative, 0], dividend non-negative
        assert_eq!(outputs[0], [f64::NEG_INFINITY, 0.0]);
        eval.run(&[[-6.0, 0.0], [0.0, 3.0]], &mut outputs).unwrap(); // Divisor [0, positive], dividend has negative part
        assert_eq!(outputs[0], [f64::NEG_INFINITY, 0.0]);
        eval.run(&[[2.0, 4.0], [-1.0, 1.0]], &mut outputs).unwrap(); // Divisor crossing zero strictly - ENTIRE
        assert_eq!(outputs[0], [f64::NEG_INFINITY, f64::INFINITY]);
        eval.run(&[[6.0, 6.0], [2.0, 2.0]], &mut outputs).unwrap(); // Point interval division
        assert_eq!(outputs[0], [3.0, 3.0]);
    }

    #[test]
    fn t_jit_interval_pow_comprehensive_f64() {
        use crate::assert_float_eq;
        let tree = deftree!(pow 'x 'y).unwrap();
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<f64>(&context, "xy").unwrap();
        let mut outputs = [[f64::NAN, f64::NAN]];
        // Helper function to check if interval is empty (NaN, NaN)
        let is_empty = |output: [f64; 2]| output[0].is_nan() && output[1].is_nan();
        // Helper function to check if interval is entire (-inf, inf)
        let is_entire =
            |output: [f64; 2]| output[0] == f64::NEG_INFINITY && output[1] == f64::INFINITY;
        // Test 1: NaN cases
        eval.run(&[[f64::NAN, f64::NAN], [2.0, 3.0]], &mut outputs)
            .unwrap();
        assert!(
            is_empty(outputs[0]),
            "NaN base should produce empty interval"
        );
        eval.run(&[[2.0, 3.0], [f64::NAN, f64::NAN]], &mut outputs)
            .unwrap();
        assert!(
            is_empty(outputs[0]),
            "NaN exponent should produce empty interval"
        );
        // Test 2: Exponent = 0
        eval.run(&[[2.0, 5.0], [0.0, 0.0]], &mut outputs).unwrap();
        assert_float_eq!(outputs[0][0], 1.0, 1e-12);
        assert_float_eq!(outputs[0][1], 1.0, 1e-12);
        // Test 3: Exponent = 2 (squaring), base crossing zero
        eval.run(&[[-3.0, 2.0], [2.0, 2.0]], &mut outputs).unwrap();
        assert_float_eq!(outputs[0][0], 0.0, 1e-12);
        assert_float_eq!(outputs[0][1], 9.0, 1e-12);
        // Test 4: Positive even integer exponent
        eval.run(&[[-2.0, 3.0], [4.0, 4.0]], &mut outputs).unwrap();
        assert_float_eq!(outputs[0][0], 0.0, 1e-12);
        assert_float_eq!(outputs[0][1], 81.0, 1e-12);
        // Test 5: Negative even integer exponent, non-zero base
        eval.run(&[[2.0, 4.0], [-2.0, -2.0]], &mut outputs).unwrap();
        assert_float_eq!(outputs[0][0], 1.0 / 16.0, 1e-12);
        assert_float_eq!(outputs[0][1], 1.0 / 4.0, 1e-12);
        // Test 6: Negative even integer exponent with zero base
        eval.run(&[[0.0, 0.0], [-2.0, -2.0]], &mut outputs).unwrap();
        assert!(is_empty(outputs[0]), "0^(-2) should be empty");
        // Test 7: Positive odd integer exponent
        eval.run(&[[-2.0, 3.0], [3.0, 3.0]], &mut outputs).unwrap();
        assert_float_eq!(outputs[0][0], -8.0, 1e-12);
        assert_float_eq!(outputs[0][1], 27.0, 1e-12);
        // Test 8: Odd negative integer exponent crossing zero
        eval.run(&[[-2.0, 3.0], [-3.0, -3.0]], &mut outputs)
            .unwrap();
        assert!(
            is_entire(outputs[0]),
            "Base crossing zero with negative odd exponent should be entire"
        );
        // Test 9: Odd negative integer exponent, not crossing zero
        eval.run(&[[2.0, 4.0], [-3.0, -3.0]], &mut outputs).unwrap();
        assert_float_eq!(outputs[0][0], 1.0 / 64.0, 1e-12);
        assert_float_eq!(outputs[0][1], 1.0 / 8.0, 1e-12);
        // Test 10: Base entirely below 1, positive rational exponent
        eval.run(&[[0.2, 0.8], [1.5, 2.5]], &mut outputs).unwrap();
        assert_float_eq!(outputs[0][0], 0.2f64.powf(2.5), 1e-12);
        assert_float_eq!(outputs[0][1], 0.8f64.powf(1.5), 1e-12);
        // Test 11: Base entirely above 1, positive rational exponent
        eval.run(&[[2.0, 4.0], [1.5, 2.5]], &mut outputs).unwrap();
        assert_float_eq!(outputs[0][0], 2.0f64.powf(1.5), 1e-12);
        assert_float_eq!(outputs[0][1], 4.0f64.powf(2.5), 1e-12);
        // Test 12: Base straddling 1, positive rational exponent
        eval.run(&[[0.5, 2.0], [1.5, 2.5]], &mut outputs).unwrap();
        assert_float_eq!(outputs[0][0], 0.5f64.powf(2.5), 1e-12);
        assert_float_eq!(outputs[0][1], 2.0f64.powf(2.5), 1e-12);
        // Test 13: Base entirely below 1, negative rational exponent
        eval.run(&[[0.2, 0.8], [-2.5, -1.5]], &mut outputs).unwrap();
        assert_float_eq!(outputs[0][0], 0.8f64.powf(-1.5), 1e-12);
        assert_float_eq!(outputs[0][1], 0.2f64.powf(-2.5), 1e-12);
        // Test 14: Base entirely above 1, negative rational exponent
        eval.run(&[[2.0, 4.0], [-2.5, -1.5]], &mut outputs).unwrap();
        assert_float_eq!(outputs[0][0], 4.0f64.powf(-2.5), 1e-12);
        assert_float_eq!(outputs[0][1], 2.0f64.powf(-1.5), 1e-12);
        // Test 15: Base straddling 1, negative rational exponent
        eval.run(&[[0.5, 2.0], [-2.5, -1.5]], &mut outputs).unwrap();
        assert_float_eq!(outputs[0][0], 2.0f64.powf(-2.5), 1e-12);
        assert_float_eq!(outputs[0][1], 0.5f64.powf(-2.5), 1e-12);
        // Test 16: Exponent crossing zero
        eval.run(&[[2.0, 3.0], [-1.0, 1.0]], &mut outputs).unwrap();
        assert_float_eq!(
            outputs[0][0],
            2.0f64.powf(-1.0).min(3.0f64.powf(-1.0)),
            1e-12
        );
        assert_float_eq!(outputs[0][1], 2.0f64.powf(1.0).max(3.0f64.powf(1.0)), 1e-12);
        // Test 17: Zero base with positive exponent
        eval.run(&[[0.0, 0.0], [2.0, 3.0]], &mut outputs).unwrap();
        assert_float_eq!(outputs[0][0], 0.0, 1e-12);
        assert_float_eq!(outputs[0][1], 0.0, 1e-12);
        // Test 18: Zero upper bound of base with negative exponent
        eval.run(&[[0.0, 0.0], [-2.5, -1.5]], &mut outputs).unwrap();
        assert!(is_empty(outputs[0]), "0^(-x) should be empty");
        // Test 19: Negative base with rational exponent (clamped to 0+)
        eval.run(&[[-2.0, -0.5], [1.5, 2.5]], &mut outputs).unwrap();
        assert_float_eq!(outputs[0][0], 0.0, 1e-12);
        assert_float_eq!(outputs[0][1], 0.0, 1e-12);
        // Test 20: Negative base crossing to positive with rational exponent
        eval.run(&[[-1.0, 2.0], [1.5, 2.5]], &mut outputs).unwrap();
        assert_float_eq!(outputs[0][0], 0.0, 1e-12);
        assert_float_eq!(outputs[0][1], 2.0f64.powf(2.5), 1e-12);
        // Test 21: Zero base with zero exponent
        eval.run(&[[0.0, 0.0], [0.0, 0.0]], &mut outputs).unwrap();
        assert_float_eq!(outputs[0][0], 1.0, 1e-12);
        assert_float_eq!(outputs[0][1], 1.0, 1e-12);
        // Test 22: Base with small positive values, large positive exponent
        eval.run(&[[0.1, 0.2], [10.0, 10.0]], &mut outputs).unwrap();
        assert_float_eq!(outputs[0][0], 0.1f64.powi(10), 1e-20);
        assert_float_eq!(outputs[0][1], 0.2f64.powi(10), 1e-20);
        // Test 23: Singleton base and exponent
        eval.run(&[[2.0, 2.0], [3.0, 3.0]], &mut outputs).unwrap();
        assert_float_eq!(outputs[0][0], 8.0, 1e-12);
        assert_float_eq!(outputs[0][1], 8.0, 1e-12);
    }

    #[test]
    fn t_jit_interval_comparisons_greater() {
        // Test > and >= operators
        check_interval_eval(
            deftree!(if (> 'x 'y) (+ 'x 1.) (- 'y 1.)).unwrap(),
            &[('x', 0., 5.), ('y', 3., 8.)],
            20,
            5,
        );
        check_interval_eval(
            deftree!(if (>= 'x 'y) (+ 'x 1.) (- 'y 1.)).unwrap(),
            &[('x', 0., 5.), ('y', 3., 8.)],
            20,
            5,
        );
    }

    #[test]
    fn t_jit_interval_comparison_edge_cases() {
        // Test with NaN, touching intervals, and exact overlaps
        check_interval_eval(
            deftree!(if (== 'x 'y) (+ 'x 10.) (- 'x 10.)).unwrap(),
            &[('x', 1., 5.), ('y', 1., 5.)],
            20,
            5,
        );
        check_interval_eval(
            deftree!(if (!= 'x 'y) (+ 'x 10.) (- 'x 10.)).unwrap(),
            &[('x', 1., 5.), ('y', 6., 10.)],
            20,
            5,
        );
    }

    #[test]
    fn t_jit_interval_subtract() {
        check_interval_eval(
            deftree!(- 'x 'y).unwrap(),
            &[('x', -5., 5.), ('y', -3., 7.)],
            20,
            5,
        );
    }

    #[test]
    fn t_jit_interval_multiply_signs() {
        // Test multiplication with different sign combinations
        check_interval_eval(
            deftree!(* 'x 'y).unwrap(),
            &[('x', -3., 3.), ('y', -5., 5.)],
            20,
            5,
        );
    }

    #[test]
    fn t_jit_interval_boolean_edge_cases() {
        // Test And with all combinations
        check_interval_eval(
            deftree!(if (and (> 'x 0) (< 'y 5)) (* 'x 2.) (+ 'y 3.)).unwrap(),
            &[('x', -1., 1.), ('y', 3., 6.)],
            20,
            5,
        );
        // Test Or with all combinations
        check_interval_eval(
            deftree!(if (or (> 'x 5) (< 'y 0)) (/ 'x 2.) (* 'y 2.)).unwrap(),
            &[('x', 4., 6.), ('y', -1., 1.)],
            20,
            5,
        );
        // Test Not with uncertain input
        check_interval_eval(
            deftree!(if (not (> 'x 0)) (+ 'x 5.) (- 'x 5.)).unwrap(),
            &[('x', -2., 2.)],
            20,
            5,
        );
    }

    #[test]
    fn t_jit_interval_mul_with_self_optimization() {
        let tree = deftree!(* 'x 'x).unwrap().compacted().unwrap();
        assert_eq!(tree.symbols().len(), 1);
        let context = JitContext::default();
        let eval = tree.jit_compile_interval::<f64>(&context, "x").unwrap();
        let mut outputs = vec![[f64::NAN; 2]];
        eval.run(&[[-2.0, 3.0]], &mut outputs).unwrap();
        let result = outputs[0];
        assert_float_eq!(result[0], 0.0);
        assert_float_eq!(result[1], 9.0);
    }

    #[test]
    fn t_jit_interval_random_circles_comparison() {
        // Compare JIT and non-JIT evaluation results.
        const XRANGE: (f64, f64) = (0.0, 128.0);
        const YRANGE: (f64, f64) = (0.0, 128.0);
        let tree = test_util::random_circles(XRANGE, YRANGE, (2.56, 12.8), 100);
        let mut rng = StdRng::seed_from_u64(42);
        let mut eval = IntervalEvaluator::new(&tree);
        let context = JitContext::default();
        let params = "xy";
        let jit_eval = tree
            .jit_compile_interval::<f64>(&context, params)
            .expect("Cannot JIT compile an interval evaluator");
        for _ in 0..100 {
            // Sample a random interval.
            let interval = [XRANGE, YRANGE].map(|range| {
                let mut bounds =
                    [0, 1].map(|_| range.0 + rng.random::<f64>() * (range.1 - range.0));
                if bounds[0] > bounds[1] {
                    bounds.swap(0, 1);
                }
                bounds
            });
            // Do the non-jit eval.
            for (label, interval) in params.chars().zip(interval.iter()) {
                eval.set_value(
                    label,
                    Interval::from_scalar(interval[0], interval[1])
                        .expect("Cannot create interval"),
                )
            }
            let result = eval.run().expect("Failed to run non-jit interval eval");
            assert_eq!(result.len(), 1);
            let mut jit_result = [[f64::NAN, f64::NAN]];
            jit_eval
                .run(&interval, &mut jit_result)
                .expect("Failed to run jit eval");
            let result = result[0]
                .scalar()
                .expect("Cannot retrieve scalar result from non-jit eval");
            let jit_result = jit_result[0];
            assert_float_eq!(
                jit_result[0],
                result.0,
                EPS,
                "The lower bound does not match"
            );
            assert_float_eq!(
                jit_result[1],
                result.1,
                EPS,
                "The upper bound does not match"
            );
        }
    }
}
