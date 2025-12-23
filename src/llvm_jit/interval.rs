use super::{
    JitContext, NumberType, build_float_unary_intrinsic, build_vec_binary_intrinsic,
    build_vec_unary_intrinsic,
};
use crate::{
    BinaryOp::*, Error, Node::*, TernaryOp::*, Tree, UnaryOp::*, Value, interval::IntervalClass,
    llvm_jit::JitCompiler,
};
use inkwell::{
    AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel,
    builder::Builder,
    context::Context,
    execution_engine::JitFunction,
    module::Module,
    types::{FloatType, IntType, VectorType},
    values::{BasicValue, BasicValueEnum, FloatValue, FunctionValue, IntValue, VectorValue},
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
        let i32_type = context.i32_type();
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
                        builder,
                        i32_type,
                        ni,
                        &format!("reg_{ni}"),
                    )?
                    .as_basic_value_enum(),
                    Sqrt => build_interval_sqrt(
                        regs[*input].into_vector_value(),
                        builder,
                        &compiler.module,
                        float_type,
                        i32_type,
                        ni,
                    )?
                    .as_basic_value_enum(),
                    Abs => build_interval_abs(
                        regs[*input].into_vector_value(),
                        builder,
                        &compiler.module,
                        float_type,
                        i32_type,
                        ni,
                    )?
                    .as_basic_value_enum(),
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
                                    i32_type.const_int(0, false),
                                    &format!("extract_lo_{ni}"),
                                )?
                                .into_float_value(),
                            builder
                                .build_extract_element(
                                    ireg,
                                    i32_type.const_int(1, false),
                                    &format!("extract_lo_{ni}"),
                                )?
                                .into_float_value(),
                        );
                        let qlo = builder
                            .build_extract_element(
                                qinterval,
                                i32_type.const_int(0, false),
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
                                            i32_type.const_int(1, false),
                                            &format!("q_extract_0_{ni}"),
                                        )?
                                        .into_float_value(),
                                    qlo,
                                    &format!("nval_sub_{ni}"),
                                )?,
                                &format!("nval_{ni}"),
                            )?
                            .into_float_value();
                        let qval = build_float_rem_euclid(
                            qlo,
                            float_type.const_float(4.0),
                            builder,
                            &format!("q_rem_euclid_val_{ni}"),
                            ni,
                        )?;
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
                            build_interval_flip(sin_base, builder, i32_type, ni)?,
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
                                i32_type.const_int(0, false),
                                &format!("out_val_case_3_{ni}"),
                            )?,
                            builder.build_insert_element(
                                full_range,
                                build_vec_unary_intrinsic(
                                    builder,
                                    &compiler.module,
                                    "llvm.vector.reduce.fmax.*",
                                    &format!("case_3_max_reduce_{ni}"),
                                    sin_base,
                                )?
                                .into_float_value(),
                                i32_type.const_int(1, false),
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
                            build_check_interval_empty(ireg, builder, &compiler.module, ni)?,
                            VectorType::const_vector(&[
                                float_type.const_float(f64::NAN),
                                float_type.const_float(f64::NAN),
                            ])
                            .as_basic_value_enum(),
                            out,
                            &format!("reg_{ni}"),
                        )?
                    }
                    Cos => {
                        let ireg = regs[*input].into_vector_value();
                        let qinterval = build_vec_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.floor.*",
                            &format!("intermediate_floor_{ni}"),
                            builder.build_float_div(
                                ireg,
                                VectorType::const_vector(&[
                                    float_type.const_float(PI),
                                    float_type.const_float(PI),
                                ]),
                                &format!("div_pi_{ni}"),
                            )?,
                        )?
                        .into_vector_value();
                        let (lo, hi) = (
                            builder
                                .build_extract_element(
                                    ireg,
                                    i32_type.const_int(0, false),
                                    &format!("extract_lo_{ni}"),
                                )?
                                .into_float_value(),
                            builder
                                .build_extract_element(
                                    ireg,
                                    i32_type.const_int(1, false),
                                    &format!("extract_lo_{ni}"),
                                )?
                                .into_float_value(),
                        );
                        let qlo = builder
                            .build_extract_element(
                                qinterval,
                                i32_type.const_int(0, false),
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
                                            i32_type.const_int(1, false),
                                            &format!("q_extract_0_{ni}"),
                                        )?
                                        .into_float_value(),
                                    qlo,
                                    &format!("nval_sub_{ni}"),
                                )?,
                                &format!("nval_{ni}"),
                            )?
                            .into_float_value();
                        let qval = builder
                            .build_select(
                                builder.build_float_compare(
                                    FloatPredicate::UEQ,
                                    qlo,
                                    builder.build_float_mul(
                                        float_type.const_float(2.0),
                                        build_float_unary_intrinsic(
                                            builder,
                                            &compiler.module,
                                            "llvm.floor.*",
                                            &format!("intermediate_qval_floor_{ni}"),
                                            builder.build_float_mul(
                                                qlo,
                                                float_type.const_float(0.5),
                                                &format!("qval_half_mul_{ni}"),
                                            )?,
                                        )?
                                        .into_float_value(),
                                        &format!("qval_doubling_{ni}"),
                                    )?,
                                    &format!("qval_comparison_{ni}"),
                                )?,
                                float_type.const_float(0.0),
                                float_type.const_float(1.0),
                                &format!("qval_{ni}"),
                            )?
                            .into_float_value();
                        let q_zero = builder.build_float_compare(
                            FloatPredicate::UEQ,
                            qval,
                            float_type.const_float(0.0),
                            &format!("qval_is_zero_{ni}"),
                        )?;
                        let full_range = VectorType::const_vector(&[
                            float_type.const_float(-1.0),
                            float_type.const_float(1.0),
                        ]);
                        let cos_base = build_vec_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.cos.*",
                            &format!("sin_base_{ni}"),
                            ireg,
                        )?
                        .into_vector_value();
                        let out = builder.build_select(
                            builder.build_float_compare(
                                FloatPredicate::UEQ,
                                nval,
                                float_type.const_float(0.0),
                                &format!("nval_zero_compare_{ni}"),
                            )?,
                            builder
                                .build_select(
                                    q_zero,
                                    build_interval_flip(cos_base, builder, i32_type, ni)?,
                                    cos_base,
                                    &format!("edge_case_1_{ni}"),
                                )?
                                .into_vector_value(),
                            builder
                                .build_select(
                                    builder.build_float_compare(
                                        FloatPredicate::ULE,
                                        nval,
                                        float_type.const_float(1.0),
                                        &format!("nval_one_compare_{ni}"),
                                    )?,
                                    builder
                                        .build_select(
                                            q_zero,
                                            builder.build_insert_element(
                                                full_range,
                                                build_vec_unary_intrinsic(
                                                    builder,
                                                    &compiler.module,
                                                    "llvm.vector.reduce.fmax.*",
                                                    &format!("case_3_max_reduce_{ni}"),
                                                    cos_base,
                                                )?
                                                .into_float_value(),
                                                i32_type.const_int(1, false),
                                                &format!("out_val_case_2_{ni}"),
                                            )?,
                                            builder.build_insert_element(
                                                full_range,
                                                build_vec_unary_intrinsic(
                                                    builder,
                                                    &compiler.module,
                                                    "llvm.vector.reduce.fmin.*",
                                                    &format!("case_3_min_reduce_{ni}"),
                                                    cos_base,
                                                )?
                                                .into_float_value(),
                                                i32_type.const_int(0, false),
                                                &format!("out_val_case_3_{ni}"),
                                            )?,
                                            &format!("nval_cases_{ni}"),
                                        )?
                                        .into_vector_value(),
                                    full_range,
                                    &format!("out_val_edge_case_0_{ni}"),
                                )?
                                .into_vector_value(),
                            &format!("out_val_{ni}"),
                        )?;
                        builder.build_select(
                            build_check_interval_empty(ireg, builder, &compiler.module, ni)?,
                            VectorType::const_vector(&[
                                float_type.const_float(f64::NAN),
                                float_type.const_float(f64::NAN),
                            ])
                            .as_basic_value_enum(),
                            out,
                            &format!("reg_{ni}"),
                        )?
                    }
                    Tan => {
                        let ireg = regs[*input].into_vector_value();
                        let lo = builder
                            .build_extract_element(
                                ireg,
                                i32_type.const_int(0, false),
                                &format!("tan_width_rhs_{ni}"),
                            )?
                            .into_float_value();
                        let width = builder.build_float_sub(
                            builder
                                .build_extract_element(
                                    ireg,
                                    i32_type.const_int(1, false),
                                    &format!("tan_width_lhs_{ni}"),
                                )?
                                .into_float_value(),
                            lo,
                            &format!("tan_width_{ni}"),
                        )?;
                        let everything = VectorType::const_vector(&[
                            float_type.const_float(f64::NEG_INFINITY),
                            float_type.const_float(f64::INFINITY),
                        ]);
                        let out = builder.build_select(
                            builder.build_float_compare(
                                FloatPredicate::UGE,
                                width,
                                float_type.const_float(PI),
                                &format!("tan_pi_compare_{ni}"),
                            )?,
                            everything,
                            {
                                // Shift lo to an equivalent value in -pi/2 to pi/2.
                                let lo = builder.build_float_sub(
                                    build_float_rem_euclid(
                                        builder.build_float_add(
                                            lo,
                                            float_type.const_float(FRAC_PI_2),
                                            &format!("tan_pi_shift_add_{ni}"),
                                        )?,
                                        float_type.const_float(PI),
                                        builder,
                                        &format!("tan_rem_euclid_{ni}"),
                                        ni,
                                    )?,
                                    float_type.const_float(FRAC_PI_2),
                                    &format!("tan_shifted_lo_{ni}"),
                                )?;
                                let hi = builder.build_float_add(
                                    lo,
                                    width,
                                    &format!("tan_shifted_hi_{ni}"),
                                )?;
                                builder
                                    .build_select(
                                        builder.build_float_compare(
                                            FloatPredicate::UGE,
                                            hi,
                                            float_type.const_float(FRAC_PI_2),
                                            &format!("tan_second_compare_{ni}"),
                                        )?,
                                        everything,
                                        {
                                            let sin = build_vec_unary_intrinsic(
                                                builder,
                                                &compiler.module,
                                                "llvm.sin.*",
                                                "sin_call",
                                                regs[*input].into_vector_value(),
                                            )?;
                                            let cos = build_vec_unary_intrinsic(
                                                builder,
                                                &compiler.module,
                                                "llvm.cos.*",
                                                "cos_call",
                                                regs[*input].into_vector_value(),
                                            )?;
                                            builder.build_float_div(
                                                sin.into_vector_value(),
                                                cos.into_vector_value(),
                                                &format!("reg_{ni}"),
                                            )?
                                        },
                                        &format!("tan_regular_tan_{ni}"),
                                    )?
                                    .into_vector_value()
                            },
                            &format!("reg_{ni}"),
                        )?;
                        builder.build_select(
                            build_check_interval_empty(ireg, builder, &compiler.module, ni)?,
                            VectorType::const_vector(&[
                                float_type.const_float(f64::NAN),
                                float_type.const_float(f64::NAN),
                            ])
                            .as_basic_value_enum(),
                            out,
                            &format!("reg_{ni}"),
                        )?
                    }
                    Log => {
                        let ireg = regs[*input].into_vector_value();
                        let is_neg = builder.build_float_compare(
                            FloatPredicate::ULE,
                            ireg,
                            interval_type.const_zero(),
                            &format!("log_neg_compare_{ni}"),
                        )?;
                        let log_base = build_vec_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.log.*",
                            "log_call",
                            regs[*input].into_vector_value(),
                        )?
                        .into_vector_value();
                        builder.build_select(
                            builder
                                .build_extract_element(
                                    is_neg,
                                    i32_type.const_int(1, false),
                                    &format!("log_hi_neg_check_{ni}"),
                                )?
                                .into_int_value(),
                            VectorType::const_vector(&[
                                float_type.const_float(f64::NAN),
                                float_type.const_float(f64::NAN),
                            ]),
                            builder
                                .build_select(
                                    builder
                                        .build_extract_element(
                                            is_neg,
                                            i32_type.const_int(0, false),
                                            &format!("log_hi_neg_check_{ni}"),
                                        )?
                                        .into_int_value(),
                                    builder.build_insert_element(
                                        log_base,
                                        float_type.const_float(f64::NEG_INFINITY),
                                        i32_type.const_int(0, false),
                                        &format!("log_range_across_zero_{ni}"),
                                    )?,
                                    log_base,
                                    &format!("log_simple_case_{ni}"),
                                )?
                                .into_vector_value(),
                            &format!("reg_{ni}"),
                        )?
                    }
                    Exp => build_vec_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.exp.*",
                        &format!("exp_call_{ni}"),
                        regs[*input].into_vector_value(),
                    )?,
                    Floor => build_vec_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.floor.*",
                        &format!("floor_call_{ni}"),
                        regs[*input].into_vector_value(),
                    )?,
                    Not => return Err(Error::TypeMismatch),
                },
                Binary(op, lhs, rhs) => match op {
                    Add => builder
                        .build_float_add(
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                            &format!("reg_{ni}"),
                        )?
                        .as_basic_value_enum(),
                    Subtract => builder
                        .build_float_sub(
                            regs[*lhs].into_vector_value(),
                            build_interval_flip(
                                regs[*rhs].into_vector_value(),
                                builder,
                                i32_type,
                                ni,
                            )?,
                            &format!("reg_{ni}"),
                        )?
                        .as_basic_value_enum(),
                    Multiply => build_interval_mul(
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        builder,
                        &compiler.module,
                        i32_type,
                        ni,
                    )?
                    .as_basic_value_enum(),
                    Divide => build_interval_div(
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                        builder,
                        &compiler.module,
                        i32_type,
                        float_type,
                        ni,
                    )?
                    .as_basic_value_enum(),
                    Pow => todo!(),
                    Min => build_vec_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.minnum.*",
                        &format!("min_call_{ni}"),
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                    )?,
                    Max => build_vec_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.maxnum.*",
                        &format!("max_call_{ni}"),
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                    )?,
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

fn build_check_interval_spanning_zero<'ctx>(
    input: VectorValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    i32_type: IntType<'ctx>,
    prefix: &str,
    index: usize,
) -> Result<IntValue<'ctx>, Error> {
    let is_neg = builder.build_float_compare(
        FloatPredicate::ULT,
        input,
        input.get_type().const_zero(),
        &format!("{prefix}_zero_spanning_check_{index}"),
    )?;
    Ok(build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.vector.reduce.and.*",
        &format!("{prefix}_zero_spanning_reduce_{index}"),
        builder.build_int_compare(
            IntPredicate::NE,
            is_neg,
            build_interval_flip(is_neg, builder, i32_type, index)?,
            &format!("{prefix}_zero_spanning_flip_comparison_{index}"),
        )?,
    )?
    .into_int_value())
}

fn build_interval_pow<'ctx>(
    lhs: VectorValue<'ctx>,
    rhs: VectorValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    i32_type: IntType<'ctx>,
    flt_type: FloatType<'ctx>,
    index: usize,
    function: FunctionValue<'ctx>,
    context: &'ctx Context,
) -> Result<VectorValue<'ctx>, Error> {
    let is_any_nan = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.vector.reduce.or.*",
        &format!("pow_nan_check_{index}"),
        builder.build_shuffle_vector(
            lhs,
            rhs,
            VectorType::const_vector(&[
                i32_type.const_int(0, false),
                i32_type.const_int(1, false),
                i32_type.const_int(2, false),
                i32_type.const_int(3, false),
            ]),
            &format!("pow_nan_check_concat_{index}"),
        )?,
    )?
    .into_int_value();
    let is_exponent_zero = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.vector.reduce.and.*",
        &format!("pow_zero_check_reduce_{index}"),
        builder.build_float_compare(
            FloatPredicate::UEQ,
            rhs,
            rhs.get_type().const_zero(),
            &format!("pow_zero_check_{index}"),
        )?,
    )?
    .into_int_value();
    let is_square = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.vector.reduce.and.*",
        &format!("pow_square_check_reduce_{index}"),
        builder.build_float_compare(
            FloatPredicate::UEQ,
            rhs,
            VectorType::const_vector(&[flt_type.const_float(2.0), flt_type.const_float(2.0)]),
            &format!("pow_square_check_{index}"),
        )?,
    )?
    .into_int_value();
    let rhs_floor = build_vec_unary_intrinsic(
        builder,
        &module,
        "llvm.floor.*",
        &format!("pow_integer_check_floor_call_{index}"),
        rhs,
    )?
    .into_vector_value();
    let is_exponent_singleton_integer = build_vec_unary_intrinsic(
        builder,
        module,
        "llvm.vector.reduce.and.*",
        &format!("pow_integer_check_reduce_{index}"),
        builder.build_float_compare(
            FloatPredicate::UEQ,
            rhs_floor,
            build_interval_flip(rhs, builder, i32_type, index)?,
            &format!("pow_integer_check_compare_{index}"),
        )?,
    )?
    .into_int_value();
    let everything = VectorType::const_vector(&[
        flt_type.const_float(f64::NEG_INFINITY),
        flt_type.const_float(f64::INFINITY),
    ]);
    let integer_bb =
        context.append_basic_block(function, &format!("pow_singleton_integer_exponent_{index}"));
    let general_bb = context.append_basic_block(function, &format!("pow_general_exponent_{index}"));
    let merge_bb = context.append_basic_block(function, &format!("pow_outer_merge_{index}"));
    builder.build_conditional_branch(is_exponent_singleton_integer, integer_bb, general_bb)?;
    let integer_case: VectorValue<'ctx> = {
        builder.position_at_end(integer_bb);
        let exponent = builder.build_float_to_signed_int(
            builder
                .build_extract_element(
                    rhs_floor,
                    i32_type.const_zero(),
                    &format!("pow_extract_floor_{index}"),
                )?
                .into_float_value(),
            i32_type,
            &format!("pow_exponent_to_integer_convert_{index}"),
        )?;
        let is_odd = builder.build_and(
            exponent,
            i32_type.const_int(1, false),
            &format!("pow_integer_exp_odd_check_{index}"),
        )?;
        todo!();
    };
    let general_case: VectorValue<'ctx> = {
        builder.position_at_end(general_bb);
        todo!();
    };
    builder.position_at_end(merge_bb);
    let phi = builder.build_phi(lhs.get_type(), &format!("outer_branch_phi_{index}"))?;
    phi.add_incoming(&[(&integer_case, integer_bb), (&general_case, general_bb)]);
    Ok(builder
        .build_select(
            is_any_nan,
            VectorType::const_vector(&[
                flt_type.const_float(f64::NAN),
                flt_type.const_float(f64::NAN),
            ]),
            builder
                .build_select(
                    is_exponent_zero,
                    VectorType::const_vector(&[
                        flt_type.const_float(1.0),
                        flt_type.const_float(1.0),
                    ]),
                    builder
                        .build_select(
                            is_square,
                            {
                                let sqbase = builder.build_float_mul(
                                    lhs,
                                    lhs,
                                    &format!("pow_lhs_square_base_{index}"),
                                )?;
                                builder
                                    .build_select(
                                        build_check_interval_spanning_zero(
                                            lhs,
                                            builder,
                                            module,
                                            i32_type,
                                            "pow_square_case",
                                            index,
                                        )?,
                                        builder.build_insert_element(
                                            sqbase.get_type().const_zero(),
                                            build_vec_unary_intrinsic(
                                                builder,
                                                module,
                                                "llvm.vector.reduce.fmax.*",
                                                &format!(
                                                    "pow_square_case_zero_spanning_max_{index}"
                                                ),
                                                sqbase,
                                            )?
                                            .into_float_value(),
                                            i32_type.const_int(1, false),
                                            &format!("pow_square_insert_elem_{index}"),
                                        )?,
                                        sqbase,
                                        &format!("pow_square_case_{index}"),
                                    )?
                                    .into_vector_value()
                            },
                            phi.as_basic_value().into_vector_value(),
                            &format!("pow_square_check_select_{index}"),
                        )?
                        .into_vector_value(),
                    &format!("pow_zero_check_select_{index}"),
                )?
                .into_vector_value(),
            &format!("pow_nan_check_select_{index}"),
        )?
        .into_vector_value())
}

fn build_interval_abs<'ctx>(
    ireg: VectorValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    float_type: FloatType<'ctx>,
    i32_type: IntType<'ctx>,
    ni: usize,
) -> Result<VectorValue<'ctx>, Error> {
    let lt_zero = builder.build_float_compare(
        FloatPredicate::ULT,
        ireg,
        VectorType::const_vector(&[float_type.const_float(0.), float_type.const_float(0.)]),
        &format!("lt_zero_{ni}"),
    )?;
    Ok(builder
        .build_select(
            builder
                .build_extract_element(
                    lt_zero,
                    i32_type.const_int(1, false),
                    &format!("first_lt_zero_{ni}"),
                )?
                .into_int_value(),
            // (-hi, -lo)
            build_interval_negate(ireg, builder, i32_type, ni, &format!("intermediate_1_{ni}"))?,
            builder
                .build_select(
                    builder
                        .build_extract_element(
                            lt_zero,
                            i32_type.const_int(0, false),
                            &format!("first_lt_zero_{ni}"),
                        )?
                        .into_int_value(),
                    // (0.0, max(abs(lo), abs(hi)))
                    builder.build_insert_element(
                        ireg.get_type().const_zero(),
                        build_vec_unary_intrinsic(
                            builder,
                            module,
                            "llvm.vector.reduce.fmax.*",
                            &format!("fmax_reduce_call_{ni}"),
                            build_vec_unary_intrinsic(
                                builder,
                                module,
                                "llvm.fabs.*",
                                &format!("abs_call_{ni}"),
                                ireg,
                            )?
                            .into_vector_value(),
                        )?
                        .into_float_value(),
                        i32_type.const_int(1, false),
                        &format!("intermediate_2_{ni}"),
                    )?,
                    // (lo, hi),
                    ireg,
                    &format!("intermediate_3_{ni}"),
                )?
                .into_vector_value(),
            &format!("reg_{ni}"),
        )?
        .into_vector_value())
}

fn build_interval_sqrt<'ctx>(
    input: VectorValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    flt_type: FloatType<'ctx>,
    i32_type: IntType<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    Ok(builder
        .build_select(
            // Check each lane for NaN, then reduce to check if this interval is empty.
            build_check_interval_empty(input, builder, module, index)?,
            // The interval is empty, so return an emtpy (NaN) interval.
            VectorType::const_vector(&[
                flt_type.const_float(f64::NAN),
                flt_type.const_float(f64::NAN),
            ])
            .as_basic_value_enum(),
            {
                // Interval is not empty.
                let lt_zero = builder.build_float_compare(
                    FloatPredicate::ULT,
                    input,
                    VectorType::const_vector(&[flt_type.const_float(0.), flt_type.const_float(0.)]),
                    &format!("lt_zero_{index}"),
                )?;
                let sqrt = build_vec_unary_intrinsic(
                    builder,
                    module,
                    "llvm.sqrt.*",
                    &format!("sqrt_call_{index}"),
                    build_vec_unary_intrinsic(
                        builder,
                        module,
                        "llvm.fabs.*",
                        &format!("fabs_call_{index}"),
                        input,
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
                            i32_type.const_int(0, false),
                            &format!("first_lt_zero_{index}"),
                        )?
                        .into_int_value(),
                    builder.build_select(
                        builder
                            .build_extract_element(
                                lt_zero,
                                i32_type.const_int(1, false),
                                &format!("second_lt_zero_{index}"),
                            )?
                            .into_int_value(),
                        VectorType::const_vector(&[
                            flt_type.const_float(f64::NAN),
                            flt_type.const_float(f64::NAN),
                        ]),
                        builder.build_float_mul(
                            sqrt,
                            VectorType::const_vector(&[
                                flt_type.const_float(0.0),
                                flt_type.const_float(1.0),
                            ]),
                            &format!("sqrt_domain_clipping_{index}"),
                        )?,
                        &format!("sqrt_edge_case_{index}"),
                    )?,
                    sqrt.as_basic_value_enum(),
                    &format!("sqrt_branching_{index}"),
                )?
            },
            &format!("reg_{index}"),
        )?
        .into_vector_value())
}

fn build_interval_flip<'ctx>(
    input: VectorValue<'ctx>,
    builder: &'ctx Builder,
    i32_type: IntType<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    Ok(builder.build_shuffle_vector(
        input,
        input.get_type().get_undef(),
        VectorType::const_vector(&[i32_type.const_int(1, false), i32_type.const_int(0, false)]),
        &format!("out_val_case_2_{index}"),
    )?)
}

fn build_interval_compose<'ctx>(
    lo: FloatValue<'ctx>,
    hi: FloatValue<'ctx>,
    builder: &'ctx Builder,
    i32_type: IntType<'ctx>,
    suffix: &str,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    Ok(builder.build_insert_element(
        builder.build_insert_element(
            lo.get_type().vec_type(2).const_zero(),
            lo,
            i32_type.const_int(0, false),
            &format!("interval_compose_{suffix}_lo_{index}"),
        )?,
        hi,
        i32_type.const_int(1, false),
        &format!("interval_compose_{suffix}_hi_{index}"),
    )?)
}

fn build_interval_div<'ctx>(
    lhs: VectorValue<'ctx>,
    rhs: VectorValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    i32_type: IntType<'ctx>,
    flt_type: FloatType<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    use crate::interval::IntervalClass::*;
    let mask = builder.build_int_add(
        builder.build_int_mul(
            build_interval_classify(lhs, builder, module, i32_type, index)?,
            i32_type.const_int(7, false),
            &format!("interval_div_mask_imul_{index}"),
        )?,
        build_interval_classify(rhs, builder, module, i32_type, index)?,
        &format!("interval_div_mask_{index}"),
    )?;
    let straight = builder.build_float_div(lhs, rhs, &format!("interval_div_straight_{index}"))?;
    let cross = builder.build_float_div(
        lhs,
        build_interval_flip(rhs, builder, i32_type, index)?,
        &format!("interval_div_cross_{index}"),
    )?;
    let combos = builder.build_shuffle_vector(
        straight,
        cross,
        VectorType::const_vector(&[
            i32_type.const_int(0, false),
            i32_type.const_int(1, false),
            i32_type.const_int(2, false),
            i32_type.const_int(3, false),
        ]),
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
        VectorType::const_vector(&[
            flt_type.const_float(f64::NEG_INFINITY),
            flt_type.const_float(f64::INFINITY),
        ]),
        lhs.get_type().const_zero(),
        builder.build_shuffle_vector(
            combos,
            combos.get_type().get_undef(),
            VectorType::const_vector(&[i32_type.const_int(1, false), i32_type.const_int(2, false)]),
            &format!("interval_div_case_spanning_negative_{index}"),
        )?,
        builder.build_shuffle_vector(
            combos,
            combos.get_type().get_undef(),
            VectorType::const_vector(&[i32_type.const_int(0, false), i32_type.const_int(3, false)]),
            &format!("interval_div_case_spanning_positive_{index}"),
        )?,
        build_interval_compose(
            builder
                .build_extract_element(
                    combos,
                    i32_type.const_int(3, false),
                    &format!("interval_div_case_neg_neg_zero_intermediate_0_{index}"),
                )?
                .into_float_value(),
            flt_type.const_float(f64::INFINITY),
            builder,
            i32_type,
            "case_neg_neg_zero",
            index,
        )?,
        builder.build_shuffle_vector(
            combos,
            combos.get_type().get_undef(),
            VectorType::const_vector(&[i32_type.const_int(3, false), i32_type.const_int(2, false)]),
            &format!("interval_div_case_neg_zero_neg_{index}"),
        )?,
        build_interval_compose(
            flt_type.const_float(f64::NEG_INFINITY),
            builder
                .build_extract_element(
                    combos,
                    i32_type.const_int(1, false),
                    &format!("interval_div_case_neg_zero_positive_upper_{index}"),
                )?
                .into_float_value(),
            builder,
            i32_type,
            "interval_div_case_neg_zero_positive",
            index,
        )?,
        straight,
        build_interval_compose(
            flt_type.const_float(f64::NEG_INFINITY),
            builder
                .build_extract_element(
                    combos,
                    i32_type.const_int(0, false),
                    &format!("interval_div_case_zero_positive_neg_{index}"),
                )?
                .into_float_value(),
            builder,
            i32_type,
            "interval_div_case_zero_positive_neg",
            index,
        )?,
        build_interval_flip(straight, builder, i32_type, index)?,
        build_interval_compose(
            builder
                .build_extract_element(
                    combos,
                    i32_type.const_int(2, false),
                    &format!("interval_div_case_zero_positive_extract_{index}"),
                )?
                .into_float_value(),
            flt_type.const_float(f64::INFINITY),
            builder,
            i32_type,
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
        .fold(
            Ok(VectorType::const_vector(&[
                flt_type.const_float(f64::NAN),
                flt_type.const_float(f64::NAN),
            ])),
            |acc, (i, (cases, out))| match acc {
                Ok(acc) => {
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
                }
                Err(e) => Err(e),
            },
        )
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
    i32_type: IntType<'ctx>,
    index: usize,
) -> Result<IntValue<'ctx>, Error> {
    let (is_empty, is_neg, is_eq) = (
        build_check_interval_empty(input, builder, module, index)?,
        builder.build_float_compare(
            FloatPredicate::ULT,
            input,
            input.get_type().const_zero(),
            &format!("interval_classify_neg_check_{index}"),
        )?,
        builder.build_float_compare(
            FloatPredicate::UEQ,
            input,
            input.get_type().const_zero(),
            &format!("interval_classify_zero_check_{index}"),
        )?,
    );
    let (lneg, rneg, leq, req) = (
        builder
            .build_extract_element(
                is_neg,
                i32_type.const_int(0, false),
                &format!("interval_classify_left_neg_{index}"),
            )?
            .into_int_value(),
        builder
            .build_extract_element(
                is_neg,
                i32_type.const_int(1, false),
                &format!("interval_classify_right_neg_{index}"),
            )?
            .into_int_value(),
        builder
            .build_extract_element(
                is_eq,
                i32_type.const_int(0, false),
                &format!("interval_classify_left_eq_{index}"),
            )?
            .into_int_value(),
        builder
            .build_extract_element(
                is_eq,
                i32_type.const_int(1, false),
                &format!("interval_classify_right_eq_{index}"),
            )?
            .into_int_value(),
    );
    Ok(builder
        .build_select(
            is_empty,
            i32_type.const_zero(),
            builder
                .build_select(
                    lneg,
                    builder
                        .build_select(
                            rneg,
                            i32_type.const_int(1, false),
                            builder
                                .build_select(
                                    req,
                                    i32_type.const_int(2, false),
                                    i32_type.const_int(4, false),
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
                                    i32_type.const_int(3, false),
                                    i32_type.const_int(5, false),
                                    &format!("interval_classify_leq_req_cases_{index}"),
                                )?
                                .into_int_value(),
                            i32_type.const_int(6, false),
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
    lhs: VectorValue<'ctx>,
    rhs: VectorValue<'ctx>,
    builder: &'ctx Builder,
    module: &'ctx Module,
    i32_type: IntType<'ctx>,
    index: usize,
) -> Result<VectorValue<'ctx>, Error> {
    let straight = builder.build_float_mul(lhs, rhs, &format!("mul_straight_{index}"))?;
    let cross = builder.build_float_mul(
        lhs,
        build_interval_flip(rhs, builder, i32_type, index)?,
        &format!("mul_cross_{index}"),
    )?;
    let concat = builder.build_shuffle_vector(
        straight,
        cross,
        VectorType::const_vector(&[
            i32_type.const_int(0, false),
            i32_type.const_int(1, false),
            i32_type.const_int(2, false),
            i32_type.const_int(3, false),
        ]),
        &format!("mul_concat_candidates_{index}"),
    )?;
    Ok(build_interval_compose(
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
        i32_type,
        "interval_mul_compose",
        index,
    )?)
}

fn build_interval_negate<'ctx>(
    input: VectorValue<'ctx>,
    builder: &'ctx Builder,
    i32_type: IntType<'ctx>,
    index: usize,
    name: &str,
) -> Result<VectorValue<'ctx>, Error> {
    Ok(builder.build_shuffle_vector(
        builder.build_float_neg(input, &format!("negate_{index}"))?,
        input.get_type().get_undef(),
        VectorType::const_vector(&[i32_type.const_int(1, false), i32_type.const_int(0, false)]),
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
    name: &str,
    index: usize,
) -> Result<FloatValue<'ctx>, Error> {
    let qval = builder.build_float_rem(lhs, rhs, &format!("q_rem_val_{index}"))?;
    Ok(builder
        .build_select(
            builder.build_float_compare(
                FloatPredicate::ULT,
                qval,
                lhs.get_type().const_zero(),
                &format!("rem_euclid_compare_{index}"),
            )?,
            builder.build_float_add(
                qval,
                lhs.get_type().const_float(4.0),
                &format!("rem_euclid_correction_{index}"),
            )?,
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
            outputs[0],
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
}
