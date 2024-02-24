use inkwell::{
    builder::Builder,
    execution_engine::JitFunction,
    intrinsics::Intrinsic,
    module::Module,
    types::{BasicTypeEnum, VectorType},
    values::{BasicMetadataValueEnum, BasicValueEnum},
    AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel,
};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::fmt::Debug;

use super::{JitCompiler, JitContext};
use crate::{
    error::Error,
    tree::{BinaryOp::*, Node::*, TernaryOp::*, Tree, UnaryOp::*, Value::*},
};

const SIMD_VEC_SIZE: usize = 4;

#[repr(C)]
#[derive(Copy, Clone)]
union WideFloat {
    vals: [f64; SIMD_VEC_SIZE],
    reg: __m256d,
}

impl WideFloat {
    pub fn from(val: f64) -> WideFloat {
        WideFloat {
            vals: [val; SIMD_VEC_SIZE],
        }
    }

    pub fn set(&mut self, val: f64, idx: usize) {
        unsafe {
            self.vals[idx] = val;
        }
    }

    pub fn ptr(&self) -> *const __m256d {
        unsafe { &self.reg as *const __m256d }
    }

    pub fn ptr_mut(&mut self) -> *mut __m256d {
        unsafe { &mut self.reg as *mut __m256d }
    }
}

impl Debug for WideFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe { write!(f, "{:?}", self.vals) }
    }
}

type UnsafeFuncType = unsafe extern "C" fn(*const __m256d, *mut __m256d, u64);

pub struct JitSimdArrayEvaluator<'ctx> {
    func: JitFunction<'ctx, UnsafeFuncType>,
    num_inputs: usize,
    num_outputs: usize,
    num_eval: usize,
    inputs: Vec<WideFloat>,
    outputs: Vec<WideFloat>,
}

impl<'ctx> JitSimdArrayEvaluator<'ctx> {
    pub fn create(
        func: JitFunction<'ctx, UnsafeFuncType>,
        num_inputs: usize,
        num_outputs: usize,
    ) -> JitSimdArrayEvaluator<'ctx> {
        JitSimdArrayEvaluator {
            func,
            num_inputs,
            num_outputs,
            num_eval: 0,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn push(&mut self, sample: &[f64]) {
        let index = self.num_eval % SIMD_VEC_SIZE;
        if index == 0 {
            self.inputs
                .extend(std::iter::repeat(WideFloat::from(f64::NAN)).take(self.num_inputs));
            self.outputs
                .extend(std::iter::repeat(WideFloat::from(f64::NAN)).take(self.num_outputs));
        }
        let inpsize = self.inputs.len();
        for (reg, val) in self.inputs[(inpsize - self.num_inputs)..]
            .iter_mut()
            .zip(sample.iter())
        {
            reg.set(*val, index);
        }
        self.num_eval += 1;
    }

    pub fn clear(&mut self) {
        self.inputs.clear();
        self.outputs.clear();
    }

    fn num_regs(&self) -> usize {
        (self.num_eval / SIMD_VEC_SIZE)
            + if self.num_eval % SIMD_VEC_SIZE > 0 {
                1
            } else {
                0
            }
    }

    pub fn run(&mut self, dst: &mut Vec<f64>) {
        unsafe {
            self.func.call(
                self.inputs[0].ptr(),
                self.outputs[0].ptr_mut(),
                self.num_regs() as u64,
            );
        }
        dst.clear();
        let mut offset = 0;
        let mut num_vals = 0;
        while offset < self.outputs.len() && num_vals < self.num_eval {
            for i in 0..SIMD_VEC_SIZE {
                for wf in &self.outputs[offset..(offset + self.num_outputs)] {
                    unsafe {
                        dst.push(wf.vals[i]);
                    }
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
    pub fn jit_compile_array<'ctx>(
        &'ctx self,
        context: &'ctx JitContext,
    ) -> Result<JitSimdArrayEvaluator, Error> {
        const FUNC_NAME: &str = "eiche_func";
        let num_roots = self.num_roots();
        let symbols = self.symbols();
        let context = &context.inner;
        let compiler = JitCompiler::new(context)?;
        let builder = &compiler.builder;
        let f64_type = context.f64_type();
        let i64_type = context.i64_type();
        let fvec_type = f64_type.vec_type(SIMD_VEC_SIZE as u32);
        let bool_type = context.bool_type();
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
                    Bool(val) => BasicValueEnum::VectorValue(VectorType::const_vector(
                        &[bool_type.const_int(if *val { 1 } else { 0 }, false); SIMD_VEC_SIZE],
                    )),
                    Scalar(val) => BasicValueEnum::VectorValue(VectorType::const_vector(
                        &[f64_type.const_float(*val); SIMD_VEC_SIZE],
                    )),
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
        Ok(JitSimdArrayEvaluator::create(
            func,
            symbols.len(),
            num_roots,
        ))
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
    use crate::deftree;

    use super::*;

    #[test]
    fn t_mul() {
        let tree = deftree!(* x y).unwrap();
        let context = JitContext::default();
        let mut eval = tree.jit_compile_array(&context).unwrap();
        const N_SAMPLES: usize = 100;
        let mut expected = Vec::with_capacity(N_SAMPLES * N_SAMPLES);
        for xi in 0..N_SAMPLES {
            let x = 0.1 + (xi as f64) * 0.1;
            for yi in 0..N_SAMPLES {
                let y = 0.1 + (yi as f64) * 0.1;
                eval.push(&[x, y]);
                expected.push(x * y);
            }
        }
        assert_eq!(eval.num_eval, N_SAMPLES * N_SAMPLES);
        let mut actual = Vec::with_capacity(N_SAMPLES * N_SAMPLES);
        eval.run(&mut actual);
        assert_eq!(actual, expected);
    }
}