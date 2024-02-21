use inkwell::{
    execution_engine::JitFunction, types::VectorType, values::BasicValueEnum, AddressSpace,
    IntPredicate, OptimizationLevel,
};

use crate::{
    error::Error,
    tree::{Node::*, Tree, Value::*},
};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{JitCompiler, JitContext};

type UnsafeFuncType = unsafe extern "C" fn(*const __m256d, *mut __m256d, u64);

pub struct JitSimdArrayEvaluator<'ctx> {
    func: JitFunction<'ctx, UnsafeFuncType>,
    num_inputs: usize,
}

impl Tree {
    pub fn jit_compile_array<'ctx>(
        &'ctx self,
        context: &'ctx JitContext,
    ) -> Result<JitSimdArrayEvaluator, Error> {
        const FUNC_NAME: &str = "eiche_func";
        const SIMD_VEC_SIZE: u32 = 4;
        let num_roots = self.num_roots();
        let symbols = self.symbols();
        let context = &context.inner;
        let compiler = JitCompiler::new(context)?;
        let builder = &compiler.builder;
        let f64_type = context.f64_type();
        let i64_type = context.i64_type();
        let fvec_type = f64_type.vec_type(SIMD_VEC_SIZE);
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
                        &[bool_type.const_int(if *val { 1 } else { 0 }, false);
                            SIMD_VEC_SIZE as usize],
                    )),
                    Scalar(val) => BasicValueEnum::VectorValue(VectorType::const_vector(
                        &[f64_type.const_float(*val); SIMD_VEC_SIZE as usize],
                    )),
                },
                Symbol(_) => todo!(),
                Unary(_, _) => todo!(),
                Binary(_, _, _) => todo!(),
                Ternary(_, _, _, _) => todo!(),
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
        Ok(JitSimdArrayEvaluator {
            func,
            num_inputs: symbols.len(),
        })
    }
}
