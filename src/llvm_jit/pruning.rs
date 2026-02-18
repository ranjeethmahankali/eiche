use super::{
    JitCompiler, build_vec_unary_intrinsic, fast_math,
    interval::{self, Constants},
    simd_array::{self, JitSimdBuffers, SimdVec, Wide},
    single,
};
use crate::{
    BinaryOp::*,
    Error,
    Node::{self, *},
    TernaryOp::*,
    Tree, Value,
    analyze::DependencyTable,
    tree::is_node_scalar,
};
use inkwell::{
    AddressSpace, IntPredicate, OptimizationLevel,
    basic_block::BasicBlock,
    builder::Builder,
    execution_engine::{ExecutionEngine, JitFunction},
    module::Module,
    types::{BasicType, IntType, VectorType},
    values::{
        BasicValue, BasicValueEnum, FunctionValue, IntValue, PhiValue, PointerValue, VectorValue,
    },
};
use std::{
    collections::{HashMap, HashSet},
    ffi::c_void,
    marker::PhantomData,
    ops::Range,
};

use super::{JitContext, NumberType};

type NativePruningIntervalFunc = unsafe extern "C" fn(
    *const c_void, // Inputs
    *mut c_void,   // Outputs
    *mut u32,      // Signals - tells future pruned evaluations how to skip and jump.
);

type NativeSingleFunc = unsafe extern "C" fn(
    *const c_void, // Inputs,
    *mut c_void,   // Outputs,
    *const u32,    // Signals,
);

type NativeSimdFunc = unsafe extern "C" fn(
    *const c_void, // Inputs
    *mut c_void,   // Outputs
    u64,           // Number of evals.
    *const u32,    // Signals,
);

type NativeIntervalFunc = unsafe extern "C" fn(
    *const c_void, // Inputs
    *mut c_void,   // Outputs
    *const u32,    // Signals,
);

pub struct JitPruner<'ctx, T: NumberType> {
    func: JitFunction<'ctx, NativePruningIntervalFunc>,
    n_inputs: usize,
    n_outputs: usize,
    n_signals: usize,
    params: String,
    blocks: Box<[Block]>,
    merge_block_map: HashMap<usize, usize>,
    block_signal_map: Box<[usize]>,
    tree: Tree,
    phantom: PhantomData<T>,
}

pub struct JitPruningFn<'ctx, T: NumberType> {
    func: JitFunction<'ctx, NativeSingleFunc>,
    n_inputs: usize,
    n_outputs: usize,
    n_signals: usize,
    _phantom: PhantomData<T>,
}

pub struct JitPruningSimdFn<'ctx, T: NumberType> {
    func: JitFunction<'ctx, NativeSimdFunc>,
    n_signals: usize,
    phantom: PhantomData<T>,
}

pub struct JitPruningIntervalFn<'ctx, T: NumberType> {
    func: JitFunction<'ctx, NativeIntervalFunc>,
    n_inputs: usize,
    n_outputs: usize,
    n_signals: usize,
    _phantom: PhantomData<T>,
}

pub struct JitPrunerSync<'ctx, T: NumberType> {
    func: NativePruningIntervalFunc,
    n_inputs: usize,
    n_outputs: usize,
    n_signals: usize,
    phantom: PhantomData<&'ctx JitPruner<'ctx, T>>,
}

pub struct JitPruningFnSync<'ctx, T: NumberType> {
    func: NativeSingleFunc,
    n_inputs: usize,
    n_outputs: usize,
    n_signals: usize,
    phantom: PhantomData<&'ctx JitPruningFn<'ctx, T>>,
}

pub struct JitPruningSimdFnSync<'ctx, T: NumberType> {
    func: NativeSimdFunc,
    n_signals: usize,
    phantom: PhantomData<&'ctx JitPruningSimdFn<'ctx, T>>,
}

pub struct JitPruningIntervalFnSync<'ctx, T: NumberType> {
    func: NativeIntervalFunc,
    n_inputs: usize,
    n_outputs: usize,
    n_signals: usize,
    phantom: PhantomData<&'ctx JitPruningIntervalFn<'ctx, T>>,
}

impl<'ctx, T> JitPruningSimdFn<'ctx, T>
where
    Wide: SimdVec<T>,
    T: NumberType,
{
    pub fn run(&self, buf: &mut JitSimdBuffers<T>, signals: &[u32]) -> Result<(), Error> {
        if self.n_signals != signals.len() {
            return Err(Error::InputSizeMismatch(self.n_signals, signals.len()));
        }
        // SAFETY: We just checked above.
        unsafe { self.run_unchecked(buf, signals) }
        Ok(())
    }

    /// # Safety
    ///
    /// This is the same as [`run`] except the caller has to ensure
    /// the buffers of the right size.
    pub unsafe fn run_unchecked(&self, buf: &mut JitSimdBuffers<T>, signals: &[u32]) {
        // SAFETY: Calling a raw function pointer. `JitSimdBuffers` is a safe
        // wrapper that populates the inputs correctly via it's public API, and
        // knows the correct number of SIMD iterations required. For signals, we
        // told the user they are responsible.
        unsafe {
            self.func.call(
                buf.inputs.as_ptr().cast(),
                buf.outputs.as_mut_ptr().cast(),
                buf.num_simd_iters() as u64,
                signals.as_ptr(),
            );
        }
    }

    pub fn as_sync(&'ctx self) -> JitPruningSimdFnSync<'ctx, T> {
        // SAFETY: Accessing the raw function pointer. This is ok, because
        // this borrows from Self, which owns an Rc reference to the
        // execution engine that owns the block of executable memory to
        // which the function pointer points.
        JitPruningSimdFnSync {
            func: unsafe { self.func.as_raw() },
            n_signals: self.n_signals,
            phantom: PhantomData,
        }
    }
}

unsafe impl<'ctx, T> Sync for JitPruningSimdFnSync<'ctx, T>
where
    Wide: SimdVec<T>,
    T: NumberType,
{
}

impl<'ctx, T> JitPruningSimdFnSync<'ctx, T>
where
    Wide: SimdVec<T>,
    T: NumberType,
{
    pub fn run(&self, buf: &mut JitSimdBuffers<T>, signals: &[u32]) -> Result<(), Error> {
        if self.n_signals != signals.len() {
            return Err(Error::InputSizeMismatch(self.n_signals, signals.len()));
        }
        // SAFETY: We just checked above.
        unsafe { self.run_unchecked(buf, signals) }
        Ok(())
    }

    /// # Safety
    ///
    /// This is the same as [`run`] except the caller has to ensure
    /// the buffers of the right size.
    pub unsafe fn run_unchecked(&self, buf: &mut JitSimdBuffers<T>, signals: &[u32]) {
        // SAFETY: Calling a raw function pointer. `JitSimdBuffers` is a safe
        // wrapper that populates the inputs correctly via it's public API, and
        // knows the correct number of SIMD iterations required. For signals, we
        // told the user they are responsible.
        unsafe {
            (self.func)(
                buf.inputs.as_ptr().cast(),
                buf.outputs.as_mut_ptr().cast(),
                buf.num_simd_iters() as u64,
                signals.as_ptr(),
            );
        }
    }
}

impl<'ctx, T: NumberType> JitPruningIntervalFn<'ctx, T> {
    pub fn run(
        &self,
        inputs: &[[T; 2]],
        outputs: &mut [[T; 2]],
        signals: &[u32],
    ) -> Result<(), Error> {
        if inputs.len() != self.n_inputs {
            return Err(Error::InputSizeMismatch(inputs.len(), self.n_inputs));
        } else if outputs.len() != self.n_outputs {
            return Err(Error::OutputSizeMismatch(outputs.len(), self.n_outputs));
        } else if signals.len() != self.n_signals {
            return Err(Error::OutputSizeMismatch(signals.len(), self.n_signals));
        }
        // # SAFETY:  We just checked the bounds above.
        unsafe { self.run_unchecked(inputs, outputs, signals) }
        Ok(())
    }

    /// # Safety
    ///
    /// This is the same as [`run`], except the user has to ensure the
    /// buffers are all the right size.
    pub unsafe fn run_unchecked(&self, inputs: &[[T; 2]], outputs: &mut [[T; 2]], signals: &[u32]) {
        // # SAFETY: We told the caller it's their fault.
        unsafe {
            self.func.call(
                inputs.as_ptr().cast(),
                outputs.as_mut_ptr().cast(),
                signals.as_ptr().cast(),
            )
        }
    }

    pub fn as_sync(&'ctx self) -> JitPruningIntervalFnSync<'ctx, T> {
        JitPruningIntervalFnSync {
            // SAFETY: Accessing the raw function pointer. This is ok, because
            // this borrows from Self, which owns an Rc reference to the
            // execution engine that owns the block of executable memory to
            // which the function pointer points.
            func: unsafe { self.func.as_raw() },
            n_inputs: self.n_inputs,
            n_outputs: self.n_outputs,
            n_signals: self.n_signals,
            phantom: PhantomData,
        }
    }
}

unsafe impl<'ctx, T: NumberType> Sync for JitPruningIntervalFnSync<'ctx, T> {}

impl<'ctx, T: NumberType> JitPruningIntervalFnSync<'ctx, T> {
    pub fn run(
        &self,
        inputs: &[[T; 2]],
        outputs: &mut [[T; 2]],
        signals: &[u32],
    ) -> Result<(), Error> {
        if inputs.len() != self.n_inputs {
            return Err(Error::InputSizeMismatch(inputs.len(), self.n_inputs));
        } else if outputs.len() != self.n_outputs {
            return Err(Error::OutputSizeMismatch(outputs.len(), self.n_outputs));
        } else if signals.len() != self.n_signals {
            return Err(Error::OutputSizeMismatch(signals.len(), self.n_signals));
        }
        // # SAFETY:  We just checked the bounds above.
        unsafe { self.run_unchecked(inputs, outputs, signals) }
        Ok(())
    }

    /// # Safety
    ///
    /// This is the same as [`run`], except the user has to ensure the
    /// buffers are all the right size.
    pub unsafe fn run_unchecked(&self, inputs: &[[T; 2]], outputs: &mut [[T; 2]], signals: &[u32]) {
        // # SAFETY: We told the caller it's their fault.
        unsafe {
            (self.func)(
                inputs.as_ptr().cast(),
                outputs.as_mut_ptr().cast(),
                signals.as_ptr().cast(),
            )
        }
    }
}

impl<'ctx, T: NumberType> JitPruningFn<'ctx, T> {
    pub fn run(&self, inputs: &[T], outputs: &mut [T], signals: &[u32]) -> Result<(), Error> {
        if inputs.len() != self.n_inputs {
            return Err(Error::InputSizeMismatch(inputs.len(), self.n_inputs));
        } else if outputs.len() != self.n_outputs {
            return Err(Error::OutputSizeMismatch(outputs.len(), self.n_outputs));
        } else if signals.len() != self.n_signals {
            return Err(Error::InputSizeMismatch(signals.len(), self.n_signals));
        }
        // # SAFETY: We just checked above.
        unsafe { self.run_unchecked(inputs, outputs, signals) }
        Ok(())
    }

    /// # Safety
    ///
    /// This is the same as [`run`], except the user has to ensure the
    /// buffers are all the right size.
    pub unsafe fn run_unchecked(&self, inputs: &[T], outputs: &mut [T], signals: &[u32]) {
        // # SAFETY: We told the caller it's their fault.
        unsafe {
            self.func.call(
                inputs.as_ptr().cast(),
                outputs.as_mut_ptr().cast(),
                signals.as_ptr().cast(),
            )
        }
    }

    pub fn as_sync(&'ctx self) -> JitPruningFnSync<'ctx, T> {
        JitPruningFnSync {
            // SAFETY: Accessing the raw function pointer. This is ok, because
            // this borrows from Self, which owns an Rc reference to the
            // execution engine that owns the block of executable memory to
            // which the function pointer points.
            func: unsafe { self.func.as_raw() },
            n_inputs: self.n_inputs,
            n_outputs: self.n_outputs,
            n_signals: self.n_signals,
            phantom: PhantomData,
        }
    }
}

unsafe impl<'ctx, T: NumberType> Sync for JitPruningFnSync<'ctx, T> {}

impl<'ctx, T: NumberType> JitPruningFnSync<'ctx, T> {
    pub fn run(&self, inputs: &[T], outputs: &mut [T], signals: &[u32]) -> Result<(), Error> {
        if inputs.len() != self.n_inputs {
            return Err(Error::InputSizeMismatch(inputs.len(), self.n_inputs));
        } else if outputs.len() != self.n_outputs {
            return Err(Error::OutputSizeMismatch(outputs.len(), self.n_outputs));
        } else if signals.len() != self.n_signals {
            return Err(Error::InputSizeMismatch(signals.len(), self.n_signals));
        }
        // # SAFETY: We just checked above.
        unsafe { self.run_unchecked(inputs, outputs, signals) }
        Ok(())
    }

    /// # Safety
    ///
    /// This is the same as [`run`], except the user has to ensure the
    /// buffers are all the right size.
    pub unsafe fn run_unchecked(&self, inputs: &[T], outputs: &mut [T], signals: &[u32]) {
        // # SAFETY: We told the caller it's their fault.
        unsafe {
            (self.func)(
                inputs.as_ptr().cast(),
                outputs.as_mut_ptr().cast(),
                signals.as_ptr().cast(),
            )
        }
    }
}

impl<'ctx, T: NumberType> JitPruner<'ctx, T> {
    pub fn num_signals(&self) -> usize {
        self.n_signals
    }

    pub fn run(
        &self,
        inputs: &[[T; 2]],
        outputs: &mut [[T; 2]],
        signals: &mut [u32],
    ) -> Result<(), Error> {
        if inputs.len() != self.n_inputs {
            return Err(Error::InputSizeMismatch(inputs.len(), self.n_inputs));
        } else if outputs.len() != self.n_outputs {
            return Err(Error::OutputSizeMismatch(outputs.len(), self.n_outputs));
        } else if signals.len() != self.n_signals {
            return Err(Error::OutputSizeMismatch(signals.len(), self.n_signals));
        }
        // # SAFETY: We just checked above.
        unsafe { self.run_unchecked(inputs, outputs, signals) }
        Ok(())
    }

    /// # Safety
    ///
    /// This is the same as [`run`], except the user has to ensure the
    /// buffers are all the right size.
    pub unsafe fn run_unchecked(
        &self,
        inputs: &[[T; 2]],
        outputs: &mut [[T; 2]],
        signals: &mut [u32],
    ) {
        // # SAFETY: We told the caller it's their fault.
        unsafe {
            self.func.call(
                inputs.as_ptr().cast(),
                outputs.as_mut_ptr().cast(),
                signals.as_mut_ptr().cast(),
            )
        }
    }

    pub fn compile_single_func(
        &self,
        context: &'ctx JitContext,
    ) -> Result<JitPruningFn<'ctx, T>, Error> {
        // No need to check if the tree has a valid scalar output, because we
        // already checked that when we made this pruner.
        let func_name = context.new_func_name::<T>(Some("pruning"));
        let context = &context.inner;
        let compiler = JitCompiler::new(context)?;
        let builder = &compiler.builder;
        let flt_type = T::jit_type(context);
        let ptr_type = context.ptr_type(AddressSpace::default());
        let mut constants = Constants::create::<T>(context);
        let fn_type = context
            .void_type()
            .fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
        let function = compiler.module.add_function(&func_name, fn_type, None);
        compiler.set_attributes(function, context)?;
        builder.position_at_end(context.append_basic_block(function, "entry"));
        let mut regs = Vec::<BasicValueEnum>::with_capacity(self.tree.len());
        let mut bbs: Box<[BasicBlock]> = self
            .blocks
            .iter()
            .enumerate()
            .map(|(bi, block)| {
                context.append_basic_block(
                    function,
                    &format!(
                        "{}_bb_{bi}",
                        match block {
                            Block::Code(_) => "code",
                            Block::Branch(_) => "branch",
                            Block::Merge(_) => "merge",
                        }
                    ),
                )
            })
            .collect();
        let signals_arg = function
            .get_nth_param(2)
            .ok_or(Error::JitCompilationError(
                "Cannot read output address".to_string(),
            ))?
            .into_pointer_value();
        let signal_ptrs: Result<Vec<PointerValue>, inkwell::builder::BuilderError> = (0..self
            .n_signals)
            .map(|si| unsafe {
                builder.build_gep(
                    context.i32_type(),
                    signals_arg,
                    &[constants.int_32(si as u32, false)],
                    &format!("signal_ptr_{}", si),
                )
            })
            .collect();
        let signal_ptrs = signal_ptrs?.into_boxed_slice();
        if let Some(first) = bbs.first() {
            builder.build_unconditional_branch(*first)?;
        }
        let PhiNodes { phis, phi_map } = init_merge_phi(
            &self.blocks,
            builder,
            &bbs,
            flt_type,
            context.bool_type(),
            self.tree.nodes(),
        )?;
        let mut merge_list = Vec::<Incoming>::new(); // Keep track of phi nodes that need to be merged.
        let nan_val = flt_type.const_float(f64::NAN).as_basic_value_enum();
        for (bi, block) in self.blocks.iter().enumerate() {
            builder.position_at_end(bbs[bi]);
            match block {
                Block::Code(range) => {
                    for (node, index) in self.tree.nodes()[range.clone()]
                        .iter()
                        .copied()
                        .zip(range.clone())
                    {
                        let reg = single::build_op(
                            single::BuildArgs {
                                nodes: self.tree.nodes(),
                                params: &self.params,
                                float_type: flt_type,
                                function,
                                regs: &regs,
                                node,
                                index,
                            },
                            builder,
                            &compiler.module,
                        )?;
                        regs.push(reg);
                        // We may have created new blocks when building the op. So we overwrite the basic block.
                        if let Some(bb) = builder.get_insert_block() {
                            bbs[bi] = bb;
                        }
                    }
                    if let Some(next_bb) = bbs.get(bi + 1) {
                        builder.build_unconditional_branch(*next_bb)?;
                    }
                }
                Block::Branch(jumps) => {
                    let si = self.block_signal_map[bi];
                    let signal = builder
                        .build_load(
                            context.i32_type(),
                            signal_ptrs[si],
                            &format!("load_signal_{si}"),
                        )?
                        .into_int_value();
                    let mut cases = Vec::<(IntValue, BasicBlock)>::new();
                    let mut prev_target = usize::MAX;
                    let mut prev_val = None;
                    let mut index = 0u32;
                    for jump in jumps.iter() {
                        // TODO: use chunk_by.
                        if prev_target != jump.target || prev_val != Some(jump.alternate.clone()) {
                            // New case.
                            index += 1;
                            let (case_bb, incoming_bb) = match jump.alternate {
                                Alternate::None => {
                                    let mbi = self
                                        .merge_block_map
                                        .get(&jump.target)
                                        .expect("We must find a match. This is a bug.");
                                    (bbs[*mbi], bbs[bi])
                                }
                                Alternate::Node(_) | Alternate::Constant(_) => {
                                    let bb = context.append_basic_block(
                                        function,
                                        &format!("branch_{bi}_case_{si}_bb"),
                                    );
                                    (bb, bb)
                                }
                            };
                            cases.push((constants.int_32(index, false), case_bb));
                            merge_list.push(Incoming {
                                target: jump.target,
                                basic_block: incoming_bb,
                                alternate: jump.alternate.clone(),
                            });
                            prev_target = jump.target;
                            prev_val = Some(jump.alternate.clone());
                        }
                    }
                    builder.build_switch(signal, bbs[bi + 1], &cases)?;
                }
                Block::Merge(target) => {
                    build_merges(
                        &phis,
                        &phi_map,
                        &mut merge_list,
                        &self.merge_block_map,
                        &bbs,
                        |ni| {
                            if is_node_scalar(self.tree.nodes(), ni) {
                                nan_val
                            } else {
                                context.bool_type().const_zero().as_basic_value_enum()
                            }
                        },
                        builder,
                        &mut regs,
                        |value| single::build_const::<T>(context, value),
                        *target,
                    )?;
                }
            }
        }
        // Copy the outputs.
        let outputs = function
            .get_nth_param(1)
            .ok_or(Error::JitCompilationError(
                "Cannot write to outputs".to_string(),
            ))?
            .into_pointer_value();
        for (i, reg) in regs[(self.tree.len() - self.tree.num_roots())..]
            .iter()
            .enumerate()
        {
            // SAFETY: GEP can segfault if the index is out of bounds. The
            // offset calculation looks pretty solid, and is thoroughly tested.
            let dst = unsafe {
                builder.build_gep(
                    flt_type,
                    outputs,
                    &[context.i64_type().const_int(i as u64, false)],
                    &format!("output_{i}"),
                )?
            };
            builder.build_store(dst, *reg)?;
        }
        builder.build_return(None)?;
        compiler.run_passes("mem2reg,instcombine,reassociate,gvn,instcombine,slp-vectorizer,instcombine,simplifycfg,adce")?;
        let engine = compiler
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .map_err(|_| Error::CannotCreateJitModule)?;
        // SAFETY: The signature is correct, and well tested. The function
        // pointer should never be invalidated, because we allocated a dedicated
        // execution engine, with it's own block of executable memory, that will
        // live as long as the function wrapper lives.
        let func = unsafe { engine.get_function(&func_name)? };
        Ok(JitPruningFn {
            func,
            n_inputs: self.n_inputs,
            n_outputs: self.n_outputs,
            n_signals: self.n_signals,
            _phantom: PhantomData,
        })
    }

    pub fn compile_interval_func(
        &self,
        context: &'ctx JitContext,
    ) -> Result<JitPruningIntervalFn<'ctx, T>, Error> {
        let PrunerInfo {
            engine,
            func_name,
            n_inputs,
            n_outputs,
            n_signals,
            ..
        } = compile_pruner_impl::<T, false>(
            &self.tree,
            context,
            &self.blocks,
            &self.params,
            f64::NAN,
        )?;
        // SAFETY: The signature is correct, and well tested. The function
        // pointer should never be invalidated, because we allocated a dedicated
        // execution engine, with it's own block of executable memory, that will
        // live as long as the function wrapper lives.
        let func = unsafe { engine.get_function(&func_name)? };
        Ok(JitPruningIntervalFn {
            func,
            n_inputs,
            n_outputs,
            n_signals,
            _phantom: PhantomData,
        })
    }

    pub fn compile_simd_func(
        &self,
        context: &'ctx JitContext,
    ) -> Result<JitPruningSimdFn<'ctx, T>, Error>
    where
        Wide: SimdVec<T>,
    {
        // No need to check if the tree has a valid scalar output, because we
        // already checked that when we made this pruner.
        let func_name = context.new_func_name::<T>(Some("array"));
        let context = &context.inner;
        let compiler = JitCompiler::new(context)?;
        let builder = &compiler.builder;
        let flt_type = <Wide as SimdVec<T>>::float_type(context);
        let i64_type = context.i64_type();
        let flt_vec_type = flt_type.vec_type(<Wide as SimdVec<T>>::SIMD_VEC_SIZE as u32);
        let bool_vec_type = context
            .bool_type()
            .vec_type(<Wide as SimdVec<T>>::SIMD_VEC_SIZE as u32);
        let ptr_type = context.ptr_type(AddressSpace::default());
        let fn_type = context.void_type().fn_type(
            &[
                ptr_type.into(),
                ptr_type.into(),
                i64_type.into(),
                ptr_type.into(),
            ],
            false,
        );
        let function = compiler.module.add_function(&func_name, fn_type, None);
        compiler.set_attributes(function, context)?;
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
        let mut regs: Vec<BasicValueEnum> = Vec::with_capacity(self.tree.len());
        let mut bbs: Box<[BasicBlock]> = self
            .blocks
            .iter()
            .enumerate()
            .map(|(bi, block)| {
                context.append_basic_block(
                    function,
                    &format!(
                        "{}_bb_{bi}",
                        match block {
                            Block::Code(_) => "code",
                            Block::Branch(_) => "branch",
                            Block::Merge(_) => "merge",
                        }
                    ),
                )
            })
            .collect();
        let signals_arg = function
            .get_nth_param(3)
            .ok_or(Error::JitCompilationError(
                "Cannot read output address".to_string(),
            ))?
            .into_pointer_value();
        let signal_ptrs: Result<Vec<PointerValue>, inkwell::builder::BuilderError> = (0..self
            .n_signals)
            .map(|si| unsafe {
                builder.build_gep(
                    context.i32_type(),
                    signals_arg,
                    &[context.i32_type().const_int(si as u64, false)],
                    &format!("signal_ptr_{}", si),
                )
            })
            .collect();
        let signal_ptrs = signal_ptrs?.into_boxed_slice();
        builder.build_unconditional_branch(loop_block)?;
        // Start the loop
        builder.position_at_end(loop_block);
        let loop_index_phi = builder.build_phi(i64_type, "counter_phi")?;
        loop_index_phi.add_incoming(&[(&i64_type.const_int(0, false), start_block)]);
        let loop_index = loop_index_phi.as_basic_value().into_int_value();
        if let Some(first) = bbs.first() {
            builder.build_unconditional_branch(*first)?;
        }
        let PhiNodes { phis, phi_map } = init_merge_phi(
            &self.blocks,
            builder,
            &bbs,
            flt_vec_type,
            bool_vec_type,
            self.tree.nodes(),
        )?;
        let mut merge_list = Vec::<Incoming>::new();
        let nan_vec = <Wide as SimdVec<T>>::const_float(f64::NAN, context);
        for (bi, block) in self.blocks.iter().enumerate() {
            builder.position_at_end(bbs[bi]);
            match block {
                Block::Code(range) => {
                    for (node_index, node) in
                        self.tree.nodes()[range.clone()].iter().copied().enumerate()
                    {
                        let reg = simd_array::build_op(
                            simd_array::BuildArgs {
                                nodes: self.tree.nodes(),
                                params: &self.params,
                                context,
                                fvec_type: flt_vec_type,
                                inputs,
                                loop_index,
                                regs: &regs,
                                node_index,
                                node,
                            },
                            builder,
                            &compiler.module,
                        )?;
                        regs.push(reg);
                        // We may have created new blocks when building the op. So we overwrite the basic block.
                        if let Some(bb) = builder.get_insert_block() {
                            bbs[bi] = bb;
                        }
                    }
                    if let Some(next_bb) = bbs.get(bi + 1) {
                        builder.build_unconditional_branch(*next_bb)?;
                    }
                }
                Block::Branch(jumps) => {
                    let si = self.block_signal_map[bi];
                    let signal = builder
                        .build_load(
                            context.i32_type(),
                            signal_ptrs[si],
                            &format!("load_signal_{si}"),
                        )?
                        .into_int_value();
                    let mut cases = Vec::<(IntValue, BasicBlock)>::new();
                    let mut prev_target = usize::MAX;
                    let mut prev_val = None;
                    let mut index = 0u32;
                    for jump in jumps.iter() {
                        // TODO: use chunk_by.
                        if prev_target != jump.target || prev_val != Some(jump.alternate.clone()) {
                            // New case.
                            index += 1;
                            let (case_bb, incoming_bb) = match jump.alternate {
                                Alternate::None => {
                                    let mbi = self
                                        .merge_block_map
                                        .get(&jump.target)
                                        .expect("We must find a match. This is a bug.");
                                    (bbs[*mbi], bbs[bi])
                                }
                                Alternate::Node(_) | Alternate::Constant(_) => {
                                    let bb = context.append_basic_block(
                                        function,
                                        &format!("branch_{bi}_case_{si}_bb"),
                                    );
                                    (bb, bb)
                                }
                            };
                            cases
                                .push((context.i32_type().const_int(index as u64, false), case_bb));
                            merge_list.push(Incoming {
                                target: jump.target,
                                basic_block: incoming_bb,
                                alternate: jump.alternate.clone(),
                            });
                            prev_target = jump.target;
                            prev_val = Some(jump.alternate.clone());
                        }
                    }
                    builder.build_switch(signal, bbs[bi + 1], &cases)?;
                }
                Block::Merge(target) => {
                    build_merges(
                        &phis,
                        &phi_map,
                        &mut merge_list,
                        &self.merge_block_map,
                        &bbs,
                        |ni| {
                            if is_node_scalar(self.tree.nodes(), ni) {
                                nan_vec
                            } else {
                                bool_vec_type.const_zero().as_basic_value_enum()
                            }
                        },
                        builder,
                        &mut regs,
                        |value| match value {
                            Value::Bool(val) => <Wide as SimdVec<T>>::const_bool(val, context),
                            Value::Scalar(val) => <Wide as SimdVec<T>>::const_float(val, context),
                        },
                        *target,
                    )?;
                }
            }
        }
        // Copy the outputs.
        let outputs = function
            .get_nth_param(1)
            .ok_or(Error::JitCompilationError(
                "Cannot read output address".to_string(),
            ))?
            .into_pointer_value();
        for (i, reg) in regs[(self.tree.len() - self.tree.num_roots())..]
            .iter()
            .enumerate()
        {
            let offset = builder.build_int_add(
                builder.build_int_mul(
                    loop_index,
                    i64_type.const_int(self.tree.num_roots() as u64, false),
                    "offset_mul",
                )?,
                i64_type.const_int(i as u64, false),
                "offset_add",
            )?;
            // SAFETY: GEP can segfault if the index is out of bounds. The
            // offset calculation looks pretty solid, and is thoroughly tested.
            let dst = unsafe {
                builder.build_gep(flt_vec_type, outputs, &[offset], &format!("output_{i}"))?
            };
            builder.build_store(dst, *reg)?;
        }
        // Check to see if the loop should go on.
        let loop_check_bb = context.append_basic_block(function, "loop_check_bb");
        builder.build_unconditional_branch(loop_check_bb)?;
        builder.position_at_end(loop_check_bb);
        let next = builder.build_int_add(loop_index, i64_type.const_int(1, false), "increment")?;
        let cmp = builder.build_int_compare(IntPredicate::ULT, next, eval_len, "loop-check")?;
        builder.build_conditional_branch(cmp, loop_block, end_block)?;
        loop_index_phi.add_incoming(&[(&next, loop_check_bb)]);
        // End loop and return.
        builder.position_at_end(end_block);
        builder.build_return(None)?;
        compiler.run_passes("mem2reg,instcombine,reassociate,gvn,simplifycfg,adce,instcombine")?;
        let engine = compiler
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .map_err(|_| Error::CannotCreateJitModule)?;
        // SAFETY: The signature is correct, and well tested. The function
        // pointer should never be invalidated, because we allocated a dedicated
        // execution engine, with it's own block of executable memory, that will
        // live as long as the function wrapper lives.
        let func = unsafe { engine.get_function(&func_name)? };
        Ok(JitPruningSimdFn::<T> {
            func,
            n_signals: signal_ptrs.len(),
            phantom: PhantomData,
        })
    }

    pub fn as_sync(&'ctx self) -> JitPrunerSync<'ctx, T> {
        JitPrunerSync {
            // SAFETY: Accessing the raw function pointer. This is ok, because
            // this borrows from Self, which owns an Rc reference to the
            // execution engine that owns the block of executable memory to
            // which the function pointer points.
            func: unsafe { self.func.as_raw() },
            n_inputs: self.n_inputs,
            n_outputs: self.n_outputs,
            n_signals: self.n_signals,
            phantom: PhantomData,
        }
    }
}

unsafe impl<'ctx, T: NumberType> Sync for JitPrunerSync<'ctx, T> {}

impl<'ctx, T: NumberType> JitPrunerSync<'ctx, T> {
    pub fn run(
        &self,
        inputs: &[[T; 2]],
        outputs: &mut [[T; 2]],
        signals: &mut [u32],
    ) -> Result<(), Error> {
        if inputs.len() != self.n_inputs {
            return Err(Error::InputSizeMismatch(inputs.len(), self.n_inputs));
        } else if outputs.len() != self.n_outputs {
            return Err(Error::OutputSizeMismatch(outputs.len(), self.n_outputs));
        } else if signals.len() != self.n_signals {
            return Err(Error::OutputSizeMismatch(signals.len(), self.n_signals));
        }
        // # SAFETY: We just checked above.
        unsafe { self.run_unchecked(inputs, outputs, signals) }
        Ok(())
    }

    /// # Safety
    ///
    /// This is the same as [`run`], except the user has to ensure the
    /// buffers are all the right size.
    pub unsafe fn run_unchecked(
        &self,
        inputs: &[[T; 2]],
        outputs: &mut [[T; 2]],
        signals: &mut [u32],
    ) {
        // # SAFETY: We told the caller it's their fault.
        unsafe {
            (self.func)(
                inputs.as_ptr().cast(),
                outputs.as_mut_ptr().cast(),
                signals.as_mut_ptr().cast(),
            )
        }
    }
}

impl Tree {
    pub fn jit_compile_pruner<'ctx, T: NumberType>(
        &'ctx self,
        context: &'ctx JitContext,
        params: &str,
        pruning_threshold: usize,
        eps: f64,
    ) -> Result<JitPruner<'ctx, T>, Error> {
        if !self.is_scalar() {
            // Only support scalar output trees.
            return Err(Error::TypeMismatch);
        }
        let (tree, ndom) = self.control_dependence_sorted()?;
        let deps = DependencyTable::from_tree(&tree);
        let claims = make_claims(&tree, &ndom, &deps, pruning_threshold)?;
        let claims: HashSet<(usize, usize)> = HashSet::from_iter(claims.into_iter().map(
            |Claim {
                 claimed,
                 claimant,
                 kind: _,
             }| (claimed, claimant),
        ));
        let interrupts = make_interrupts(&tree, &ndom, &claims, pruning_threshold)?;
        let blocks = make_blocks(interrupts, tree.len())?;
        let PrunerInfo {
            engine,
            func_name,
            n_inputs,
            n_outputs,
            n_signals,
            params,
            merge_block_map,
            block_signal_map,
        } = compile_pruner_impl::<T, true>(&tree, context, &blocks, params, eps)?;
        // SAFETY: The signature is correct, and well tested. The function
        // pointer should never be invalidated, because we allocated a dedicated
        // execution engine, with it's own block of executable memory, that will
        // live as long as the function wrapper lives.
        let func = unsafe { engine.get_function(&func_name)? };
        Ok(JitPruner {
            func,
            n_inputs,
            n_outputs,
            n_signals,
            params: params.to_string(),
            blocks,
            merge_block_map,
            block_signal_map,
            tree,
            phantom: PhantomData,
        })
    }
}

struct PrunerInfo<'ctx> {
    engine: ExecutionEngine<'ctx>,
    func_name: String,
    n_inputs: usize,
    n_outputs: usize,
    n_signals: usize,
    params: String,
    merge_block_map: HashMap<usize, usize>,
    block_signal_map: Box<[usize]>,
}

fn compile_pruner_impl<'ctx, T: NumberType, const WITH_NOTIFY: bool>(
    tree: &Tree,
    context: &'ctx JitContext,
    blocks: &[Block],
    params: &str,
    eps: f64,
) -> Result<PrunerInfo<'ctx>, Error> {
    let (code_block_map, merge_block_map) = reverse_lookup(blocks);
    let ranges = interval::compute_ranges(tree)?;
    let func_name = context.new_func_name::<T>(Some("pruner"));
    let context = &context.inner;
    let compiler = JitCompiler::new(context)?;
    let builder = &compiler.builder;
    let interval_type = T::jit_type(context).vec_type(2);
    let ptr_type = context.ptr_type(AddressSpace::default());
    let mut constants = Constants::create::<T>(context);
    let fn_type = context
        .void_type()
        .fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
    let function = compiler.module.add_function(&func_name, fn_type, None);
    compiler.set_attributes(function, context)?;
    builder.position_at_end(context.append_basic_block(function, "entry"));
    let mut regs = Vec::<BasicValueEnum>::with_capacity(tree.len());
    let mut bbs: Box<[BasicBlock]> = blocks
        .iter()
        .enumerate()
        .map(|(bi, block)| {
            context.append_basic_block(
                function,
                &format!(
                    "{}_bb_{bi}",
                    match block {
                        Block::Code(_) => "code",
                        Block::Branch(_) => "branch",
                        Block::Merge(_) => "merge",
                    }
                ),
            )
        })
        .collect();
    let SignalPointers {
        pointers: signal_ptrs,
        block_signal_map,
    } = init_signal_ptrs(
        blocks,
        function
            .get_nth_param(2)
            .ok_or(Error::JitCompilationError(
                "Cannot read output address".to_string(),
            ))?
            .into_pointer_value(),
        context.i32_type(),
        builder,
        &mut constants,
    )?;
    let eps = eps.abs();
    let eps = VectorType::const_vector(&[constants.float(-eps), constants.float(eps)]);
    if let Some(first) = bbs.first() {
        builder.build_unconditional_branch(*first)?;
    }
    let PhiNodes { phis, phi_map } = init_merge_phi(
        blocks,
        builder,
        &bbs,
        interval_type,
        context.bool_type().vec_type(2),
        tree.nodes(),
    )?;
    let mut merge_list = Vec::<Incoming>::new();
    let mut all_notifications = Vec::<Notification>::new();
    let mut temp_notify = Vec::<Notification>::new();
    let mut notified = HashSet::<(usize, u32)>::new();
    let nan_interval = constants.float_vec([f64::NAN; 2]).as_basic_value_enum();
    for (bi, block) in blocks.iter().enumerate() {
        builder.position_at_end(bbs[bi]);
        match block {
            Block::Code(range) => {
                for (index, node) in tree.nodes()[range.clone()].iter().copied().enumerate() {
                    let reg = interval::build_op::<T>(
                        interval::BuildArgs {
                            nodes: tree.nodes(),
                            params,
                            ranges: ranges.as_ref(),
                            regs: &regs,
                            constants: &mut constants,
                            interval_type,
                            function,
                            node,
                            index,
                        },
                        builder,
                        &compiler.module,
                    )?;
                    regs.push(fast_math(reg));
                    // We may have created new blocks when building the op. So we overwrite the basic block.
                    if let Some(bb) = builder.get_insert_block() {
                        bbs[bi] = bb;
                    }
                }
                if WITH_NOTIFY {
                    // Notify upstream.
                    temp_notify.clear();
                    all_notifications.retain(
                        |Notification {
                             src_inst,
                             dst_signal,
                             signal,
                             kind,
                         }| {
                            if *src_inst == range.end - 1 {
                                temp_notify.push(Notification {
                                    src_inst: *src_inst,
                                    dst_signal: *dst_signal,
                                    signal: *signal,
                                    kind: *kind,
                                });
                                false
                            } else {
                                true
                            }
                        },
                    );
                    temp_notify.drain(..).try_fold(
                        (),
                        |_,
                         Notification {
                             src_inst,
                             dst_signal,
                             signal,
                             kind,
                         }|
                         -> Result<(), Error> {
                            let ci = *code_block_map
                                .get(&src_inst)
                                .expect("Code map is not complete. This is a bug.");
                            let bb = bbs[ci];
                            builder.position_at_end(bb);
                            let first = notified.insert((dst_signal, signal));
                            bbs[ci] = build_notify(
                                *tree.node(src_inst),
                                src_inst,
                                kind,
                                signal,
                                signal_ptrs[dst_signal],
                                first,
                                &regs,
                                builder,
                                &compiler.module,
                                &mut constants,
                                function,
                                eps,
                            )?;
                            Ok(())
                        },
                    )?;
                }
            }
            Block::Branch(jumps) => {
                let si = block_signal_map[bi];
                let signal = builder
                    .build_load(
                        context.i32_type(),
                        signal_ptrs[si],
                        &format!("load_signal_{si}"),
                    )?
                    .into_int_value();
                let mut cases = Vec::<(IntValue, BasicBlock)>::new();
                let mut prev_target = usize::MAX;
                let mut prev_val = None;
                let mut index = 0u32;
                for jump in jumps.iter() {
                    if prev_target != jump.target || prev_val != Some(jump.alternate.clone()) {
                        // New case.
                        index += 1;
                        let (case_bb, incoming_bb) = match jump.alternate {
                            Alternate::None => {
                                let mbi = merge_block_map
                                    .get(&jump.target)
                                    .expect("We must find a match. This is a bug.");
                                (bbs[*mbi], bbs[bi])
                            }
                            Alternate::Node(_) | Alternate::Constant(_) => {
                                let bb = context.append_basic_block(
                                    function,
                                    &format!("branch_{bi}_case_{si}_bb"),
                                );
                                (bb, bb)
                            }
                        };
                        cases.push((constants.int_32(index, false), case_bb));
                        merge_list.push(Incoming {
                            target: jump.target,
                            basic_block: incoming_bb,
                            alternate: jump.alternate.clone(),
                        });
                        prev_target = jump.target;
                        prev_val = Some(jump.alternate.clone());
                    }
                    if WITH_NOTIFY {
                        all_notifications.push(Notification {
                            src_inst: jump.owner,
                            dst_signal: si,
                            signal: index,
                            kind: jump.trigger,
                        });
                    }
                }
                builder.build_switch(signal, bbs[bi + 1], &cases)?;
            }
            Block::Merge(target) => {
                build_merges(
                    &phis,
                    &phi_map,
                    &mut merge_list,
                    &merge_block_map,
                    &bbs,
                    |ni| {
                        if is_node_scalar(tree.nodes(), ni) {
                            nan_interval
                        } else {
                            context
                                .bool_type()
                                .vec_type(2)
                                .const_zero()
                                .as_basic_value_enum()
                        }
                    },
                    builder,
                    &mut regs,
                    |value| interval::build_const(&mut constants, value),
                    *target,
                )?;
            }
        }
    }
    assert!(
        merge_list.is_empty(),
        "All merges should be processed by now. This is a bug otherwise"
    );
    // Branch out of all code blocks.
    let mut done = vec![false; bbs.len()];
    for ci in code_block_map.values() {
        if let Some(next) = bbs.get(ci + 1)
            && !std::mem::replace(&mut done[*ci], true)
        {
            builder.position_at_end(bbs[*ci]);
            builder.build_unconditional_branch(*next)?;
        }
    }
    if let Some(last) = bbs.last() {
        builder.position_at_end(*last);
    }
    // Compile instructions to copy the outputs to the out argument.
    let outputs = function
        .get_nth_param(1)
        .ok_or(Error::JitCompilationError(
            "Cannot read output address".to_string(),
        ))?
        .into_pointer_value();
    for (i, reg) in regs[(tree.len() - tree.num_roots())..].iter().enumerate() {
        // # SAFETY: This is unit tested a lot. If this fails, we segfault.
        let dst = unsafe {
            builder.build_gep(
                interval_type,
                outputs,
                &[constants.int_32(i as u32, false)],
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
                Error::JitCompilationError(format!("Cannot set alignment when storing output: {e}"))
            })?;
    }
    builder.build_return(None)?;
    compiler.run_passes("mem2reg,instcombine,reassociate,gvn,simplifycfg,adce,instcombine")?;
    let engine = compiler
        .module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .map_err(|_| Error::CannotCreateJitModule)?;
    Ok(PrunerInfo {
        engine,
        func_name,
        n_inputs: params.len(),
        n_outputs: tree.num_roots(),
        n_signals: signal_ptrs.len(),
        params: params.to_string(),
        merge_block_map,
        block_signal_map,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_merges<
    'ctx,
    FnGetNaN: Fn(usize) -> BasicValueEnum<'ctx>,
    FnMakeConst: FnMut(Value) -> BasicValueEnum<'ctx>,
>(
    phis: &[PhiValue<'ctx>],
    phi_map: &[usize],
    merge_list: &mut Vec<Incoming<'ctx>>,
    merge_map: &HashMap<usize, usize>,
    bbs: &[BasicBlock<'ctx>],
    get_nan: FnGetNaN,
    builder: &'ctx Builder,
    regs: &mut [BasicValueEnum<'ctx>],
    mut build_const_fn: FnMakeConst,
    target: usize,
) -> Result<(), Error> {
    let bi = merge_map
        .get(&target)
        .copied()
        .expect("This is a bug. We must find a block index here.");
    let phi = phis[phi_map[bi]];
    for Incoming {
        target,
        basic_block,
        alternate,
    } in merge_list.iter().filter(|m| m.target == target)
    {
        let val = match alternate {
            Alternate::None => get_nan(*target),
            Alternate::Node(ni) => {
                builder.position_at_end(*basic_block);
                builder.build_unconditional_branch(bbs[bi])?;
                regs[*ni]
            }
            Alternate::Constant(value) => {
                builder.position_at_end(*basic_block);
                let out = build_const_fn(*value);
                builder.build_unconditional_branch(bbs[bi])?;
                out
            }
        };
        phi.add_incoming(&[(&val as &dyn BasicValue, *basic_block)]);
    }
    // Default path when nothing is pruned.
    phi.add_incoming(&[(&regs[target], bbs[bi - 1])]);
    builder.position_at_end(bbs[bi]);
    regs[target] = phi.as_basic_value();
    if let Some(next_bb) = bbs.get(bi + 1) {
        builder.build_unconditional_branch(*next_bb)?;
    }
    // Clean up.
    merge_list.retain(|m| m.target != target);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_notify<'ctx>(
    node: Node,
    index: usize,
    trigger: PruneKind,
    signal: u32,
    signal_ptr: PointerValue<'ctx>,
    first: bool,
    regs: &[BasicValueEnum<'ctx>],
    builder: &'ctx Builder,
    module: &'ctx Module,
    constants: &mut Constants<'ctx>,
    function: FunctionValue<'ctx>,
    eps: VectorValue<'ctx>,
) -> Result<BasicBlock<'ctx>, Error> {
    let cond = match node {
        Binary(op, lhs, rhs) => {
            let left = builder.build_float_add(
                regs[lhs].into_vector_value(),
                eps,
                &format!("lhs_inflate_{index}"),
            )?;
            let right = builder.build_float_add(
                regs[rhs].into_vector_value(),
                eps,
                &format!("rhs_inflate_{index}"),
            )?;
            let interval::InequalityFlags {
                either_empty,
                strictly_before,
                strictly_after,
                touching,
            } = interval::build_interval_inequality_flags(
                left, right, builder, module, constants, index,
            )?;
            let not_empty = builder.build_not(either_empty, &format!("not_empty_{index}"))?;
            let before = builder.build_or(
                strictly_before,
                builder
                    .build_extract_element(
                        touching,
                        constants.int_32(1, false),
                        &format!("extract_touching_left_{index}"),
                    )?
                    .into_int_value(),
                &format!("before_check_{index}"),
            )?;
            let after = builder.build_or(
                strictly_after,
                builder
                    .build_extract_element(
                        touching,
                        constants.int_32(0, false),
                        &format!("extract_touching_right_{index}"),
                    )?
                    .into_int_value(),
                &format!("after_check_{index}"),
            )?;
            let strictly_before = builder.build_and(
                not_empty,
                strictly_before,
                &format!("not_empty_combine_flags_{index}"),
            )?;
            let strictly_after = builder.build_and(
                not_empty,
                strictly_after,
                &format!("not_empty_combine_flags_{index}"),
            )?;
            let before = builder.build_and(
                not_empty,
                before,
                &format!("not_empty_combine_flags_{index}"),
            )?;
            let after = builder.build_and(
                not_empty,
                after,
                &format!("not_empty_combine_flags_{index}"),
            )?;
            match (op, trigger) {
                (Min, PruneKind::Right)
                | (Max, PruneKind::Left)
                | (LessOrEqual, PruneKind::Constant(Value::Bool(true)))
                | (Greater, PruneKind::Constant(Value::Bool(false)))
                | (GreaterOrEqual, PruneKind::Constant(Value::Bool(false))) => before,
                (Min, PruneKind::Left)
                | (Max, PruneKind::Right)
                | (GreaterOrEqual, PruneKind::Constant(Value::Bool(true)))
                | (Less, PruneKind::Constant(Value::Bool(false)))
                | (LessOrEqual, PruneKind::Constant(Value::Bool(false))) => after,
                (Less, PruneKind::Constant(Value::Bool(true))) => strictly_before,
                (Greater, PruneKind::Constant(Value::Bool(true))) => strictly_after,
                _ => unreachable!("Invalid node / trigger combo. This is a bug."),
            }
        }
        Ternary(Choose, cond, _, _) => match trigger {
            PruneKind::Right => build_vec_unary_intrinsic(
                builder,
                module,
                "llvm.vector.reduce.and.*",
                &format!("all_true_check_{index}"),
                regs[cond].into_vector_value(),
            )?
            .into_int_value(),
            PruneKind::Left => build_vec_unary_intrinsic(
                builder,
                module,
                "llvm.vector.reduce.and.*",
                &format!("all_false_check_{index}"),
                builder.build_not(
                    regs[cond].into_vector_value(),
                    &format!("always_false_check_flip_{index}"),
                )?,
            )?
            .into_int_value(),
            _ => unreachable!("Invalid node / trigger combo. This is a bug."),
        },
        _ => unreachable!("Unsupported node type. This is a bug."),
    };
    let then_bb = module
        .get_context()
        .append_basic_block(function, &format!("notify_block_{index}"));
    let merge_bb = module
        .get_context()
        .append_basic_block(function, &format!("notify_merge_{index}"));
    builder.build_conditional_branch(cond, then_bb, merge_bb)?;
    builder.position_at_end(then_bb);
    let signal = constants.int_32(signal, false);
    let signal = if first {
        signal
    } else {
        let old = builder
            .build_load(
                module.get_context().i32_type(),
                signal_ptr,
                &format!("notify_combine_load_{index}"),
            )?
            .into_int_value();
        builder.build_and(old, signal, &format!("notify_combine_{index}"))?
    };
    builder.build_store(signal_ptr, signal)?;
    builder.build_unconditional_branch(merge_bb)?;
    Ok(merge_bb)
}

fn reverse_lookup(blocks: &[Block]) -> (HashMap<usize, usize>, HashMap<usize, usize>) {
    let mut code_map = HashMap::default();
    let mut merge_map = HashMap::default();
    for (bi, block) in blocks.iter().enumerate() {
        match block {
            Block::Code(range) => {
                code_map.extend(range.clone().map(|inst| (inst, bi)));
            }
            Block::Branch(_) => {}
            Block::Merge(after) => {
                merge_map.insert(*after, bi);
            }
        };
    }
    (code_map, merge_map)
}

#[derive(Debug)]
struct Incoming<'ctx> {
    target: usize,
    basic_block: BasicBlock<'ctx>,
    alternate: Alternate,
}

#[derive(Debug)]
struct Notification {
    src_inst: usize,
    dst_signal: usize,
    signal: u32,
    kind: PruneKind,
}

struct PhiNodes<'ctx> {
    phis: Box<[PhiValue<'ctx>]>,
    phi_map: Box<[usize]>,
}

fn init_merge_phi<'ctx, TPhi: BasicType<'ctx> + Copy, TBool: BasicType<'ctx> + Copy>(
    blocks: &[Block],
    builder: &'ctx Builder,
    bbs: &[BasicBlock<'ctx>],
    out_type: TPhi,
    bool_type: TBool,
    nodes: &[Node],
) -> Result<PhiNodes<'ctx>, Error> {
    let original_bb = builder.get_insert_block();
    let (phis, indices) = blocks.iter().zip(bbs.iter()).try_fold(
        (Vec::<PhiValue>::new(), Vec::<usize>::new()),
        |(mut phis, mut indices), (block, bb)| -> Result<(Vec<PhiValue>, Vec<usize>), Error> {
            match block {
                Block::Code(_) | Block::Branch(_) => {
                    indices.push(usize::MAX);
                    Ok((phis, indices))
                }
                Block::Merge(after) => {
                    let index = phis.len();
                    builder.position_at_end(*bb);
                    let phi = if is_node_scalar(nodes, *after) {
                        builder.build_phi(out_type, &format!("merge_phi_{index}"))?
                    } else {
                        builder.build_phi(bool_type, &format!("merge_phi_{index}"))?
                    };
                    phis.push(phi);
                    indices.push(index);
                    Ok((phis, indices))
                }
            }
        },
    )?;
    if let Some(bb) = original_bb {
        builder.position_at_end(bb);
    }
    Ok(PhiNodes {
        phis: phis.into_boxed_slice(),
        phi_map: indices.into_boxed_slice(),
    })
}

struct SignalPointers<'ctx> {
    pointers: Box<[PointerValue<'ctx>]>,
    block_signal_map: Box<[usize]>,
}

fn init_signal_ptrs<'ctx>(
    blocks: &[Block],
    signals: PointerValue<'ctx>,
    i32_type: IntType<'ctx>,
    builder: &'ctx Builder,
    constants: &mut Constants<'ctx>,
) -> Result<SignalPointers<'ctx>, Error> {
    let (ptrs, indices) = blocks.iter().try_fold(
        (Vec::<PointerValue>::new(), Vec::<usize>::new()),
        |(mut ptrs, mut indices), block| -> Result<(Vec<PointerValue>, Vec<usize>), Error> {
            match block {
                Block::Code(_) | Block::Merge(_) => {
                    indices.push(usize::MAX);
                    Ok((ptrs, indices))
                }
                Block::Branch(_) => {
                    let index = ptrs.len();
                    indices.push(index);
                    let ptr = unsafe {
                        builder.build_gep(
                            i32_type,
                            signals,
                            &[constants.int_32(index as u32, false)],
                            &format!("signal_ptr_{}", index),
                        )?
                    };
                    ptrs.push(ptr);
                    Ok((ptrs, indices))
                }
            }
        },
    )?;
    Ok(SignalPointers {
        pointers: ptrs.into_boxed_slice(),
        block_signal_map: indices.into_boxed_slice(),
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Alternate {
    None,
    Node(usize),
    Constant(Value),
}

impl PartialOrd for Alternate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Alternate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        match (self, other) {
            (Alternate::None, Alternate::None) => Equal,
            (Alternate::None, Alternate::Node(_))
            | (Alternate::None, Alternate::Constant(_))
            | (Alternate::Node(_), Alternate::Constant(_)) => Less,
            (Alternate::Node(_), Alternate::None)
            | (Alternate::Constant(_), Alternate::None)
            | (Alternate::Constant(_), Alternate::Node(_)) => Greater,
            (Alternate::Node(a), Alternate::Node(b)) => a.cmp(b),
            (Alternate::Constant(a), Alternate::Constant(b)) => match a.partial_cmp(b) {
                Some(cmp) => cmp,
                None => Equal,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Interrupt {
    Jump {
        before_node: usize,
        target: usize,
        alternate: Alternate,
        owner: usize,
        trigger: PruneKind,
    },
    Land {
        after_node: usize,
    },
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum PruneKind {
    None,
    Left,
    Right,
    Constant(Value),
}

#[derive(Clone, Debug)]
struct Jump {
    before_node: usize,
    target: usize,
    alternate: Alternate,
    owner: usize,
    trigger: PruneKind,
}

#[derive(Debug)]
enum Block {
    Code(Range<usize>),
    Branch(Vec<Jump>),
    Merge(usize),
}

fn make_blocks(interrupts: Box<[Interrupt]>, n_nodes: usize) -> Result<Box<[Block]>, Error> {
    enum PartialBlock {
        Code(usize),
        Branch(Vec<Jump>),
    }
    let (mut blocks, last) = interrupts.into_iter().fold(
        (Vec::<Block>::new(), PartialBlock::Code(0)),
        |(mut blocks, partial), current| match (partial, current) {
            (
                PartialBlock::Code(start),
                Interrupt::Jump {
                    before_node,
                    target,
                    alternate,
                    owner,
                    trigger,
                },
            ) => {
                if start < before_node {
                    blocks.push(Block::Code(start..before_node));
                }
                (
                    blocks,
                    PartialBlock::Branch(vec![Jump {
                        before_node,
                        target,
                        alternate,
                        owner,
                        trigger,
                    }]),
                )
            }
            (PartialBlock::Code(start), Interrupt::Land { after_node }) => {
                if start < after_node + 1 {
                    blocks.push(Block::Code(start..(after_node + 1)));
                }
                blocks.push(Block::Merge(after_node));
                (blocks, PartialBlock::Code(after_node + 1))
            }
            (
                PartialBlock::Branch(mut jumps),
                Interrupt::Jump {
                    before_node,
                    target,
                    alternate,
                    owner,
                    trigger,
                },
            ) => match jumps.last().map(|j| j.before_node) {
                Some(prev) if prev != before_node => {
                    blocks.push(Block::Branch(jumps));
                    blocks.push(Block::Code(prev..before_node));
                    (
                        blocks,
                        PartialBlock::Branch(vec![Jump {
                            before_node,
                            target,
                            alternate,
                            owner,
                            trigger,
                        }]),
                    )
                }
                _ => {
                    jumps.push(Jump {
                        before_node,
                        target,
                        alternate,
                        owner,
                        trigger,
                    });
                    (blocks, PartialBlock::Branch(jumps))
                }
            },
            (PartialBlock::Branch(jumps), Interrupt::Land { after_node }) => {
                let start = match jumps.last() {
                    Some(j) => j.before_node,
                    None => unreachable!("Invalid interrupts. This is a bug."),
                };
                blocks.push(Block::Branch(jumps));
                blocks.push(Block::Code(start..(after_node + 1)));
                blocks.push(Block::Merge(after_node));
                (blocks, PartialBlock::Code(after_node + 1))
            }
        },
    );
    match last {
        PartialBlock::Code(start) if start < n_nodes => {
            blocks.push(Block::Code(start..n_nodes));
        }
        PartialBlock::Branch(_) => {
            unreachable!("Last block should not be a branch block. This is a bug")
        }
        _ => {}
    };
    Ok(blocks.into_boxed_slice())
}

fn push_land(dst: &mut Vec<Interrupt>, after_node: usize, land_map: &mut [Option<usize>]) {
    let mapped = &mut land_map[after_node];
    if mapped.is_none() {
        let idx = dst.len();
        dst.push(Interrupt::Land { after_node });
        *mapped = Some(idx);
    }
}

fn make_interrupts(
    tree: &Tree,
    ndom: &[usize],
    claims: &HashSet<(usize, usize)>,
    threshold: usize,
) -> Result<Box<[Interrupt]>, Error> {
    let mut interrupts = Vec::<Interrupt>::with_capacity(tree.len() / 2);
    let mut land_map: Vec<Option<usize>> = vec![None; tree.len()];
    for (ni, node) in tree.nodes().iter().enumerate() {
        match node {
            Binary(Min | Max, lhs, rhs) => {
                let ldom = ndom[*lhs];
                let rdom = ndom[*rhs];
                let lskip = claims.contains(&(*lhs, ni));
                let rskip = claims.contains(&(*rhs, ni));
                if lskip {
                    push_land(&mut interrupts, *lhs, &mut land_map);
                    interrupts.push(Interrupt::Jump {
                        before_node: *lhs - ldom,
                        target: *lhs,
                        alternate: Alternate::None,
                        owner: ni,
                        trigger: PruneKind::Left,
                    });
                }
                if rskip {
                    push_land(&mut interrupts, *rhs, &mut land_map);
                    interrupts.push(Interrupt::Jump {
                        before_node: *rhs - rdom,
                        target: *rhs,
                        alternate: Alternate::None,
                        owner: ni,
                        trigger: PruneKind::Right,
                    });
                }
                if lskip || rskip {
                    push_land(&mut interrupts, ni, &mut land_map);
                    if lskip {
                        interrupts.push(Interrupt::Jump {
                            before_node: ni,
                            target: ni,
                            alternate: Alternate::Node(*rhs),
                            owner: ni,
                            trigger: PruneKind::Left,
                        });
                    }
                    if rskip {
                        interrupts.push(Interrupt::Jump {
                            before_node: ni,
                            target: ni,
                            alternate: Alternate::Node(*lhs),
                            owner: ni,
                            trigger: PruneKind::Right,
                        });
                    }
                }
            }
            Binary(Less | LessOrEqual | Greater | GreaterOrEqual, _lhs, _rhs) => {
                let dom = ndom[ni];
                if dom > threshold {
                    push_land(&mut interrupts, ni, &mut land_map);
                    interrupts.push(Interrupt::Jump {
                        before_node: ni - dom,
                        target: ni,
                        alternate: Alternate::Constant(Value::Bool(true)),
                        owner: ni,
                        trigger: PruneKind::Constant(Value::Bool(true)),
                    });
                    interrupts.push(Interrupt::Jump {
                        before_node: ni - dom,
                        target: ni,
                        alternate: Alternate::Constant(Value::Bool(false)),
                        owner: ni,
                        trigger: PruneKind::Constant(Value::Bool(false)),
                    });
                }
            }
            Ternary(Choose, _cond, tt, ff) => {
                let ttdom = ndom[*tt];
                let ffdom = ndom[*ff];
                let tskip = claims.contains(&(*tt, ni));
                let fskip = claims.contains(&(*tt, ni));
                if tskip {
                    push_land(&mut interrupts, *tt, &mut land_map);
                    interrupts.push(Interrupt::Jump {
                        before_node: *tt - ttdom,
                        target: *tt,
                        alternate: Alternate::None,
                        owner: ni,
                        trigger: PruneKind::Left,
                    });
                }
                if fskip {
                    push_land(&mut interrupts, *ff, &mut land_map);
                    interrupts.push(Interrupt::Jump {
                        before_node: *ff - ffdom,
                        target: *ff,
                        alternate: Alternate::None,
                        owner: ni,
                        trigger: PruneKind::Right,
                    });
                }
                if tskip || fskip {
                    push_land(&mut interrupts, ni, &mut land_map);
                    if tskip {
                        interrupts.push(Interrupt::Jump {
                            before_node: ni,
                            target: ni,
                            alternate: Alternate::Node(*ff),
                            owner: ni,
                            trigger: PruneKind::Left,
                        });
                    }
                    if fskip {
                        interrupts.push(Interrupt::Jump {
                            before_node: ni,
                            target: ni,
                            alternate: Alternate::Node(*tt),
                            owner: ni,
                            trigger: PruneKind::Right,
                        });
                    }
                }
            }
            _ => continue,
        }
    }
    interrupts.sort_by(|a, b| -> std::cmp::Ordering {
        match (a, b) {
            (
                Interrupt::Jump {
                    before_node: lbn,
                    target: lt,
                    alternate: la,
                    owner: lo,
                    ..
                },
                Interrupt::Jump {
                    before_node: rbn,
                    target: rt,
                    alternate: ra,
                    owner: ro,
                    ..
                },
            ) => (lbn, std::cmp::Reverse(lt), la, lo).cmp(&(rbn, std::cmp::Reverse(rt), ra, ro)),
            (Interrupt::Jump { before_node, .. }, Interrupt::Land { after_node }) => {
                (before_node, 0).cmp(&(after_node, 1))
            }
            (Interrupt::Land { after_node }, Interrupt::Jump { before_node, .. }) => {
                (after_node, 1).cmp(&(before_node, 0))
            }
            (Interrupt::Land { after_node: la }, Interrupt::Land { after_node: ra }) => la.cmp(ra),
        }
    });
    Ok(interrupts.into_boxed_slice())
}

#[derive(Eq, PartialEq, Debug)]
struct Claim {
    claimed: usize,
    claimant: usize,
    kind: PruneKind,
}

/**
# Core Idea

Claims are how we decide which nodes have the pruning rights for which other
nodes. When a node is pruned, it's dominated node range is skipped during
evaluation.

Say a node A claims rights over a prunable node P. Then later, another node B
also claims pruning rights over P. If B dominates A, then A's claims are
revoked. Similarly, any other claims on rights over P, by nodes that are
dominated by B are revoked. After this, only remaining claims over P are from
nodes that don't dominate each other. These are all alternate paths from P to
the root. For P to get pruned at runtime, all the claimants should agree at
runtime that P can be pruned.
*/
fn make_claims(
    tree: &Tree,
    ndom: &[usize],
    deps: &DependencyTable,
    threshold: usize,
) -> Result<Box<[Claim]>, Error> {
    use std::cmp::Ordering;
    let mut claims = tree
        .nodes()
        .iter()
        .enumerate()
        .fold(Vec::new(), |mut claims, (ni, node)| match node {
            Constant(_) | Symbol(_) => claims,
            Unary(_, input) => {
                claims.push(Some((*input, ni, PruneKind::None)));
                claims
            }
            Binary(Min | Max, lhs, rhs) => {
                if ndom[*lhs] > threshold && !deps.is_needed_by(*lhs, *rhs) {
                    claims.push(Some((*lhs, ni, PruneKind::Left)));
                }
                if ndom[*rhs] > threshold && !deps.is_needed_by(*rhs, *lhs) {
                    claims.push(Some((*rhs, ni, PruneKind::Right)));
                }
                claims
            }
            Binary(Less | LessOrEqual | Greater | GreaterOrEqual, lhs, rhs)
                if ndom[ni] > threshold =>
            {
                claims.push(Some((ni, ni, PruneKind::Constant(Value::Bool(true)))));
                claims.push(Some((ni, ni, PruneKind::Constant(Value::Bool(false)))));
                claims
            }
            Binary(_, lhs, rhs) => {
                claims.push(Some((*lhs, ni, PruneKind::None)));
                claims.push(Some((*rhs, ni, PruneKind::None)));
                claims
            }
            Ternary(op, _cond, tt, ff) => match op {
                Choose => {
                    if ndom[*tt] > threshold {
                        claims.push(Some((*tt, ni, PruneKind::Left)));
                    }
                    if ndom[*ff] > threshold {
                        claims.push(Some((*ff, ni, PruneKind::Right)));
                    }
                    claims
                }
            },
        });
    claims.sort_by(|a, b| match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(..)) => Ordering::Less,
        (Some(..), None) => Ordering::Greater,
        (Some((la, lb, _)), Some((ra, rb, _))) => (la, lb).cmp(&(ra, rb)),
    });
    // Revoke claims if also claimed by a dominating node.
    for chunk in claims.chunk_by_mut(|a, b| match (a, b) {
        (None, None) => true,
        (None, Some(_)) | (Some(_), None) => false,
        (Some((la, _, _)), Some((ra, _, _))) => la == ra,
    }) {
        // We start at the last claimant and iterate backwards. For each
        // claimant, we revoke any claims by another claimant that is dominated
        // by this claimant.
        let mut i = chunk.len() - 1;
        while i > 0 {
            let (left, right) = chunk.split_at_mut(i);
            i -= 1;
            let claimant = match right[0] {
                Some((_, c, _)) => c,
                None => continue,
            };
            let dom = (claimant - ndom[claimant])..claimant;
            for check in left.iter_mut() {
                if match check {
                    Some((_, c, _)) => dom.contains(c),
                    None => continue,
                } {
                    *check = None;
                }
            }
        }
        // Any claimant that survived the above process has irrevokable pruning
        // rights. If any of them says the node cannot be pruned, then the
        // pruning rights of the rest are revoked.
        if chunk
            .iter()
            .any(|c| matches!(c, Some((_, _, PruneKind::None))))
        {
            chunk.fill(None);
            continue;
        }
    }
    let mut claims = claims.into_iter().flatten().collect::<Vec<_>>();
    // If any node owns it's own pruning rights, then we revoke the rights of
    // other nodes over that node. This is because if the node is capable of
    // choosing when to prune itself, other nodes' opinion about it is
    // irrelevant. We can reconsider this decision later.
    let self_owned: HashSet<usize> = HashSet::from_iter(
        claims
            .iter()
            .filter_map(|(a, b, _)| if *a == *b { Some(*a) } else { None }),
    );
    claims.retain(|(a, b, _)| *a == *b || !self_owned.contains(a));
    Ok(claims
        .into_iter()
        .map(|(claimed, claimant, kind)| Claim {
            claimed,
            claimant,
            kind,
        })
        .collect())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{assert_float_eq, deftree, test_util};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    #[test]
    fn t_claim() {
        // Case 1.
        let tree = deftree!(min (+ 'x 1) (+ 'y (min (+ 'x 1) (+ 'y 1))))
            .unwrap()
            .compacted()
            .unwrap();
        let (tree, ndom) = tree.control_dependence_sorted().unwrap();
        let deps = DependencyTable::from_tree(&tree);
        let claims = make_claims(&tree, &ndom, &deps, 0).unwrap();
        assert_eq!(
            claims.as_ref(),
            &[
                Claim {
                    claimed: 2,
                    claimant: 5,
                    kind: PruneKind::Left
                },
                Claim {
                    claimed: 6,
                    claimant: 7,
                    kind: PruneKind::Right
                }
            ]
        );
        // Case 2.
        let tree = deftree!(min
                            (rem (- (- 'x) 0.016_422) 0.011_2)
                            (- (- 'x) 0.016_422))
        .unwrap()
        .compacted()
        .unwrap();
        let (tree, ndom) = tree.control_dependence_sorted().unwrap();
        let deps = DependencyTable::from_tree(&tree);
        let claims = make_claims(&tree, &ndom, &deps, 0).unwrap();
        assert_eq!(
            claims.as_ref(),
            &[Claim {
                claimed: 5,
                claimant: 6,
                kind: PruneKind::Left
            }]
        );
        // Test case I discovered when benchmarking the hex test case.
        let tree = Tree::read_from(
            "1 1 # output dims
var x
neg 0 # 1
float 3f90d0edc3bd5992 # 2: 0.016422
sub 1 2 # 3
float 3f86f0068db8bac7 # 4: 0.0112
rem 3 4 # 5
min 5 3 # 6
float 3feed288ce703afb # 7: 0.9632
sub 3 7 # 8
max 6 8 # 9
add 9 2 # 10
sub 10 4 # 11
float 3f76f0068db8bac7 # 12: 0.0056
sub 11 12 # 13
abs 13 # 14
float 3f9999999999999a # 15: 0.025
sub 14 15 # 16
float 0 # 17: 0
max 16 17 # 18
mul 18 18 # 19
var y
abs 20 # 21
float 3fcb333333333333 # 22: 0.2125
sub 21 22 # 23
max 23 17 # 24
mul 24 24 # 25
add 25 19 # 26
var z
abs 27 # 28
sub 28 22 # 29
max 29 17 # 30
mul 30 30 # 31
add 31 26 # 32
sqrt 32 # 33
max 23 29 # 34
max 16 34 # 35
min 17 35 # 36
# outputs
add 36 33 # 37
"
            .as_bytes(),
        )
        .unwrap()
        .compacted()
        .unwrap();
        let (tree, ndom) = tree.control_dependence_sorted().unwrap();
        let deps = DependencyTable::from_tree(&tree);
        let claims = make_claims(&tree, &ndom, &deps, 0).unwrap();
        assert_eq!(
            claims.as_ref(),
            &[
                Claim {
                    claimed: 6,
                    claimant: 9,
                    kind: PruneKind::Left
                },
                Claim {
                    claimed: 8,
                    claimant: 9,
                    kind: PruneKind::Right
                },
                Claim {
                    claimed: 16,
                    claimant: 25,
                    kind: PruneKind::Left
                },
                Claim {
                    claimed: 16,
                    claimant: 35,
                    kind: PruneKind::Left
                },
                Claim {
                    claimed: 21,
                    claimant: 27,
                    kind: PruneKind::Left
                },
                Claim {
                    claimed: 21,
                    claimant: 34,
                    kind: PruneKind::Left
                },
                Claim {
                    claimed: 24,
                    claimant: 30,
                    kind: PruneKind::Left
                },
                Claim {
                    claimed: 24,
                    claimant: 34,
                    kind: PruneKind::Right
                },
                Claim {
                    claimed: 35,
                    claimant: 36,
                    kind: PruneKind::Right
                }
            ]
        );
        // Self owning.
        let tree = deftree!(if (< (sqrt (+ 1 (pow 'x 2))) 0) (+ 1 'x) (+ 1 (pow 'x 2)))
            .unwrap()
            .compacted()
            .unwrap();
        let (tree, ndom) = tree.control_dependence_sorted().unwrap();
        let deps = DependencyTable::from_tree(&tree);
        let claims = make_claims(&tree, &ndom, &deps, 0).unwrap();
        assert_eq!(
            claims.as_ref(),
            &[
                Claim {
                    claimed: 4,
                    claimant: 9,
                    kind: PruneKind::Right
                },
                Claim {
                    claimed: 7,
                    claimant: 7,
                    kind: PruneKind::Constant(Value::Bool(true))
                },
                Claim {
                    claimed: 7,
                    claimant: 7,
                    kind: PruneKind::Constant(Value::Bool(false))
                }
            ]
        );
    }

    #[test]
    fn t_pruning_two_circles() {
        let tree = deftree!(min
                            (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5)
                            (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5))
        .unwrap();
        let ctx = JitContext::default();
        let eval = tree.jit_compile_pruner::<f64>(&ctx, "xy", 8, 1e-6).unwrap();
        assert_eq!(eval.n_signals, 3);
        assert_eq!(eval.n_inputs, 2);
        assert_eq!(eval.n_outputs, 1);
        // Prune the RHS with an interval to the left of the origin.
        let mut outputs = [[f64::NAN; 2]];
        let mut signals = [0u32; 3];
        eval.run(&[[-2.0, -1.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_eq!(&signals, &[0, 1, 1]);
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        // Reset and test the other side of the origin.
        signals.fill(0u32);
        outputs[0].fill(f64::NAN);
        eval.run(&[[1.0, 2.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        assert_eq!(&signals, &[1, 0, 2]);
    }

    #[test]
    fn t_pruning_choose_two_circles() {
        let tree = deftree!(if (< 'x 0)
                            (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5)
                            (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5))
        .unwrap();
        let ctx = JitContext::default();
        let eval = tree.jit_compile_pruner::<f64>(&ctx, "xy", 8, 1e-6).unwrap();
        assert_eq!(eval.n_signals, 3);
        assert_eq!(eval.n_inputs, 2);
        assert_eq!(eval.n_outputs, 1);
        // Prune the RHS with an interval to the left of the origin.
        let mut outputs = [[f64::NAN; 2]];
        let mut signals = [0u32; 3];
        eval.run(&[[-2.0, -1.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_eq!(&signals, &[0, 1, 1]);
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        // Reset and test the other side of the origin.
        signals.fill(0u32);
        outputs[0].fill(f64::NAN);
        eval.run(&[[1.0, 2.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        assert_eq!(&signals, &[1, 0, 2]);
    }

    #[test]
    fn t_pruning_three_circles() {
        let tree = deftree!(min
                            (- (sqrt (+ (pow 'x 2) (pow (- 'y 1) 2))) 1.5)
                            (min
                             (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5)
                             (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5)))
        .unwrap();
        let ctx = JitContext::default();
        let eval = tree.jit_compile_pruner::<f64>(&ctx, "xy", 8, 1e-6).unwrap();
        assert_eq!(eval.n_signals, 5);
        assert_eq!(eval.n_inputs, 2);
        assert_eq!(eval.n_outputs, 1);
        // Prune the RHS with an interval to the left of the origin.
        let mut outputs = [[f64::NAN; 2]];
        let mut signals = [0u32; 5];
        eval.run(&[[-2.0, -1.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_eq!(&signals, &[0, 0, 1, 1, 0]);
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        // Reset and test the other side of the origin.
        signals.fill(0u32);
        outputs[0].fill(f64::NAN);
        eval.run(&[[1.0, 2.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        assert_eq!(&signals, &[0, 2, 0, 2, 0]);
    }

    #[test]
    fn t_pruning_two_circles_compacted() {
        let tree = deftree!(min
                            (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5)
                            (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5))
        .unwrap()
        .compacted()
        .unwrap();
        let ctx = JitContext::default();
        // Because the tree is compacted, more nodes are shared, and the min
        // nodes dominate fewer nodes. So we lower the pruning threshold for this test.
        let eval = tree.jit_compile_pruner::<f64>(&ctx, "xy", 3, 1e-6).unwrap();
        assert_eq!(eval.n_signals, 3);
        assert_eq!(eval.n_inputs, 2);
        assert_eq!(eval.n_outputs, 1);
        // Prune the RHS with an interval to the left of the origin.
        let mut outputs = [[f64::NAN; 2]];
        let mut signals = [0u32; 3];
        eval.run(&[[-2.0, -1.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_eq!(&signals, &[0, 1, 1]);
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        // Reset and test the other side of the origin.
        signals.fill(0u32);
        outputs[0].fill(f64::NAN);
        eval.run(&[[1.0, 2.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        assert_eq!(&signals, &[1, 0, 2]);
    }

    #[test]
    fn t_pruning_three_circles_compacted() {
        let tree = deftree!(min
                            (- (sqrt (+ (pow 'x 2) (pow (- 'y 1) 2))) 1.5)
                            (min
                             (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5)
                             (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5)))
        .unwrap()
        .compacted()
        .unwrap();
        let ctx = JitContext::default();
        let eval = tree.jit_compile_pruner::<f64>(&ctx, "xy", 3, 1e-6).unwrap();
        assert_eq!(eval.n_signals, 6);
        assert_eq!(eval.n_inputs, 2);
        assert_eq!(eval.n_outputs, 1);
        // Prune the RHS with an interval to the left of the origin.
        let mut outputs = [[f64::NAN; 2]];
        let mut signals = [0u32; 6];
        eval.run(&[[-2.0, -1.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_eq!(&signals, &[0, 0, 1, 1, 0, 0]);
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        // Reset and test the other side of the origin.
        signals.fill(0u32);
        outputs[0].fill(f64::NAN);
        eval.run(&[[1.0, 2.0], [-1.0, 1.0]], &mut outputs, &mut signals)
            .unwrap();
        assert_float_eq!(outputs[0][0], -1.5);
        assert_float_eq!(outputs[0][1], -0.08578643762690485);
        assert_eq!(&signals, &[0, 1, 0, 2, 0, 0]);
    }

    fn check_pruned_eval_impl<T: NumberType>(
        tree: &Tree,
        pruning_threshold: usize,
        vardata: &[(char, f64, f64)],
        eps: f64,
    ) where
        Wide: SimdVec<T>,
    {
        let params: String = vardata.iter().map(|(c, _, _)| *c).collect();
        let context = JitContext::default();
        let interval_eval = tree
            .jit_compile_interval::<T>(&context, &params)
            .expect("Unable to compile interval eval");
        let eval = tree
            .jit_compile::<T>(&context, &params)
            .expect("Unable to compile single eval");
        let simd_eval = tree.jit_compile_array(&context, &params).unwrap();
        let pruner = tree
            .jit_compile_pruner::<T>(&context, &params, pruning_threshold, eps)
            .expect("Unable to compile a JIT pruner");
        let pruned_interval_eval = pruner
            .compile_interval_func(&context)
            .expect("Unable to compile pruned interval evaluator");
        let pruned_eval = pruner
            .compile_single_func(&context)
            .expect("Unable to compile a pruned single eval");
        let pruned_simd_eval = pruner.compile_simd_func(&context).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let n_outputs = tree.num_roots();
        const N_INTERVALS: usize = 32;
        const N_QUERIES: usize = 32;
        // Reusable buffers for evaluations.
        let mut interval = Vec::<[T; 2]>::new();
        let mut iout_pruned = Vec::<[T; 2]>::new();
        let mut iout = Vec::<[T; 2]>::new();
        let mut sample = Vec::<T>::new();
        let mut out = Vec::<T>::new();
        let mut out_pruned = Vec::<T>::new();
        let mut signals = Vec::<u32>::new();
        let mut simd_buf = JitSimdBuffers::<T>::new(tree);
        let mut simd_pruned_buf = JitSimdBuffers::<T>::new(tree);
        for _ in 0..N_INTERVALS {
            // Sample a random interval.
            interval.clear();
            interval.extend(vardata.iter().map(|(_, lo, hi)| {
                let mut bounds = [0, 1].map(|_| lo + rng.random::<f64>() * (hi - lo));
                if bounds[0] > bounds[1] {
                    bounds.swap(0, 1);
                }
                bounds.map(|b| T::from_f64(b))
            }));
            // Ensure the pruner and interval eval report the same value.
            iout.clear();
            iout.resize(n_outputs, [T::nan(); 2]);
            interval_eval.run(&interval, &mut iout).unwrap();
            // Compare the pruner.
            iout_pruned.clear();
            iout_pruned.resize(n_outputs, [T::nan(); 2]);
            signals.clear();
            signals.resize(pruner.n_signals, 0u32);
            pruner
                .run(&interval, &mut iout_pruned, &mut signals)
                .unwrap();
            // Compare intervals.
            for (i, j) in iout.iter().zip(iout_pruned.iter()) {
                for (i, j) in i.iter().zip(j.iter()) {
                    assert_float_eq!(
                        i.to_f64(),
                        j.to_f64(),
                        eps,
                        "Comparing the pruner with interval eval"
                    );
                }
            }
            // Compare the pruned interval eval.
            iout_pruned.fill([T::nan(); 2]);
            pruned_interval_eval
                .run(&interval, &mut iout_pruned, &signals)
                .unwrap();
            for (i, j) in iout.iter().zip(iout_pruned.iter()) {
                for (i, j) in i.iter().zip(j.iter()) {
                    assert_float_eq!(
                        i.to_f64(),
                        j.to_f64(),
                        eps,
                        "Comparing pruned interval eval"
                    );
                }
            }
            // Now sample that interval and compare pruned and un-pruned evaluations.
            simd_buf.clear();
            simd_pruned_buf.clear();
            for _ in 0..N_QUERIES {
                sample.clear();
                sample.extend(interval.iter().map(|i| {
                    T::from_f64(
                        i[0].to_f64() + rng.random::<f64>() * (i[1].to_f64() - i[0].to_f64()),
                    )
                }));
                simd_buf.pack(&sample).unwrap();
                simd_pruned_buf.pack(&sample).unwrap();
                out_pruned.clear();
                out_pruned.resize(n_outputs, T::nan());
                pruned_eval.run(&sample, &mut out_pruned, &signals).unwrap();
                out.clear();
                out.resize(n_outputs, T::nan());
                eval.run(&sample, &mut out).unwrap();
                for (i, j) in out_pruned.iter().zip(out.iter()) {
                    assert_float_eq!(
                        i.to_f64(),
                        j.to_f64(),
                        eps,
                        "Pruned single eval compare: {:?}; Signals: {:?}\n",
                        sample,
                        signals
                    );
                }
            }
            // Now compare the simd evaluations with and without pruning.
            simd_eval.run(&mut simd_buf);
            pruned_simd_eval
                .run(&mut simd_pruned_buf, &signals)
                .unwrap();
            let mut actual = simd_pruned_buf.unpack_outputs();
            let mut expected = simd_buf.unpack_outputs();
            loop {
                match (actual.next(), expected.next()) {
                    (None, None) => break,
                    (None, Some(_)) | (Some(_), None) => {
                        panic!("The two evaluations returned unequal number of outputs")
                    }
                    (Some(i), Some(j)) => assert_float_eq!(
                        i.to_f64(),
                        j.to_f64(),
                        eps,
                        "Comparing simd evals within the interval"
                    ),
                }
            }
        }
    }

    fn check_pruned_eval(
        tree: Tree,
        pruning_threshold: usize,
        vardata: &[(char, f64, f64)],
        eps: f64,
    ) {
        let tree = tree.compacted().unwrap();
        check_pruned_eval_impl::<f32>(&tree, pruning_threshold, vardata, eps);
        check_pruned_eval_impl::<f64>(&tree, pruning_threshold, vardata, eps);
    }

    #[test]
    fn t_circles() {
        check_pruned_eval(
            deftree!(min
                     (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5)
                     (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5))
            .unwrap()
            .compacted()
            .unwrap(),
            3,
            &[('x', -10.0, 10.0), ('y', -10.0, 10.0)],
            1e-16,
        );
        check_pruned_eval(
            deftree!(min
                     (- (sqrt (+ (pow 'x 2) (pow (- 'y 1) 2))) 1.5)
                     (min
                      (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5)
                      (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5)))
            .unwrap()
            .compacted()
            .unwrap(),
            3,
            &[('x', -10.0, 10.0), ('y', -10.0, 10.0)],
            1e-16,
        );
    }

    #[test]
    fn t_choose_two_circles() {
        check_pruned_eval(
            deftree!(if (< 'x 0)
                            (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5)
                            (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5))
            .unwrap(),
            3,
            &[('x', -10.0, 10.0), ('y', -10.0, 10.0)],
            1e-6,
        );
    }

    #[test]
    fn t_interval_tree_2() {
        check_pruned_eval(
            deftree!(
                (max (min
                      (- (sqrt (+ (+ (pow (- 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 2.75)
                      (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 4.))
                 (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (+ 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 5.25))
            )
                .unwrap(),
            8,
            &[('x', -10., 10.), ('y', -9., 10.), ('z', -11., 12.)],
            1e-5,
        );
    }

    #[test]
    fn t_interval_trees_concat_1() {
        check_pruned_eval(
            deftree!(concat
                     (/ (pow (log (+ (sin 'x) 2.)) 3.) (+ (cos 'x) 2.))
                     (+ 'x 'y)
                     ((max (min
                            (- (sqrt (+ (+ (pow (- 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 2.75)
                            (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 4.))
                       (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (+ 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 5.25))
            )).unwrap(), 8,
            &[('x', -5.0, 5.0), ('y', -5.0, 5.0), ('z', -5.0, 5.0)],
            1e-6,
        );
    }

    #[test]
    fn t_interval_choose() {
        check_pruned_eval(
            deftree!(if (> 'x 0) 'x (- 'x)).unwrap(),
            0,
            &[('x', -10., 10.)],
            1e-16,
        );
        check_pruned_eval(
            deftree!(if (< 'x 0) (- 'x) 'x).unwrap(),
            0,
            &[('x', -10., 10.)],
            1e-16,
        );
    }

    #[test]
    fn t_jit_interval_boolean_ops() {
        check_pruned_eval(
            deftree!(if (and (> 'x 0) (< 'y 5)) (- 'x 2.) (+ 'y 1.5)).unwrap(),
            0,
            &[('x', -2., 3.), ('y', 2., 7.)],
            1e-16,
        );
        check_pruned_eval(
            deftree!(if (or (> 'x 5) (< 'y 0)) (- 'x 2.) (+ 'y 1.5)).unwrap(),
            0,
            &[('x', -2., 3.), ('y', 2., 7.)],
            1e-16,
        );
        check_pruned_eval(
            deftree!(if (not (> 'x 0)) (- 'x 2.) 1.5).unwrap(),
            0,
            &[('x', -5., 5.)],
            1e-16,
        );
    }

    #[test]
    fn t_rational_pow_cond() {
        check_pruned_eval(
            deftree!(if (< (pow 'a 2.1) 1)
                            (- (sqrt (+ (pow (+ 'x 1) 2) (pow 'y 2))) 1.5)
                            (- (sqrt (+ (pow (- 'x 1) 2) (pow 'y 2))) 1.5))
            .unwrap()
            .compacted()
            .unwrap(),
            0,
            &[('a', 0.0, 10.0), ('x', -10.0, 10.0), ('y', -10.0, 10.0)],
            1e-16,
        );
    }

    /**
    The large implicit benchmark hex.vm I used in another project was showing
    inconsistencies in the output values when I use pruning. I managed to reduce
    the size of that large tree to a more manageable repro of the issue. This
    unit test is that tree. Adding a test here to help fix the problem and
    ensure it doesn't regress in the future.
     */
    #[test]
    fn t_hex_test_case_reduced() {
        let tree = deftree!(min
                            (rem (- (- 'x)
                                  0.016_422)
                             0.011_2)
                            (- (- 'x) 0.016_422))
        .unwrap()
        .compacted()
        .unwrap();
        // Setup the buffers;
        let threshold = tree.len() / 10;
        let input: [f32; 3] = [-0.11246335, -0.022609971, -0.0885];
        let interval: [[f32; 2]; 3] = [
            [-0.11246335, -0.07492669],
            [-0.045131966, -0.022609971],
            [-0.08850001, -0.08849999],
        ];
        let mut iout = [[f32::NAN; 2]];
        let mut output = [f32::NAN];
        // Evaluate without pruning for baseline.
        let context = JitContext::default();
        let eval = tree.jit_compile(&context, "xyz").unwrap();
        eval.run(&input, &mut output).unwrap();
        let expected = output[0];
        // Evaluate with pruning.
        let pruner = tree
            .jit_compile_pruner::<f32>(&context, "xyz", threshold, 1e-6)
            .unwrap();
        let prune_eval = pruner.compile_single_func(&context).unwrap();
        let mut signals = vec![0u32; pruner.n_signals];
        pruner.run(&interval, &mut iout, &mut signals).unwrap();
        output.fill(f32::NAN);
        prune_eval.run(&input, &mut output, &signals).unwrap();
        let actual = output[0];
        assert!(
            actual.signum() == expected.signum() && (actual - expected).abs() < f32::EPSILON,
            "

These two values must be equal. This bug was fixed at some point by checking for
dependencies between mutually exclusive code blocks when pruning.

"
        );
    }

    #[test]
    fn t_compare_pruned_eval_random_circles() {
        type ImageBuffer = image::ImageBuffer<image::Luma<u8>, Vec<u8>>;
        const DIMS: u32 = 1 << 7;
        const DIMS_F64: f64 = DIMS as f64;
        const RAD_RANGE: (f64, f64) = (0.02 * DIMS_F64, 0.1 * DIMS_F64);
        const DIM_INTERVAL: u32 = 1 << 5;
        const NUM_CIRCLES: usize = 64;
        const PRUNE_THRESHOLD: usize = 32;
        let tree = test_util::random_circles_sorted(
            (0., DIMS_F64),
            (0., DIMS_F64),
            RAD_RANGE,
            NUM_CIRCLES,
        )
        .compacted()
        .unwrap();
        let context = JitContext::default();
        let expected_image = {
            let mut image = ImageBuffer::new(DIMS, DIMS);
            let eval = tree.jit_compile::<f64>(&context, "xy").unwrap();
            for y in 0..DIMS {
                let mut pos = [f64::NAN, y as f64 + 0.5];
                for x in 0..DIMS {
                    pos[0] = x as f64 + 0.5;
                    let mut out = [f64::NAN];
                    eval.run(&pos, &mut out).unwrap();
                    let val = out[0];
                    *image.get_pixel_mut(x, y) = image::Luma([if val < 0. {
                        f64::min((-val / RAD_RANGE.1) * 255., 255.) as u8
                    } else {
                        f64::min(((RAD_RANGE.1 - val) / RAD_RANGE.1) * 255., 255.) as u8
                    }]);
                }
            }
            image
        };
        let pruned_image = {
            let mut image = ImageBuffer::new(DIMS, DIMS);
            let pruner = tree
                .jit_compile_pruner::<f64>(&context, "xy", PRUNE_THRESHOLD, 1e-6)
                .unwrap();
            let eval = pruner.compile_single_func(&context).unwrap();
            let mut signals = vec![0u32; pruner.n_signals].into_boxed_slice();
            for yi in (0..DIMS).step_by(DIM_INTERVAL as usize) {
                let mut interval = [[f64::NAN; 2], [yi as f64, (yi + DIM_INTERVAL) as f64]];
                for xi in (0..DIMS).step_by(DIM_INTERVAL as usize) {
                    interval[0] = [xi as f64, (xi + DIM_INTERVAL) as f64];
                    let mut iout = [[f64::NAN; 2]];
                    signals.fill(0u32);
                    pruner.run(&interval, &mut iout, &mut signals).unwrap();
                    for y in yi..(yi + DIM_INTERVAL) {
                        let mut pos = [f64::NAN, y as f64 + 0.5];
                        for x in xi..(xi + DIM_INTERVAL) {
                            pos[0] = x as f64 + 0.5;
                            let mut out = [f64::NAN];
                            eval.run(&pos, &mut out, &signals).unwrap();
                            let val = out[0];
                            *image.get_pixel_mut(x, y) = image::Luma([if val < 0. {
                                f64::min((-val / RAD_RANGE.1) * 255., 255.) as u8
                            } else {
                                f64::min(((RAD_RANGE.1 - val) / RAD_RANGE.1) * 255., 255.) as u8
                            }]);
                        }
                    }
                }
            }
            image
        };
        assert_eq!(pruned_image.width(), expected_image.width());
        assert_eq!(pruned_image.height(), expected_image.height());
        assert!(
            pruned_image
                .iter()
                .zip(expected_image.iter())
                .all(|(left, right)| left == right)
        );
    }
}
