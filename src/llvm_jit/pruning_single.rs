use super::{JitCompiler, JitContext, NumberType};
use crate::{BinaryOp::*, Error, Node::*, TernaryOp::*, Tree, UnaryOp::*, Value::*};
use inkwell::{AddressSpace, values::BasicValueEnum};
use std::ffi::c_void;

type UnsafePruningFuncType = unsafe extern "C" fn(
    *const c_void,    // Inputs
    *mut c_void,      // Outputs
    *const *const i8, // Jump targets
    *const i8,        // Traffic signals.
);

struct JumpTable {
    targets: Box<[usize]>,
    offsets: Box<[usize]>,
    prunable: Box<[bool]>,
}

impl JumpTable {
    fn from_counts(counts: &[usize], count_threshold: usize) -> Self {
        let pairs = {
            let mut pairs: Vec<_> = counts
                .iter()
                .enumerate()
                .filter_map(|(i, c)| {
                    if *c >= count_threshold {
                        Some((i - c, i))
                    } else {
                        None
                    }
                })
                .collect();
            pairs.sort();
            pairs
        };
        let mut targets = Vec::with_capacity(counts.len());
        let mut offsets = Vec::with_capacity(counts.len());
        let mut iter = pairs.iter().peekable();
        for ni in 0..counts.len() {
            offsets.push(targets.len());
            while let Some((_, target)) = iter.next_if(|(i, _)| *i == ni) {
                targets.push(*target);
            }
        }
        let prunable = {
            let mut prunable = vec![false; offsets.len()];
            for t in targets.iter() {
                prunable[*t] = true;
            }
            prunable.into_boxed_slice()
        };
        Self {
            targets: targets.into_boxed_slice(),
            offsets: offsets.into_boxed_slice(),
            prunable,
        }
    }

    fn get_targets(&self, node: usize) -> &[usize] {
        let start = self.offsets[node];
        let stop = self
            .offsets
            .get(node + 1)
            .cloned()
            .unwrap_or(self.targets.len());
        &self.targets[start..stop]
    }

    fn is_prunable(&self, node: usize) -> bool {
        self.prunable[node]
    }
}

impl Tree {
    pub fn jit_compile_with_pruning<'ctx, T>(
        &'ctx self,
        context: &'ctx JitContext,
        prune_threshold: usize,
    ) -> Result<(), Error>
    where
        T: NumberType,
    {
        let (tree, counts) = self.control_dependence_sorted()?;
        let jtable = JumpTable::from_counts(&counts, prune_threshold);
        let signal_offsets: Vec<_> = (0..tree.len())
            .map(|i| jtable.is_prunable(i))
            .scan(0usize, |scan, current| {
                let prev = *scan;
                *scan += if current { 1 } else { 0 };
                Some(prev)
            })
            .collect();
        let num_signals = signal_offsets.last().cloned().ok_or(Error::EmptyTree)?;
        debug_assert_eq!(
            (0..tree.len()).filter(|i| jtable.is_prunable(*i)).count(),
            num_signals,
            "The number of prunable nodes should be equal to the number of signals computed via scan."
        );
        const FUNC_NAME: &str = "eiche_pruning_func";
        let num_roots = tree.num_roots();
        let symbols = tree.symbols();
        let context = &context.inner;
        let compiler = JitCompiler::new(context)?;
        let builder = &compiler.builder;
        let float_type = T::jit_type(context);
        let float_ptr_type = context.ptr_type(AddressSpace::default());
        let bool_type = context.bool_type();
        let fn_type = context
            .void_type()
            .fn_type(&[float_ptr_type.into(), float_ptr_type.into()], false);
        let function = compiler.module.add_function(FUNC_NAME, fn_type, None);
        builder.position_at_end(context.append_basic_block(function, "entry"));
        let mut regs: Vec<BasicValueEnum> = Vec::with_capacity(tree.len());
        for (ni, node) in tree.nodes().iter().enumerate() {
            let reg = match node {
                Constant(val) => match val {
                    Bool(val) => BasicValueEnum::IntValue(
                        bool_type.const_int(if *val { 1 } else { 0 }, false),
                    ),
                    Scalar(val) => BasicValueEnum::FloatValue(float_type.const_float(*val)),
                },
                Symbol(label) => {
                    let inputs = function
                        .get_first_param()
                        .ok_or(Error::JitCompilationError("Cannot read inputs".to_string()))?
                        .into_pointer_value();
                    let ptr = unsafe {
                        builder.build_gep(
                            float_type,
                            inputs,
                            &[context.i64_type().const_int(
                                symbols.iter().position(|c| c == label).ok_or(
                                    Error::JitCompilationError("Cannot find symbol".to_string()),
                                )? as u64,
                                false,
                            )],
                            &format!("arg_{}", *label),
                        )?
                    };
                    builder.build_load(float_type, ptr, &format!("reg_{}", *label))?
                }
                _ => todo!("Not implemented"),
            };
        }
        todo!("Not Implemented");
    }
}
