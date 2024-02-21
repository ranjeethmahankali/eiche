use inkwell::{builder::Builder, context::Context, module::Module, passes::PassManager};

use crate::error::Error;

pub struct JitContext {
    inner: Context,
}

impl Default for JitContext {
    fn default() -> Self {
        JitContext {
            inner: Context::create(),
        }
    }
}

struct JitCompiler<'ctx> {
    module: Module<'ctx>,
    builder: Builder<'ctx>,
}

impl<'ctx> JitCompiler<'ctx> {
    pub fn new(context: &'ctx Context) -> Result<JitCompiler<'ctx>, Error> {
        let module = context.create_module("eiche_module");
        Ok(JitCompiler {
            module,
            builder: context.create_builder(),
        })
    }

    fn run_passes(&self) {
        // Run optimization passes.
        let fpm = PassManager::create(());
        fpm.add_instruction_combining_pass();
        fpm.add_reassociate_pass();
        fpm.add_gvn_pass();
        fpm.add_cfg_simplification_pass();
        fpm.add_basic_alias_analysis_pass();
        fpm.add_promote_memory_to_register_pass();
        fpm.run_on(&self.module);
    }
}

pub mod single;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod simd_array;
