use std::path::Path;

use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    passes::PassManager,
    targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine},
    OptimizationLevel,
};

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
    machine: TargetMachine,
}

impl<'ctx> JitCompiler<'ctx> {
    pub fn new(context: &'ctx Context) -> Result<JitCompiler<'ctx>, Error> {
        Target::initialize_native(&InitializationConfig::default())
            .expect("Failed to initialize native target");
        let reloc = RelocMode::Default;
        let model = CodeModel::JITDefault;
        let triple = TargetMachine::get_default_triple();
        let cpu = TargetMachine::get_host_cpu_name().to_string();
        let target = Target::from_triple(&triple).unwrap();
        let features = TargetMachine::get_host_cpu_features().to_string();
        let machine = target
            .create_target_machine(
                &triple,
                &cpu,
                &features,
                OptimizationLevel::Aggressive,
                reloc,
                model,
            )
            .unwrap();
        let module = context.create_module("eiche_module");
        module.set_triple(&machine.get_triple());
        module.set_data_layout(&machine.get_target_data().get_data_layout());
        Ok(JitCompiler {
            module,
            builder: context.create_builder(),
            machine,
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

    #[allow(dead_code)]
    pub fn write_asm(&self, path: &Path) {
        self.machine
            .write_to_file(&self.module, FileType::Assembly, path)
            .unwrap();
    }
}

pub mod single;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod simd_array;
