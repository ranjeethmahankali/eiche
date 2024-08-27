use crate::error::Error;
use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    passes::PassManager,
    targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple},
    OptimizationLevel,
};
use std::path::Path;

/// Context for comiling a tree to LLVM. This is just a thin wrapper around
/// inkwell Context.
pub struct JitContext {
    inner: Context,
}

struct JitCompiler<'ctx> {
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    machine: TargetMachine,
}

impl Default for JitContext {
    fn default() -> Self {
        JitContext {
            inner: Context::create(),
        }
    }
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

    /// Run optimization passes.
    fn run_passes(&self) {
        let fpm = PassManager::create(());
        fpm.add_instruction_combining_pass();
        fpm.add_reassociate_pass();
        fpm.add_gvn_pass();
        fpm.add_cfg_simplification_pass();
        fpm.add_basic_alias_analysis_pass();
        fpm.add_promote_memory_to_register_pass();
        fpm.run_on(&self.module);
    }

    /// Write out the compiled assembly to file specified by `path`.
    #[allow(dead_code)]
    pub fn write_asm(&self, path: &Path) {
        let triple = TargetTriple::create("x86_64-pc-windows-msvc");
        let reloc = RelocMode::Default;
        let model = CodeModel::JITDefault;
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
        machine
            .write_to_file(&self.module, FileType::Object, path)
            .unwrap();
    }

    #[allow(dead_code)]
    pub fn write_llvm_ir(&self, path: &Path) {
        match self.module.print_to_file(path) {
            Ok(_) => {}
            Err(err) => eprintln!("Error when printing LLVM IR to a file: {}", err),
        }
    }
}

pub mod single;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod simd_array;
