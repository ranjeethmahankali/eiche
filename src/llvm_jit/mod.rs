use crate::error::Error;
use inkwell::{
    OptimizationLevel,
    builder::{Builder, BuilderError},
    context::Context,
    execution_engine::FunctionLookupError,
    intrinsics::Intrinsic,
    module::Module,
    passes::PassManager,
    targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine},
    types::{BasicTypeEnum, FloatType, IntType},
    values::{BasicMetadataValueEnum, BasicValueEnum, FloatValue, VectorValue},
};
use std::{
    cell::RefCell,
    fmt::Debug,
    ops::{Add, AddAssign, Div, DivAssign, MulAssign, Neg, Sub, SubAssign},
    path::Path,
};

/// Context for comiling a tree to LLVM. This is just a thin wrapper around
/// inkwell Context.
pub struct JitContext {
    inner: Context,
    numfuncs: RefCell<usize>,
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
            numfuncs: Default::default(),
        }
    }
}

impl JitContext {
    fn new_func_name<T: NumberType>(&self, suffix: Option<&str>) -> String {
        let mut nf = self.numfuncs.borrow_mut();
        let idx = *nf;
        *nf += 1;
        format!("func_{}_{}_{}", idx, T::type_str(), suffix.unwrap_or(""))
    }
}

impl From<BuilderError> for Error {
    fn from(value: BuilderError) -> Self {
        Error::JitCompilationError(value.to_string())
    }
}

impl From<FunctionLookupError> for Error {
    fn from(value: FunctionLookupError) -> Self {
        Error::JitCompilationError(value.to_string())
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
        self.machine
            .write_to_file(&self.module, FileType::Assembly, path)
            .unwrap();
    }
}

pub trait NumberType:
    Copy
    + PartialOrd
    + Neg
    + Div
    + DivAssign
    + Add
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + MulAssign
    + Debug
{
    fn nan() -> Self;

    fn jit_type(context: &Context) -> FloatType<'_>;

    fn jit_int_type(context: &Context) -> IntType<'_>;

    fn from_f64(val: f64) -> Self;

    fn min(a: Self, b: Self) -> Self;

    fn max(a: Self, b: Self) -> Self;

    fn type_str() -> &'static str;

    fn is_nan(&self) -> bool;
}

impl NumberType for f32 {
    fn nan() -> Self {
        f32::NAN
    }

    fn jit_type(context: &Context) -> FloatType<'_> {
        context.f32_type()
    }

    fn from_f64(val: f64) -> Self {
        val as f32
    }

    fn min(a: Self, b: Self) -> Self {
        f32::min(a, b)
    }

    fn max(a: Self, b: Self) -> Self {
        f32::max(a, b)
    }

    fn type_str() -> &'static str {
        "f32"
    }

    fn jit_int_type(context: &Context) -> IntType<'_> {
        context.i32_type()
    }

    fn is_nan(&self) -> bool {
        f32::is_nan(*self)
    }
}

impl NumberType for f64 {
    fn nan() -> Self {
        f64::NAN
    }

    fn jit_type(context: &Context) -> FloatType<'_> {
        context.f64_type()
    }

    fn from_f64(val: f64) -> Self {
        val
    }

    fn min(a: Self, b: Self) -> Self {
        f64::min(a, b)
    }

    fn max(a: Self, b: Self) -> Self {
        f64::max(a, b)
    }

    fn type_str() -> &'static str {
        "f64"
    }

    fn jit_int_type(context: &Context) -> IntType<'_> {
        context.i64_type()
    }

    fn is_nan(&self) -> bool {
        f64::is_nan(*self)
    }
}

fn build_vec_unary_intrinsic<'ctx>(
    builder: &'ctx Builder,
    module: &'ctx Module,
    name: &'static str,
    call_name: &str,
    input: VectorValue<'ctx>,
) -> Result<BasicValueEnum<'ctx>, Error> {
    let intrinsic = Intrinsic::find(name).ok_or(Error::CannotCompileIntrinsic(name))?;
    let intrinsic_fn = intrinsic
        .get_declaration(module, &[BasicTypeEnum::VectorType(input.get_type())])
        .ok_or(Error::CannotCompileIntrinsic(name))?;
    builder
        .build_call(
            intrinsic_fn,
            &[BasicMetadataValueEnum::VectorValue(input)],
            call_name,
        )
        .map_err(|_| Error::CannotCompileIntrinsic(name))?
        .try_as_basic_value()
        .left()
        .ok_or(Error::CannotCompileIntrinsic(name))
}

fn build_vec_binary_intrinsic<'ctx>(
    builder: &'ctx Builder,
    module: &'ctx Module,
    name: &'static str,
    call_name: &str,
    lhs: VectorValue<'ctx>,
    rhs: VectorValue<'ctx>,
) -> Result<BasicValueEnum<'ctx>, Error> {
    let intrinsic = Intrinsic::find(name).ok_or(Error::CannotCompileIntrinsic(name))?;
    let intrinsic_fn = intrinsic
        .get_declaration(
            module,
            &[
                BasicTypeEnum::VectorType(lhs.get_type()),
                BasicTypeEnum::VectorType(rhs.get_type()),
            ],
        )
        .ok_or(Error::CannotCompileIntrinsic(name))?;
    builder
        .build_call(
            intrinsic_fn,
            &[
                BasicMetadataValueEnum::VectorValue(lhs),
                BasicMetadataValueEnum::VectorValue(rhs),
            ],
            call_name,
        )
        .map_err(|_| Error::CannotCompileIntrinsic(name))?
        .try_as_basic_value()
        .left()
        .ok_or(Error::CannotCompileIntrinsic(name))
}

fn build_float_unary_intrinsic<'ctx>(
    builder: &'ctx Builder,
    module: &'ctx Module,
    name: &'static str,
    call_name: &str,
    input: FloatValue<'ctx>,
) -> Result<BasicValueEnum<'ctx>, Error> {
    let intrinsic = Intrinsic::find(name).ok_or(Error::CannotCompileIntrinsic(name))?;
    let intrinsic_fn = intrinsic
        .get_declaration(module, &[BasicTypeEnum::FloatType(input.get_type())])
        .ok_or(Error::CannotCompileIntrinsic(name))?;
    builder
        .build_call(
            intrinsic_fn,
            &[BasicMetadataValueEnum::FloatValue(input)],
            call_name,
        )
        .map_err(|_| Error::CannotCompileIntrinsic(name))?
        .try_as_basic_value()
        .left()
        .ok_or(Error::CannotCompileIntrinsic(name))
}

fn build_float_binary_intrinsic<'ctx>(
    builder: &'ctx Builder,
    module: &'ctx Module,
    name: &'static str,
    call_name: &str,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    float_type: FloatType<'ctx>,
) -> Result<BasicValueEnum<'ctx>, Error> {
    let intrinsic = Intrinsic::find(name).ok_or(Error::CannotCompileIntrinsic(name))?;
    let intrinsic_fn = intrinsic
        .get_declaration(
            module,
            &[
                BasicTypeEnum::FloatType(float_type),
                BasicTypeEnum::FloatType(float_type),
            ],
        )
        .ok_or(Error::CannotCompileIntrinsic(name))?;
    builder
        .build_call(
            intrinsic_fn,
            &[
                BasicMetadataValueEnum::FloatValue(lhs.into_float_value()),
                BasicMetadataValueEnum::FloatValue(rhs.into_float_value()),
            ],
            call_name,
        )
        .map_err(|_| Error::CannotCompileIntrinsic(name))?
        .try_as_basic_value()
        .left()
        .ok_or(Error::CannotCompileIntrinsic(name))
}

pub mod single;

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
pub mod simd_array;

pub mod interval;
