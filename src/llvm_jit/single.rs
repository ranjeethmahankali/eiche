use super::{JitCompiler, JitContext};
use crate::{
    error::Error,
    tree::{BinaryOp::*, Node::*, TernaryOp::*, Tree, UnaryOp::*, Value::*},
};
use inkwell::{
    builder::Builder,
    execution_engine::JitFunction,
    intrinsics::Intrinsic,
    module::Module,
    types::{BasicTypeEnum, FloatType},
    values::{BasicMetadataValueEnum, BasicValueEnum},
    AddressSpace, FloatPredicate, OptimizationLevel,
};

type UnsafeFuncType = unsafe extern "C" fn(*const f64, *mut f64);

/// This represents a JIT compiled tree. This is a wrapper around the JIT compiled native function.
pub struct JitEvaluator<'ctx> {
    func: JitFunction<'ctx, UnsafeFuncType>,
    num_inputs: usize,
    outputs: Vec<f64>,
}

impl<'ctx> JitEvaluator<'ctx> {
    pub fn create(
        func: JitFunction<'ctx, UnsafeFuncType>,
        num_inputs: usize,
        num_outputs: usize,
    ) -> JitEvaluator<'ctx> {
        // Construct evaluator
        let eval = JitEvaluator {
            func,
            num_inputs,
            outputs: vec![f64::NAN; num_outputs],
        };
        eval
    }
}

impl Tree {
    /// JIT compile a tree and return a native evaluator.
    pub fn jit_compile<'ctx>(&'ctx self, context: &'ctx JitContext) -> Result<JitEvaluator, Error> {
        const FUNC_NAME: &str = "eiche_func";
        let num_roots = self.num_roots();
        let symbols = self.symbols();
        let context = &context.inner;
        let compiler = JitCompiler::new(context)?;
        let builder = &compiler.builder;
        let f64_type = context.f64_type();
        let f64_ptr_type = f64_type.ptr_type(AddressSpace::default());
        let bool_type = context.bool_type();
        let fn_type = context
            .void_type()
            .fn_type(&[f64_ptr_type.into(), f64_ptr_type.into()], false);
        let function = compiler.module.add_function(FUNC_NAME, fn_type, None);
        let basic_block = context.append_basic_block(function, "entry");
        builder.position_at_end(basic_block);
        let mut regs: Vec<BasicValueEnum> = Vec::with_capacity(self.len());
        for (ni, node) in self.nodes().iter().enumerate() {
            let reg = match node {
                Constant(val) => match val {
                    Bool(val) => BasicValueEnum::IntValue(
                        bool_type.const_int(if *val { 1 } else { 0 }, false),
                    ),
                    Scalar(val) => BasicValueEnum::FloatValue(f64_type.const_float(*val)),
                },
                Symbol(label) => {
                    let inputs = function
                        .get_first_param()
                        .ok_or(Error::JitCompilationError("Cannot read inputs".to_string()))?
                        .into_pointer_value();
                    let ptr = unsafe {
                        builder
                            .build_gep(
                                inputs,
                                &[context.i64_type().const_int(
                                    symbols.iter().position(|c| c == label).ok_or(
                                        Error::JitCompilationError(
                                            "Cannot find symbol".to_string(),
                                        ),
                                    )? as u64,
                                    false,
                                )],
                                &format!("arg_{}", *label),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?
                    };
                    builder
                        .build_load(ptr, &format!("reg_{}", *label))
                        .map_err(|e| Error::JitCompilationError(e.to_string()))?
                }
                Unary(op, input) => match op {
                    Negate => BasicValueEnum::FloatValue(
                        builder
                            .build_float_neg(
                                regs[*input].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    Sqrt => build_float_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.sqrt.*",
                        "sqrt_call",
                        regs[*input],
                        f64_type,
                    )?,
                    Abs => build_float_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.fabs.*",
                        "abs_call",
                        regs[*input],
                        f64_type,
                    )?,
                    Sin => build_float_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.sin.*",
                        "sin_call",
                        regs[*input],
                        f64_type,
                    )?,
                    Cos => build_float_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.cos.*",
                        "cos_call",
                        regs[*input],
                        f64_type,
                    )?,
                    Tan => {
                        let sin = build_float_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.sin.*",
                            "sin_call",
                            regs[*input],
                            f64_type,
                        )?;
                        let cos = build_float_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.cos.*",
                            "cos_call",
                            regs[*input],
                            f64_type,
                        )?;
                        BasicValueEnum::FloatValue(
                            builder
                                .build_float_div(
                                    sin.into_float_value(),
                                    cos.into_float_value(),
                                    &format!("reg_{}", ni),
                                )
                                .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                        )
                    }
                    Log => build_float_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.log.*",
                        "log_call",
                        regs[*input],
                        f64_type,
                    )?,
                    Exp => build_float_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.exp.*",
                        "exp_call",
                        regs[*input],
                        f64_type,
                    )?,
                    Not => BasicValueEnum::IntValue(
                        builder
                            .build_not(regs[*input].into_int_value(), &format!("reg_{}", ni))
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                },
                Binary(op, lhs, rhs) => match op {
                    Add => BasicValueEnum::FloatValue(
                        builder
                            .build_float_add(
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    Subtract => BasicValueEnum::FloatValue(
                        builder
                            .build_float_sub(
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    Multiply => BasicValueEnum::FloatValue(
                        builder
                            .build_float_mul(
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    Divide => BasicValueEnum::FloatValue(
                        builder
                            .build_float_div(
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    Pow => build_float_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.pow.*",
                        "pow_call",
                        regs[*lhs],
                        regs[*rhs],
                        f64_type,
                    )?,
                    Min => build_float_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.minnum.*",
                        "min_call",
                        regs[*lhs],
                        regs[*rhs],
                        f64_type,
                    )?,
                    Max => build_float_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.maxnum.*",
                        "max_call",
                        regs[*lhs],
                        regs[*rhs],
                        f64_type,
                    )?,
                    Less => BasicValueEnum::IntValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::ULT,
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    LessOrEqual => BasicValueEnum::IntValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::ULE,
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    Equal => BasicValueEnum::IntValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::UEQ,
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    NotEqual => BasicValueEnum::IntValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::UNE,
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    Greater => BasicValueEnum::IntValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::UGT,
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    GreaterOrEqual => BasicValueEnum::IntValue(
                        builder
                            .build_float_compare(
                                FloatPredicate::UGE,
                                regs[*lhs].into_float_value(),
                                regs[*rhs].into_float_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    And => BasicValueEnum::IntValue(
                        builder
                            .build_and(
                                regs[*lhs].into_int_value(),
                                regs[*rhs].into_int_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                    Or => BasicValueEnum::IntValue(
                        builder
                            .build_or(
                                regs[*lhs].into_int_value(),
                                regs[*rhs].into_int_value(),
                                &format!("reg_{}", ni),
                            )
                            .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                    ),
                },
                Ternary(op, a, b, c) => match op {
                    Choose => builder
                        .build_select(
                            regs[*a].into_int_value(),
                            regs[*b].into_float_value(),
                            regs[*c].into_float_value(),
                            &format!("reg_{}", ni),
                        )
                        .map_err(|e| Error::JitCompilationError(e.to_string()))?,
                },
            };
            regs.push(reg);
        }
        // Copy the outputs.
        let outputs = function
            .get_last_param()
            .ok_or(Error::JitCompilationError(
                "Cannot write to outputs".to_string(),
            ))?
            .into_pointer_value();
        for (i, reg) in regs[(self.len() - num_roots)..].iter().enumerate() {
            let dst = unsafe {
                builder
                    .build_gep(
                        outputs,
                        &[context.i64_type().const_int(i as u64, false)],
                        &format!("output_{}", i),
                    )
                    .map_err(|e| Error::JitCompilationError(e.to_string()))?
            };
            builder
                .build_store(dst, *reg)
                .map_err(|e| Error::JitCompilationError(e.to_string()))?;
        }
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
        return Ok(JitEvaluator::create(func, symbols.len(), num_roots));
    }
}

fn build_float_unary_intrinsic<'ctx>(
    builder: &'ctx Builder,
    module: &'ctx Module,
    name: &'static str,
    call_name: &'static str,
    input: BasicValueEnum<'ctx>,
    f64_type: FloatType<'ctx>,
) -> Result<BasicValueEnum<'ctx>, Error> {
    let intrinsic = Intrinsic::find(name).ok_or(Error::CannotCompileIntrinsic(name))?;
    let intrinsic_fn = intrinsic
        .get_declaration(module, &[BasicTypeEnum::FloatType(f64_type)])
        .ok_or(Error::CannotCompileIntrinsic(name))?;
    builder
        .build_call(
            intrinsic_fn,
            &[BasicMetadataValueEnum::FloatValue(input.into_float_value())],
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
    call_name: &'static str,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    f64_type: FloatType<'ctx>,
) -> Result<BasicValueEnum<'ctx>, Error> {
    let intrinsic = Intrinsic::find(name).ok_or(Error::CannotCompileIntrinsic(name))?;
    let intrinsic_fn = intrinsic
        .get_declaration(
            module,
            &[
                BasicTypeEnum::FloatType(f64_type),
                BasicTypeEnum::FloatType(f64_type),
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

impl<'ctx> JitEvaluator<'ctx> {
    /// Run the JIT evaluator with the given input values. The number of input
    /// values is expected to be the same as the number of variables in the
    /// tree. The variables are substituted with the input values in the same
    /// order as returned by calling `tree.symbols()` which was compiled to
    /// produce this evaluator.
    pub fn run(&mut self, inputs: &[f64]) -> Result<&[f64], Error> {
        if inputs.len() != self.num_inputs {
            return Err(Error::InputSizeMismatch(inputs.len(), self.num_inputs));
        }
        unsafe { self.func.call(inputs.as_ptr(), self.outputs.as_mut_ptr()) };
        Ok(&self.outputs)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{deftree, test::util::check_value_eval};

    fn check_jit_eval(tree: &Tree, vardata: &[(char, f64, f64)], samples_per_var: usize, eps: f64) {
        let context = JitContext::default();
        let mut jiteval = tree.jit_compile(&context).unwrap();
        check_value_eval(
            tree.clone(),
            |inputs: &[f64], outputs: &mut [f64]| {
                let results = jiteval.run(inputs).unwrap();
                outputs.copy_from_slice(results);
            },
            vardata,
            samples_per_var,
            eps,
        );
    }

    #[test]
    fn t_prod_sum() {
        check_jit_eval(
            &deftree!(concat (+ x y) (* x y)).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.)],
            100,
            0.,
        );
    }

    #[test]
    fn t_sub_div() {
        check_jit_eval(
            &deftree!(concat (- x y) (/ x y)).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.)],
            20,
            0.,
        );
    }

    #[test]
    fn t_pow() {
        check_jit_eval(&deftree!(pow x 2).unwrap(), &[('x', -10., -10.)], 100, 0.);
    }

    #[test]
    fn t_sqrt() {
        check_jit_eval(&deftree!(sqrt x).unwrap(), &[('x', 0.01, 10.)], 100, 0.);
    }

    #[test]
    fn t_circle() {
        check_jit_eval(
            &deftree!(- (sqrt (+ (pow x 2) (pow y 2))) 3).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.)],
            20,
            0.,
        );
    }

    #[test]
    fn t_sum_3() {
        check_jit_eval(
            &deftree!(+ (+ x 3) (+ y z)).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            5,
            0.,
        );
    }

    #[test]
    fn t_sphere() {
        check_jit_eval(
            &deftree!(- (sqrt (+ (pow x 2) (+ (pow y 2) (pow z 2)))) 3).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            0.,
        )
    }

    #[test]
    fn t_negate() {
        check_jit_eval(
            &deftree!(* (- x) (+ y z)).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            0.,
        );
    }

    #[test]
    fn t_abs() {
        check_jit_eval(
            &deftree!(* (abs x) (+ (abs y) (abs z))).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            0.,
        );
    }

    #[test]
    fn t_trigonometry() {
        check_jit_eval(
            &deftree!(/ (+ (sin x) (cos y)) (+ 0.27 (pow (tan z) 2))).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            1e-14,
        );
    }

    #[test]
    fn t_log_exp() {
        check_jit_eval(
            &deftree!(/ (+ 1 (log x)) (+ 1 (exp y))).unwrap(),
            &[('x', 0.1, 5.), ('y', 0.1, 5.)],
            10,
            0.,
        );
    }

    #[test]
    fn t_min_max() {
        check_jit_eval(
            &deftree!(
                (max (min
                      (- (sqrt (+ (+ (pow (- x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 2.75)
                      (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (- y 3.) 2.)) (pow (- z 4.) 2.))) 4.))
                 (- (sqrt (+ (+ (pow (+ x 2.) 2.) (pow (+ y 3.) 2.)) (pow (- z 4.) 2.))) 5.25))
            )
            .unwrap(),
            &[('x', -10., 10.), ('y', -9., 10.), ('z', -11., 12.)],
            20,
            0.,
        );
    }

    #[test]
    fn t_choose() {
        check_jit_eval(
            &deftree!(if (> x 0) x (-x)).unwrap(),
            &[('x', -10., 10.)],
            100,
            0.,
        );
        check_jit_eval(
            &deftree!(if (< x 0) (- x) x).unwrap(),
            &[('x', -10., 10.)],
            100,
            0.,
        );
    }

    #[test]
    fn t_or_and() {
        check_jit_eval(
            &deftree!(if (and (> x 0) (< x 1)) (* 2 x) 1).unwrap(),
            &[('x', -3., 3.)],
            100,
            0.,
        );
    }

    #[test]
    fn t_not() {
        check_jit_eval(
            &deftree!(if (not (> x 0)) (- (pow x 3) (pow y 3)) (+ (pow x 2) (pow y 2))).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.)],
            100,
            1e-14,
        );
    }
}

#[cfg(test)]
mod perft {
    use super::*;
    use crate::{
        dedup::Deduplicater,
        deftree,
        eval::ValueEvaluator,
        prune::Pruner,
        test::util::assert_float_eq,
        // test::util::assert_float_eq,
        tree::{min, MaybeTree, Tree},
    };
    use rand::{rngs::StdRng, SeedableRng};
    use std::time::{Duration, Instant};

    fn _sample_range(range: (f64, f64), rng: &mut StdRng) -> f64 {
        use rand::Rng;
        range.0 + rng.gen::<f64>() * (range.1 - range.0)
    }
    const _RADIUS_RANGE: (f64, f64) = (0.2, 2.);
    const _X_RANGE: (f64, f64) = (0., 100.);
    const _Y_RANGE: (f64, f64) = (0., 100.);
    const _Z_RANGE: (f64, f64) = (0., 100.);
    const _N_SPHERES: usize = 5000;
    const _N_QUERIES: usize = 5000;

    fn _sphere_union() -> Tree {
        let mut rng = StdRng::seed_from_u64(42);
        let mut make_sphere = || -> MaybeTree {
            deftree!(- (sqrt (+ (+
                                 (pow (- x (const _sample_range(_X_RANGE, &mut rng))) 2)
                                 (pow (- y (const _sample_range(_Y_RANGE, &mut rng))) 2))
                              (pow (- z (const _sample_range(_Z_RANGE, &mut rng))) 2)))
                     (const _sample_range(_RADIUS_RANGE, &mut rng)))
        };
        let mut tree = make_sphere();
        for _ in 1.._N_SPHERES {
            tree = min(tree, make_sphere());
        }
        let tree = tree.unwrap();
        assert_eq!(tree.dims(), (1, 1));
        tree
    }

    fn _benchmark_eval(
        values: &mut Vec<f64>,
        queries: &[[f64; 3]],
        eval: &mut ValueEvaluator,
    ) -> Duration {
        let before = Instant::now();
        values.extend(queries.iter().map(|coords| {
            eval.set_value('x', coords[0].into());
            eval.set_value('y', coords[1].into());
            eval.set_value('z', coords[2].into());
            let results = eval.run().unwrap();
            results[0].scalar().unwrap()
        }));
        Instant::now() - before
    }

    fn _benchmark_jit(
        values: &mut Vec<f64>,
        queries: &[[f64; 3]],
        eval: &mut JitEvaluator,
    ) -> Duration {
        let before = Instant::now();
        values.extend(queries.iter().map(|coords| {
            let results = eval.run(coords).unwrap();
            results[0]
        }));
        Instant::now() - before
    }

    // Run this function to bench mark the performance
    fn _t_perft() {
        let mut rng = StdRng::seed_from_u64(234);
        let queries: Vec<[f64; 3]> = (0.._N_QUERIES)
            .map(|_| {
                [
                    _sample_range(_X_RANGE, &mut rng),
                    _sample_range(_Y_RANGE, &mut rng),
                    _sample_range(_Z_RANGE, &mut rng),
                ]
            })
            .collect();
        let before = Instant::now();
        let tree = {
            let mut dedup = Deduplicater::new();
            let mut pruner = Pruner::new();
            _sphere_union()
                .fold()
                .unwrap()
                .deduplicate(&mut dedup)
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
        };
        println!(
            "Tree creation time: {}ms",
            (Instant::now() - before).as_millis()
        );
        let mut values1: Vec<f64> = Vec::with_capacity(_N_QUERIES);
        let mut eval = ValueEvaluator::new(&tree);
        println!(
            "Tree has {} nodes, evaluator allocated {} registers",
            tree.len(),
            eval.num_regs()
        );
        let evaltime = _benchmark_eval(&mut values1, &queries, &mut eval);
        println!("Evaluator time: {}ms", evaltime.as_millis());
        let mut values2: Vec<f64> = Vec::with_capacity(_N_QUERIES);
        let context = JitContext::default();
        let mut jiteval = {
            let before = Instant::now();
            let jiteval = tree.jit_compile(&context).unwrap();
            println!(
                "Compilation time: {}ms",
                (Instant::now() - before).as_millis()
            );
            jiteval
        };
        let jittime = _benchmark_jit(&mut values2, &queries, &mut jiteval);
        println!("Jit time: {}ms", jittime.as_millis());
        let ratio = evaltime.as_millis() as f64 / jittime.as_millis() as f64;
        println!("Ratio: {}", ratio);
        assert_eq!(values1.len(), values2.len());
        for (l, r) in values1.iter().zip(values2.iter()) {
            assert_float_eq!(l, r, 1e-15);
        }
    }
}
