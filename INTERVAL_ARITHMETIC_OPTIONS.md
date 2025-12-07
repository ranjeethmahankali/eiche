# Pure Rust Interval Arithmetic Implementation Options

## Current State of inari

### What Works WITHOUT GMP (Pure Rust + SIMD)
- Basic arithmetic: `+`, `-`, `*`, `/`, `sqrt`, `sqr`, `recip`
- Uses inline assembly to control CPU rounding modes:
  - **x86-64**: MXCSR register (see `src/simd/x86_64/avx_fma.rs`)
  - **AArch64**: FPCR register (see `src/simd/aarch64.rs`)
- Achieves correct directed rounding for hardware operations
- Fast, tight bounds, no dependencies

### What Requires GMP/MPFR
- **Transcendental functions**: sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, asinh, acosh, atanh, exp, exp2, exp10, ln, log2, log10, pow, powi
- **Reason**: No CPU instructions with directed rounding for these functions
- Standard library `f64::sin()` etc. do NOT respect rounding modes

## Why Setting Rounding Modes Doesn't Help stdlib Functions

Standard library transcendental functions:
- Call into libc or use intrinsics
- Use complex internal algorithms
- **Ignore CPU rounding mode** - aim for ~1 ULP accuracy but not correctly-rounded results
- Cannot be used for interval arithmetic even with rounding modes set

## Implementation Approaches for Transcendental Functions

### 1. Range Reduction + Polynomial Approximation (RECOMMENDED)

**See working examples in this repo:**
- `examples/exp.rs` - Implements exp, exp2, exp10
- `examples/log.rs` - Implements ln, log2, log10

**Technique:**
1. **Range reduction**: Decompose input into manageable range
   - exp: `exp(x) = 2^(lg(e)*x)`, then `2^x = 2^a * 2^b` where `a ∈ ℤ, b ∈ [-1/2, 1/2]`
   - log: `lg(x) = a + lg(b)` where `x = 2^a * b`, `b` in small range
2. **Taylor polynomial** on reduced range with proven error bounds
   - exp: 14-term series gives error < 2^-53
   - log: 22-term series gives error < 2^-53
3. **Interval arithmetic** with directed rounding for all operations
4. **Combine results** using exact transformations

**Advantages:**
- Pure Rust, no dependencies
- Mathematically rigorous with proven error bounds
- Fast (precomputed coefficients)
- You understand the math completely
- Easy to translate to JIT machine code

**Tradeoffs vs MPFR:**
- Bounds are valid but slightly looser (not tightest possible)
- Must implement each function individually
- Need mathematical analysis for error bounds

**For your use case (JIT compilation):** This is ideal - you can directly translate the polynomial evaluation into machine code.

### 2. Use Pure Rust Library: astro-float

**Library:** https://github.com/stencillogic/astro-float

**Features:**
- Arbitrary precision floating-point
- **Has `RoundingMode::Up` and `RoundingMode::Down`** - exactly what you need
- All transcendental functions with directed rounding:
  - Exponential/log: exp, ln, log, log2, log10
  - Trig: sin, cos, tan, asin, acos, atan
  - Hyperbolic: sinh, cosh, tanh, asinh, acosh, atanh
  - Power: sqrt, cbrt, pow, powi
- Pure Rust, Windows-friendly
- Actively maintained

**Example:**
```rust
use astro_float::{BigFloat, Consts, RoundingMode, ctx::Context};

let mut ctx = Context::new(128, RoundingMode::Down,
    Consts::new().unwrap(), -10000, 10000);

let x = BigFloat::from_f64(1.5, 128);
let sin_lower = x.sin(128, RoundingMode::Down, &mut ctx);
let sin_upper = x.sin(128, RoundingMode::Up, &mut ctx);
```

**Tradeoffs:**
- Dependency on external library
- Not as tight as MPFR (but close)
- Won't help with JIT compilation (black box)

### 3. Other Pure Rust Libraries
- **dashu**: GMP/MPFR alternative, less mature for interval arithmetic
- **malachite**: MPFR algorithms in Rust, less clear on directed rounding
- **mpmfnum**: Has directed rounding, sparse docs on transcendental functions

## Implementing Rounding Modes in Rust

**For basic arithmetic (you can do this yourself):**

```rust
use std::arch::asm;

fn add_round_up(mut x: f64, y: f64) -> f64 {
    unsafe {
        asm!(
            "sub rsp, 8",
            "vstmxcsr [rsp]",              // Save MXCSR
            "mov eax, [rsp]",
            "and eax, 0xFFFF9FFF",         // Clear rounding bits
            "or eax, 0x00006000",          // Set round-up
            "mov [rsp+4], eax",
            "vldmxcsr [rsp+4]",            // Load new MXCSR
            "vaddsd {x}, {x}, {y}",        // Add with round-up
            "vldmxcsr [rsp]",              // Restore MXCSR
            "add rsp, 8",
            x = inout(xmm_reg) x,
            y = in(xmm_reg) y,
            out("eax") _,
        );
    }
    x
}
```

**This works for:** +, -, *, /, sqrt, FMA (CPU instructions)
**This does NOT work for:** sin, cos, exp, log, etc. (no CPU instructions)

## Recommended Path for Your Goals

### Option A: Implement Math Yourself (Best for JIT)
1. Start with basic arithmetic using inline asm (see `src/simd/`)
2. Port `examples/exp.rs` and `examples/log.rs` into library
3. Implement trig functions using similar range reduction + Taylor series
4. Reference papers for error bounds (e.g., Oishi & Kashiwagi cited in examples)
5. All code will be transparent for JIT translation

**Effort:** High initial investment, complete control
**Accuracy:** Valid bounds, slightly looser than MPFR
**Performance:** Excellent (especially with SIMD)

### Option B: Use astro-float Initially
1. Replace GMP with astro-float for Windows builds
2. Learn from their implementations
3. Gradually replace with your own implementations for JIT

**Effort:** Low initial investment
**Accuracy:** Good (close to MPFR)
**Performance:** Good (but black box for JIT)

## Key Mathematical Resources

**From inari examples:**
- Oishi, S., & Kashiwagi M. (2018). 数学関数の精度保証. In S. Oishi (Ed.), 精度保証付き数値計算の基礎 (pp. 91-107). コロナ社.

**Core techniques:**
- Range reduction (critical for tight bounds)
- Taylor series with Lagrange remainder bounds
- Rigorous error analysis
- All arithmetic done with interval operations

## Performance Notes

**What makes inari fast:**
- SIMD operations for basic arithmetic (see `src/simd/`)
- Minimal branching
- Tight integration with hardware rounding modes
- Precomputed constants

**For JIT compilation:**
- Range reduction is table lookups + integer ops
- Polynomial evaluation is just FMA chains
- Can inline everything and eliminate overhead
- Perfect fit for code generation

## Building on Windows

**Current problem:** GMP/MPFR are C libraries, painful to build on MSVC
**Solutions:**
- Use pure Rust (Option A: DIY, Option B: astro-float)
- Both compile trivially on Windows with cargo

## Summary

For your goals (understand the math, implement yourself, eventual JIT):
1. Study `examples/exp.rs` and `examples/log.rs` - they're well-documented
2. Implement basic operations using inline asm (copy from `src/simd/`)
3. Port the example implementations into the library
4. Extend to other transcendental functions using similar techniques
5. All code will be transparent and JIT-friendly

The math is not complex - it's undergraduate calculus + error analysis. The examples show it's ~100 lines per function family.
