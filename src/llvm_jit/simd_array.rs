use super::{
    JitCompiler, JitContext, NumberType, build_vec_binary_intrinsic, build_vec_unary_intrinsic,
};
use crate::{
    error::Error,
    tree::{BinaryOp::*, Node::*, TernaryOp::*, Tree, UnaryOp::*, Value::*},
};
use inkwell::{
    AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel,
    context::Context,
    execution_engine::JitFunction,
    types::{FloatType, VectorType},
    values::{BasicValue, BasicValueEnum},
};
use std::{ffi::c_void, marker::PhantomData, mem::size_of};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub type SimdType64 = __m256d;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub type SimdType32 = __m256;

#[cfg(target_arch = "aarch64")]
type SimdType64 = float64x2x2_t;
#[cfg(target_arch = "aarch64")]
type SimdType32 = float32x4x2_t;

const SIMD_F32_SIZE: usize = size_of::<SimdType64>() / size_of::<f32>();
const SIMD_F64_SIZE: usize = size_of::<SimdType64>() / size_of::<f64>();

/// Thin wrapper around a simd floating point value. The union makes it easier
/// to access the individual floating point numbers.
#[repr(C)]
#[derive(Copy, Clone)]
pub union Wide {
    valsf32: [f32; SIMD_F32_SIZE],
    valsf64: [f64; SIMD_F64_SIZE],
    valsu32: [u32; SIMD_F32_SIZE],
    valsu64: [u64; SIMD_F64_SIZE],
    reg64: SimdType64,
    reg32: SimdType32,
}

/// This trait exists to allow reuse of code between f32 and f64 types with
/// generics. i.e. this enables sharing the code to compile and run the compiled
/// tree for both f32 and f64. This could represent a simd vector of f64 values,
/// or that of twice as many f32 values.
pub trait SimdVec<T>
where
    T: Copy,
{
    /// The number of values of type T in the wide simd type.
    const SIMD_VEC_SIZE: usize;

    /// Broadcast the value to all lanes of a simd vec and return in.
    fn broadcast(val: T) -> Wide;

    /// Get a simd vector filled with NaNs.
    fn nan() -> Wide;

    /// Set the entry at `idx` to value `val`.
    fn set(&mut self, val: T, idx: usize);

    /// Get the value at index `idx`.
    fn get(&self, idx: usize) -> T;

    /// Get the type of float, either f32 or f64.
    fn float_type(context: &Context) -> FloatType<'_>;

    /// Get a constant value with all entries in the simd vector populated with `val`.
    fn const_float(val: f64, context: &Context) -> BasicValueEnum<'_>;

    /// Get a constant value with all entries in the simd vector populated with `val`.
    fn const_bool(val: bool, context: &Context) -> BasicValueEnum<'_>;

    fn mul(a: Self, b: Self) -> Self;

    fn div(a: Self, b: Self) -> Self;

    fn add(a: Self, b: Self) -> Self;

    fn mul_add(a: Self, b: Self, c: Self) -> Self;

    fn sub(a: Self, b: Self) -> Self;

    fn neg(a: Self) -> Self;

    fn lt(a: Self, b: Self) -> Wide;

    fn eq(a: Self, b: Self) -> Wide;

    fn gt(a: Self, b: Self) -> Wide;

    fn and(a: Self, b: Self) -> Wide;

    fn or(a: Self, b: Self) -> Wide;

    fn check_bool(a: Self, lane: usize) -> bool;

    /**
    Check if the given simd lane is set to a non-zero integer value. This is
    useful for checking a specific lane after doing a simd comparison or other
    Boolean operation.

    # Safety

    This function doesn't do any bounds checking. It is the caller's
    responsibility to make sure the lane index is within bounds.
     */
    unsafe fn check_bool_unchecked(a: Self, lane: usize) -> bool;

    fn abs(a: Self) -> Self;

    fn recip_sqrt(a: Self) -> Self;

    fn max(a: Self, b: Self) -> Self;

    fn min(a: Self, b: Self) -> Self;
}

impl SimdVec<f32> for Wide {
    const SIMD_VEC_SIZE: usize = SIMD_F32_SIZE;

    fn broadcast(val: f32) -> Wide {
        Wide {
            valsf32: [val; <Self as SimdVec<f32>>::SIMD_VEC_SIZE],
        }
    }

    fn nan() -> Wide {
        Self::broadcast(f32::NAN)
    }

    fn set(&mut self, val: f32, idx: usize) {
        // SAFETY: accessing the union. valsf32 is correct.
        unsafe { self.valsf32[idx] = val }
    }

    fn get(&self, idx: usize) -> f32 {
        // SAFETY: accesing the union, valsf32 is correct.
        unsafe { self.valsf32[idx] }
    }

    fn float_type(context: &Context) -> FloatType<'_> {
        context.f32_type()
    }

    fn const_float(val: f64, context: &Context) -> BasicValueEnum<'_> {
        BasicValueEnum::VectorValue(VectorType::const_vector(
            &[<Self as SimdVec<f32>>::float_type(context).const_float(val);
                <Self as SimdVec<f32>>::SIMD_VEC_SIZE],
        ))
    }

    fn const_bool(val: bool, context: &Context) -> BasicValueEnum<'_> {
        BasicValueEnum::VectorValue(VectorType::const_vector(
            &[context
                .bool_type()
                .const_int(if val { 1 } else { 0 }, false);
                <Self as SimdVec<f32>>::SIMD_VEC_SIZE],
        ))
    }

    fn mul(a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrnsics. reg32 is correct.
                reg32: unsafe { _mm256_mul_ps(a.reg32, b.reg32) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrnsics. reg32 is correct.
                reg32: unsafe {
                    float32x4x2_t(
                        vmulq_f32(a.reg32.0, b.reg32.0),
                        vmulq_f32(a.reg32.1, b.reg32.1),
                    )
                },
            }
        }
    }

    fn div(a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrnsics. reg32 is correct.
                reg32: unsafe { _mm256_div_ps(a.reg32, b.reg32) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg32: unsafe {
                    float32x4x2_t(
                        vdivq_f32(a.reg32.0, b.reg32.0),
                        vdivq_f32(a.reg32.1, b.reg32.1),
                    )
                },
            }
        }
    }

    fn add(a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg32: unsafe { _mm256_add_ps(a.reg32, b.reg32) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg32: unsafe {
                    float32x4x2_t(
                        vaddq_f32(a.reg32.0, b.reg32.0),
                        vaddq_f32(a.reg32.1, b.reg32.1),
                    )
                },
            }
        }
    }

    fn sub(a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg32: unsafe { _mm256_sub_ps(a.reg32, b.reg32) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg32: unsafe {
                    float32x4x2_t(
                        vsubq_f32(a.reg32.0, b.reg32.0),
                        vsubq_f32(a.reg32.1, b.reg32.1),
                    )
                },
            }
        }
    }

    fn mul_add(a: Self, b: Self, c: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg32: unsafe { _mm256_fmadd_ps(a.reg32, b.reg32, c.reg32) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg32: unsafe {
                    float32x4x2_t(
                        vfmaq_f32(c.reg32.0, a.reg32.0, b.reg32.0),
                        vfmaq_f32(c.reg32.1, a.reg32.1, b.reg32.1),
                    )
                },
            }
        }
    }

    fn neg(a: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg32: unsafe { _mm256_sub_ps(_mm256_setzero_ps(), a.reg32) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg32: unsafe { float32x4x2_t(vnegq_f32(a.reg32.0), vnegq_f32(a.reg32.1)) },
            }
        }
    }

    fn lt(a: Self, b: Self) -> Wide {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg32: unsafe { _mm256_cmp_ps::<_CMP_LT_OQ>(a.reg32, b.reg32) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg32: unsafe {
                    float32x4x2_t(
                        vreinterpretq_f32_u32(vcltq_f32(a.reg32.0, b.reg32.0)),
                        vreinterpretq_f32_u32(vcltq_f32(a.reg32.1, b.reg32.1)),
                    )
                },
            }
        }
    }

    fn eq(a: Self, b: Self) -> Wide {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg32: unsafe { _mm256_cmp_ps::<_CMP_EQ_OQ>(a.reg32, b.reg32) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg32: unsafe {
                    float32x4x2_t(
                        vreinterpretq_f32_u32(vceqq_f32(a.reg32.0, b.reg32.0)),
                        vreinterpretq_f32_u32(vceqq_f32(a.reg32.1, b.reg32.1)),
                    )
                },
            }
        }
    }

    fn gt(a: Self, b: Self) -> Wide {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg32: unsafe { _mm256_cmp_ps::<_CMP_GT_OQ>(a.reg32, b.reg32) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg32: unsafe {
                    float32x4x2_t(
                        vreinterpretq_f32_u32(vcgtq_f32(a.reg32.0, b.reg32.0)),
                        vreinterpretq_f32_u32(vcgtq_f32(a.reg32.1, b.reg32.1)),
                    )
                },
            }
        }
    }

    fn and(a: Self, b: Self) -> Wide {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg32: unsafe { _mm256_and_ps(a.reg32, b.reg32) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg32: unsafe {
                    float32x4x2_t(
                        vreinterpretq_f32_u32(vandq_u32(
                            vreinterpretq_u32_f32(a.reg32.0),
                            vreinterpretq_u32_f32(b.reg32.0),
                        )),
                        vreinterpretq_f32_u32(vandq_u32(
                            vreinterpretq_u32_f32(a.reg32.1),
                            vreinterpretq_u32_f32(b.reg32.1),
                        )),
                    )
                },
            }
        }
    }

    fn or(a: Self, b: Self) -> Wide {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg32: unsafe { _mm256_or_ps(a.reg32, b.reg32) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg32: unsafe {
                    float32x4x2_t(
                        vreinterpretq_f32_u32(vorrq_u32(
                            vreinterpretq_u32_f32(a.reg32.0),
                            vreinterpretq_u32_f32(b.reg32.0),
                        )),
                        vreinterpretq_f32_u32(vorrq_u32(
                            vreinterpretq_u32_f32(a.reg32.1),
                            vreinterpretq_u32_f32(b.reg32.1),
                        )),
                    )
                },
            }
        }
    }

    fn check_bool(a: Self, lane: usize) -> bool {
        // SAFETY: union access. valsu32 is correct.
        unsafe { a.valsu32[lane] != 0 }
    }

    /** # Safety

    See the trait function for more details: [`SimdVec::check_bool_unchecked`].
     */
    unsafe fn check_bool_unchecked(a: Self, lane: usize) -> bool {
        // SAFETY: union access. valsu32 is correct.
        unsafe { *a.valsu32.get_unchecked(lane) != 0 }
    }

    fn abs(a: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg32 is correct.
                // Clear the sign bit.
                reg32: unsafe { _mm256_andnot_ps(_mm256_set1_ps(-0.0f32), a.reg32) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg32 is correct.
                reg32: unsafe { float32x4x2_t(vabsq_f32(a.reg32.0), vabsq_f32(a.reg32.1)) },
            }
        }
    }

    fn recip_sqrt(a: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg32 is correct.
                reg32: unsafe { _mm256_rsqrt_ps(a.reg32) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg32 is correct.
                reg32: unsafe { float32x4x2_t(vrsqrteq_f32(a.reg32.0), vrsqrteq_f32(a.reg32.1)) },
            }
        }
    }

    fn max(a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg32 is correct.
                reg32: unsafe { _mm256_max_ps(a.reg32, b.reg32) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg32 is correct.
                reg32: unsafe {
                    float32x4x2_t(
                        vmaxq_f32(a.reg32.0, b.reg32.0),
                        vmaxq_f32(a.reg32.1, b.reg32.1),
                    )
                },
            }
        }
    }

    fn min(a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg32 is correct.
                reg32: unsafe { _mm256_min_ps(a.reg32, b.reg32) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg32 is correct.
                reg32: unsafe {
                    float32x4x2_t(
                        vminq_f32(a.reg32.0, b.reg32.0),
                        vminq_f32(a.reg32.1, b.reg32.1),
                    )
                },
            }
        }
    }
}

impl SimdVec<f64> for Wide {
    const SIMD_VEC_SIZE: usize = SIMD_F64_SIZE;

    fn broadcast(val: f64) -> Wide {
        Wide {
            valsf64: [val; <Self as SimdVec<f64>>::SIMD_VEC_SIZE],
        }
    }

    fn nan() -> Wide {
        Self::broadcast(f64::NAN)
    }

    fn set(&mut self, val: f64, idx: usize) {
        // SAFETY: union access. valsf64 is correct.
        unsafe { self.valsf64[idx] = val }
    }

    fn get(&self, idx: usize) -> f64 {
        // SAFETY: union access. valsf64 is correct.
        unsafe { self.valsf64[idx] }
    }

    fn float_type(context: &Context) -> FloatType<'_> {
        context.f64_type()
    }

    fn const_float(val: f64, context: &Context) -> BasicValueEnum<'_> {
        BasicValueEnum::VectorValue(VectorType::const_vector(
            &[<Self as SimdVec<f64>>::float_type(context).const_float(val);
                <Self as SimdVec<f64>>::SIMD_VEC_SIZE],
        ))
    }

    fn const_bool(val: bool, context: &Context) -> BasicValueEnum<'_> {
        BasicValueEnum::VectorValue(VectorType::const_vector(
            &[context
                .bool_type()
                .const_int(if val { 1 } else { 0 }, false);
                <Self as SimdVec<f64>>::SIMD_VEC_SIZE],
        ))
    }

    fn mul(a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe { _mm256_mul_pd(a.reg64, b.reg64) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe {
                    float64x2x2_t(
                        vmulq_f64(a.reg64.0, b.reg64.0),
                        vmulq_f64(a.reg64.1, b.reg64.1),
                    )
                },
            }
        }
    }

    fn div(a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe { _mm256_div_pd(a.reg64, b.reg64) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe {
                    float64x2x2_t(
                        vdivq_f64(a.reg64.0, b.reg64.0),
                        vdivq_f64(a.reg64.1, b.reg64.1),
                    )
                },
            }
        }
    }

    fn add(a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe { _mm256_add_pd(a.reg64, b.reg64) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe {
                    float64x2x2_t(
                        vaddq_f64(a.reg64.0, b.reg64.0),
                        vaddq_f64(a.reg64.1, b.reg64.1),
                    )
                },
            }
        }
    }

    fn sub(a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe { _mm256_sub_pd(a.reg64, b.reg64) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe {
                    float64x2x2_t(
                        vsubq_f64(a.reg64.0, b.reg64.0),
                        vsubq_f64(a.reg64.1, b.reg64.1),
                    )
                },
            }
        }
    }

    fn mul_add(a: Self, b: Self, c: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe { _mm256_fmadd_pd(a.reg64, b.reg64, c.reg64) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe {
                    float64x2x2_t(
                        vfmaq_f64(c.reg64.0, a.reg64.0, b.reg64.0),
                        vfmaq_f64(c.reg64.1, a.reg64.1, b.reg64.1),
                    )
                },
            }
        }
    }

    fn neg(a: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe { _mm256_sub_pd(_mm256_setzero_pd(), a.reg64) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe { float64x2x2_t(vnegq_f64(a.reg64.0), vnegq_f64(a.reg64.1)) },
            }
        }
    }

    fn lt(a: Self, b: Self) -> Wide {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe { _mm256_cmp_pd::<_CMP_LT_OQ>(a.reg64, b.reg64) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe {
                    float64x2x2_t(
                        vreinterpretq_f64_u64(vcltq_f64(a.reg64.0, b.reg64.0)),
                        vreinterpretq_f64_u64(vcltq_f64(a.reg64.1, b.reg64.1)),
                    )
                },
            }
        }
    }

    fn eq(a: Self, b: Self) -> Wide {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe { _mm256_cmp_pd::<_CMP_EQ_OQ>(a.reg64, b.reg64) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe {
                    float64x2x2_t(
                        vreinterpretq_f64_u64(vceqq_f64(a.reg64.0, b.reg64.0)),
                        vreinterpretq_f64_u64(vceqq_f64(a.reg64.1, b.reg64.1)),
                    )
                },
            }
        }
    }

    fn gt(a: Self, b: Self) -> Wide {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe { _mm256_cmp_pd::<_CMP_GT_OQ>(a.reg64, b.reg64) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe {
                    float64x2x2_t(
                        vreinterpretq_f64_u64(vcgtq_f64(a.reg64.0, b.reg64.0)),
                        vreinterpretq_f64_u64(vcgtq_f64(a.reg64.1, b.reg64.1)),
                    )
                },
            }
        }
    }

    fn and(a: Self, b: Self) -> Wide {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe { _mm256_and_pd(a.reg64, b.reg64) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe {
                    float64x2x2_t(
                        vreinterpretq_f64_u64(vandq_u64(
                            vreinterpretq_u64_f64(a.reg64.0),
                            vreinterpretq_u64_f64(b.reg64.0),
                        )),
                        vreinterpretq_f64_u64(vandq_u64(
                            vreinterpretq_u64_f64(a.reg64.1),
                            vreinterpretq_u64_f64(b.reg64.1),
                        )),
                    )
                },
            }
        }
    }

    fn or(a: Self, b: Self) -> Wide {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe { _mm256_or_pd(a.reg64, b.reg64) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe {
                    float64x2x2_t(
                        vreinterpretq_f64_u64(vorrq_u64(
                            vreinterpretq_u64_f64(a.reg64.0),
                            vreinterpretq_u64_f64(b.reg64.0),
                        )),
                        vreinterpretq_f64_u64(vorrq_u64(
                            vreinterpretq_u64_f64(a.reg64.1),
                            vreinterpretq_u64_f64(b.reg64.1),
                        )),
                    )
                },
            }
        }
    }

    fn check_bool(a: Self, lane: usize) -> bool {
        // SAFETY: valsu64 is correct.
        unsafe { a.valsu64[lane] != 0 }
    }

    /** # Safety

    See the trait function for more details: [`SimdVec::check_bool_unchecked`].
     */
    unsafe fn check_bool_unchecked(a: Self, lane: usize) -> bool {
        // SAFETY: valsu64 is correct.
        unsafe { *a.valsu64.get_unchecked(lane) != 0 }
    }

    fn abs(a: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                // Clear the sign bit.
                reg64: unsafe { _mm256_andnot_pd(_mm256_set1_pd(-0.0f64), a.reg64) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe { float64x2x2_t(vabsq_f64(a.reg64.0), vabsq_f64(a.reg64.1)) },
            }
        }
    }

    fn recip_sqrt(a: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe { _mm256_div_pd(_mm256_set1_pd(1.), _mm256_sqrt_pd(a.reg64)) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe { float64x2x2_t(vrsqrteq_f64(a.reg64.0), vrsqrteq_f64(a.reg64.1)) },
            }
        }
    }

    fn max(a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe { _mm256_max_pd(a.reg64, b.reg64) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe {
                    float64x2x2_t(
                        vmaxq_f64(a.reg64.0, b.reg64.0),
                        vmaxq_f64(a.reg64.1, b.reg64.1),
                    )
                },
            }
        }
    }

    fn min(a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe { _mm256_min_pd(a.reg64, b.reg64) },
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Wide {
                // SAFETY: SIMD intrinsics. reg64 is correct.
                reg64: unsafe {
                    float64x2x2_t(
                        vminq_f64(a.reg64.0, b.reg64.0),
                        vminq_f64(a.reg64.1, b.reg64.1),
                    )
                },
            }
        }
    }
}

impl From<f32> for Wide {
    fn from(value: f32) -> Self {
        <Self as SimdVec<f32>>::broadcast(value)
    }
}

impl From<f64> for Wide {
    fn from(value: f64) -> Self {
        <Self as SimdVec<f64>>::broadcast(value)
    }
}

pub type NativeSimdFunc = unsafe extern "C" fn(*const c_void, *mut c_void, u64);

/// Thin wrapper around the compiled native JIT function to do simd evaluations.
pub struct JitSimdFn<'ctx, T>
where
    Wide: SimdVec<T>,
    T: NumberType,
{
    func: JitFunction<'ctx, NativeSimdFunc>,
    phantom: PhantomData<T>,
}

/**
`JitSimdFn` is not thread safe, because it contains the executable memory where
the JIT machine code resides, somewhere inside the Execution Engine. LLVM
doesn't implement the `Send` trait for this block of memory, because it doesn't
know what's in the JIT machine code, it doesn't know if that code itself is
thread safe, or has side effects. This `JitSimdFnSync` can be pulled out of a
`JitSimdFn`, via the `.as_async()` function, and is thread safe. It implements
the `Send` trait. This is OK, because we know the machine code represents a
mathematical expression without any side effects. So we pull out the function
pointer and wrap it in this struct, that can be shared across threads. Still the
execution engine held inside the original `JitSmdFn` needs to outlive this sync
wrapper, because it owns the block of executable memory. To guarantee that, this
structs pseudo borrows (via a phantom) from the `JitSimdFn`. It has to be done
via a phantom othwerwise we can't implement The Sync trait on this.
*/
pub struct JitSimdFnSync<'ctx, T>
where
    Wide: SimdVec<T>,
    T: NumberType,
{
    func: NativeSimdFunc,
    phantom: PhantomData<&'ctx JitSimdFn<'ctx, T>>,
}

unsafe impl<'ctx, T> Sync for JitSimdFnSync<'ctx, T>
where
    Wide: SimdVec<T>,
    T: NumberType,
{
}

pub struct JitSimdBuffers<T>
where
    Wide: SimdVec<T>,
    T: NumberType,
{
    num_samples: usize,
    num_inputs: usize,
    num_outputs: usize,
    inputs: Vec<Wide>,
    outputs: Vec<Wide>,
    phantom: PhantomData<T>, // This only exists to specialize the type for type T.
}

impl<'ctx, T> JitSimdFn<'ctx, T>
where
    Wide: SimdVec<T>,
    T: NumberType,
{
    pub fn run(&self, buf: &mut JitSimdBuffers<T>) {
        // SAFETY: Calling a raw function pointer. `JitSimdBuffers` is a safe
        // wrapper that populates the inputs correctly via it's public API, and
        // knows the correct number of SIMD iterations required.
        unsafe {
            self.func.call(
                buf.inputs.as_ptr().cast(),
                buf.outputs.as_mut_ptr().cast(),
                buf.num_simd_iters() as u64,
            );
        }
    }

    pub fn as_sync(&'ctx self) -> JitSimdFnSync<'ctx, T> {
        JitSimdFnSync {
            // SAFETY: Accessing the raw function pointer. This is ok, because
            // this borrows from Self, which owns an Rc reference to the
            // execution engine that owns the block of executable memory to
            // which the function pointer points.
            func: unsafe { self.func.as_raw() },
            phantom: PhantomData,
        }
    }
}

impl<'ctx, T> JitSimdFnSync<'ctx, T>
where
    Wide: SimdVec<T>,
    T: NumberType,
{
    pub fn run(&self, buf: &mut JitSimdBuffers<T>) {
        // SAFETY: Calling a raw function pointer. `JitSimdBuffers` is a safe
        // wrapper that populates the inputs correctly via it's public API, and
        // knows the correct number of SIMD iterations required.
        unsafe {
            self.run_raw(
                buf.inputs.as_ptr().cast(),
                buf.outputs.as_mut_ptr().cast(),
                buf.num_simd_iters(),
            );
        }
    }

    /**
    Same as [`run`], except this version doesn't do bounds checking, because it
    works with pointers. The inputs must be packed in the right order. Say a
    tree has three inputs `a`, `b`, and `c`. The first `Wide` should contain the
    first `SIMD_VEC_SIZE` values of `a` (e.g. either 4 or 8 depending on
    precision). The next `Wide` should contain the next `SIMD_VEC_SIZE` of `b`,
    and so on. The data ust be interleaved in this way to ensure the inputs of
    one iteration are together in memory.

    # Safety

    It is the caller's responsibility to make sure the length of the input slice
    matches the number of symbols of the tree, and the length of the outputs
    slice matches the number of roots of the tree used to create this JIT
    function. Further more, because this JIT array function supports multiple
    iterations, the slices must be large enough to accomodate that. The inputs
    slice must be `n_symbols * num_iters` long, and the output slice must be
    `n_roots * num_iters` long.
     */
    pub unsafe fn run_raw(&self, inputs: *const Wide, outputs: *mut Wide, num_iters: usize) {
        // SAFETY: Calling a raw functin pointer. We told the caller it's their
        // responsibility to make sure inputs are correct.
        unsafe {
            (self.func)(inputs.cast(), outputs.cast(), num_iters as u64);
        }
    }
}

impl<T> JitSimdBuffers<T>
where
    Wide: SimdVec<T>,
    T: NumberType,
{
    const SIMD_VEC_SIZE: usize = <Wide as SimdVec<T>>::SIMD_VEC_SIZE;

    pub fn new(tree: &Tree) -> Self {
        Self {
            num_samples: 0,
            num_inputs: tree.symbols().len(),
            num_outputs: tree.num_roots(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            phantom: PhantomData,
        }
    }

    pub fn reset_for_tree(&mut self, tree: &Tree) {
        self.num_samples = 0;
        self.num_inputs = tree.symbols().len();
        self.num_outputs = tree.num_roots();
        self.inputs.clear();
        self.outputs.clear();
    }

    fn num_simd_iters(&self) -> usize {
        (self.num_samples / Self::SIMD_VEC_SIZE)
            + if self.num_samples.is_multiple_of(Self::SIMD_VEC_SIZE) {
                0
            } else {
                1
            }
    }

    /// Push a new set of input values. The length of `sample` is expected to be
    /// the same as the number of symbols in the tree that was compiled to
    /// produce this JIT evaluator. The values are substituted into the
    /// variables in the same order as they are returned by calling
    /// `tree.symbols` on the tree that produced this JIT evaluator.
    pub fn pack(&mut self, sample: &[T]) -> Result<(), Error> {
        if sample.len() != self.num_inputs {
            return Err(Error::InputSizeMismatch(sample.len(), self.num_inputs));
        }
        let index = self.num_samples % Self::SIMD_VEC_SIZE;
        if index == 0 {
            self.inputs.extend(std::iter::repeat_n(
                <Wide as SimdVec<T>>::nan(),
                self.num_inputs,
            ));
            self.outputs.extend(std::iter::repeat_n(
                <Wide as SimdVec<T>>::nan(),
                self.num_outputs,
            ));
        }
        let inpsize = self.inputs.len();
        for (reg, val) in self.inputs[(inpsize - self.num_inputs)..]
            .iter_mut()
            .zip(sample.iter())
        {
            <Wide as SimdVec<T>>::set(reg, *val, index);
        }
        self.num_samples += 1;
        Ok(())
    }

    /// Clear all inputs and outputs.
    pub fn clear(&mut self) {
        self.inputs.clear();
        self.clear_outputs();
    }

    pub fn clear_outputs(&mut self) {
        self.outputs.clear();
        self.num_samples = 0;
    }

    pub fn unpack_outputs(&self) -> impl Iterator<Item = T> {
        debug_assert_eq!(self.outputs.len() % self.num_outputs, 0);
        self.outputs
            .chunks_exact(self.num_outputs)
            .flat_map(|chunk| {
                (0..Self::SIMD_VEC_SIZE).flat_map(|lane| {
                    chunk
                        .iter()
                        .map(move |simd| <Wide as SimdVec<T>>::get(simd, lane))
                })
            })
            .take(self.num_samples * self.num_outputs)
    }
}

impl Tree {
    /// Compile the tree for doing native simd calculations.
    pub fn jit_compile_array<'ctx, T>(
        &'ctx self,
        context: &'ctx JitContext,
        symbols: &str,
    ) -> Result<JitSimdFn<'ctx, T>, Error>
    where
        Wide: SimdVec<T>,
        T: NumberType,
    {
        if !self.is_scalar() {
            // Only support scalar output trees.
            return Err(Error::TypeMismatch);
        }
        let num_roots = self.num_roots();
        let func_name = context.new_func_name::<T>(Some("array"));
        let context = &context.inner;
        let compiler = JitCompiler::new(context)?;
        let builder = &compiler.builder;
        let float_type = <Wide as SimdVec<T>>::float_type(context);
        let i64_type = context.i64_type();
        let fvec_type = float_type.vec_type(<Wide as SimdVec<T>>::SIMD_VEC_SIZE as u32);
        let fptr_type = context.ptr_type(AddressSpace::default());
        let fn_type = context.void_type().fn_type(
            &[fptr_type.into(), fptr_type.into(), i64_type.into()],
            false,
        );
        let function = compiler.module.add_function(&func_name, fn_type, None);
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
        builder.build_unconditional_branch(loop_block)?;
        // Start the loop
        builder.position_at_end(loop_block);
        let phi = builder.build_phi(i64_type, "counter_phi")?;
        phi.add_incoming(&[(&i64_type.const_int(0, false), start_block)]);
        let index = phi.as_basic_value().into_int_value();
        let mut regs: Vec<BasicValueEnum> = Vec::with_capacity(self.len());
        for (ni, node) in self.nodes().iter().enumerate() {
            let reg = match node {
                Constant(val) => match val {
                    Bool(val) => <Wide as SimdVec<T>>::const_bool(*val, context),
                    Scalar(val) => <Wide as SimdVec<T>>::const_float(*val, context),
                },
                Symbol(label) => {
                    let offset = builder.build_int_add(
                        builder.build_int_mul(
                            index,
                            i64_type.const_int(symbols.len() as u64, false),
                            &format!("input_offset_mul_{label}"),
                        )?,
                        i64_type.const_int(
                            symbols.chars().position(|c| c == *label).ok_or(
                                Error::JitCompilationError("Cannot find symbol".to_string()),
                            )? as u64,
                            false,
                        ),
                        &format!("input_offset_add_{label}"),
                    )?;
                    builder.build_load(
                        fvec_type,
                        // SAFETY: GEP can segfault if the index is out of
                        // bounds. The offset calculation looks pretty solid,
                        // and is thoroughly tested.
                        unsafe {
                            builder.build_gep(fvec_type, inputs, &[offset], &format!("arg_{label}"))
                        }?,
                        &format!("arg_{label}"),
                    )?
                }
                Unary(op, input) => match op {
                    Negate => builder
                        .build_float_neg(regs[*input].into_vector_value(), &format!("reg_{ni}"))?
                        .as_basic_value_enum(),
                    Sqrt => build_vec_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.sqrt.*",
                        "sqrt_call",
                        regs[*input].into_vector_value(),
                    )?,
                    Abs => build_vec_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.fabs.*",
                        "abs_call",
                        regs[*input].into_vector_value(),
                    )?,
                    Sin => build_vec_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.sin.*",
                        "sin_call",
                        regs[*input].into_vector_value(),
                    )?,
                    Cos => build_vec_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.cos.*",
                        "cos_call",
                        regs[*input].into_vector_value(),
                    )?,
                    Tan => {
                        let sin = build_vec_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.sin.*",
                            "sin_call",
                            regs[*input].into_vector_value(),
                        )?;
                        let cos = build_vec_unary_intrinsic(
                            builder,
                            &compiler.module,
                            "llvm.cos.*",
                            "cos_call",
                            regs[*input].into_vector_value(),
                        )?;
                        builder
                            .build_float_div(
                                sin.into_vector_value(),
                                cos.into_vector_value(),
                                &format!("reg_{ni}"),
                            )?
                            .as_basic_value_enum()
                    }
                    Log => build_vec_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.log.*",
                        "log_call",
                        regs[*input].into_vector_value(),
                    )?,
                    Exp => build_vec_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.exp.*",
                        &format!("exp_call_{ni}"),
                        regs[*input].into_vector_value(),
                    )?,
                    Floor => build_vec_unary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.floor.*",
                        &format!("floor_call_{ni}"),
                        regs[*input].into_vector_value(),
                    )?,
                    Not => builder
                        .build_not(regs[*input].into_vector_value(), &format!("reg_{ni}"))?
                        .as_basic_value_enum(),
                },
                Binary(op, lhs, rhs) => match op {
                    Add => builder
                        .build_float_add(
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                            &format!("reg_{ni}"),
                        )?
                        .as_basic_value_enum(),
                    Subtract => builder
                        .build_float_sub(
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                            &format!("reg_{ni}"),
                        )?
                        .as_basic_value_enum(),
                    Multiply => builder
                        .build_float_mul(
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                            &format!("reg_{ni}"),
                        )?
                        .as_basic_value_enum(),
                    Divide => builder
                        .build_float_div(
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                            &format!("reg_{ni}"),
                        )?
                        .as_basic_value_enum(),
                    Pow => build_vec_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.pow.*",
                        &format!("pow_call_{ni}"),
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                    )?,
                    Min => build_vec_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.minnum.*",
                        &format!("min_call_{ni}"),
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                    )?,
                    Max => build_vec_binary_intrinsic(
                        builder,
                        &compiler.module,
                        "llvm.maxnum.*",
                        &format!("max_call_{ni}"),
                        regs[*lhs].into_vector_value(),
                        regs[*rhs].into_vector_value(),
                    )?,
                    Remainder => builder
                        .build_float_rem(
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                            &format!("reg_{ni}"),
                        )?
                        .as_basic_value_enum(),
                    Less => builder
                        .build_float_compare(
                            FloatPredicate::ULT,
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                            &format!("reg_{ni}"),
                        )?
                        .as_basic_value_enum(),
                    LessOrEqual => builder
                        .build_float_compare(
                            FloatPredicate::ULE,
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                            &format!("reg_{ni}"),
                        )?
                        .as_basic_value_enum(),
                    Equal => builder
                        .build_float_compare(
                            FloatPredicate::UEQ,
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                            &format!("reg_{ni}"),
                        )?
                        .as_basic_value_enum(),
                    NotEqual => builder
                        .build_float_compare(
                            FloatPredicate::UNE,
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                            &format!("reg_{ni}"),
                        )?
                        .as_basic_value_enum(),
                    Greater => builder
                        .build_float_compare(
                            FloatPredicate::UGT,
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                            &format!("reg_{ni}"),
                        )?
                        .as_basic_value_enum(),
                    GreaterOrEqual => builder
                        .build_float_compare(
                            FloatPredicate::UGE,
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                            &format!("reg_{ni}"),
                        )?
                        .as_basic_value_enum(),
                    And => builder
                        .build_and(
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                            &format!("reg_{ni}"),
                        )?
                        .as_basic_value_enum(),
                    Or => builder
                        .build_or(
                            regs[*lhs].into_vector_value(),
                            regs[*rhs].into_vector_value(),
                            &format!("reg_{ni}"),
                        )?
                        .as_basic_value_enum(),
                },
                Ternary(op, a, b, c) => match op {
                    Choose => builder.build_select(
                        regs[*a].into_vector_value(),
                        regs[*b].into_vector_value(),
                        regs[*c].into_vector_value(),
                        &format!("reg_{ni}"),
                    )?,
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
            let offset = builder.build_int_add(
                builder.build_int_mul(
                    index,
                    i64_type.const_int(num_roots as u64, false),
                    "offset_mul",
                )?,
                i64_type.const_int(i as u64, false),
                "offset_add",
            )?;
            // SAFETY: GEP can segfault if the index is out of bounds. The
            // offset calculation looks pretty solid, and is thoroughly tested.
            let dst = unsafe {
                builder.build_gep(fvec_type, outputs, &[offset], &format!("output_{i}"))?
            };
            builder.build_store(dst, *reg)?;
        }
        // Check to see if the loop should go on.
        let next = builder.build_int_add(index, i64_type.const_int(1, false), "increment")?;
        phi.add_incoming(&[(&next, loop_block)]);
        let cmp = builder.build_int_compare(IntPredicate::ULT, next, eval_len, "loop-check")?;
        builder.build_conditional_branch(cmp, loop_block, end_block)?;
        // End loop and return.
        builder.position_at_end(end_block);
        builder.build_return(None)?;
        compiler.run_passes();
        let engine = compiler
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .map_err(|_| Error::CannotCreateJitModule)?;
        // SAFETY: The signature is correct, and well tested. The function
        // pointer should never be invalidated, because we allocated a dedicated
        // execution engine, with it's own block of executable memory, that will
        // live as long as the function wrapper lives.
        let func = unsafe { engine.get_function(&func_name)? };
        Ok(JitSimdFn::<T> {
            func,
            phantom: PhantomData,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{assert_float_eq, deftree, eval::ValueEvaluator, test_util::Sampler};

    fn check_jit_eval(
        tree: &Tree,
        vardata: &[(char, f64, f64)],
        samples_per_var: usize,
        eps64: f64,
        eps32: f32,
    ) {
        let context = JitContext::default();
        let params: String = vardata.iter().map(|(c, ..)| *c).collect();
        let eval64 = tree.jit_compile_array::<f64>(&context, &params).unwrap();
        let eval32 = tree.jit_compile_array::<f32>(&context, &params).unwrap();
        let mut buf64 = JitSimdBuffers::<f64>::new(tree);
        let mut buf32 = JitSimdBuffers::<f32>::new(tree);
        let mut eval = ValueEvaluator::new(tree);
        let mut sampler = Sampler::new(vardata, samples_per_var, 42);
        let mut expected = Vec::with_capacity(
            tree.num_roots() * usize::pow(samples_per_var, vardata.len() as u32),
        );
        let mut sample32 = Vec::new(); // Temporary storage.
        while let Some(sample) = sampler.next() {
            for (label, value) in vardata.iter().map(|(label, ..)| *label).zip(sample.iter()) {
                eval.set_value(label, (*value).into());
            }
            expected.extend(
                eval.run()
                    .unwrap()
                    .iter()
                    .map(|value| value.scalar().unwrap()),
            );
            buf64.pack(sample).unwrap();
            {
                // f32
                sample32.clear();
                sample32.extend(sample.iter().map(|s| *s as f32));
                buf32.pack(&sample32).unwrap();
            }
        }
        {
            // Run and check f64.
            eval64.run(&mut buf64);
            let actual: Vec<_> = buf64.unpack_outputs().collect();
            assert_eq!(actual.len(), expected.len());
            for (l, r) in actual.iter().zip(expected.iter()) {
                assert_float_eq!(l, r, eps64);
            }
        }
        {
            // Run and check f32.
            eval32.run(&mut buf32);
            let actual: Vec<_> = buf32.unpack_outputs().collect();
            assert_eq!(actual.len(), expected.len());
            for (l, r) in actual.iter().zip(expected.iter()) {
                assert_float_eq!(*l as f64, r, eps32 as f64);
            }
        }
    }

    #[test]
    fn t_mul() {
        check_jit_eval(
            &deftree!(* 'x 'y).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.)],
            20,
            0.,
            1e-5,
        );
    }

    #[test]
    fn t_prod_sum() {
        check_jit_eval(
            &deftree!(concat (+ 'x 'y) (* 'x 'y)).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.)],
            100,
            0.,
            1e-4,
        );
    }

    #[test]
    fn t_sub_div() {
        check_jit_eval(
            &deftree!(concat (- 'x 'y) (/ 'x 'y)).unwrap(),
            &[('x', -10., 10.), ('y', -10., 10.)],
            20,
            0.,
            1e-4,
        );
    }

    #[test]
    fn t_pow() {
        check_jit_eval(
            &deftree!(pow 'x 2).unwrap(),
            &[('x', -10., -10.)],
            100,
            0.,
            1e-6,
        );
    }

    #[test]
    fn t_sqrt() {
        check_jit_eval(
            &deftree!(sqrt 'x).unwrap(),
            &[('x', 0.01, 10.)],
            100,
            0.,
            1e-6,
        );
    }

    #[test]
    fn t_circle() {
        check_jit_eval(
            &deftree!(- (sqrt (+ (pow 'x 2) (pow 'y 2))) 3).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.)],
            20,
            0.,
            1e-6,
        );
    }

    #[test]
    fn t_sum_3() {
        check_jit_eval(
            &deftree!(+ (+ 'x 3) (+ 'y 'z)).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            5,
            0.,
            1e-5,
        );
    }

    #[test]
    fn t_sphere() {
        check_jit_eval(
            &deftree!(- (sqrt (+ (pow 'x 2) (+ (pow 'y 2) (pow 'z 2)))) 3).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            0.,
            1e-6,
        );
    }

    #[test]
    fn t_negate() {
        check_jit_eval(
            &deftree!(* (- 'x) (+ 'y 'z)).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            0.,
            1e-5,
        );
    }

    #[test]
    fn t_abs() {
        check_jit_eval(
            &deftree!(* (abs 'x) (+ (abs 'y) (abs 'z))).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            0.,
            1e-5,
        );
    }

    #[test]
    fn t_trigonometry() {
        check_jit_eval(
            &deftree!(/ (+ (sin 'x) (cos 'y)) (+ 0.27 (pow (tan 'z) 2))).unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.), ('z', -5., 5.)],
            10,
            1e-14,
            1e-5,
        );
    }

    #[test]
    fn t_log_exp() {
        check_jit_eval(
            &deftree!(/ (+ 1 (log 'x)) (+ 1 (exp 'y))).unwrap(),
            &[('x', 0.1, 5.), ('y', 0.1, 5.)],
            10,
            0.,
            1e-6,
        );
    }

    #[test]
    fn t_min_max() {
        check_jit_eval(
            &deftree!(
                (max (min
                      (- (sqrt (+ (+ (pow (- 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 2.75)
                      (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (- 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 4.))
                 (- (sqrt (+ (+ (pow (+ 'x 2.) 2.) (pow (+ 'y 3.) 2.)) (pow (- 'z 4.) 2.))) 5.25))
            )
            .unwrap(),
            &[('x', -10., 10.), ('y', -9., 10.), ('z', -11., 12.)],
            20,
            0.,
            1e-5,
        );
    }

    #[test]
    fn t_floor() {
        check_jit_eval(
            &deftree!(floor (+ (pow 'x 2) (sin 'x))).unwrap(),
            &[('x', -5., 5.)],
            100,
            0.,
            1e-6,
        );
    }

    #[test]
    fn t_remainder() {
        check_jit_eval(
            &deftree!(rem (pow 'x 2) (+ 2 (sin 'x))).unwrap(),
            &[('x', 1., 5.)],
            100,
            1e-15,
            1e-5,
        );
    }

    #[test]
    fn t_choose() {
        check_jit_eval(
            &deftree!(if (> 'x 0) 'x (- 'x)).unwrap(),
            &[('x', -10., 10.)],
            100,
            0.,
            1e-6,
        );
        check_jit_eval(
            &deftree!(if (< 'x 0) (- 'x) 'x).unwrap(),
            &[('x', -10., 10.)],
            100,
            0.,
            1e-6,
        );
    }

    #[test]
    fn t_or_and() {
        check_jit_eval(
            &deftree!(if (and (> 'x 0) (< 'x 1)) (* 2 'x) 1).unwrap(),
            &[('x', -3., 3.)],
            100,
            0.,
            1e-6,
        );
    }

    #[test]
    fn t_not() {
        check_jit_eval(
            &deftree!(if (not (> 'x 0)) (- (pow 'x 3) (pow 'y 3)) (+ (pow 'x 2) (pow 'y 2)))
                .unwrap(),
            &[('x', -5., 5.), ('y', -5., 5.)],
            100,
            1e-14,
            1e-4,
        );
    }
}

#[cfg(test)]
mod sphere_test {
    use super::*;
    use crate::{
        assert_float_eq,
        dedup::Deduplicater,
        deftree,
        eval::ValueEvaluator,
        prune::Pruner,
        tree::{Tree, min},
    };
    use rand::{SeedableRng, rngs::StdRng};

    fn sample_range(range: (f64, f64), rng: &mut StdRng) -> f64 {
        use rand::Rng;
        range.0 + rng.random::<f64>() * (range.1 - range.0)
    }
    const RADIUS_RANGE: (f64, f64) = (0.2, 2.);
    const X_RANGE: (f64, f64) = (0., 100.);
    const Y_RANGE: (f64, f64) = (0., 100.);
    const Z_RANGE: (f64, f64) = (0., 100.);
    const N_SPHERES: usize = 500;
    const N_QUERIES: usize = 500;

    fn sphere_union() -> Tree {
        let mut rng = StdRng::seed_from_u64(42);
        let mut make_sphere = || -> Result<Tree, Error> {
            deftree!(- (sqrt (+ (+
                                 (pow (- 'x (const sample_range(X_RANGE, &mut rng))) 2)
                                 (pow (- 'y (const sample_range(Y_RANGE, &mut rng))) 2))
                              (pow (- 'z (const sample_range(Z_RANGE, &mut rng))) 2)))
                     (const sample_range(RADIUS_RANGE, &mut rng)))
        };
        let mut tree = make_sphere();
        for _ in 1..N_SPHERES {
            tree = min(tree, make_sphere());
        }
        let tree = tree.unwrap();
        assert_eq!(tree.dims(), (1, 1));
        tree
    }

    #[test]
    fn t_compare_jit_simd() {
        let mut rng = StdRng::seed_from_u64(234);
        let queries: Vec<[f64; 3]> = (0..N_QUERIES)
            .map(|_| {
                [
                    sample_range(X_RANGE, &mut rng),
                    sample_range(Y_RANGE, &mut rng),
                    sample_range(Z_RANGE, &mut rng),
                ]
            })
            .collect();
        let tree = {
            let mut dedup = Deduplicater::new();
            let mut pruner = Pruner::new();
            sphere_union()
                .fold()
                .unwrap()
                .deduplicate(&mut dedup)
                .unwrap()
                .prune(&mut pruner)
                .unwrap()
        };
        let mut val_eval: Vec<f64> = Vec::with_capacity(N_QUERIES);
        {
            let mut eval = ValueEvaluator::new(&tree);
            val_eval.extend(queries.iter().map(|coords| {
                eval.set_value('x', coords[0].into());
                eval.set_value('y', coords[1].into());
                eval.set_value('z', coords[2].into());
                let results = eval.run().unwrap();
                results[0].scalar().unwrap()
            }));
        }
        let val_jit: Vec<_> = {
            let context = JitContext::default();
            let eval = tree.jit_compile_array(&context, "xyz").unwrap();
            let mut buf = JitSimdBuffers::new(&tree);
            for q in queries {
                buf.pack(&q).unwrap();
            }
            eval.run(&mut buf);
            buf.unpack_outputs().collect()
        };
        assert_eq!(val_eval.len(), val_jit.len());
        for (l, r) in val_eval.iter().zip(val_jit.iter()) {
            assert_float_eq!(l, r, 1e-15);
        }
    }
}

#[cfg(test)]
mod simd_ops_test {
    use super::*;

    #[test]
    fn t_wfloat_f32_mul() {
        let a = Wide {
            valsf32: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        };
        let b = Wide {
            valsf32: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        let result = <Wide as SimdVec<f32>>::mul(a, b);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsf32[0], 2.0);
            assert_eq!(result.valsf32[1], 6.0);
            assert_eq!(result.valsf32[2], 12.0);
            assert_eq!(result.valsf32[3], 20.0);
            assert_eq!(result.valsf32[4], 30.0);
            assert_eq!(result.valsf32[5], 42.0);
            assert_eq!(result.valsf32[6], 56.0);
            assert_eq!(result.valsf32[7], 72.0);
        }
    }

    #[test]
    fn t_wfloat_f32_add() {
        let a = Wide {
            valsf32: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        };
        let b = Wide {
            valsf32: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        };
        let result = <Wide as SimdVec<f32>>::add(a, b);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsf32[0], 11.0);
            assert_eq!(result.valsf32[1], 22.0);
            assert_eq!(result.valsf32[2], 33.0);
            assert_eq!(result.valsf32[3], 44.0);
            assert_eq!(result.valsf32[4], 55.0);
            assert_eq!(result.valsf32[5], 66.0);
            assert_eq!(result.valsf32[6], 77.0);
            assert_eq!(result.valsf32[7], 88.0);
        }
    }

    #[test]
    fn t_wfloat_f32_mul_add() {
        let a = Wide {
            valsf32: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        };
        let b = Wide {
            valsf32: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        let c = Wide {
            valsf32: [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        };
        let result = <Wide as SimdVec<f32>>::mul_add(a, b, c);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsf32[0], 12.0); // 1*2 + 10
            assert_eq!(result.valsf32[1], 16.0); // 2*3 + 10
            assert_eq!(result.valsf32[2], 22.0); // 3*4 + 10
            assert_eq!(result.valsf32[3], 30.0); // 4*5 + 10
            assert_eq!(result.valsf32[4], 40.0); // 5*6 + 10
            assert_eq!(result.valsf32[5], 52.0); // 6*7 + 10
            assert_eq!(result.valsf32[6], 66.0); // 7*8 + 10
            assert_eq!(result.valsf32[7], 82.0); // 8*9 + 10
        }
    }

    #[test]
    fn t_wfloat_f64_mul() {
        let a = Wide {
            valsf64: [1.0, 2.0, 3.0, 4.0],
        };
        let b = Wide {
            valsf64: [5.0, 6.0, 7.0, 8.0],
        };
        let result = <Wide as SimdVec<f64>>::mul(a, b);
        unsafe {
            assert_eq!(result.valsf64[0], 5.0);
            assert_eq!(result.valsf64[1], 12.0);
            assert_eq!(result.valsf64[2], 21.0);
            assert_eq!(result.valsf64[3], 32.0);
        }
    }

    #[test]
    fn t_wfloat_f64_add() {
        let a = Wide {
            valsf64: [1.5, 2.5, 3.5, 4.5],
        };
        let b = Wide {
            valsf64: [10.5, 20.5, 30.5, 40.5],
        };
        let result = <Wide as SimdVec<f64>>::add(a, b);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsf64[0], 12.0);
            assert_eq!(result.valsf64[1], 23.0);
            assert_eq!(result.valsf64[2], 34.0);
            assert_eq!(result.valsf64[3], 45.0);
        }
    }

    #[test]
    fn t_wfloat_f64_mul_add() {
        let a = Wide {
            valsf64: [2.0, 3.0, 4.0, 5.0],
        };
        let b = Wide {
            valsf64: [10.0, 10.0, 10.0, 10.0],
        };
        let c = Wide {
            valsf64: [1.0, 2.0, 3.0, 4.0],
        };
        let result = <Wide as SimdVec<f64>>::mul_add(a, b, c);
        unsafe {
            assert_eq!(result.valsf64[0], 21.0); // 2*10 + 1
            assert_eq!(result.valsf64[1], 32.0); // 3*10 + 2
            assert_eq!(result.valsf64[2], 43.0); // 4*10 + 3
            assert_eq!(result.valsf64[3], 54.0); // 5*10 + 4
        }
    }

    #[test]
    fn t_wfloat_f32_neg() {
        let a = Wide {
            valsf32: [1.0, -2.0, 3.5, -4.5, 0.0, 10.5, -7.25, 8.75],
        };
        let result = <Wide as SimdVec<f32>>::neg(a);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsf32[0], -1.0);
            assert_eq!(result.valsf32[1], 2.0);
            assert_eq!(result.valsf32[2], -3.5);
            assert_eq!(result.valsf32[3], 4.5);
            assert_eq!(result.valsf32[4], -0.0);
            assert_eq!(result.valsf32[5], -10.5);
            assert_eq!(result.valsf32[6], 7.25);
            assert_eq!(result.valsf32[7], -8.75);
        }
    }

    #[test]
    fn t_wfloat_f64_neg() {
        let a = Wide {
            valsf64: [1.5, -2.75, 0.0, -10.25],
        };
        let result = <Wide as SimdVec<f64>>::neg(a);
        unsafe {
            assert_eq!(result.valsf64[0], -1.5);
            assert_eq!(result.valsf64[1], 2.75);
            assert_eq!(result.valsf64[2], -0.0);
            assert_eq!(result.valsf64[3], 10.25);
        }
    }

    #[test]
    fn t_wfloat_f32_lt() {
        let a = Wide {
            valsf32: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        };
        let b = Wide {
            valsf32: [2.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0],
        };
        let result = <Wide as SimdVec<f32>>::lt(a, b);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsu32[0], 0xFFFFFFFF);
            assert_eq!(result.valsu32[1], 0x00000000);
            assert_eq!(result.valsu32[2], 0x00000000);
            assert_eq!(result.valsu32[3], 0x00000000);
            assert_eq!(result.valsu32[4], 0xFFFFFFFF);
            assert_eq!(result.valsu32[5], 0x00000000);
            assert_eq!(result.valsu32[6], 0x00000000);
            assert_eq!(result.valsu32[7], 0x00000000);
        }
    }

    #[test]
    fn t_wfloat_f32_eq() {
        let a = Wide {
            valsf32: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        };
        let b = Wide {
            valsf32: [2.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0],
        };
        let result = <Wide as SimdVec<f32>>::eq(a, b);
        unsafe {
            assert_eq!(result.valsu32[0], 0x00000000);
            assert_eq!(result.valsu32[1], 0xFFFFFFFF);
            assert_eq!(result.valsu32[2], 0x00000000);
            assert_eq!(result.valsu32[3], 0x00000000);
            assert_eq!(result.valsu32[4], 0x00000000);
            assert_eq!(result.valsu32[5], 0xFFFFFFFF);
            assert_eq!(result.valsu32[6], 0x00000000);
            assert_eq!(result.valsu32[7], 0x00000000);
        }
    }

    #[test]
    fn t_wfloat_f32_gt() {
        let a = Wide {
            valsf32: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        };
        let b = Wide {
            valsf32: [2.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0],
        };
        let result = <Wide as SimdVec<f32>>::gt(a, b);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsu32[0], 0x00000000);
            assert_eq!(result.valsu32[1], 0x00000000);
            assert_eq!(result.valsu32[2], 0xFFFFFFFF);
            assert_eq!(result.valsu32[3], 0xFFFFFFFF);
            assert_eq!(result.valsu32[4], 0x00000000);
            assert_eq!(result.valsu32[5], 0x00000000);
            assert_eq!(result.valsu32[6], 0xFFFFFFFF);
            assert_eq!(result.valsu32[7], 0xFFFFFFFF);
        }
    }

    #[test]
    fn t_wfloat_f64_lt() {
        let a = Wide {
            valsf64: [1.0, 2.0, 3.0, 4.0],
        };
        let b = Wide {
            valsf64: [2.0, 2.0, 2.0, 2.0],
        };
        let result = <Wide as SimdVec<f64>>::lt(a, b);
        unsafe {
            assert_eq!(result.valsu64[0], 0xFFFFFFFFFFFFFFFF);
            assert_eq!(result.valsu64[1], 0x0000000000000000);
            assert_eq!(result.valsu64[2], 0x0000000000000000);
            assert_eq!(result.valsu64[3], 0x0000000000000000);
        }
    }

    #[test]
    fn t_wfloat_f64_eq() {
        let a = Wide {
            valsf64: [1.0, 2.0, 3.0, 4.0],
        };
        let b = Wide {
            valsf64: [2.0, 2.0, 2.0, 2.0],
        };
        let result = <Wide as SimdVec<f64>>::eq(a, b);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsu64[0], 0x0000000000000000);
            assert_eq!(result.valsu64[1], 0xFFFFFFFFFFFFFFFF);
            assert_eq!(result.valsu64[2], 0x0000000000000000);
            assert_eq!(result.valsu64[3], 0x0000000000000000);
        }
    }

    #[test]
    fn t_wfloat_f64_gt() {
        let a = Wide {
            valsf64: [1.0, 2.0, 3.0, 4.0],
        };
        let b = Wide {
            valsf64: [2.0, 2.0, 2.0, 2.0],
        };
        let result = <Wide as SimdVec<f64>>::gt(a, b);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsu64[0], 0x0000000000000000);
            assert_eq!(result.valsu64[1], 0x0000000000000000);
            assert_eq!(result.valsu64[2], 0xFFFFFFFFFFFFFFFF);
            assert_eq!(result.valsu64[3], 0xFFFFFFFFFFFFFFFF);
        }
    }

    #[test]
    fn t_wfloat_f32_and() {
        let a = Wide {
            valsu32: [
                0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                0x00000000,
            ],
        };
        let b = Wide {
            valsu32: [
                0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000,
                0x00000000,
            ],
        };
        let result = <Wide as SimdVec<f32>>::and(a, b);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsu32[0], 0xFFFFFFFF);
            assert_eq!(result.valsu32[1], 0x00000000);
            assert_eq!(result.valsu32[2], 0x00000000);
            assert_eq!(result.valsu32[3], 0x00000000);
            assert_eq!(result.valsu32[4], 0xFFFFFFFF);
            assert_eq!(result.valsu32[5], 0x00000000);
            assert_eq!(result.valsu32[6], 0x00000000);
            assert_eq!(result.valsu32[7], 0x00000000);
        }
    }

    #[test]
    fn t_wfloat_f32_or() {
        let a = Wide {
            valsu32: [
                0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                0x00000000,
            ],
        };
        let b = Wide {
            valsu32: [
                0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000,
                0x00000000,
            ],
        };
        let result = <Wide as SimdVec<f32>>::or(a, b);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsu32[0], 0xFFFFFFFF);
            assert_eq!(result.valsu32[1], 0xFFFFFFFF);
            assert_eq!(result.valsu32[2], 0xFFFFFFFF);
            assert_eq!(result.valsu32[3], 0x00000000);
            assert_eq!(result.valsu32[4], 0xFFFFFFFF);
            assert_eq!(result.valsu32[5], 0xFFFFFFFF);
            assert_eq!(result.valsu32[6], 0xFFFFFFFF);
            assert_eq!(result.valsu32[7], 0x00000000);
        }
    }

    #[test]
    fn t_wfloat_f64_and() {
        let a = Wide {
            valsu64: [
                0xFFFFFFFFFFFFFFFF,
                0x0000000000000000,
                0xFFFFFFFFFFFFFFFF,
                0x0000000000000000,
            ],
        };
        let b = Wide {
            valsu64: [
                0xFFFFFFFFFFFFFFFF,
                0xFFFFFFFFFFFFFFFF,
                0x0000000000000000,
                0x0000000000000000,
            ],
        };
        let result = <Wide as SimdVec<f64>>::and(a, b);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsu64[0], 0xFFFFFFFFFFFFFFFF);
            assert_eq!(result.valsu64[1], 0x0000000000000000);
            assert_eq!(result.valsu64[2], 0x0000000000000000);
            assert_eq!(result.valsu64[3], 0x0000000000000000);
        }
    }

    #[test]
    fn t_wfloat_f64_or() {
        let a = Wide {
            valsu64: [
                0xFFFFFFFFFFFFFFFF,
                0x0000000000000000,
                0xFFFFFFFFFFFFFFFF,
                0x0000000000000000,
            ],
        };
        let b = Wide {
            valsu64: [
                0xFFFFFFFFFFFFFFFF,
                0xFFFFFFFFFFFFFFFF,
                0x0000000000000000,
                0x0000000000000000,
            ],
        };
        let result = <Wide as SimdVec<f64>>::or(a, b);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsu64[0], 0xFFFFFFFFFFFFFFFF);
            assert_eq!(result.valsu64[1], 0xFFFFFFFFFFFFFFFF);
            assert_eq!(result.valsu64[2], 0xFFFFFFFFFFFFFFFF);
            assert_eq!(result.valsu64[3], 0x0000000000000000);
        }
    }

    #[test]
    fn t_wfloat_f32_check_bool() {
        let mask = Wide {
            valsu32: [
                0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0x12345678, 0x00000000, 0xABCDEF00,
                0x00000000,
            ],
        };
        assert!(<Wide as SimdVec<f32>>::check_bool(mask, 0));
        assert!(!<Wide as SimdVec<f32>>::check_bool(mask, 1));
        assert!(<Wide as SimdVec<f32>>::check_bool(mask, 2));
        assert!(!<Wide as SimdVec<f32>>::check_bool(mask, 3));
        assert!(<Wide as SimdVec<f32>>::check_bool(mask, 4));
        assert!(!<Wide as SimdVec<f32>>::check_bool(mask, 5));
        assert!(<Wide as SimdVec<f32>>::check_bool(mask, 6));
        assert!(!<Wide as SimdVec<f32>>::check_bool(mask, 7));
    }

    #[test]
    fn t_wfloat_f32_check_bool_unchecked() {
        let mask = Wide {
            valsu32: [
                0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0x12345678, 0x00000000, 0xABCDEF00,
                0x00000000,
            ],
        };
        // SAFETY: Union access to assert values.
        unsafe {
            assert!(<Wide as SimdVec<f32>>::check_bool_unchecked(mask, 0));
            assert!(!<Wide as SimdVec<f32>>::check_bool_unchecked(mask, 1));
            assert!(<Wide as SimdVec<f32>>::check_bool_unchecked(mask, 2));
            assert!(!<Wide as SimdVec<f32>>::check_bool_unchecked(mask, 3));
            assert!(<Wide as SimdVec<f32>>::check_bool_unchecked(mask, 4));
            assert!(!<Wide as SimdVec<f32>>::check_bool_unchecked(mask, 5));
            assert!(<Wide as SimdVec<f32>>::check_bool_unchecked(mask, 6));
            assert!(!<Wide as SimdVec<f32>>::check_bool_unchecked(mask, 7));
        }
    }

    #[test]
    #[should_panic]
    fn t_wfloat_f32_check_bool_bounds() {
        let mask = Wide {
            valsu32: [
                0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0x12345678, 0x00000000, 0xABCDEF00,
                0x00000000,
            ],
        };
        <Wide as SimdVec<f32>>::check_bool(mask, 8);
    }

    #[test]
    fn t_wfloat_f64_check_bool() {
        let mask = Wide {
            valsu64: [
                0xFFFFFFFFFFFFFFFF,
                0x0000000000000000,
                0x123456789ABCDEF0,
                0x0000000000000000,
            ],
        };
        assert!(<Wide as SimdVec<f64>>::check_bool(mask, 0));
        assert!(!<Wide as SimdVec<f64>>::check_bool(mask, 1));
        assert!(<Wide as SimdVec<f64>>::check_bool(mask, 2));
        assert!(!<Wide as SimdVec<f64>>::check_bool(mask, 3));
    }

    #[test]
    fn t_wfloat_f64_check_bool_unchecked() {
        let mask = Wide {
            valsu64: [
                0xFFFFFFFFFFFFFFFF,
                0x0000000000000000,
                0x123456789ABCDEF0,
                0x0000000000000000,
            ],
        };
        // SAFETY: Union access to assert values.
        unsafe {
            assert!(<Wide as SimdVec<f64>>::check_bool_unchecked(mask, 0));
            assert!(!<Wide as SimdVec<f64>>::check_bool_unchecked(mask, 1));
            assert!(<Wide as SimdVec<f64>>::check_bool_unchecked(mask, 2));
            assert!(!<Wide as SimdVec<f64>>::check_bool_unchecked(mask, 3));
        }
    }

    #[test]
    #[should_panic]
    fn t_wfloat_f64_check_bool_bounds() {
        let mask = Wide {
            valsu64: [
                0xFFFFFFFFFFFFFFFF,
                0x0000000000000000,
                0x123456789ABCDEF0,
                0x0000000000000000,
            ],
        };
        <Wide as SimdVec<f64>>::check_bool(mask, 4);
    }

    #[test]
    fn t_wfloat_f32_abs() {
        let a = Wide {
            valsf32: [-1.0, 2.0, -3.5, 4.25, -5.75, 6.125, -7.875, 8.0],
        };
        let result = <Wide as SimdVec<f32>>::abs(a);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsf32[0], 1.0);
            assert_eq!(result.valsf32[1], 2.0);
            assert_eq!(result.valsf32[2], 3.5);
            assert_eq!(result.valsf32[3], 4.25);
            assert_eq!(result.valsf32[4], 5.75);
            assert_eq!(result.valsf32[5], 6.125);
            assert_eq!(result.valsf32[6], 7.875);
            assert_eq!(result.valsf32[7], 8.0);
        }
    }

    #[test]
    fn t_wfloat_f64_abs() {
        let a = Wide {
            valsf64: [-1.5, 2.25, -3.125, 4.0625],
        };
        let result = <Wide as SimdVec<f64>>::abs(a);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsf64[0], 1.5);
            assert_eq!(result.valsf64[1], 2.25);
            assert_eq!(result.valsf64[2], 3.125);
            assert_eq!(result.valsf64[3], 4.0625);
        }
    }

    #[test]
    fn t_wfloat_f32_recip_sqrt() {
        let a = Wide {
            valsf32: [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0],
        };
        let result = <Wide as SimdVec<f32>>::recip_sqrt(a);
        // SAFETY: Union access to assert values.
        unsafe {
            // Check that recip_sqrt gives approximately 1/sqrt(x)
            // Using loose tolerance since rsqrt is an approximation
            assert!((result.valsf32[0] - 1.0).abs() < 0.01); // 1/sqrt(1) = 1
            assert!((result.valsf32[1] - 0.5).abs() < 0.01); // 1/sqrt(4) = 0.5
            assert!((result.valsf32[2] - (1.0 / 3.0)).abs() < 0.01); // 1/sqrt(9) = 1/3
            assert!((result.valsf32[3] - 0.25).abs() < 0.01); // 1/sqrt(16) = 0.25
            assert!((result.valsf32[4] - 0.2).abs() < 0.01); // 1/sqrt(25) = 0.2
            assert!((result.valsf32[5] - (1.0 / 6.0)).abs() < 0.01); // 1/sqrt(36) = 1/6
            assert!((result.valsf32[6] - (1.0 / 7.0)).abs() < 0.01); // 1/sqrt(49) = 1/7
            assert!((result.valsf32[7] - 0.125).abs() < 0.01); // 1/sqrt(64) = 0.125
        }
    }

    #[test]
    fn t_wfloat_f64_recip_sqrt() {
        let a = Wide {
            valsf64: [1.0, 4.0, 9.0, 16.0],
        };
        let result = <Wide as SimdVec<f64>>::recip_sqrt(a);
        // SAFETY: Union access to assert values.
        unsafe {
            // Check that recip_sqrt gives approximately 1/sqrt(x)
            // Using loose tolerance since rsqrt is an approximation
            assert!((result.valsf64[0] - 1.0).abs() < 0.01); // 1/sqrt(1) = 1
            assert!((result.valsf64[1] - 0.5).abs() < 0.01); // 1/sqrt(4) = 0.5
            assert!((result.valsf64[2] - (1.0 / 3.0)).abs() < 0.01); // 1/sqrt(9) = 1/3
            assert!((result.valsf64[3] - 0.25).abs() < 0.01); // 1/sqrt(16) = 0.25
        }
    }

    #[test]
    fn t_wfloat_f32_sub() {
        let a = Wide {
            valsf32: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        };
        let b = Wide {
            valsf32: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        };
        let result = <Wide as SimdVec<f32>>::sub(a, b);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsf32[0], 9.0);
            assert_eq!(result.valsf32[1], 18.0);
            assert_eq!(result.valsf32[2], 27.0);
            assert_eq!(result.valsf32[3], 36.0);
            assert_eq!(result.valsf32[4], 45.0);
            assert_eq!(result.valsf32[5], 54.0);
            assert_eq!(result.valsf32[6], 63.0);
            assert_eq!(result.valsf32[7], 72.0);
        }
    }

    #[test]
    fn t_wfloat_f64_sub() {
        let a = Wide {
            valsf64: [10.5, 20.25, 30.125, 40.0625],
        };
        let b = Wide {
            valsf64: [1.5, 2.25, 3.125, 4.0625],
        };
        let result = <Wide as SimdVec<f64>>::sub(a, b);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsf64[0], 9.0);
            assert_eq!(result.valsf64[1], 18.0);
            assert_eq!(result.valsf64[2], 27.0);
            assert_eq!(result.valsf64[3], 36.0);
        }
    }

    #[test]
    fn t_wfloat_f32_div() {
        let a = Wide {
            valsf32: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        };
        let b = Wide {
            valsf32: [2.0, 4.0, 5.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        };
        let result = <Wide as SimdVec<f32>>::div(a, b);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsf32[0], 5.0); // 10.0 / 2.0
            assert_eq!(result.valsf32[1], 5.0); // 20.0 / 4.0
            assert_eq!(result.valsf32[2], 6.0); // 30.0 / 5.0
            assert_eq!(result.valsf32[3], 5.0); // 40.0 / 8.0
            assert_eq!(result.valsf32[4], 5.0); // 50.0 / 10.0
            assert_eq!(result.valsf32[5], 5.0); // 60.0 / 12.0
            assert_eq!(result.valsf32[6], 5.0); // 70.0 / 14.0
            assert_eq!(result.valsf32[7], 5.0); // 80.0 / 16.0
        }
    }

    #[test]
    fn t_wfloat_f64_div() {
        let a = Wide {
            valsf64: [12.0, 24.0, 36.0, 48.0],
        };
        let b = Wide {
            valsf64: [3.0, 6.0, 9.0, 12.0],
        };
        let result = <Wide as SimdVec<f64>>::div(a, b);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsf64[0], 4.0); // 12.0 / 3.0
            assert_eq!(result.valsf64[1], 4.0); // 24.0 / 6.0
            assert_eq!(result.valsf64[2], 4.0); // 36.0 / 9.0
            assert_eq!(result.valsf64[3], 4.0); // 48.0 / 12.0
        }
    }

    #[test]
    fn t_wfloat_f32_max() {
        let a = Wide {
            valsf32: [1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 6.0],
        };
        let b = Wide {
            valsf32: [4.0, 2.0, 7.0, 1.0, 6.0, 3.0, 8.0, 5.0],
        };
        let result = <Wide as SimdVec<f32>>::max(a, b);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsf32[0], 4.0); // max(1.0, 4.0) = 4.0
            assert_eq!(result.valsf32[1], 5.0); // max(5.0, 2.0) = 5.0
            assert_eq!(result.valsf32[2], 7.0); // max(3.0, 7.0) = 7.0
            assert_eq!(result.valsf32[3], 8.0); // max(8.0, 1.0) = 8.0
            assert_eq!(result.valsf32[4], 6.0); // max(2.0, 6.0) = 6.0
            assert_eq!(result.valsf32[5], 9.0); // max(9.0, 3.0) = 9.0
            assert_eq!(result.valsf32[6], 8.0); // max(4.0, 8.0) = 8.0
            assert_eq!(result.valsf32[7], 6.0); // max(6.0, 5.0) = 6.0
        }
    }

    #[test]
    fn t_wfloat_f64_max() {
        let a = Wide {
            valsf64: [1.5, 8.25, 3.125, 6.0],
        };
        let b = Wide {
            valsf64: [2.75, 4.5, 9.875, 1.25],
        };
        let result = <Wide as SimdVec<f64>>::max(a, b);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsf64[0], 2.75); // max(1.5, 2.75) = 2.75
            assert_eq!(result.valsf64[1], 8.25); // max(8.25, 4.5) = 8.25
            assert_eq!(result.valsf64[2], 9.875); // max(3.125, 9.875) = 9.875
            assert_eq!(result.valsf64[3], 6.0); // max(6.0, 1.25) = 6.0
        }
    }

    #[test]
    fn t_wfloat_f32_min() {
        let a = Wide {
            valsf32: [1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 6.0],
        };
        let b = Wide {
            valsf32: [4.0, 2.0, 7.0, 1.0, 6.0, 3.0, 8.0, 5.0],
        };
        let result = <Wide as SimdVec<f32>>::min(a, b);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsf32[0], 1.0); // min(1.0, 4.0) = 4.0
            assert_eq!(result.valsf32[1], 2.0); // min(5.0, 2.0) = 5.0
            assert_eq!(result.valsf32[2], 3.0); // min(3.0, 7.0) = 7.0
            assert_eq!(result.valsf32[3], 1.0); // min(8.0, 1.0) = 8.0
            assert_eq!(result.valsf32[4], 2.0); // min(2.0, 6.0) = 6.0
            assert_eq!(result.valsf32[5], 3.0); // min(9.0, 3.0) = 9.0
            assert_eq!(result.valsf32[6], 4.0); // min(4.0, 8.0) = 8.0
            assert_eq!(result.valsf32[7], 5.0); // min(6.0, 5.0) = 6.0
        }
    }

    #[test]
    fn t_wfloat_f64_min() {
        let a = Wide {
            valsf64: [1.5, 8.25, 3.125, 6.0],
        };
        let b = Wide {
            valsf64: [2.75, 4.5, 9.875, 1.25],
        };
        let result = <Wide as SimdVec<f64>>::min(a, b);
        // SAFETY: Union access to assert values.
        unsafe {
            assert_eq!(result.valsf64[0], 1.5); // min(1.5, 2.75) = 2.75
            assert_eq!(result.valsf64[1], 4.5); // min(8.25, 4.5) = 8.25
            assert_eq!(result.valsf64[2], 3.125); // min(3.125, 9.875) = 9.875
            assert_eq!(result.valsf64[3], 1.25); // min(6.0, 1.25) = 6.0
        }
    }
}
