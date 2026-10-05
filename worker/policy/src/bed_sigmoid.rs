//! One-logit mask sigmoid. Not a model, decoder, or inference entrypoint.
//!
//! Numeric authority is `seg_postprocess._sigmoid` on NumPy 2.4.6 when its
//! float32 exp dispatch is X86_V3 (AVX2/FMA3). The private exponential is a
//! scalar transcription of that graph's rounding: ordinary quadrant mul/add/sub,
//! explicit f32 FMAs, one f32 divide, then wrapping exponent-bit scaling.
//! It does not emulate warning flags, and NaN/infinity results are only the
//! numeric intermediates of this function.

/*
Diagnostic scalar translation of NumPy v2.4.6's AVX2/FMA3 exp graph:
numpy/_core/src/umath/loops_exponent_log.dispatch.c.src (simd_exp_FLOAT,
fma_scalef_ps, simd_range_reduction) and npy_simd_data.h.
Not an adopted runtime kernel. It preserves explicit f32 rounding/FMA and
bitwise exponent scaling, but does not reproduce NumPy warning/flag reporting.

Copyright (c) 2005-2025, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

const XMAX: f32 = f32::from_bits(0x42b1_7218);
const XMIN: f32 = f32::from_bits(0xc2cf_f1b5);
const MAGIC: f32 = f32::from_bits(0x4b40_0000);
const LOG2E: f32 = f32::from_bits(0x3fb8_aa3b);
const C1: f32 = f32::from_bits(0xbf31_7200);
const C2: f32 = f32::from_bits(0xb5bf_be8e);
const P0: f32 = f32::from_bits(0x3f80_0000);
const P1: f32 = f32::from_bits(0x3f39_cbd5);
const P2: f32 = f32::from_bits(0x3e7d_4c58);
const P3: f32 = f32::from_bits(0x3d51_7d8c);
const P4: f32 = f32::from_bits(0x3bdd_7159);
const P5: f32 = f32::from_bits(0x3a05_3dd8);
const Q0: f32 = f32::from_bits(0x3f80_0000);
const Q1: f32 = f32::from_bits(0xbe8c_6857);
const Q2: f32 = f32::from_bits(0x3cb0_e832);
const CANONICAL_NAN: f32 = f32::from_bits(0x7fc0_0000);
const POSITIVE_ZERO: f32 = f32::from_bits(0);

/// Finite inputs that pass the exp bounds reduce to a shift of at most 25.
/// That keeps the denormal power-of-two correction inside a 32-bit shift.
fn scale(mut fraction: f32, mut quadrant: i32) -> f32 {
    let denormal = quadrant <= -125;
    let difference = if denormal { -125 - quadrant } else { 0 };
    if denormal {
        quadrant = -125;
    }
    let bits = fraction
        .to_bits()
        .wrapping_add(quadrant.cast_unsigned().wrapping_shl(23));
    fraction = f32::from_bits(bits);
    if denormal {
        fraction /= (1u32 << difference.cast_unsigned()) as f32;
    }
    fraction
}

fn exponential(mut value: f32) -> f32 {
    if value.is_nan() {
        return CANONICAL_NAN;
    }
    if value >= XMAX {
        return f32::from_bits(0x7f80_0000);
    }
    if value <= XMIN {
        return POSITIVE_ZERO;
    }
    let product = value * LOG2E;
    let biased = product + MAGIC;
    let quadrant = biased - MAGIC;
    value = quadrant.mul_add(C1, value);
    value = quadrant.mul_add(C2, value);
    value = quadrant.mul_add(POSITIVE_ZERO, value);
    let mut numerator = P5.mul_add(value, P4);
    numerator = numerator.mul_add(value, P3);
    numerator = numerator.mul_add(value, P2);
    numerator = numerator.mul_add(value, P1);
    numerator = numerator.mul_add(value, P0);
    let mut denominator = Q2.mul_add(value, Q1);
    denominator = denominator.mul_add(value, Q0);
    scale(numerator / denominator, quadrant as i32)
}

/// Both signed zeros take the `>= 0` branch. IEEE compares them equal;
/// the sign bit alone would not. The add and the divide each round once.
pub fn bed_mask_sigmoid(value: f32) -> f32 {
    let positive = value >= 0.0;
    let exponent = exponential(if positive { -value } else { value });
    let denominator = 1.0 + exponent;
    let numerator = if positive { 1.0 } else { exponent };
    numerator / denominator
}
