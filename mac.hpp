#ifndef MAC_HPP
#define MAC_HPP

#include "utils.hpp"
#include <ap_int.h>
#include <stdint.h>
#include <string.h>

// ============================================================
// Multiplier selection (exact + approximate modes)
// Set APPROX_MUL_MODE:
//   0 = exact multiplication (native c*d)
//   1 = approximate: BFloat16 (emulated by truncation)
//   2 = approximate: 4:2 compressors (Design 1) in CSA tree
//   3 = approximate: 4:2 compressors (Design 2) in CSA tree
//   4 = approximate: DQ4:2C1 (approx mode behavior)
//   5 = approximate: DQ4:2C2 (approx mode behavior)
//   6 = approximate: DQ4:2C3 (approx mode behavior)
//   7 = approximate: DQ4:2C4 (approx mode behavior)
//   8  = approximate: MA approximation 1 (A8)
//   9  = approximate: MA approximation 2 (A8)
//   10 = approximate: MA approximation 3 (A8)
//   11 = approximate: MA approximation 4 (A8)
//   12 = approximate: MA approximation 5 (A8)
// ============================================================

#ifndef APPROX_MUL_MODE
#define APPROX_MUL_MODE 0
#endif

// ------------------------------------------------------------
// Mode 1 helper: emulate BFloat16 by truncating float32 mantissa
// ------------------------------------------------------------
inline float float_to_bfloat16(float x) {
#pragma HLS inline
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    u &= 0xFFFF0000u;
    memcpy(&x, &u, sizeof(x));
    return x;
}

// ------------------------------------------------------------
// Approx compressors (Design 1 / Design 2)
// ------------------------------------------------------------
inline void approx_comp42_design1(ap_uint<1> x1, ap_uint<1> x2,
    ap_uint<1> x3, ap_uint<1> x4,
    ap_uint<1> Cin,
    ap_uint<1>& sum,
    ap_uint<1>& carry,
    ap_uint<1>& Cout) {
#pragma HLS inline
    ap_uint<1> p12 = x1 ^ x2;
    ap_uint<1> p34 = x3 ^ x4;

    sum = (~Cin) & ((~p12) | (~p34));
    carry = Cin;
    Cout = (x1 & x2) | (x3 & x4);
}

inline void approx_comp42_design2(ap_uint<1> x1, ap_uint<1> x2,
    ap_uint<1> x3, ap_uint<1> x4,
    ap_uint<1>& sum,
    ap_uint<1>& carry,
    ap_uint<1>& Cout /*ignored*/) {
#pragma HLS inline
    ap_uint<1> p12 = x1 ^ x2;
    ap_uint<1> p34 = x3 ^ x4;

    sum = ((~p12) | (~p34));
    carry = (x1 & x2) | (x3 & x4);
    Cout = 0;
}

// ------------------------------------------------------------
// DQ4:2Cx (approx-mode behavior)
// ------------------------------------------------------------
inline void dq42c1(ap_uint<1> x1, ap_uint<1>, ap_uint<1>, ap_uint<1> x4,
    ap_uint<1>& sum, ap_uint<1>& carry, ap_uint<1>& Cout) {
#pragma HLS inline
    sum = x1;
    carry = x4;
    Cout = 0;
}

inline void dq42c2(ap_uint<1> x1, ap_uint<1>, ap_uint<1> x3, ap_uint<1> x4,
    ap_uint<1>& sum, ap_uint<1>& carry, ap_uint<1>& Cout) {
#pragma HLS inline
    sum = x1;
    carry = x4;
    Cout = x3;
}

inline void dq42c3(ap_uint<1> x1, ap_uint<1> x2, ap_uint<1> x3, ap_uint<1> x4,
    ap_uint<1>& sum, ap_uint<1>& carry, ap_uint<1>& Cout) {
#pragma HLS inline
    ap_uint<1> p12 = x1 ^ x2;
    ap_uint<1> p34 = x3 ^ x4;
    sum = ~(p12 & p34);
    carry = p12;   // proxy
    Cout = 0;
}

inline void dq42c4(ap_uint<1> x1, ap_uint<1> x2, ap_uint<1> x3, ap_uint<1> x4,
    ap_uint<1>& sum, ap_uint<1>& carry, ap_uint<1>& Cout) {
#pragma HLS inline
    ap_uint<1> p12 = x1 ^ x2;
    ap_uint<1> p34 = x3 ^ x4;
    carry = (x1 & x2) | (x3 & x4);
    sum = p12 | p34;
    Cout = 0;
}
// ------------------------------------------------------------
// Approximate Mirror Adder full-adder behavior (A8): modes 8..12
// Implemented as a 1-bit FA used inside CSA reduction.
// ------------------------------------------------------------
inline void ma_fa(ap_uint<1> A, ap_uint<1> B, ap_uint<1> Cin,
    ap_uint<1>& Sum, ap_uint<1>& Cout) {
#pragma HLS inline

    // idx = {A,B,Cin} as 0..7
    unsigned idx = ((unsigned)A << 2) | ((unsigned)B << 1) | (unsigned)Cin;

#if (APPROX_MUL_MODE == 8)
    // MA Approximation 1 (A8): 1 error in Cout, 2 errors in Sum (Table I).
    // NOTE: Replace these LUTs with the exact Table I entries if you want bit-exact behavior.
    static const ap_uint<1> SUM_LUT[8] = { 0,1,0,0,0,0,0,1 };
    static const ap_uint<1> COUT_LUT[8] = { 0,0,0,1,0,1,1,0 };
    Sum = SUM_LUT[idx];
    Cout = COUT_LUT[idx];

#elif (APPROX_MUL_MODE == 9)
    // MA Approximation 2 (A8): Cout exact; Sum = ~Cout (2 mismatches at 000 and 111).
    ap_uint<1> cout_exact = (A & B) | (Cin & (A ^ B));
    Cout = cout_exact;
    Sum = ~cout_exact;

#elif (APPROX_MUL_MODE == 10)
    // MA Approximation 3 (A8): combines ideas of approx 1 and 2 (Table I).
    static const ap_uint<1> COUT_LUT[8] = { 0,0,0,1,0,1,1,0 };
    Cout = COUT_LUT[idx];
    Sum = ~Cout;

#elif (APPROX_MUL_MODE == 11)
    // MA Approximation 4 (A8): Cout = A; Sum similar to approx 1 (Table I).
    Cout = A;
    static const ap_uint<1> SUM_LUT[8] = { 0,1,0,0,0,0,0,1 };
    Sum = SUM_LUT[idx];

#elif (APPROX_MUL_MODE == 12)
    // MA Approximation 5 (A8), choice 2: Sum = B, Cout = A (Table II).
    Sum = B;
    Cout = A;

#else
    // Accurate FA (fallback)
    Sum = A ^ B ^ Cin;
    Cout = (A & B) | (Cin & (A ^ B));
#endif
}
// ------------------------------------------------------------
// Unsigned CSA multiplier (modes 2..7 apply inside CSA reduce)
// ------------------------------------------------------------
template<int WA, int WB>
ap_uint<WA + WB> mul_csa_comp42_unsigned(ap_uint<WA> a, ap_uint<WB> b) {
#pragma HLS inline
    constexpr int WP = WA + WB;
    constexpr int MAXH = (WA < WB ? WA : WB) + 4;

    ap_uint<1> heap[WP + 1][MAXH];
    ap_uint<8> cnt[WP + 1];
#pragma HLS array_partition variable=cnt complete dim=1

    for (int k = 0; k < WP + 1; k++) {
#pragma HLS unroll
        cnt[k] = 0;
    }

    // Partial products
    for (int i = 0; i < WA; i++) {
        for (int j = 0; j < WB; j++) {
#pragma HLS pipeline II=1
            ap_uint<1> pp = a[i] & b[j];
            int col = i + j;
            heap[col][cnt[col]++] = pp;
        }
    }

    // CSA reduction
    for (int col = 0; col < WP; col++) {
#pragma HLS pipeline II=1
        while (cnt[col] > 2) {
#pragma HLS loop_tripcount min=0 max=64

#if (APPROX_MUL_MODE == 2)
            // Design 1
            if (cnt[col] >= 5) {
                ap_uint<1> Cin = heap[col][--cnt[col]];
                ap_uint<1> x4 = heap[col][--cnt[col]];
                ap_uint<1> x3 = heap[col][--cnt[col]];
                ap_uint<1> x2 = heap[col][--cnt[col]];
                ap_uint<1> x1 = heap[col][--cnt[col]];
                ap_uint<1> s, c, co;
                approx_comp42_design1(x1, x2, x3, x4, Cin, s, c, co);
                heap[col][cnt[col]++] = s;
                heap[col + 1][cnt[col + 1]++] = c;
                heap[col + 1][cnt[col + 1]++] = co;
            }
            else if (cnt[col] >= 4) {
                ap_uint<1> x4 = heap[col][--cnt[col]];
                ap_uint<1> x3 = heap[col][--cnt[col]];
                ap_uint<1> x2 = heap[col][--cnt[col]];
                ap_uint<1> x1 = heap[col][--cnt[col]];
                ap_uint<1> s, c, co;
                approx_comp42_design1(x1, x2, x3, x4, (ap_uint<1>)0, s, c, co);
                heap[col][cnt[col]++] = s;
                heap[col + 1][cnt[col + 1]++] = c;
                heap[col + 1][cnt[col + 1]++] = co;
            }
            else {
                ap_uint<1> x3 = heap[col][--cnt[col]];
                ap_uint<1> x2 = heap[col][--cnt[col]];
                ap_uint<1> x1 = heap[col][--cnt[col]];
                ap_uint<1> s = x1 ^ x2 ^ x3;
                ap_uint<1> c = (x1 & x2) | (x1 & x3) | (x2 & x3);
                heap[col][cnt[col]++] = s;
                heap[col + 1][cnt[col + 1]++] = c;
            }

#elif (APPROX_MUL_MODE == 3)
            // Design 2
            if (cnt[col] >= 4) {
                ap_uint<1> x4 = heap[col][--cnt[col]];
                ap_uint<1> x3 = heap[col][--cnt[col]];
                ap_uint<1> x2 = heap[col][--cnt[col]];
                ap_uint<1> x1 = heap[col][--cnt[col]];
                ap_uint<1> s, c, co;
                approx_comp42_design2(x1, x2, x3, x4, s, c, co);
                heap[col][cnt[col]++] = s;
                heap[col + 1][cnt[col + 1]++] = c;
            }
            else {
                ap_uint<1> x3 = heap[col][--cnt[col]];
                ap_uint<1> x2 = heap[col][--cnt[col]];
                ap_uint<1> x1 = heap[col][--cnt[col]];
                ap_uint<1> s = x1 ^ x2 ^ x3;
                ap_uint<1> c = (x1 & x2) | (x1 & x3) | (x2 & x3);
                heap[col][cnt[col]++] = s;
                heap[col + 1][cnt[col + 1]++] = c;
            }

#elif (APPROX_MUL_MODE >= 4) && (APPROX_MUL_MODE <= 7)
            // DQ4:2Cx
            if (cnt[col] >= 4) {
                ap_uint<1> x4 = heap[col][--cnt[col]];
                ap_uint<1> x3 = heap[col][--cnt[col]];
                ap_uint<1> x2 = heap[col][--cnt[col]];
                ap_uint<1> x1 = heap[col][--cnt[col]];
                ap_uint<1> s, c, co;

#if (APPROX_MUL_MODE == 4)
                dq42c1(x1, x2, x3, x4, s, c, co);
#elif (APPROX_MUL_MODE == 5)
                dq42c2(x1, x2, x3, x4, s, c, co);
#elif (APPROX_MUL_MODE == 6)
                dq42c3(x1, x2, x3, x4, s, c, co);
#elif (APPROX_MUL_MODE == 7)
                dq42c4(x1, x2, x3, x4, s, c, co);
#endif
                heap[col][cnt[col]++] = s;
                heap[col + 1][cnt[col + 1]++] = c;
                if (co) heap[col + 1][cnt[col + 1]++] = co;
            }
            else {
                ap_uint<1> x3 = heap[col][--cnt[col]];
                ap_uint<1> x2 = heap[col][--cnt[col]];
                ap_uint<1> x1 = heap[col][--cnt[col]];
                ap_uint<1> s = x1 ^ x2 ^ x3;
                ap_uint<1> c = (x1 & x2) | (x1 & x3) | (x2 & x3);
                heap[col][cnt[col]++] = s;
                heap[col + 1][cnt[col + 1]++] = c;
            }
#elif (APPROX_MUL_MODE >= 8) && (APPROX_MUL_MODE <= 12)
            // MA approximate full adders inside CSA reduction (A8)
            // Reduce column by consuming 3 bits at a time.
            ap_uint<1> x3 = heap[col][--cnt[col]];
            ap_uint<1> x2 = heap[col][--cnt[col]];
            ap_uint<1> x1 = heap[col][--cnt[col]];

            ap_uint<1> s, c;
            ma_fa(x1, x2, x3, s, c);

            heap[col][cnt[col]++] = s;
            heap[col + 1][cnt[col + 1]++] = c;
#else
            // Fallback exact reduction (so modes 0/1 compile cleanly):
            // consume 3 bits -> exact FA -> push sum in same col, carry in col+1
            ap_uint<1> x3 = heap[col][--cnt[col]];
            ap_uint<1> x2 = heap[col][--cnt[col]];
            ap_uint<1> x1 = heap[col][--cnt[col]];

            ap_uint<1> s = x1 ^ x2 ^ x3;
            ap_uint<1> c = (x1 & x2) | (x1 & x3) | (x2 & x3);

            heap[col][cnt[col]++] = s;
            heap[col + 1][cnt[col + 1]++] = c;
#endif
        }
    }

    // Final two-row add
    ap_uint<WP> r0 = 0, r1 = 0;
    for (int col = 0; col < WP; col++) {
#pragma HLS unroll
        if (cnt[col] > 0) r0[col] = heap[col][0];
        if (cnt[col] > 1) r1[col] = heap[col][1];
    }
    return (ap_uint<WP>)(r0 + r1);
}

// ------------------------------------------------------------
// Overloads: safe signed/unsigned handling
// ------------------------------------------------------------
template<int WA, int WB>
ap_uint<WA + WB> mul_csa_comp42(ap_uint<WA> a, ap_uint<WB> b) {
#pragma HLS inline
    return mul_csa_comp42_unsigned<WA, WB>(a, b);
}

template<int WA, int WB>
ap_int<WA + WB> mul_csa_comp42(ap_int<WA> a, ap_int<WB> b) {
#pragma HLS inline
    bool neg = (a < 0) ^ (b < 0);
    ap_uint<WA> ua = (a < 0) ? (ap_uint<WA>)(-a) : (ap_uint<WA>)a;
    ap_uint<WB> ub = (b < 0) ? (ap_uint<WB>)(-b) : (ap_uint<WB>)b;
    ap_uint<WA + WB> um = mul_csa_comp42_unsigned<WA, WB>(ua, ub);
    ap_int<WA + WB>  sm = (ap_int<WA + WB>)um;
    return neg ? (ap_int<WA + WB>)(-sm) : sm;
}

template<int WA, int WB>
ap_int<WA + WB> mul_csa_comp42(ap_int<WA> a, ap_uint<WB> b) {
#pragma HLS inline
    bool neg = (a < 0);
    ap_uint<WA> ua = (a < 0) ? (ap_uint<WA>)(-a) : (ap_uint<WA>)a;
    ap_uint<WA + WB> um = mul_csa_comp42_unsigned<WA, WB>(ua, b);
    ap_int<WA + WB>  sm = (ap_int<WA + WB>)um;
    return neg ? (ap_int<WA + WB>)(-sm) : sm;
}

template<int WA, int WB>
ap_int<WA + WB> mul_csa_comp42(ap_uint<WA> a, ap_int<WB> b) {
#pragma HLS inline
    bool neg = (b < 0);
    ap_uint<WB> ub = (b < 0) ? (ap_uint<WB>)(-b) : (ap_uint<WB>)b;
    ap_uint<WA + WB> um = mul_csa_comp42_unsigned<WA, WB>(a, ub);
    ap_int<WA + WB>  sm = (ap_int<WA + WB>)um;
    return neg ? (ap_int<WA + WB>)(-sm) : sm;
}

// ------------------------------------------------------------
// approx_mul selector
// ------------------------------------------------------------
template<typename TC, typename TD>
auto approx_mul(TC const& c, TD const& d) -> decltype(c* d) {
#pragma HLS inline
#if (APPROX_MUL_MODE == 0)
    return c * d;
#elif (APPROX_MUL_MODE == 1)
    float cb = float_to_bfloat16((float)c);
    float db = float_to_bfloat16((float)d);
    float pb = cb * db;
    return (decltype(c * d))pb;
#elif (APPROX_MUL_MODE >= 2) && (APPROX_MUL_MODE <= 12)
    return (decltype(c * d))mul_csa_comp42(c, d);
#else
#   error "Unsupported APPROX_MUL_MODE"
#endif
}

// ======================= mul wrappers =======================
template<typename TC, typename TD>
auto mul(TC const& c, TD const& d, ap_resource_dflt const&) -> decltype(c* d) {
#pragma HLS inline
    return approx_mul(c, d);
}

template<typename TC, typename TD>
auto mul(TC const& c, TD const& d, ap_resource_lut const&) -> decltype(c* d) {
#pragma HLS inline
    decltype(c * d) const res = approx_mul(c, d);
#if (APPROX_MUL_MODE <= 1)
#pragma HLS BIND_OP variable=res op=mul impl=fabric
#endif
    return res;
}

template<typename TC, typename TD>
auto mul(TC const& c, TD const& d, ap_resource_dsp const&) -> decltype(c* d) {
#pragma HLS inline
    decltype(c * d) const res = approx_mul(c, d);
#if (APPROX_MUL_MODE <= 1)
#pragma HLS BIND_OP variable=res op=mul impl=dsp
#endif
    return res;
}

// ======================== MAC blocks ========================
template<unsigned N, typename T, typename TC, typename TD, typename R>
T mac(T const& a, TC const& c, TD const& d, R const& r, unsigned mmv) {
#pragma HLS inline
    T res = a;
    for (unsigned i = 0; i < N; i++) {
#pragma HLS unroll
        res += mul(c[i], d(i, mmv), r);
    }
    return res;
}

template<unsigned N, typename T, typename TC, typename TD, typename R>
T mac(T const& a, TC const& c, TD const& d, R const& r) {
#pragma HLS inline
    T res = a;
    for (unsigned i = 0; i < N; i++) {
#pragma HLS unroll
        res += mul(c[i], d[i], r);
    }
    return res;
}

template<unsigned N, typename T, typename TC, typename TD>
inline T mac(T const& a, TC const& c, TD const& d) {
#pragma HLS inline
    return mac<N>(a, c, d, ap_resource_dflt());
}

#endif // MAC_HPP