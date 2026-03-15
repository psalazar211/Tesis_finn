/******************************************************************************
 *  Copyright (c) 2019, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/

/*****************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Thomas B. Preusser <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *           Christoph Doehring <cdoehrin@xilinx.com>
 *
 *  \file mac.hpp
 *
 *  Library of templated HLS functions for BNN deployment.
 *  This file lists a set of convenience funtions used to implement
 *  multipliers with selectable implementation resource
 *
 *  This project has received funding from the European Union's Framework
 *  Programme for Research and Innovation Horizon 2020 (2014-2020) under
 *  the Marie Skłodowska-Curie Grant Agreement No. 751339.
 *
 *****************************************************************************/
 
/*****************************************************************************
 * MAC operation template:
 *
 *   mac<N, T, TC, TD>(T a, TC c[N], TD d[N])
 *      = a + SUM_{i=0}^{N-1} c(i)*d(i)
 *
 * All template arguments but N can typically be inferred.
 *
 *   mac<ap_uint<14>>(0, c, d)
 *****************************************************************************/
 
#ifndef MAC_HPP
#define MAC_HPP

#include "utils.hpp"
#include <ap_bfloat16.h>

 // ============================================================
 // Multiplier selection (exact + 2 approximate examples)
 // Set APPROX_MUL_MODE:
 //   0 = exact (c*d)
 //   1 = approximate: truncate K LSBs of product
 //   2 = approximate: truncate KA/KB LSBs of operands before multiply
 // ============================================================

#ifndef APPROX_MUL_MODE
#define APPROX_MUL_MODE 0
#endif

//For mode 2 (BFLOAT16 ), the multiplier is implemented by casting the operands to bfloat16, multiplying, and returning the result as the original datatype. This effectively truncates the product to 8 bits of precision in the mantissa, which can save resources while still providing good accuracy for many applications.
#include <ap_int.h>

inline float float_to_bfloat16(float x) {
#pragma HLS inline
    union {
        float f;
        ap_uint<32> u;
    } v;

    v.f = x;
    v.u &= 0xFFFF0000; // keep sign + exponent + 7 MSB mantissa
    return v.f;
}
// For mode 2 (truncate operands LSBs)
#ifndef APPROX_TRUNC_A_K
#define APPROX_TRUNC_A_K 1
#endif
#ifndef APPROX_TRUNC_B_K
#define APPROX_TRUNC_B_K 1
#endif

template <typename TC, typename TD>
auto approx_mul(TC const& c, TD const& d) -> decltype(c* d) {
#pragma HLS inline
#if (APPROX_MUL_MODE == 0)
    // Exact
    return c * d;

#elif (APPROX_MUL_MODE == 1)
    // Approx #1 (BFloat16 emulated)
    float cb = float_to_bfloat16((float)c);
    float db = float_to_bfloat16((float)d);
    float pb = cb * db;
    return (decltype(c * d))pb;

#elif (APPROX_MUL_MODE == 2)
    // Approx #2: truncate LSBs of operands before multiply
    auto cc = c;
    auto dd = d;
    cc = (cc >> APPROX_TRUNC_A_K) << APPROX_TRUNC_A_K;
    dd = (dd >> APPROX_TRUNC_B_K) << APPROX_TRUNC_B_K;
    return cc * dd;

#else
#   error "Unsupported APPROX_MUL_MODE"
#endif
}


/**
 * \brief      Multipliy operation between 2 operands, HLS choose the best resource
 * 
 * The same multiply operation can be implemented using multiple Vivado HLS pragmas to select the 
 * hardware resource to be used:
 * ap_resource_dflt will let HLS choose the best one
 * ap_resource_lut will force HLS to implement the multiplier in LUTs
 * ap_resource_dsp will force HLS to implement the multiplier in DSP48
 *
 * \tparam     TC    First operand datatype (weights)
 * \tparam     TD    Second operand datatype (input)
 * 
 * \param      c     First operand (array of weights)
 * \param      d     Second operand (array of input activation)
 * \param      r     Resource type for the hardware implementation of the MAC block
 *
 * \return     Result of the multiply operation
 */
template<typename TC, typename TD>
auto mul(TC const &c, TD const &d, ap_resource_dflt const&) -> decltype(c*d) {
#pragma HLS inline
  auto r = approx_mul(c, d);
  return  r;
}

/**
 * \brief      Multiply operation between 2 operands, implemented in LUT
 * 
 * The same multiply operation can be implemented using multiple Vivado HLS pragmas to select the 
 * hardware resource to be used:
 * ap_resource_dflt will let HLS choose the best one
 * ap_resource_lut will force HLS to implement the multiplier in LUTs
 * ap_resource_dsp will force HLS to implement the multiplier in DSP48
 *
 * \tparam     TC    First operand datatype (weights)
 * \tparam     TD    Second operand datatype (input)
 * 
 * \param      c     First operand (array of weights)
 * \param      d     Second operand (array of input activation)
 * \param      r     Resource type for the hardware implementation of the MAC block
 *
 * \return     Result of the multiply operation
 */
template<typename TC, typename TD>
auto mul(TC const &c, TD const &d, ap_resource_lut const&) -> decltype(c*d) {
#pragma HLS inline
  decltype(c*d) const  res = approx_mul(c, d);
#pragma HLS BIND_OP variable=res op=mul impl=fabric
  return  res;
}

/**
 * \brief      Multipliy operation between 2 operands, implemented in a DSP48
 * 
 * The same multiply operation can be implemented using multiple Vivado HLS pragmas to select the 
 * hardware resource to be used:
 * ap_resource_dflt will let HLS choose the best one
 * ap_resource_lut will force HLS to implement the multiplier in LUTs
 * ap_resource_dsp will force HLS to implement the multiplier in DSP48
 *
 * \tparam     TC    First operand datatype (weights)
 * \tparam     TD    Second operand datatype (input)
 * 
 * \param      c     First operand (array of weights)
 * \param      d     Second operand (array of input activation)
 * \param      r     Resource type for the hardware implementation of the MAC block
 *
 * \return     Result of the multiply operation
 */
template<typename TC, typename TD>
auto mul(TC const &c, TD const &d, ap_resource_dsp const&) -> decltype(c*d) {
#pragma HLS inline
  decltype(c*d) const res = approx_mul(c, d);
#pragma HLS BIND_OP variable=res op=mul impl=dsp
  return  res;
}

/**
 * \brief      MAC with selectable implementation resource, used by Matrix_Vector_Activate_Batch
 *
 * \tparam     N     Number of MAC to be performed (equals to SIMD in mvau)
 * \tparam     T     Accumulator datatype
 * \tparam     TC    First operand datatype (weights)
 * \tparam     TD    Second operand datatype (input)
 * \tparam     R     Datatype for the resource used for FPGA implementation of the MAC  - safely deducible from the paramaters
 * 
 * \param      a     Initialization value of the accumulation
 * \param      c     First operand (array of weights)
 * \param      d     Second operand (array of input activation)
 * \param      r     Resource type for the hardware implementation of the MAC block
 * \param      mmv   MMV value to address accumulator and activation
 *
 * \return     Result of the MAC operation
 */
template<unsigned N, typename T, typename TC, typename TD, typename R>
T mac(T const &a, TC const &c, TD const &d, R const &r, unsigned mmv) {
#pragma HLS inline
  T  res = a;
  for(unsigned  i = 0; i < N; i++) {
#pragma HLS unroll
    res += mul(c[i], d(i,mmv), r);
  }
  return  res;
}

/**
 * \brief      MAC with selectable implementation resource, used by Matrix_Vector_Activate_Batch
 *
 * \tparam     N     Number of MAC to be performed (equals to SIMD in mvau)
 * \tparam     T     Accumulator datatype
 * \tparam     TC    First operand datatype (weights)
 * \tparam     TD    Second operand datatype (input)
 * \tparam     R     Datatype for the resource used for FPGA implementation of the MAC  - safely deducible from the paramaters
 * 
 * \param      a     Initialization value of the accumulation
 * \param      c     First operand (array of weights)
 * \param      d     Second operand (array of input activation)
 * \param      r     Resource type for the hardware implementation of the MAC block
 *
 * \return     Result of the MAC operation
 */
template<unsigned N, typename T, typename TC, typename TD, typename R>
T mac(T const &a, TC const &c, TD const &d, R const &r) {
#pragma HLS inline
  T  res = a;
  for(unsigned  i = 0; i < N; i++) {
#pragma HLS unroll
    res += mul(c[i], d[i], r);
  }
  return  res;
}
template<unsigned N, typename T, typename TC, typename TD>
inline T mac(T const &a, TC const &c, TD const &d) {
#pragma HLS inline
  return  mac<N>(a, c, d, ap_resource_dflt());
}

#endif
