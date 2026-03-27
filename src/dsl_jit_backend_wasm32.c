/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "dsl_jit_bridge_contract.h"
#include "dsl_jit_runtime_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if ME_USE_WASM32_JIT

#include "libtcc.h"

#if !ME_WASM32_SIDE_MODULE
#include <emscripten.h>
#endif

#if !ME_WASM32_SIDE_MODULE
EM_JS(int, me_wasm_jit_instantiate,
      (const unsigned char *wasm_bytes, int wasm_len, int bridge_lookup_fn_idx), {
    var src = HEAPU8.subarray(wasm_bytes, wasm_bytes + wasm_len);
    var enc = new TextEncoder();
    var dec = new TextDecoder();
    function readULEB(buf, pos) {
        var r = 0, s = 0, b;
        do { b = buf[pos++]; r |= (b & 0x7f) << s; s += 7; } while (b & 0x80);
        return [r, pos];
    }
    function encULEB(v) {
        var a = [];
        do { var b = v & 0x7f; v >>>= 7; if (v) b |= 0x80; a.push(b); } while (v);
        return a;
    }
    function encStr(s) {
        var b = enc.encode(s);
        return encULEB(b.length).concat(Array.from(b));
    }
    function readName(buf, pos) {
        var t = readULEB(buf, pos);
        var n = t[0];
        pos = t[1];
        var s = dec.decode(buf.subarray(pos, pos + n));
        return [s, pos + n];
    }
    function skipLimits(buf, pos) {
        var t = readULEB(buf, pos);
        var flags = t[0];
        pos = t[1];
        t = readULEB(buf, pos);
        pos = t[1];
        if (flags & 0x01) {
            t = readULEB(buf, pos);
            pos = t[1];
        }
        return pos;
    }
    function encMemoryImport() {
        var imp = [];
        imp = imp.concat(encStr("env"), encStr("memory"));
        imp.push(0x02, 0x00);
        imp = imp.concat(encULEB(256));
        return imp;
    }
    function buildImportSecWithMemory() {
        var body = encULEB(1);
        body = body.concat(encMemoryImport());
        var sec = [0x02];
        sec = sec.concat(encULEB(body.length));
        return sec.concat(body);
    }
    function patchImportSec(secData) {
        var pos = 0;
        var t = readULEB(secData, pos);
        var nimports = t[0];
        pos = t[1];
        var entries = [];
        var hasEnvMemory = false;
        for (var i = 0; i < nimports; i++) {
            var start = pos;
            var moduleName = "";
            var fieldName = "";
            t = readName(secData, pos);
            moduleName = t[0];
            pos = t[1];
            t = readName(secData, pos);
            fieldName = t[0];
            pos = t[1];
            var kind = secData[pos++];
            if (kind === 0x00) {
                t = readULEB(secData, pos);
                pos = t[1];
            }
            else if (kind === 0x01) {
                pos++;
                pos = skipLimits(secData, pos);
            }
            else if (kind === 0x02) {
                pos = skipLimits(secData, pos);
                if (moduleName === "env" && fieldName === "memory") {
                    hasEnvMemory = true;
                }
            }
            else if (kind === 0x03) {
                pos += 2;
            }
            else {
                throw new Error("unsupported wasm import kind " + kind);
            }
            entries.push(Array.from(secData.subarray(start, pos)));
        }
        if (!hasEnvMemory) {
            entries.push(encMemoryImport());
        }
        var body = encULEB(entries.length);
        for (var ei = 0; ei < entries.length; ei++) {
            body = body.concat(entries[ei]);
        }
        var sec = [0x02];
        sec = sec.concat(encULEB(body.length));
        return sec.concat(body);
    }
    function buildEnvImports() {
        var bridgeLookup = null;
        var bridgeCache = Object.create(null);
        if (bridge_lookup_fn_idx) {
            bridgeLookup = wasmTable.get(bridge_lookup_fn_idx);
        }
        function lookupBridge(name) {
            if (!bridgeLookup) {
                return null;
            }
            if (Object.prototype.hasOwnProperty.call(bridgeCache, name)) {
                return bridgeCache[name];
            }
            var sp = stackSave();
            try {
                var nbytes = lengthBytesUTF8(name) + 1;
                var namePtr = stackAlloc(nbytes);
                stringToUTF8(name, namePtr, nbytes);
                var fnIdx = bridgeLookup(namePtr) | 0;
                bridgeCache[name] = fnIdx ? wasmTable.get(fnIdx) : null;
            } finally {
                stackRestore(sp);
            }
            return bridgeCache[name];
        }
        function bindBridge(name, fallback) {
            var fn = lookupBridge(name);
            return fn ? fn : fallback;
        }
        function fdim(x, y) { return x > y ? (x - y) : 0.0; }
        function copysign(x, y) {
            if (y === 0) {
                return (1 / y === -Infinity) ? -Math.abs(x) : Math.abs(x);
            }
            return y < 0 ? -Math.abs(x) : Math.abs(x);
        }
        function ldexp(x, e) { return x * Math.pow(2.0, e); }
        function rint(x) {
            if (!isFinite(x)) {
                return x;
            }
            var n = Math.round(x);
            if (Math.abs(x - n) === 0.5) {
                n = 2 * Math.round(x / 2);
            }
            return n;
        }
        function remainder(x, y) {
            if (!isFinite(x) || !isFinite(y) || y === 0.0) {
                return NaN;
            }
            return x - y * Math.round(x / y);
        }
        function erfApprox(x) {
            var sign = x < 0 ? -1.0 : 1.0;
            x = Math.abs(x);
            var a1 = 0.254829592;
            var a2 = -0.284496736;
            var a3 = 1.421413741;
            var a4 = -1.453152027;
            var a5 = 1.061405429;
            var p = 0.3275911;
            var t = 1.0 / (1.0 + p * x);
            var y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * Math.exp(-x * x);
            return sign * y;
        }
        function erfcApprox(x) { return 1.0 - erfApprox(x); }
        function tgammaApprox(z) {
            var p = [
                676.5203681218851, -1259.1392167224028, 771.32342877765313,
                -176.61502916214059, 12.507343278686905, -0.13857109526572012,
                9.9843695780195716e-6, 1.5056327351493116e-7
            ];
            if (z < 0.5) {
                return Math.PI / (Math.sin(Math.PI * z) * tgammaApprox(1.0 - z));
            }
            z -= 1.0;
            var x = 0.99999999999980993;
            for (var i = 0; i < p.length; i++) {
                x += p[i] / (z + i + 1.0);
            }
            var t = z + p.length - 0.5;
            return Math.sqrt(2.0 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
        }
        function lgammaApprox(x) {
            var g = tgammaApprox(x);
            return Math.log(Math.abs(g));
        }
        function nextafterApprox(x, y) {
            if (isNaN(x) || isNaN(y)) {
                return NaN;
            }
            if (x === y) {
                return y;
            }
            if (x === 0.0) {
                return y > 0.0 ? Number.MIN_VALUE : -Number.MIN_VALUE;
            }
            var buf = new ArrayBuffer(8);
            var dv = new DataView(buf);
            dv.setFloat64(0, x, true);
            var bits = dv.getBigUint64(0, true);
            if ((y > x) === (x > 0.0)) {
                bits += 1n;
            }
            else {
                bits -= 1n;
            }
            dv.setBigUint64(0, bits, true);
            return dv.getFloat64(0, true);
        }
        function meJitExp10(x) { return Math.pow(10.0, x); }
        function meJitSinpi(x) { return Math.sin(Math.PI * x); }
        function meJitCospi(x) { return Math.cos(Math.PI * x); }
        var mathExp2 = Math.exp2 ? Math.exp2 : function(x) { return Math.pow(2.0, x); };
        function meJitLogaddexp(a, b) {
            var hi = a > b ? a : b;
            var lo = a > b ? b : a;
            return hi + Math.log1p(Math.exp(lo - hi));
        }
        function meJitWhere(c, x, y) { return c !== 0.0 ? x : y; }
        function vecUnaryF64(inPtr, outPtr, n, fn) {
            var ii = inPtr >> 3;
            var oo = outPtr >> 3;
            for (var i = 0; i < n; i++) {
                HEAPF64[oo + i] = fn(HEAPF64[ii + i]);
            }
        }
        function vecBinaryF64(aPtr, bPtr, outPtr, n, fn) {
            var aa = aPtr >> 3;
            var bb = bPtr >> 3;
            var oo = outPtr >> 3;
            for (var i = 0; i < n; i++) {
                HEAPF64[oo + i] = fn(HEAPF64[aa + i], HEAPF64[bb + i]);
            }
        }
        function vecUnaryF32(inPtr, outPtr, n, fn) {
            var ii = inPtr >> 2;
            var oo = outPtr >> 2;
            for (var i = 0; i < n; i++) {
                HEAPF32[oo + i] = fn(HEAPF32[ii + i]);
            }
        }
        function vecBinaryF32(aPtr, bPtr, outPtr, n, fn) {
            var aa = aPtr >> 2;
            var bb = bPtr >> 2;
            var oo = outPtr >> 2;
            for (var i = 0; i < n; i++) {
                HEAPF32[oo + i] = fn(HEAPF32[aa + i], HEAPF32[bb + i]);
            }
        }
        var env = {
            memory: wasmMemory,
            acos: Math.acos, acosh: Math.acosh, asin: Math.asin, asinh: Math.asinh,
            atan: Math.atan, atan2: Math.atan2, atanh: Math.atanh, cbrt: Math.cbrt,
            ceil: Math.ceil, copysign: copysign, cos: Math.cos, cosh: Math.cosh,
            erf: erfApprox, erfc: erfcApprox, exp: Math.exp, exp2: mathExp2,
            expm1: Math.expm1, fabs: Math.abs, fdim: fdim, floor: Math.floor,
            fma: function(a, b, c) { return a * b + c; }, fmax: Math.max, fmin: Math.min,
            fmod: function(a, b) { return a % b; }, hypot: Math.hypot, ldexp: ldexp,
            lgamma: lgammaApprox, log: Math.log, log10: Math.log10, log1p: Math.log1p,
            log2: Math.log2, nextafter: nextafterApprox, pow: Math.pow, remainder: remainder,
            rint: rint, round: Math.round, sin: Math.sin, sinh: Math.sinh, sqrt: Math.sqrt,
            tan: Math.tan, tanh: Math.tanh, tgamma: tgammaApprox, trunc: Math.trunc,
            me_jit_exp10: meJitExp10, me_jit_sinpi: meJitSinpi, me_jit_cospi: meJitCospi,
            me_jit_logaddexp: meJitLogaddexp, me_jit_where: meJitWhere
        };
        env.me_wasm32_cast_int = function(x) {
            return x < 0 ? Math.ceil(x) : Math.floor(x);
        };
        env.me_wasm32_cast_float = function(x) {
            return x;
        };
        env.me_wasm32_cast_bool = function(x) {
            return x !== 0 ? 1 : 0;
        };
        env.memset = bindBridge("memset", function(ptr, value, n) {
            if (n > 0) {
                HEAPU8.fill(value & 255, ptr, ptr + n);
            }
            return ptr | 0;
        });
        env.me_jit_exp10 = bindBridge("me_jit_exp10", env.me_jit_exp10);
        env.me_jit_sinpi = bindBridge("me_jit_sinpi", env.me_jit_sinpi);
        env.me_jit_cospi = bindBridge("me_jit_cospi", env.me_jit_cospi);
        env.me_jit_logaddexp = bindBridge("me_jit_logaddexp", env.me_jit_logaddexp);
        env.me_jit_where = bindBridge("me_jit_where", env.me_jit_where);
        env.me_jit_vec_sin_f64 = bindBridge("me_jit_vec_sin_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.sin); });
        env.me_jit_vec_cos_f64 = bindBridge("me_jit_vec_cos_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.cos); });
        env.me_jit_vec_exp_f64 = bindBridge("me_jit_vec_exp_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.exp); });
        env.me_jit_vec_log_f64 = bindBridge("me_jit_vec_log_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.log); });
        env.me_jit_vec_exp10_f64 = bindBridge("me_jit_vec_exp10_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, meJitExp10); });
        env.me_jit_vec_sinpi_f64 = bindBridge("me_jit_vec_sinpi_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, meJitSinpi); });
        env.me_jit_vec_cospi_f64 = bindBridge("me_jit_vec_cospi_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, meJitCospi); });
        env.me_jit_vec_atan2_f64 = bindBridge("me_jit_vec_atan2_f64", function(aPtr, bPtr, outPtr, n) { vecBinaryF64(aPtr, bPtr, outPtr, n, Math.atan2); });
        env.me_jit_vec_hypot_f64 = bindBridge("me_jit_vec_hypot_f64", function(aPtr, bPtr, outPtr, n) { vecBinaryF64(aPtr, bPtr, outPtr, n, Math.hypot); });
        env.me_jit_vec_pow_f64 = bindBridge("me_jit_vec_pow_f64", function(aPtr, bPtr, outPtr, n) { vecBinaryF64(aPtr, bPtr, outPtr, n, Math.pow); });
        env.me_jit_vec_fmax_f64 = bindBridge("me_jit_vec_fmax_f64", function(aPtr, bPtr, outPtr, n) { vecBinaryF64(aPtr, bPtr, outPtr, n, Math.max); });
        env.me_jit_vec_fmin_f64 = bindBridge("me_jit_vec_fmin_f64", function(aPtr, bPtr, outPtr, n) { vecBinaryF64(aPtr, bPtr, outPtr, n, Math.min); });
        env.me_jit_vec_expm1_f64 = bindBridge("me_jit_vec_expm1_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.expm1); });
        env.me_jit_vec_log10_f64 = bindBridge("me_jit_vec_log10_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.log10); });
        env.me_jit_vec_sinh_f64 = bindBridge("me_jit_vec_sinh_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.sinh); });
        env.me_jit_vec_cosh_f64 = bindBridge("me_jit_vec_cosh_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.cosh); });
        env.me_jit_vec_tanh_f64 = bindBridge("me_jit_vec_tanh_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.tanh); });
        env.me_jit_vec_asinh_f64 = bindBridge("me_jit_vec_asinh_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.asinh); });
        env.me_jit_vec_acosh_f64 = bindBridge("me_jit_vec_acosh_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.acosh); });
        env.me_jit_vec_atanh_f64 = bindBridge("me_jit_vec_atanh_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.atanh); });
        env.me_jit_vec_abs_f64 = bindBridge("me_jit_vec_abs_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.abs); });
        env.me_jit_vec_sqrt_f64 = bindBridge("me_jit_vec_sqrt_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.sqrt); });
        env.me_jit_vec_log1p_f64 = bindBridge("me_jit_vec_log1p_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.log1p); });
        env.me_jit_vec_exp2_f64 = bindBridge("me_jit_vec_exp2_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, mathExp2); });
        env.me_jit_vec_log2_f64 = bindBridge("me_jit_vec_log2_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.log2); });
        env.me_jit_vec_sin_f32 = bindBridge("me_jit_vec_sin_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.sin); });
        env.me_jit_vec_cos_f32 = bindBridge("me_jit_vec_cos_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.cos); });
        env.me_jit_vec_exp_f32 = bindBridge("me_jit_vec_exp_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.exp); });
        env.me_jit_vec_log_f32 = bindBridge("me_jit_vec_log_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.log); });
        env.me_jit_vec_exp10_f32 = bindBridge("me_jit_vec_exp10_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, meJitExp10); });
        env.me_jit_vec_sinpi_f32 = bindBridge("me_jit_vec_sinpi_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, meJitSinpi); });
        env.me_jit_vec_cospi_f32 = bindBridge("me_jit_vec_cospi_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, meJitCospi); });
        env.me_jit_vec_atan2_f32 = bindBridge("me_jit_vec_atan2_f32", function(aPtr, bPtr, outPtr, n) { vecBinaryF32(aPtr, bPtr, outPtr, n, Math.atan2); });
        env.me_jit_vec_hypot_f32 = bindBridge("me_jit_vec_hypot_f32", function(aPtr, bPtr, outPtr, n) { vecBinaryF32(aPtr, bPtr, outPtr, n, Math.hypot); });
        env.me_jit_vec_pow_f32 = bindBridge("me_jit_vec_pow_f32", function(aPtr, bPtr, outPtr, n) { vecBinaryF32(aPtr, bPtr, outPtr, n, Math.pow); });
        env.me_jit_vec_fmax_f32 = bindBridge("me_jit_vec_fmax_f32", function(aPtr, bPtr, outPtr, n) { vecBinaryF32(aPtr, bPtr, outPtr, n, Math.max); });
        env.me_jit_vec_fmin_f32 = bindBridge("me_jit_vec_fmin_f32", function(aPtr, bPtr, outPtr, n) { vecBinaryF32(aPtr, bPtr, outPtr, n, Math.min); });
        env.me_jit_vec_expm1_f32 = bindBridge("me_jit_vec_expm1_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.expm1); });
        env.me_jit_vec_log10_f32 = bindBridge("me_jit_vec_log10_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.log10); });
        env.me_jit_vec_sinh_f32 = bindBridge("me_jit_vec_sinh_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.sinh); });
        env.me_jit_vec_cosh_f32 = bindBridge("me_jit_vec_cosh_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.cosh); });
        env.me_jit_vec_tanh_f32 = bindBridge("me_jit_vec_tanh_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.tanh); });
        env.me_jit_vec_asinh_f32 = bindBridge("me_jit_vec_asinh_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.asinh); });
        env.me_jit_vec_acosh_f32 = bindBridge("me_jit_vec_acosh_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.acosh); });
        env.me_jit_vec_atanh_f32 = bindBridge("me_jit_vec_atanh_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.atanh); });
        env.me_jit_vec_abs_f32 = bindBridge("me_jit_vec_abs_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.abs); });
        env.me_jit_vec_sqrt_f32 = bindBridge("me_jit_vec_sqrt_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.sqrt); });
        env.me_jit_vec_log1p_f32 = bindBridge("me_jit_vec_log1p_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.log1p); });
        env.me_jit_vec_exp2_f32 = bindBridge("me_jit_vec_exp2_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, mathExp2); });
        env.me_jit_vec_log2_f32 = bindBridge("me_jit_vec_log2_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.log2); });
        return env;
    }
    var pos = 8, sections = [];
    while (pos < src.length) {
        var id = src[pos++];
        var tmp = readULEB(src, pos), len = tmp[0]; pos = tmp[1];
        sections.push({ id: id, data: src.subarray(pos, pos + len) });
        pos += len;
    }
    var out = [0x00,0x61,0x73,0x6d, 0x01,0x00,0x00,0x00];
    var impDone = false;
    for (var i = 0; i < sections.length; i++) {
        var s = sections[i];
        if (s.id === 5) continue;
        if (s.id === 2) {
            out = out.concat(patchImportSec(s.data));
            impDone = true;
            continue;
        }
        if (!impDone && s.id > 2) {
            out = out.concat(buildImportSecWithMemory());
            impDone = true;
        }
        if (s.id === 7) {
            var ep = 0, et = readULEB(s.data, ep), ne = et[0]; ep = et[1];
            var exps = [];
            for (var e = 0; e < ne; e++) {
                var nt = readULEB(s.data, ep), nl = nt[0]; ep = nt[1];
                var nm = dec.decode(s.data.subarray(ep, ep + nl)); ep += nl;
                var kd = s.data[ep++];
                var xt = readULEB(s.data, ep), xi = xt[0]; ep = xt[1];
                if (nm === "memory" && kd === 0x02) continue;
                exps.push({ n: nm, k: kd, i: xi });
            }
            var eb = encULEB(exps.length);
            for (var ei = 0; ei < exps.length; ei++) {
                eb = eb.concat(encStr(exps[ei].n));
                eb.push(exps[ei].k);
                eb = eb.concat(encULEB(exps[ei].i));
            }
            out.push(0x07);
            out = out.concat(encULEB(eb.length));
            out = out.concat(eb);
            continue;
        }
        out.push(s.id);
        out = out.concat(encULEB(s.data.length));
        out = out.concat(Array.from(s.data));
    }
    if (!impDone) {
        out = out.concat(buildImportSecWithMemory());
    }
    var patched = new Uint8Array(out);
    try {
        var mod = new WebAssembly.Module(patched);
        var inst = new WebAssembly.Instance(mod, { env: buildEnvImports() });
    } catch (e) {
        err("[me-wasm-jit] " + e.message);
        return 0;
    }
    var fn = inst.exports["me_dsl_jit_kernel"];
    if (!fn) { err("[me-wasm-jit] missing export"); return 0; }
    return addFunction(fn, "iiii");
});

EM_JS(void, me_wasm_jit_free_fn, (int idx), {
    if (idx) removeFunction(idx);
});
#endif

static me_wasm_jit_instantiate_helper g_me_wasm_jit_instantiate_helper = NULL;
static me_wasm_jit_free_helper g_me_wasm_jit_free_helper = NULL;

#if ME_WASM32_SIDE_MODULE
bool me_wasm_jit_helpers_available(void) {
    return g_me_wasm_jit_instantiate_helper != NULL &&
           g_me_wasm_jit_free_helper != NULL;
}
#endif

void dsl_register_wasm_jit_helpers(me_wasm_jit_instantiate_helper instantiate_helper,
                                   me_wasm_jit_free_helper free_helper) {
    g_me_wasm_jit_instantiate_helper = instantiate_helper;
    g_me_wasm_jit_free_helper = free_helper;
}

static int me_wasm_jit_instantiate_dispatch(const unsigned char *wasm_bytes, int wasm_len,
                                            int bridge_lookup_fn_idx) {
#if ME_WASM32_SIDE_MODULE
    if (!g_me_wasm_jit_instantiate_helper) {
        return 0;
    }
    return g_me_wasm_jit_instantiate_helper(wasm_bytes, wasm_len, bridge_lookup_fn_idx);
#else
    return me_wasm_jit_instantiate(wasm_bytes, wasm_len, bridge_lookup_fn_idx);
#endif
}

void dsl_wasm_jit_free_dispatch(int idx) {
#if ME_WASM32_SIDE_MODULE
    if (g_me_wasm_jit_free_helper) {
        g_me_wasm_jit_free_helper(idx);
    }
#else
    me_wasm_jit_free_fn(idx);
#endif
}

typedef struct {
    bool valid;
    uint64_t key;
    int fn_idx;
    void *scratch;
} me_dsl_jit_wasm_pos_cache_entry;

static me_dsl_jit_wasm_pos_cache_entry g_dsl_jit_wasm_pos_cache[ME_DSL_JIT_WASM_POS_CACHE_SLOTS];
static int g_dsl_jit_wasm_pos_cache_cursor = 0;

static int dsl_jit_wasm_pos_cache_find_slot(uint64_t key) {
    for (int i = 0; i < ME_DSL_JIT_WASM_POS_CACHE_SLOTS; i++) {
        if (g_dsl_jit_wasm_pos_cache[i].valid && g_dsl_jit_wasm_pos_cache[i].key == key) {
            return i;
        }
    }
    return -1;
}

static int dsl_jit_wasm_pos_cache_find_free_slot(void) {
    for (int i = 0; i < ME_DSL_JIT_WASM_POS_CACHE_SLOTS; i++) {
        if (!g_dsl_jit_wasm_pos_cache[i].valid) {
            return i;
        }
    }
    return -1;
}

bool dsl_jit_wasm_pos_cache_bind_program(me_dsl_compiled_program *program, uint64_t key) {
    if (!program) {
        return false;
    }
    int slot = dsl_jit_wasm_pos_cache_find_slot(key);
    if (slot < 0) {
        return false;
    }
    program->jit_kernel_fn = (me_dsl_jit_kernel_fn)(uintptr_t)g_dsl_jit_wasm_pos_cache[slot].fn_idx;
    program->jit_dl_handle = NULL;
    program->jit_runtime_key = key;
    program->jit_dl_handle_cached = true;
    return true;
}

static bool dsl_jit_wasm_pos_cache_store_program(me_dsl_compiled_program *program, uint64_t key,
                                                 int fn_idx, void *scratch) {
    if (!program || fn_idx == 0 || !scratch) {
        return false;
    }

    int slot = dsl_jit_wasm_pos_cache_find_slot(key);
    if (slot >= 0) {
        if (fn_idx != g_dsl_jit_wasm_pos_cache[slot].fn_idx) {
            dsl_wasm_jit_free_dispatch(fn_idx);
            free(scratch);
        }
        program->jit_kernel_fn = (me_dsl_jit_kernel_fn)(uintptr_t)g_dsl_jit_wasm_pos_cache[slot].fn_idx;
        program->jit_dl_handle = NULL;
        program->jit_runtime_key = key;
        program->jit_dl_handle_cached = true;
        return true;
    }

    slot = dsl_jit_wasm_pos_cache_find_free_slot();
    if (slot < 0) {
        slot = g_dsl_jit_wasm_pos_cache_cursor;
        g_dsl_jit_wasm_pos_cache_cursor =
            (g_dsl_jit_wasm_pos_cache_cursor + 1) % ME_DSL_JIT_WASM_POS_CACHE_SLOTS;
        if (g_dsl_jit_wasm_pos_cache[slot].valid) {
            if (g_dsl_jit_wasm_pos_cache[slot].fn_idx != 0) {
                dsl_wasm_jit_free_dispatch(g_dsl_jit_wasm_pos_cache[slot].fn_idx);
            }
            free(g_dsl_jit_wasm_pos_cache[slot].scratch);
        }
    }

    g_dsl_jit_wasm_pos_cache[slot].valid = true;
    g_dsl_jit_wasm_pos_cache[slot].key = key;
    g_dsl_jit_wasm_pos_cache[slot].fn_idx = fn_idx;
    g_dsl_jit_wasm_pos_cache[slot].scratch = scratch;

    program->jit_kernel_fn = (me_dsl_jit_kernel_fn)(uintptr_t)fn_idx;
    program->jit_dl_handle = NULL;
    program->jit_runtime_key = key;
    program->jit_dl_handle_cached = true;
    return true;
}

static void dsl_wasm_tcc_error_handler(void *opaque, const char *msg) {
    (void)opaque;
    dsl_tracef("jit tcc error: %s", msg);
}

static char *dsl_wasm32_patch_source(const char *src) {
    typedef struct {
        const char *old;
        const char *rep;
    } repl_t;
    repl_t repls[] = {
        { "#define ME_DSL_CAST_INT(x) ((int64_t)(x))",
          "extern int me_wasm32_cast_int(double);\n"
          "#define ME_DSL_CAST_INT(x) (me_wasm32_cast_int((double)(x)))" },
        { "#define ME_DSL_CAST_FLOAT(x) ((double)(x))",
          "extern double me_wasm32_cast_float(double);\n"
          "#define ME_DSL_CAST_FLOAT(x) (me_wasm32_cast_float((double)(x)))" },
        { "#define ME_DSL_CAST_BOOL(x) ((x) != 0)",
          "extern int me_wasm32_cast_bool(double);\n"
          "#define ME_DSL_CAST_BOOL(x) (me_wasm32_cast_bool((double)(x)))" },
        { "uint64_t ", "unsigned int " },
        { "(uint64_t)", "(unsigned int)" },
        { "int64_t ", "int " },
        { "(int64_t)", "(int)" },
        { "if (!output || nitems < 0) {\n"
          "        return -1;\n"
          "    }",
          "if (!output) {\n"
          "        return -1;\n"
          "    }\n"
          "    if (nitems < 0) {\n"
          "        return -1;\n"
          "    }" },
    };
    size_t src_len = strlen(src);
    size_t alloc = src_len + 2048;
    char *patched = (char *)malloc(alloc);
    if (!patched) {
        return NULL;
    }
    const char *p = src;
    char *d = patched;
    while (*p) {
        bool matched = false;
        for (size_t ri = 0; ri < sizeof(repls) / sizeof(repls[0]); ri++) {
            size_t olen = strlen(repls[ri].old);
            if (strncmp(p, repls[ri].old, olen) == 0) {
                size_t rlen = strlen(repls[ri].rep);
                if ((size_t)(d - patched) + rlen + 1 > alloc) {
                    break;
                }
                memcpy(d, repls[ri].rep, rlen);
                d += rlen;
                p += olen;
                matched = true;
                break;
            }
        }
        if (!matched) {
            *d++ = *p++;
        }
    }
    *d = '\0';
    return patched;
}

static bool dsl_wasm32_source_calls_symbol(const char *src, const char *name) {
    if (!src || !name || name[0] == '\0') {
        return false;
    }
    size_t name_len = strlen(name);
    const char *p = src;
    while (*p) {
        if (strncmp(p, "extern ", 7) == 0 ||
            strncmp(p, "static ", 7) == 0) {
            while (*p && *p != '\n') {
                p++;
            }
            if (*p) {
                p++;
            }
            continue;
        }
        if (strncmp(p, name, name_len) == 0 && p[name_len] == '(') {
            if (p == src || !((p[-1] >= 'a' && p[-1] <= 'z') ||
                              (p[-1] >= 'A' && p[-1] <= 'Z') ||
                              (p[-1] >= '0' && p[-1] <= '9') ||
                               p[-1] == '_')) {
                return true;
            }
        }
        p++;
    }
    return false;
}

typedef struct {
    const char *name;
    const void *addr;
} dsl_wasm32_symbol_binding;

#define ME_WASM32_DECL_me_dsl_jit_sig_scalar_unary_fn(name) extern double name(double);
#define ME_WASM32_DECL_me_dsl_jit_sig_scalar_binary_fn(name) extern double name(double, double);
#define ME_WASM32_DECL_me_dsl_jit_sig_scalar_ternary_fn(name) extern double name(double, double, double);
#define ME_WASM32_DECL_me_dsl_jit_sig_vec_unary_f64_fn(name) extern void name(const double *, double *, int64_t);
#define ME_WASM32_DECL_me_dsl_jit_sig_vec_binary_f64_fn(name) extern void name(const double *, const double *, double *, int64_t);
#define ME_WASM32_DECL_me_dsl_jit_sig_vec_unary_f32_fn(name) extern void name(const float *, float *, int64_t);
#define ME_WASM32_DECL_me_dsl_jit_sig_vec_binary_f32_fn(name) extern void name(const float *, const float *, float *, int64_t);
#define ME_WASM32_BRIDGE_DECL(pub_sym, bridge_fn, sig_type, decl) ME_WASM32_DECL_##sig_type(pub_sym)
ME_DSL_JIT_BRIDGE_SYMBOL_CONTRACT(ME_WASM32_BRIDGE_DECL)
#undef ME_WASM32_DECL_me_dsl_jit_sig_scalar_unary_fn
#undef ME_WASM32_DECL_me_dsl_jit_sig_scalar_binary_fn
#undef ME_WASM32_DECL_me_dsl_jit_sig_scalar_ternary_fn
#undef ME_WASM32_DECL_me_dsl_jit_sig_vec_unary_f64_fn
#undef ME_WASM32_DECL_me_dsl_jit_sig_vec_binary_f64_fn
#undef ME_WASM32_DECL_me_dsl_jit_sig_vec_unary_f32_fn
#undef ME_WASM32_DECL_me_dsl_jit_sig_vec_binary_f32_fn
#undef ME_WASM32_BRIDGE_DECL

#define ME_WASM32_BRIDGE_SYM(fn) { #fn, (const void *)&fn }
#define ME_WASM32_BRIDGE_CONTRACT_SYM(pub_sym, bridge_fn, sig_type, decl) \
    ME_WASM32_BRIDGE_SYM(pub_sym),

static const dsl_wasm32_symbol_binding dsl_wasm32_symbol_bindings[] = {
    ME_WASM32_BRIDGE_SYM(acos), ME_WASM32_BRIDGE_SYM(acosh), ME_WASM32_BRIDGE_SYM(asin),
    ME_WASM32_BRIDGE_SYM(asinh), ME_WASM32_BRIDGE_SYM(atan), ME_WASM32_BRIDGE_SYM(atan2),
    ME_WASM32_BRIDGE_SYM(atanh), ME_WASM32_BRIDGE_SYM(cbrt), ME_WASM32_BRIDGE_SYM(ceil),
    ME_WASM32_BRIDGE_SYM(copysign), ME_WASM32_BRIDGE_SYM(cos), ME_WASM32_BRIDGE_SYM(cosh),
    ME_WASM32_BRIDGE_SYM(erf), ME_WASM32_BRIDGE_SYM(erfc), ME_WASM32_BRIDGE_SYM(exp),
    ME_WASM32_BRIDGE_SYM(exp2), ME_WASM32_BRIDGE_SYM(expm1), ME_WASM32_BRIDGE_SYM(fabs),
    ME_WASM32_BRIDGE_SYM(fdim), ME_WASM32_BRIDGE_SYM(floor), ME_WASM32_BRIDGE_SYM(fma),
    ME_WASM32_BRIDGE_SYM(fmax), ME_WASM32_BRIDGE_SYM(fmin), ME_WASM32_BRIDGE_SYM(fmod),
    ME_WASM32_BRIDGE_SYM(hypot), ME_WASM32_BRIDGE_SYM(ldexp), ME_WASM32_BRIDGE_SYM(lgamma),
    ME_WASM32_BRIDGE_SYM(log), ME_WASM32_BRIDGE_SYM(log10), ME_WASM32_BRIDGE_SYM(log1p),
    ME_WASM32_BRIDGE_SYM(log2), ME_WASM32_BRIDGE_SYM(nextafter), ME_WASM32_BRIDGE_SYM(pow),
    ME_WASM32_BRIDGE_SYM(remainder), ME_WASM32_BRIDGE_SYM(rint), ME_WASM32_BRIDGE_SYM(round),
    ME_WASM32_BRIDGE_SYM(sin), ME_WASM32_BRIDGE_SYM(sinh), ME_WASM32_BRIDGE_SYM(sqrt),
    ME_WASM32_BRIDGE_SYM(tan), ME_WASM32_BRIDGE_SYM(tanh), ME_WASM32_BRIDGE_SYM(tgamma),
    ME_WASM32_BRIDGE_SYM(trunc),
    ME_WASM32_BRIDGE_SYM(memset),
    ME_DSL_JIT_BRIDGE_SYMBOL_CONTRACT(ME_WASM32_BRIDGE_CONTRACT_SYM)
};

static int dsl_wasm32_lookup_bridge_symbol(const char *name) {
    if (!name || name[0] == '\0') {
        return 0;
    }
    for (size_t i = 0; i < sizeof(dsl_wasm32_symbol_bindings) / sizeof(dsl_wasm32_symbol_bindings[0]); i++) {
        if (strcmp(name, dsl_wasm32_symbol_bindings[i].name) != 0) {
            continue;
        }
        return (int)(uintptr_t)dsl_wasm32_symbol_bindings[i].addr;
    }
    return 0;
}

static bool dsl_wasm32_register_required_symbols(TCCState *state, const char *src) {
    if (!state || !src) {
        return false;
    }
    for (size_t i = 0; i < sizeof(dsl_wasm32_symbol_bindings) / sizeof(dsl_wasm32_symbol_bindings[0]); i++) {
        if (!dsl_wasm32_source_calls_symbol(src, dsl_wasm32_symbol_bindings[i].name)) {
            continue;
        }
        if (tcc_add_symbol(state, dsl_wasm32_symbol_bindings[i].name, dsl_wasm32_symbol_bindings[i].addr) < 0) {
            dsl_tracef("jit runtime skip: tcc_add_symbol failed for '%s'", dsl_wasm32_symbol_bindings[i].name);
        }
    }
    return true;
}

#undef ME_WASM32_BRIDGE_SYM
#undef ME_WASM32_BRIDGE_CONTRACT_SYM

bool dsl_jit_compile_wasm32(me_dsl_compiled_program *program, uint64_t key) {
    if (!program || !program->jit_c_source) {
        return false;
    }
#if ME_WASM32_SIDE_MODULE
    if (!me_wasm_jit_helpers_available()) {
        dsl_tracef("jit runtime skip: side-module wasm32 helpers are not registered");
        return false;
    }
#endif
    char *patched_src = dsl_wasm32_patch_source(program->jit_c_source);
    if (!patched_src) {
        return false;
    }
    dsl_tracef("jit wasm32: source patched (%zu bytes)", strlen(patched_src));

    TCCState *state = tcc_new();
    if (!state) {
        free(patched_src);
        dsl_tracef("jit runtime skip: tcc_new failed");
        return false;
    }
    tcc_set_error_func(state, NULL, dsl_wasm_tcc_error_handler);
    tcc_set_options(state, "-nostdlib -nostdinc");

    if (tcc_set_output_type(state, TCC_OUTPUT_EXE) < 0) {
        tcc_delete(state);
        free(patched_src);
        dsl_tracef("jit runtime skip: tcc_set_output_type failed");
        return false;
    }

    void *jit_scratch = malloc(256 * 1024);
    if (!jit_scratch) {
        tcc_delete(state);
        free(patched_src);
        return false;
    }
    unsigned int jit_base = ((unsigned int)(uintptr_t)jit_scratch + 0xFFFFu) & ~0xFFFFu;
    tcc_set_wasm_data_base(state, jit_base);

    if (!dsl_wasm32_register_required_symbols(state, patched_src)) {
        tcc_delete(state);
        free(patched_src);
        free(jit_scratch);
        return false;
    }

    if (tcc_compile_string(state, patched_src) < 0) {
        dsl_tracef("jit runtime skip: tcc_compile_string failed");
        tcc_delete(state);
        free(patched_src);
        free(jit_scratch);
        return false;
    }
    free(patched_src);

    const char *wasm_path = "/tmp/me_jit_kernel.wasm";
    if (tcc_output_file(state, wasm_path) < 0) {
        tcc_delete(state);
        free(jit_scratch);
        dsl_tracef("jit runtime skip: tcc_output_file failed");
        return false;
    }
    tcc_delete(state);

    FILE *fp = fopen(wasm_path, "rb");
    if (!fp) {
        free(jit_scratch);
        dsl_tracef("jit runtime skip: cannot read wasm file");
        return false;
    }
    fseek(fp, 0, SEEK_END);
    long wasm_len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (wasm_len <= 0 || wasm_len > 1024 * 1024) {
        fclose(fp);
        free(jit_scratch);
        dsl_tracef("jit runtime skip: wasm file size %ld unexpected", wasm_len);
        return false;
    }
    unsigned char *wasm_bytes = (unsigned char *)malloc((size_t)wasm_len);
    if (!wasm_bytes) {
        fclose(fp);
        free(jit_scratch);
        return false;
    }
    if ((long)fread(wasm_bytes, 1, (size_t)wasm_len, fp) != wasm_len) {
        free(wasm_bytes);
        fclose(fp);
        free(jit_scratch);
        return false;
    }
    fclose(fp);
    remove(wasm_path);

    int fn_idx = me_wasm_jit_instantiate_dispatch(wasm_bytes, (int)wasm_len,
                                                  (int)(uintptr_t)&dsl_wasm32_lookup_bridge_symbol);
    free(wasm_bytes);
    if (fn_idx == 0) {
        free(jit_scratch);
        dsl_tracef("jit runtime skip: wasm instantiation failed");
        return false;
    }

    if (!dsl_jit_wasm_pos_cache_store_program(program, key, fn_idx, jit_scratch)) {
        program->jit_kernel_fn = (me_dsl_jit_kernel_fn)(uintptr_t)fn_idx;
        program->jit_dl_handle = jit_scratch;
        program->jit_runtime_key = key;
        program->jit_dl_handle_cached = false;
    }
    program->jit_c_error_line = 0;
    program->jit_c_error_column = 0;
    program->jit_c_error[0] = '\0';
    return true;
}

#else

bool dsl_jit_compile_wasm32(me_dsl_compiled_program *program, uint64_t key) {
    (void)program;
    (void)key;
    return false;
}

bool dsl_jit_wasm_pos_cache_bind_program(me_dsl_compiled_program *program, uint64_t key) {
    (void)program;
    (void)key;
    return false;
}

void dsl_wasm_jit_free_dispatch(int idx) {
    (void)idx;
}

void dsl_register_wasm_jit_helpers(me_wasm_jit_instantiate_helper instantiate_helper,
                                   me_wasm_jit_free_helper free_helper) {
    (void)instantiate_helper;
    (void)free_helper;
}

#endif
