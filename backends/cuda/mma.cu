// mma: what each tensor-core math format actually delivers, measured with the hardware's own MMA
// instructions and verified numerically. Spec: probes/mma/spec.md.
//
// WMMA formats (fp16, bf16, tf32, int8) load operands from global memory through the WMMA API, so
// their fragment layouts are handled by the compiler. fp8 e4m3 and fp4 e2m1 use hand-written PTX
// whose fragment layouts follow the SM80 int8 m16n8k32 encoding (fp4's k is counted in bytes, so
// one MMA covers 64 fp4 values); every format's result is checked element-wise against a host
// reference computed in exact arithmetic, so a mis-encoded operand layout fails loudly instead of
// posting a plausible throughput.
//
// fp4 needs the family-specific build target (compute_121a): with plain compute_121 the assembler
// rejects kind::f8f6f4, and the probe reports fp4 as not built rather than guessing.

#include "common/echolock.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <functional>
#include <map>
#include <mma.h>

using namespace echolock;
using namespace nvcuda;

namespace {

constexpr int kChains = 8; // independent accumulators per warp, so the MMA pipeline stays full

struct Options {
    double seconds      = 0.5; // target duration of one throughput series
    std::string formats = "fp16,bf16,tf32,int8,fp8,fp4";
    int blocks_per_sm   = 8;
};

Options parse(int argc, char** argv) {
    Args args(argc, argv,
              "usage: mma [--seconds S (0.5)] [--formats fp16,bf16,tf32,int8,fp8,fp4] "
              "[--blocks-per-sm N (8)]");
    Options o;
    for (std::string a; args.next(a);) {
        if (a == "--seconds") { o.seconds = std::stod(args.value(a)); }
        else if (a == "--formats") { o.formats = args.value(a); }
        else if (a == "--blocks-per-sm") { o.blocks_per_sm = std::stoi(args.value(a)); }
        else { args.usage_error("unknown argument " + a); }
    }
    if (o.seconds <= 0 || o.blocks_per_sm <= 0) { args.usage_error("values must be positive"); }
    return o;
}

bool wants(const Options& o, const std::string& f) {
    return o.formats.find(f) != std::string::npos;
}

// ---- exact operand patterns ---------------------------------------------------------------------
// Small values, exactly representable in every format from fp4 up, so the host reference is exact:
// fp4 grid {0, 0.5, 1, 1.5, 2}, fp8 adds nothing, fp16/bf16/tf32 take small integers.
float a_pattern(int m, int k) { return 0.5f * static_cast<float>((m * 13 + k * 7) % 5); }  // [0, 2]
float b_pattern(int k, int n) { return 0.5f * static_cast<float>((k * 11 + n * 5) % 5); }  // [0, 2]

// ---- host reference -----------------------------------------------------------------------------

// C[m][n] = scale * sum_k A[m][k] * B[k][n], in double (the values are exact in every format).
std::vector<double> reference(int m_dim, int n_dim, int k_dim, double scale,
                              float (*a_val)(int, int), float (*b_val)(int, int)) {
    std::vector<double> c(std::size_t(m_dim) * n_dim, 0.0);
    for (int m = 0; m < m_dim; ++m) {
        for (int n = 0; n < n_dim; ++n) {
            double acc = 0.0;
            for (int k = 0; k < k_dim; ++k) { acc += double(a_val(m, k)) * double(b_val(k, n)); }
            c[std::size_t(m) * n_dim + n] = scale * acc;
        }
    }
    return c;
}

// Compares a stored matrix against the reference; returns the first mismatch for the check detail.
bool matrix_ok(const float* got, const std::vector<double>& want, Obj* detail) {
    for (std::size_t i = 0; i < want.size(); ++i) {
        if (double(got[i]) != want[i]) {
            detail->uint("first_bad_index", i).num("got", got[i]).num("expected", want[i]);
            return false;
        }
    }
    return true;
}

bool matrix_ok_int(const int* got, const std::vector<double>& want, Obj* detail) {
    for (std::size_t i = 0; i < want.size(); ++i) {
        if (double(got[i]) != want[i]) {
            detail->uint("first_bad_index", i).integer("got", got[i]).num("expected", want[i]);
            return false;
        }
    }
    return true;
}

// ---- WMMA kernels ---------------------------------------------------------------------------------

template <typename FragA, typename FragB, typename FragAcc, typename ElemT, typename AccT>
__global__ void wmma_verify_kernel(const ElemT* A, const ElemT* B, AccT* D, unsigned ldm_a,
                                   unsigned ldm_b) {
    FragA a;
    FragB b;
    wmma::load_matrix_sync(a, A, ldm_a);
    wmma::load_matrix_sync(b, B, ldm_b);
    FragAcc acc;
    wmma::fill_fragment(acc, AccT(0));
    wmma::mma_sync(acc, a, b, acc);
    if (blockIdx.x == 0 && threadIdx.y == 0) {
        wmma::store_matrix_sync(D, acc, 16, wmma::mem_row_major);
    }
}

template <typename FragA, typename FragB, typename FragAcc, typename ElemT, typename AccT>
__global__ void wmma_throughput_kernel(const ElemT* A, const ElemT* B, AccT* out, unsigned ldm_a,
                                       unsigned ldm_b, int iters) {
    FragA a;
    FragB b;
    wmma::load_matrix_sync(a, A, ldm_a);
    wmma::load_matrix_sync(b, B, ldm_b);
    FragAcc acc[kChains];
    for (int j = 0; j < kChains; ++j) { wmma::fill_fragment(acc[j], AccT(0)); }
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < kChains; ++j) { wmma::mma_sync(acc[j], a, b, acc[j]); }
    }
    const std::size_t warp = std::size_t(blockIdx.x) * blockDim.y + threadIdx.y;
    for (int j = 0; j < kChains; ++j) {
        wmma::store_matrix_sync(out + (warp * kChains + std::size_t(j)) * 256, acc[j], 16,
                                wmma::mem_row_major);
    }
}

// ---- PTX mma (bf16, fp8 e4m3, fp4 e2m1) ---------------------------------------------------------
// WMMA covers fp16/tf32/int8; bf16, fp8 and fp4 go through hand-written PTX. Every fragment
// layout below follows the SM80 encodings (verified against CUTLASS's hardware-validated
// mma_traits): fp16/bf16 m16n8k16 and fp8/fp4 m16n8k32 share the same per-thread structure,
// fp4 counting k in bytes so one MMA covers 64 fp4 values. Each format's full result matrix is
// checked against an exact host reference, so a wrong layout fails loudly, never silently.

__device__ __forceinline__ void mma_bf16(float* d, const unsigned* a, const unsigned* b,
                                         const float* c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void mma_fp8(float* d, const unsigned* a, const unsigned* b,
                                        const float* c) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void mma_fp4(float* d, const unsigned* a, const unsigned* b,
                                        const float* c) {
#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ >= 1210
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#else
    (void)a; (void)b;  // not built for this target: the host checks kFp4Built and skips the section
    d[0] = c[0]; d[1] = c[1]; d[2] = c[2]; d[3] = c[3];
#endif
}

// Set at compile time so the host knows whether the fp4 instruction exists in this binary.
__device__ __constant__ int kFp4Built =
#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ >= 1210
    1;
#else
    0;
#endif

struct PtxFrags {
    unsigned a[4], b[2];
};

__device__ __forceinline__ unsigned pack4(const unsigned char* base, std::size_t row_bytes,
                                          int k_byte) {
    const unsigned char* p = base + row_bytes + std::size_t(k_byte);
    return static_cast<unsigned>(p[0]) | (static_cast<unsigned>(p[1]) << 8) |
           (static_cast<unsigned>(p[2]) << 16) | (static_cast<unsigned>(p[3]) << 24);
}

__device__ __forceinline__ unsigned bf16_pair(const __nv_bfloat16* p) {
    // Two adjacent bf16 values as one u32 (x = low half).
    __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(p);
    return *reinterpret_cast<const unsigned*>(&v);
}

__device__ __forceinline__ unsigned bf16_pair2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    __nv_bfloat162 v;
    v.x = lo; // x = low half = the first (lower-k) element
    v.y = hi;
    return *reinterpret_cast<const unsigned*>(&v);
}

// fp8/fp4 fill (m16n8k32). A is 16 rows x 32 bytes (fp8: 32 values; fp4: 64 values, 2 per byte),
// B is 32 rows x 8 bytes. Layout per the SM80 int8 m16n8k32 encoding:
//   a0 = A[g][kb 4t..+3]      a1 = A[g+8][kb 4t..+3]
//   a2 = A[g][kb 4t+16..+19]  a3 = A[g+8][kb 4t+16..+19]
//   b0 = B[kb 4t..+3][g]      b1 = B[kb 4t+16..+19][g]
__device__ __forceinline__ PtxFrags fill_frags(const unsigned char* A, const unsigned char* B) {
    const int lane = threadIdx.x;
    const int g = lane >> 2, t = lane & 3;
    PtxFrags f;
    f.a[0] = pack4(A, std::size_t(g) * 32, 4 * t);
    f.a[1] = pack4(A, std::size_t(g + 8) * 32, 4 * t);
    f.a[2] = pack4(A, std::size_t(g) * 32, 4 * t + 16);
    f.a[3] = pack4(A, std::size_t(g + 8) * 32, 4 * t + 16);
    f.b[0] = pack4(B + std::size_t(g), 8, 4 * t); // row-major B: row k_byte, col g
    f.b[1] = pack4(B + std::size_t(g), 8, 4 * t + 16);
    return f;
}

// bf16 fill (m16n8k16). A is 16x16 bf16 row-major; B is 16x8 (k x n) row-major.
//   a0 = A[g][2t..+1]      a1 = A[g+8][2t..+1]
//   a2 = A[g][2t+8..+9]    a3 = A[g+8][2t+8..+9]
//   b0 = B[2t..+1][g]      b1 = B[2t+8..+9][g]
__device__ __forceinline__ PtxFrags fill_frags_bf16(const unsigned char* A_bytes,
                                                    const unsigned char* B_bytes) {
    const auto* A = reinterpret_cast<const __nv_bfloat16*>(A_bytes);
    const auto* B = reinterpret_cast<const __nv_bfloat16*>(B_bytes);
    const int lane = threadIdx.x;
    const int g = lane >> 2, t = lane & 3;
    PtxFrags f;
    f.a[0] = bf16_pair(A + std::size_t(g) * 16 + 2 * t);
    f.a[1] = bf16_pair(A + std::size_t(g + 8) * 16 + 2 * t);
    f.a[2] = bf16_pair(A + std::size_t(g) * 16 + 2 * t + 8);
    f.a[3] = bf16_pair(A + std::size_t(g + 8) * 16 + 2 * t + 8);
    f.b[0] = bf16_pair2(B[std::size_t(2 * t) * 8 + g], B[std::size_t(2 * t + 1) * 8 + g]);
    f.b[1] = bf16_pair2(B[std::size_t(2 * t + 8) * 8 + g], B[std::size_t(2 * t + 9) * 8 + g]);
    return f;
}

__device__ __forceinline__ void store_d(float* out, const float* d) {
    const int lane = threadIdx.x;
    const int g = lane >> 2, t = lane & 3;
    // D 16x8: c0 = C[g][2t], c1 = C[g+8][2t], c2 = C[g][2t+1], c3 = C[g+8][2t+1]
    out[std::size_t(g) * 8 + 2 * t]     = d[0];
    out[std::size_t(g + 8) * 8 + 2 * t] = d[1];
    out[std::size_t(g) * 8 + 2 * t + 1] = d[2];
    out[std::size_t(g + 8) * 8 + 2 * t + 1] = d[3];
}

template <PtxFrags (*Fill)(const unsigned char*, const unsigned char*),
          void (*MMA)(float*, const unsigned*, const unsigned*, const float*)>
__global__ void ptx_verify_kernel(const unsigned char* A, const unsigned char* B, float* D) {
    const PtxFrags f = Fill(A, B);
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    MMA(d, f.a, f.b, d);
    if (blockIdx.x == 0 && threadIdx.y == 0) { store_d(D, d); }
}

template <PtxFrags (*Fill)(const unsigned char*, const unsigned char*),
          void (*MMA)(float*, const unsigned*, const unsigned*, const float*)>
__global__ void ptx_throughput_kernel(const unsigned char* A, const unsigned char* B, float* out,
                                      int iters) {
    const PtxFrags f = Fill(A, B);
    float d[kChains][4];
    for (int j = 0; j < kChains; ++j) {
        for (int i = 0; i < 4; ++i) { d[j][i] = 0.f; }
    }
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < kChains; ++j) { MMA(d[j], f.a, f.b, d[j]); }
    }
    const std::size_t warp = std::size_t(blockIdx.x) * blockDim.y + threadIdx.y;
    float* base = out + warp * std::size_t(kChains) * 128;
    for (int j = 0; j < kChains; ++j) { store_d(base + std::size_t(j) * 128, d[j]); }
}

// ---- host drivers ---------------------------------------------------------------------------------

struct SeriesResult {
    double tflops = 0;
    bool verified = false;
};

// Times `reps` launches of `launch(iters)` and checks the whole grid's stored result every time.
SeriesResult timed_series(cudaStream_t stream, const std::function<void(int)>& launch,
                          const std::function<bool(int, Obj*)>& verify, double seconds,
                          double flops_per_iter, const Obj& tags, const std::string& name) {
    cudaEvent_t t0, t1;
    EL_CUDA(cudaEventCreate(&t0));
    EL_CUDA(cudaEventCreate(&t1));
    launch(4096); // warm-up, also sizes the series
    EL_CUDA(cudaStreamSynchronize(stream));
    EL_CUDA(cudaEventRecord(t0, stream));
    launch(4096);
    EL_CUDA(cudaEventRecord(t1, stream));
    EL_CUDA(cudaEventSynchronize(t1));
    float warm_ms = 0;
    EL_CUDA(cudaEventElapsedTime(&warm_ms, t0, t1));
    const int iters = std::clamp(static_cast<int>(seconds * 1000.0 / std::max(warm_ms, 0.001f)),
                                 4096, 4000000);

    std::vector<double> per;
    int bad = 0;
    Obj first_bad;
    const int reps = 5;
    for (int r = 0; r < reps; ++r) {
        EL_CUDA(cudaEventRecord(t0, stream));
        launch(iters);
        EL_CUDA(cudaEventRecord(t1, stream));
        EL_CUDA(cudaEventSynchronize(t1));
        EL_CUDA(cudaGetLastError());
        float ms = 0;
        EL_CUDA(cudaEventElapsedTime(&ms, t0, t1));
        Obj detail;
        const bool ok = verify(iters, &detail);
        bad += ok ? 0 : 1;
        if (!ok && bad == 1) {
            first_bad = detail;          // the first failing launch's matrices, for the check detail
        }
        detail.integer("iters", iters).num("kernel_ms", ms);
        const double tflops = flops_per_iter * iters / (double(ms) * 1e-3) / 1e12;
        per.push_back(tflops);
        measure(name, tflops, "TFLOPS", tags, ok, r);
    }
    EL_CUDA(cudaEventDestroy(t0));
    EL_CUDA(cudaEventDestroy(t1));
    const Summary s = summarize(per);
    const bool ok = bad == 0;
    measure(name, s.p50, "TFLOPS", Obj(tags).obj("per_rep", summary_obj(s)), ok);
    first_bad.integer("reps", reps).integer("bad_launches", bad);
    check(name + ".result", ok, first_bad);
    return {s.p50, ok};
}

} // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    probe_start("mma", Obj().num("seconds", o.seconds).str("formats", o.formats)
                          .integer("chains", kChains).integer("blocks_per_sm", o.blocks_per_sm));
    EL_CUDA(cudaSetDevice(0));
    declare_device(0);
    cudaStream_t stream;
    EL_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    declare_compiled_arch(stream);

    const int sms     = runtime_attr(cudaDevAttrMultiProcessorCount, 0);
    const dim3 grid(sms * o.blocks_per_sm), block(32, 8);
    const std::size_t warps = std::size_t(grid.x) * block.y;
    std::map<std::string, double> tflops;

    // ---- fp16 (WMMA 16x16x16) -------------------------------------------------------------------
    if (wants(o, "fp16")) {
        const std::string fmt = "fp16";
        half A[256], B[256];
        for (int m = 0; m < 16; ++m) {
            for (int k = 0; k < 16; ++k) { A[m * 16 + k] = __float2half(a_pattern(m, k)); }
        }
        for (int k = 0; k < 16; ++k) {
            for (int n = 0; n < 16; ++n) { B[k * 16 + n] = __float2half(b_pattern(k, n)); }
        }
        half *dA = nullptr, *dB = nullptr;
        float* dD = nullptr;
        EL_CUDA(cudaMalloc(&dA, sizeof(A)));
        EL_CUDA(cudaMalloc(&dB, sizeof(B)));
        EL_CUDA(cudaMalloc(&dD, 256 * sizeof(float)));
        EL_CUDA(cudaMemcpy(dA, A, sizeof(A), cudaMemcpyHostToDevice));
        EL_CUDA(cudaMemcpy(dB, B, sizeof(B), cudaMemcpyHostToDevice));
        wmma_verify_kernel<wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>,
                           wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>,
                           wmma::fragment<wmma::accumulator, 16, 16, 16, float>, half, float>
            <<<1, block, 0, stream>>>(dA, dB, dD, 16, 16);
        EL_CUDA(cudaStreamSynchronize(stream));
        std::vector<float> got(256);
        EL_CUDA(cudaMemcpy(got.data(), dD, 256 * sizeof(float), cudaMemcpyDeviceToHost));
        Obj detail;
        check(fmt + ".matrix", matrix_ok(got.data(), reference(16, 16, 16, 1.0, a_pattern,
                                                                b_pattern), &detail),
              detail);

        float* out = nullptr;
        EL_CUDA(cudaMalloc(&out, warps * kChains * 256 * sizeof(float)));
        const auto ref = reference(16, 16, 16, 1.0, a_pattern, b_pattern);
        const auto launch = [&](int iters) {
            wmma_throughput_kernel<wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>,
                                   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>,
                                   wmma::fragment<wmma::accumulator, 16, 16, 16, float>, half,
                                   float>
                <<<grid, block, 0, stream>>>(dA, dB, out, 16, 16, iters);
        };
        const auto verify = [&](int iters, Obj* d) {
            std::vector<float> all(warps * kChains * 256);
            EL_CUDA(cudaMemcpy(all.data(), out, all.size() * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<double> want = reference(16, 16, 16, double(iters), a_pattern, b_pattern);
            // The host reference of one chain; every warp and chain must match it element-wise.
            for (std::size_t w = 0; w < warps * kChains; ++w) {
                if (!matrix_ok(all.data() + w * 256, want, d)) { return false; }
            }
            return true;
        };
        const SeriesResult s = timed_series(
            stream, launch, verify, o.seconds,
            double(warps) * kChains * 2.0 * 16 * 16 * 16,
            Obj().str("format", fmt).str("shape", "m16n16k16"), "mma_throughput");
        tflops[fmt] = s.tflops;
        EL_CUDA(cudaFree(out));
        EL_CUDA(cudaFree(dA));
        EL_CUDA(cudaFree(dB));
        EL_CUDA(cudaFree(dD));
    }

    // ---- bf16 (PTX m16n8k16; WMMA has no bf16 fragments) --------------------------------------
    if (wants(o, "bf16")) {
        const std::string fmt = "bf16";
        // A: 16x16 bf16 row-major; B: 16x8 bf16 (k x n) row-major.
        std::vector<__nv_bfloat16> A(256), B(128);
        for (int m = 0; m < 16; ++m) {
            for (int k = 0; k < 16; ++k) { A[std::size_t(m) * 16 + k] = __float2bfloat16(a_pattern(m, k)); }
        }
        for (int k = 0; k < 16; ++k) {
            for (int n = 0; n < 8; ++n) { B[std::size_t(k) * 8 + n] = __float2bfloat16(b_pattern(k, n)); }
        }
        __nv_bfloat16 *dA = nullptr, *dB = nullptr;
        float* dD = nullptr;
        EL_CUDA(cudaMalloc(&dA, A.size() * sizeof(__nv_bfloat16)));
        EL_CUDA(cudaMalloc(&dB, B.size() * sizeof(__nv_bfloat16)));
        EL_CUDA(cudaMalloc(&dD, 128 * sizeof(float)));
        EL_CUDA(cudaMemcpy(dA, A.data(), A.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        EL_CUDA(cudaMemcpy(dB, B.data(), B.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        ptx_verify_kernel<fill_frags_bf16, mma_bf16><<<1, block, 0, stream>>>(
            reinterpret_cast<const unsigned char*>(dA),
            reinterpret_cast<const unsigned char*>(dB), dD);
        EL_CUDA(cudaStreamSynchronize(stream));
        std::vector<float> got(128);
        EL_CUDA(cudaMemcpy(got.data(), dD, 128 * sizeof(float), cudaMemcpyDeviceToHost));
        Obj detail;
        check(fmt + ".matrix",
              matrix_ok(got.data(), reference(16, 8, 16, 1.0, a_pattern, b_pattern), &detail),
              detail);

        float* out = nullptr;
        EL_CUDA(cudaMalloc(&out, warps * kChains * 128 * sizeof(float)));
        const auto launch = [&](int iters) {
            ptx_throughput_kernel<fill_frags_bf16, mma_bf16><<<grid, block, 0, stream>>>(
                reinterpret_cast<const unsigned char*>(dA),
                reinterpret_cast<const unsigned char*>(dB), out, iters);
        };
        const auto verify = [&](int iters, Obj* d) {
            std::vector<float> all(warps * kChains * 128);
            EL_CUDA(cudaMemcpy(all.data(), out, all.size() * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<double> want = reference(16, 8, 16, double(iters), a_pattern, b_pattern);
            for (std::size_t w = 0; w < warps * kChains; ++w) {
                if (!matrix_ok(all.data() + w * 128, want, d)) { return false; }
            }
            return true;
        };
        const SeriesResult s = timed_series(
            stream, launch, verify, o.seconds,
            double(warps) * kChains * 2.0 * 16 * 8 * 16,
            Obj().str("format", fmt).str("shape", "m16n8k16"), "mma_throughput");
        tflops[fmt] = s.tflops;
        EL_CUDA(cudaFree(out));
        EL_CUDA(cudaFree(dA));
        EL_CUDA(cudaFree(dB));
        EL_CUDA(cudaFree(dD));
    }

    // ---- tf32 (WMMA 16x16x8) --------------------------------------------------------------------
    if (wants(o, "tf32")) {
        const std::string fmt = "tf32";
        float A[128], B[128]; // A: 16x8, B: 8x16
        for (int m = 0; m < 16; ++m) {
            for (int k = 0; k < 8; ++k) { A[m * 8 + k] = a_pattern(m, k); }
        }
        for (int k = 0; k < 8; ++k) {
            for (int n = 0; n < 16; ++n) { B[k * 16 + n] = b_pattern(k, n); }
        }
        float *dA = nullptr, *dB = nullptr, *dD = nullptr;
        EL_CUDA(cudaMalloc(&dA, sizeof(A)));
        EL_CUDA(cudaMalloc(&dB, sizeof(B)));
        EL_CUDA(cudaMalloc(&dD, 256 * sizeof(float)));
        EL_CUDA(cudaMemcpy(dA, A, sizeof(A), cudaMemcpyHostToDevice));
        EL_CUDA(cudaMemcpy(dB, B, sizeof(B), cudaMemcpyHostToDevice));
        wmma_verify_kernel<wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major>,
                           wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major>,
                           wmma::fragment<wmma::accumulator, 16, 16, 8, float>, float, float>
            <<<1, block, 0, stream>>>(dA, dB, dD, 8, 16);
        EL_CUDA(cudaStreamSynchronize(stream));
        std::vector<float> got(256);
        EL_CUDA(cudaMemcpy(got.data(), dD, 256 * sizeof(float), cudaMemcpyDeviceToHost));
        Obj detail;
        check(fmt + ".matrix",
              matrix_ok(got.data(), reference(16, 16, 8, 1.0, a_pattern, b_pattern), &detail),
              detail);

        float* out = nullptr;
        EL_CUDA(cudaMalloc(&out, warps * kChains * 256 * sizeof(float)));
        const auto launch = [&](int iters) {
            wmma_throughput_kernel<wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major>,
                                   wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major>,
                                   wmma::fragment<wmma::accumulator, 16, 16, 8, float>, float,
                                   float>
                <<<grid, block, 0, stream>>>(dA, dB, out, 8, 16, iters);
        };
        const auto verify = [&](int iters, Obj* d) {
            std::vector<float> all(warps * kChains * 256);
            EL_CUDA(cudaMemcpy(all.data(), out, all.size() * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<double> want = reference(16, 16, 8, double(iters), a_pattern, b_pattern);
            for (std::size_t w = 0; w < warps * kChains; ++w) {
                if (!matrix_ok(all.data() + w * 256, want, d)) { return false; }
            }
            return true;
        };
        const SeriesResult s = timed_series(
            stream, launch, verify, o.seconds,
            double(warps) * kChains * 2.0 * 16 * 16 * 8,
            Obj().str("format", fmt).str("shape", "m16n16k8"), "mma_throughput");
        tflops[fmt] = s.tflops;
        EL_CUDA(cudaFree(out));
        EL_CUDA(cudaFree(dA));
        EL_CUDA(cudaFree(dB));
        EL_CUDA(cudaFree(dD));
    }

    // ---- int8 (WMMA 16x16x16) -------------------------------------------------------------------
    if (wants(o, "int8")) {
        const std::string fmt = "int8";
        signed char A[256], B[256];
        for (int m = 0; m < 16; ++m) {
            for (int k = 0; k < 16; ++k) {
                A[m * 16 + k] = static_cast<signed char>(int(a_pattern(m, k) * 2) - 3); // [-3, 1]
            }
        }
        for (int k = 0; k < 16; ++k) {
            for (int n = 0; n < 16; ++n) {
                B[k * 16 + n] = static_cast<signed char>(int(b_pattern(k, n) * 2) - 2); // [-2, 2]
            }
        }
        signed char *dA = nullptr, *dB = nullptr;
        int* dD = nullptr;
        EL_CUDA(cudaMalloc(&dA, sizeof(A)));
        EL_CUDA(cudaMalloc(&dB, sizeof(B)));
        EL_CUDA(cudaMalloc(&dD, 256 * sizeof(int)));
        EL_CUDA(cudaMemcpy(dA, A, sizeof(A), cudaMemcpyHostToDevice));
        EL_CUDA(cudaMemcpy(dB, B, sizeof(B), cudaMemcpyHostToDevice));
        wmma_verify_kernel<wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major>,
                           wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major>,
                           wmma::fragment<wmma::accumulator, 16, 16, 16, int>, signed char, int>
            <<<1, block, 0, stream>>>(dA, dB, dD, 16, 16);
        EL_CUDA(cudaStreamSynchronize(stream));
        std::vector<int> got(256);
        EL_CUDA(cudaMemcpy(got.data(), dD, 256 * sizeof(int), cudaMemcpyDeviceToHost));
        Obj detail;
        check(fmt + ".matrix",
              matrix_ok_int(got.data(),
                            reference(16, 16, 16, 1.0,
                                      [](int m, int k) { return float(int(a_pattern(m, k) * 2) - 3); },
                                      [](int k, int n) { return float(int(b_pattern(k, n) * 2) - 2); }),
                            &detail),
              detail);

        int* out = nullptr;
        EL_CUDA(cudaMalloc(&out, warps * kChains * 256 * sizeof(int)));
        const auto launch = [&](int iters) {
            wmma_throughput_kernel<wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major>,
                                   wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major>,
                                   wmma::fragment<wmma::accumulator, 16, 16, 16, int>, signed char,
                                   int>
                <<<grid, block, 0, stream>>>(dA, dB, out, 16, 16, iters);
        };
        const auto verify = [&](int iters, Obj* d) {
            std::vector<int> all(warps * kChains * 256);
            EL_CUDA(cudaMemcpy(all.data(), out, all.size() * sizeof(int), cudaMemcpyDeviceToHost));
            std::vector<double> want =
                reference(16, 16, 16, double(iters),
                          [](int m, int k) { return float(int(a_pattern(m, k) * 2) - 3); },
                          [](int k, int n) { return float(int(b_pattern(k, n) * 2) - 2); });
            for (std::size_t w = 0; w < warps * kChains; ++w) {
                if (!matrix_ok_int(all.data() + w * 256, want, d)) { return false; }
            }
            return true;
        };
        const SeriesResult s = timed_series(
            stream, launch, verify, o.seconds,
            double(warps) * kChains * 2.0 * 16 * 16 * 16,
            Obj().str("format", fmt).str("shape", "m16n16k16"), "mma_throughput");
        tflops[fmt] = s.tflops;
        EL_CUDA(cudaFree(out));
        EL_CUDA(cudaFree(dA));
        EL_CUDA(cudaFree(dB));
        EL_CUDA(cudaFree(dD));
    }

    // ---- fp8 e4m3 and fp4 e2m1 (PTX m16n8k32) -----------------------------------------------------
    const auto run_ptx = [&](const std::string& fmt, int k_elements) {
        if (fmt == "fp4") {
            int built = 0;
            EL_CUDA(cudaMemcpyFromSymbol(&built, kFp4Built, sizeof(int)));
            if (!built) {
                note("fp4 not built: compile with the compute_121a family target "
                     "(make ARCH=sm_121a uses it); plain compute_121 has no kind::f8f6f4");
                return;
            }
        }
        // A: 16 rows x 32 bytes; B: 32 rows x 8 bytes. fp8 packs 1 value/byte, fp4 packs 2.
        std::vector<unsigned char> A(16 * 32), B(32 * 8);
        for (int m = 0; m < 16; ++m) {
            for (int e = 0; e < k_elements; ++e) {
                const float v = a_pattern(m, e);
                if (fmt == "fp8") {
                    A[std::size_t(m) * 32 + e] =
                        static_cast<unsigned char>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
                } else {
                    const unsigned char nib =
                        v == 0.0f ? 0u : v == 0.5f ? 1u : v == 1.0f ? 2u : v == 1.5f ? 3u : 4u;
                    A[std::size_t(m) * 32 + e / 2] |=
                        static_cast<unsigned char>(nib << ((e % 2) * 4)); // even element: low nibble
                }
            }
        }
        for (int e = 0; e < k_elements; ++e) {
            for (int n = 0; n < 8; ++n) {
                const float v = b_pattern(e, n);
                if (fmt == "fp8") {
                    B[std::size_t(e) * 8 + n] =
                        static_cast<unsigned char>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
                } else {
                    const unsigned char nib =
                        v == 0.0f ? 0u : v == 0.5f ? 1u : v == 1.0f ? 2u : v == 1.5f ? 3u : 4u;
                    // fp4 packs 2 values per byte along k: element e lives in byte e/2.
                    B[(e / 2) * 8 + n] |= static_cast<unsigned char>(nib << ((e % 2) * 4));
                }
            }
        }
        unsigned char *dA = nullptr, *dB = nullptr;
        float* dD = nullptr;
        EL_CUDA(cudaMalloc(&dA, A.size()));
        EL_CUDA(cudaMalloc(&dB, B.size()));
        EL_CUDA(cudaMalloc(&dD, 128 * sizeof(float)));
        EL_CUDA(cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice));
        EL_CUDA(cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice));
        if (fmt == "fp8") {
            ptx_verify_kernel<fill_frags, mma_fp8><<<1, block, 0, stream>>>(dA, dB, dD);
        } else {
            ptx_verify_kernel<fill_frags, mma_fp4><<<1, block, 0, stream>>>(dA, dB, dD);
        }
        EL_CUDA(cudaStreamSynchronize(stream));
        std::vector<float> got(128);
        EL_CUDA(cudaMemcpy(got.data(), dD, 128 * sizeof(float), cudaMemcpyDeviceToHost));
        Obj detail;
        check(fmt + ".matrix",
              matrix_ok(got.data(), reference(16, 8, k_elements, 1.0, a_pattern, b_pattern),
                        &detail),
              detail);

        float* out = nullptr;
        EL_CUDA(cudaMalloc(&out, warps * kChains * 128 * sizeof(float)));
        const auto launch = [&](int iters) {
            if (fmt == "fp8") {
                ptx_throughput_kernel<fill_frags, mma_fp8><<<grid, block, 0, stream>>>(dA, dB, out,
                                                                                    iters);
            } else {
                ptx_throughput_kernel<fill_frags, mma_fp4><<<grid, block, 0, stream>>>(dA, dB, out,
                                                                                    iters);
            }
        };
        const auto verify = [&](int iters, Obj* d) {
            std::vector<float> all(warps * kChains * 128);
            EL_CUDA(cudaMemcpy(all.data(), out, all.size() * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<double> want =
                reference(16, 8, k_elements, double(iters), a_pattern, b_pattern);
            for (std::size_t w = 0; w < warps * kChains; ++w) {
                if (!matrix_ok(all.data() + w * 128, want, d)) { return false; }
            }
            return true;
        };
        const SeriesResult s = timed_series(
            stream, launch, verify, o.seconds,
            double(warps) * kChains * 2.0 * 16 * 8 * k_elements,
            Obj().str("format", fmt).str("shape", fmt == "fp8" ? "m16n8k32" : "m16n8k64"),
            "mma_throughput");
        tflops[fmt] = s.tflops;
        EL_CUDA(cudaFree(out));
        EL_CUDA(cudaFree(dA));
        EL_CUDA(cudaFree(dB));
        EL_CUDA(cudaFree(dD));
    };

    if (wants(o, "fp8")) { run_ptx("fp8", 32); }
    if (wants(o, "fp4")) { run_ptx("fp4", 64); }

    // fp16-relative throughput per format: fp8 at half of fp16 reads as 2.0.
    if (tflops.count("fp16") && tflops["fp16"] > 0) {
        for (const auto& [fmt, t] : tflops) {
            measure("format_ratio", t / tflops["fp16"], "x", Obj().str("format", fmt), true);
        }
    }

    EL_CUDA(cudaStreamDestroy(stream));
    return probe_end();
}
