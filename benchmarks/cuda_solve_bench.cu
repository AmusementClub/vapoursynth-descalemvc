// Microbenchmark of banded LDL^T solve kernels on real dsmvc plans.
// Build: nvcc -O3 -std=c++17 -arch=sm_120 -o cuda_solve_bench cuda_solve_bench.cu
// Plans: see cuda_solve_plans.cpp. Run: cuda_solve_bench <file.plan> <vectors> ...
// K0 copies the shipped kernel; K1 batches its loads (same arithmetic);
// K2 is the exact partitioned solve (chunks + carried state).
// Data is column-major (element i of vector v at i*V+v), as in
// dsmvc_cuda_solve_vertical / solve_horizontal_column_major.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <type_traits>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1);} } while (0)

__device__ __forceinline__ uint32_t minu(uint32_t a, uint32_t b) { return a < b ? a : b; }

// ---- K0: verbatim copy of inverse_axis_ring<W, UA, Precomputed=true> ----
template <uint32_t W, bool UA>
__global__ void k0_baseline(uint32_t V, uint32_t size, const float *__restrict__ lower_ld,
                            const float *__restrict__ upper_l, const float *__restrict__ inverse_diagonal,
                            float *__restrict__ data) {
    const uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    float *output = data + v; const uint32_t output_stride = V;
    float recent[W]{};
    for (uint32_t index = 0U; index < size; ++index) {
        float sum = output[index * output_stride];
        const uint32_t available = minu(W, index);
#pragma unroll
        for (uint32_t distance = W; distance > 0U; --distance)
            if (distance <= available)
                sum = fmaf(-lower_ld[(distance - 1U) * size + index], recent[distance - 1U], sum);
        const float current = sum * inverse_diagonal[index];
        output[index * output_stride] = current;
#pragma unroll
        for (uint32_t d = W - 1U; d > 0U; --d) recent[d] = recent[d - 1U];
        recent[0] = current;
    }
    if (size < 2U) return;
    recent[0] = output[(size - 1U) * output_stride];
#pragma unroll
    for (uint32_t d = 1U; d < W; ++d) recent[d] = 0.0F;
    for (uint32_t reverse = size - 1U; reverse > 0U; --reverse) {
        const uint32_t index = reverse - 1U;
        const uint32_t available = minu(W, size - index - 1U);
        float sum = 0.0F;
        if constexpr (UA) {
#pragma unroll
            for (uint32_t distance = 1U; distance <= W; ++distance)
                if (distance <= available)
                    sum = fmaf(upper_l[(distance - 1U) * size + index], recent[distance - 1U], sum);
        } else {
#pragma unroll
            for (uint32_t distance = W; distance > 0U; --distance)
                if (distance <= available)
                    sum = fmaf(upper_l[(distance - 1U) * size + index], recent[distance - 1U], sum);
        }
        const float current = output[index * output_stride] - sum;
        output[index * output_stride] = current;
#pragma unroll
        for (uint32_t d = W - 1U; d > 0U; --d) recent[d] = recent[d - 1U];
        recent[0] = current;
    }
}

// ---- K1: same arithmetic, loads batched U steps ahead (bit-exact) ----
template <uint32_t W, bool UA, uint32_t U>
__global__ void k1_prefetch(uint32_t V, uint32_t size, const float *__restrict__ lower_ld,
                            const float *__restrict__ upper_l, const float *__restrict__ inverse_diagonal,
                            float *__restrict__ data) {
    const uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    float *output = data + v;
    float recent[W]{};
    for (uint32_t base = 0U; base < size; base += U) {
        float vals[U], inv[U], coef[U][W];
#pragma unroll
        for (uint32_t j = 0; j < U; ++j) {
            const uint32_t index = base + j;
            if (index < size) {
                vals[j] = output[index * V];
                inv[j] = inverse_diagonal[index];
#pragma unroll
                for (uint32_t d = 0; d < W; ++d) coef[j][d] = lower_ld[d * size + index];
            }
        }
#pragma unroll
        for (uint32_t j = 0; j < U; ++j) {
            const uint32_t index = base + j;
            if (index < size) {
                float sum = vals[j];
                const uint32_t available = minu(W, index);
#pragma unroll
                for (uint32_t distance = W; distance > 0U; --distance)
                    if (distance <= available)
                        sum = fmaf(-coef[j][distance - 1U], recent[distance - 1U], sum);
                const float current = sum * inv[j];
                output[index * V] = current;
#pragma unroll
                for (uint32_t d = W - 1U; d > 0U; --d) recent[d] = recent[d - 1U];
                recent[0] = current;
            }
        }
    }
    if (size < 2U) return;
    // recent[0] already equals output[size-1] (same value the baseline reloads).
#pragma unroll
    for (uint32_t d = 1U; d < W; ++d) recent[d] = 0.0F;
    for (uint32_t reverse = size - 1U; reverse > 0U;) {
        const uint32_t count = minu(U, reverse);
        float vals[U], coef[U][W];
#pragma unroll
        for (uint32_t j = 0; j < U; ++j) {
            if (j < count) {
                const uint32_t index = reverse - 1U - j;
                vals[j] = output[index * V];
#pragma unroll
                for (uint32_t d = 0; d < W; ++d) coef[j][d] = upper_l[d * size + index];
            }
        }
#pragma unroll
        for (uint32_t j = 0; j < U; ++j) {
            if (j < count) {
                const uint32_t index = reverse - 1U - j;
                const uint32_t available = minu(W, size - index - 1U);
                float sum = 0.0F;
                if constexpr (UA) {
#pragma unroll
                    for (uint32_t distance = 1U; distance <= W; ++distance)
                        if (distance <= available)
                            sum = fmaf(coef[j][distance - 1U], recent[distance - 1U], sum);
                } else {
#pragma unroll
                    for (uint32_t distance = W; distance > 0U; --distance)
                        if (distance <= available)
                            sum = fmaf(coef[j][distance - 1U], recent[distance - 1U], sum);
                }
                const float current = vals[j] - sum;
                output[index * V] = current;
#pragma unroll
                for (uint32_t d = W - 1U; d > 0U; --d) recent[d] = recent[d - 1U];
                recent[0] = current;
            }
        }
        reverse -= count;
    }
}

// ---- K2: exact partitioned solve. Block = 32 vectors x P chunks. ----
// Pass 1: each chunk solves from zero state (end state only). Pass 2: one
// thread per vector carries the true state across chunks with precomputed
// WxW chunk transfer matrices. Pass 3: each chunk reruns from its true state.
template <uint32_t W, bool UA, uint32_t U>
__device__ __forceinline__ void fwd_run(const float *__restrict__ lower_ld, const float *__restrict__ invd,
                                        float *__restrict__ col, uint32_t V, uint32_t size,
                                        uint32_t lo, uint32_t hi, float (&recent)[W], bool store) {
    for (uint32_t base = lo; base < hi; base += U) {
        float vals[U];
#pragma unroll
        for (uint32_t j = 0; j < U; ++j) if (base + j < hi) vals[j] = col[(base + j) * V];
#pragma unroll
        for (uint32_t j = 0; j < U; ++j) {
            const uint32_t index = base + j;
            if (index < hi) {
                float sum = vals[j];
                const uint32_t available = minu(W, index);
#pragma unroll
                for (uint32_t distance = W; distance > 0U; --distance)
                    if (distance <= available)
                        sum = fmaf(-lower_ld[(distance - 1U) * size + index], recent[distance - 1U], sum);
                const float current = sum * invd[index];
                if (store) col[index * V] = current;
#pragma unroll
                for (uint32_t d = W - 1U; d > 0U; --d) recent[d] = recent[d - 1U];
                recent[0] = current;
            }
        }
    }
}

// Processes indices hi-1 down to lo.
template <uint32_t W, bool UA, uint32_t U>
__device__ __forceinline__ void bwd_run(const float *__restrict__ upper_l,
                                        float *__restrict__ col, uint32_t V, uint32_t size,
                                        uint32_t lo, uint32_t hi, float (&recent)[W], bool store) {
    for (uint32_t top = hi; top > lo;) {
        const uint32_t count = minu(U, top - lo);
        float vals[U];
#pragma unroll
        for (uint32_t j = 0; j < U; ++j) if (j < count) vals[j] = col[(top - 1U - j) * V];
#pragma unroll
        for (uint32_t j = 0; j < U; ++j) {
            if (j < count) {
                const uint32_t index = top - 1U - j;
                const uint32_t available = minu(W, size - index - 1U);
                float sum = 0.0F;
                if constexpr (UA) {
#pragma unroll
                    for (uint32_t distance = 1U; distance <= W; ++distance)
                        if (distance <= available)
                            sum = fmaf(upper_l[(distance - 1U) * size + index], recent[distance - 1U], sum);
                } else {
#pragma unroll
                    for (uint32_t distance = W; distance > 0U; --distance)
                        if (distance <= available)
                            sum = fmaf(upper_l[(distance - 1U) * size + index], recent[distance - 1U], sum);
                }
                const float current = vals[j] - sum;
                if (store) col[index * V] = current;
#pragma unroll
                for (uint32_t d = W - 1U; d > 0U; --d) recent[d] = recent[d - 1U];
                recent[0] = current;
            }
        }
        top -= count;
    }
}

template <uint32_t W, bool UA, uint32_t P, uint32_t U>
__global__ void __launch_bounds__(32 * P) k2_chunked(
    uint32_t V, uint32_t size, uint32_t C, const float *__restrict__ lower_ld,
    const float *__restrict__ upper_l, const float *__restrict__ invd,
    const float *__restrict__ gf, const float *__restrict__ gb, float *__restrict__ data) {
    __shared__ float st[P][W][32];
    const uint32_t tx = threadIdx.x, c = threadIdx.y;
    const uint32_t v = blockIdx.x * 32U + tx;
    const bool active = v < V;
    float *col = data + (active ? v : 0U);
    const uint32_t lo = minu(size, c * C), hi = minu(size, lo + C);

    // forward, pass 1
    float recent[W];
#pragma unroll
    for (uint32_t k = 0; k < W; ++k) recent[k] = 0.0F;
    if (active && c + 1U < P) fwd_run<W, UA, U>(lower_ld, invd, col, V, size, lo, hi, recent, false);
#pragma unroll
    for (uint32_t k = 0; k < W; ++k) st[c][k][tx] = recent[k];
    __syncthreads();
    // pass 2: st[c] becomes the true incoming state of chunk c
    if (c == 0U) {
        float state[W];
#pragma unroll
        for (uint32_t k = 0; k < W; ++k) { state[k] = st[0][k][tx]; st[0][k][tx] = 0.0F; }
        for (uint32_t q = 1; q < P; ++q) {
            float next[W];
            const float *g = gf + q * W * W;
#pragma unroll
            for (uint32_t r = 0; r < W; ++r) {
                float s = st[q][r][tx];
#pragma unroll
                for (uint32_t k = 0; k < W; ++k) s = fmaf(g[r * W + k], state[k], s);
                next[r] = s;
            }
#pragma unroll
            for (uint32_t k = 0; k < W; ++k) { st[q][k][tx] = state[k]; state[k] = next[k]; }
        }
    }
    __syncthreads();
#pragma unroll
    for (uint32_t k = 0; k < W; ++k) recent[k] = st[c][k][tx];
    if (active) fwd_run<W, UA, U>(lower_ld, invd, col, V, size, lo, hi, recent, true);
    __syncthreads();  // st reuse; also makes y_{size-1} visible

    // backward. The last chunk is the true start: state (y_{size-1}, 0, ...).
    const uint32_t blo = lo, bhi = minu(hi, size - 1U);
#pragma unroll
    for (uint32_t k = 0; k < W; ++k) recent[k] = 0.0F;
    if (c + 1U == P) {
        recent[0] = active ? col[(size - 1U) * V] : 0.0F;
#pragma unroll
        for (uint32_t k = 0; k < W; ++k) st[c][k][tx] = recent[k];  // true in-state
        if (active) bwd_run<W, UA, U>(upper_l, col, V, size, blo, bhi, recent, false);
    } else if (active) {
        bwd_run<W, UA, U>(upper_l, col, V, size, blo, bhi, recent, false);
    }
    __syncthreads();
    float last_end[W];
    if (c + 1U == P) {
#pragma unroll
        for (uint32_t k = 0; k < W; ++k) last_end[k] = recent[k];
    }
    __syncthreads();
    if (c + 1U < P) {
#pragma unroll
        for (uint32_t k = 0; k < W; ++k) st[c][k][tx] = recent[k];
    }
    __shared__ float last_state[W][32];
    if (c + 1U == P) {
#pragma unroll
        for (uint32_t k = 0; k < W; ++k) last_state[k][tx] = last_end[k];
    }
    __syncthreads();
    if (c == 0U) {
        float state[W];
#pragma unroll
        for (uint32_t k = 0; k < W; ++k) state[k] = last_state[k][tx];
        for (int q = int(P) - 2; q >= 0; --q) {
            float next[W];
            const float *g = gb + uint32_t(q) * W * W;
#pragma unroll
            for (uint32_t r = 0; r < W; ++r) {
                float s = st[q][r][tx];
#pragma unroll
                for (uint32_t k = 0; k < W; ++k) s = fmaf(g[r * W + k], state[k], s);
                next[r] = s;
            }
#pragma unroll
            for (uint32_t k = 0; k < W; ++k) { st[q][k][tx] = state[k]; state[k] = next[k]; }
        }
    }
    __syncthreads();
#pragma unroll
    for (uint32_t k = 0; k < W; ++k) recent[k] = st[c][k][tx];
    if (active) bwd_run<W, UA, U>(upper_l, col, V, size, blo, bhi, recent, true);
}

// ------------------------------------------------------------------ host
struct Plan { uint32_t n, w; std::vector<float> l, u, d; };

static Plan load(const std::string &path) {
    Plan p; FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) { std::perror(path.c_str()); std::exit(1); }
    if (std::fread(&p.n, 4, 1, f) != 1 || std::fread(&p.w, 4, 1, f) != 1) std::exit(1);
    p.l.resize(size_t(p.n) * p.w); p.u.resize(p.l.size()); p.d.resize(p.n);
    if (std::fread(p.l.data(), 4, p.l.size(), f) != p.l.size()) std::exit(1);
    if (std::fread(p.u.data(), 4, p.u.size(), f) != p.u.size()) std::exit(1);
    if (std::fread(p.d.data(), 4, p.d.size(), f) != p.d.size()) std::exit(1);
    std::fclose(f); return p;
}

// Chunk transfer matrices: end ring = local + G * in ring (double, homogeneous runs).
static void transfer(const Plan &p, uint32_t P, uint32_t C, std::vector<float> &gf, std::vector<float> &gb) {
    const uint32_t n = p.n, W = p.w;
    gf.assign(size_t(P) * W * W, 0.0F); gb.assign(gf.size(), 0.0F);
    for (uint32_t q = 0; q < P; ++q) {
        const uint32_t lo = std::min(n, q * C), hi = std::min(n, lo + C);
        for (uint32_t k = 0; k < W; ++k) {
            std::vector<double> ring(W, 0.0); ring[k] = 1.0;
            for (uint32_t i = lo; i < hi; ++i) {
                double s = 0.0; const uint32_t avail = std::min(W, i);
                for (uint32_t dd = 1; dd <= avail; ++dd) s -= double(p.l[(dd - 1) * n + i]) * ring[dd - 1];
                const double cur = s * double(p.d[i]);
                for (uint32_t dd = W - 1; dd > 0; --dd) ring[dd] = ring[dd - 1];
                ring[0] = cur;
            }
            for (uint32_t r = 0; r < W; ++r) gf[(size_t(q) * W + r) * W + k] = float(ring[r]);
            std::fill(ring.begin(), ring.end(), 0.0); ring[k] = 1.0;
            const uint32_t bhi = std::min(hi, n - 1);
            for (uint32_t i = bhi; i-- > lo;) {
                double s = 0.0; const uint32_t avail = std::min(W, n - i - 1);
                for (uint32_t dd = 1; dd <= avail; ++dd) s += double(p.u[(dd - 1) * n + i]) * ring[dd - 1];
                const double cur = -s;
                for (uint32_t dd = W - 1; dd > 0; --dd) ring[dd] = ring[dd - 1];
                ring[0] = cur;
            }
            for (uint32_t r = 0; r < W; ++r) gb[(size_t(q) * W + r) * W + k] = float(ring[r]);
        }
    }
}

static void reference(const Plan &p, const float *b, double *x) {  // one vector, double
    const uint32_t n = p.n, W = p.w;
    for (uint32_t i = 0; i < n; ++i) {
        double s = b[i];
        for (uint32_t dd = 1; dd <= std::min(W, i); ++dd) s -= double(p.l[(dd - 1) * n + i]) * x[i - dd];
        x[i] = s * double(p.d[i]);
    }
    for (uint32_t i = n - 1; i-- > 0;) {
        double s = 0.0;
        for (uint32_t dd = 1; dd <= std::min(W, n - i - 1); ++dd) s += double(p.u[(dd - 1) * n + i]) * x[i + dd];
        x[i] -= s;
    }
}

template <class Launch>
static double time_us(Launch launch, float *d_data, const float *d_src, size_t bytes, int reps) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 5; ++i) { CK(cudaMemcpy(d_data, d_src, bytes, cudaMemcpyDeviceToDevice)); launch(); }
    std::vector<float> t;
    for (int i = 0; i < reps; ++i) {
        CK(cudaMemcpyAsync(d_data, d_src, bytes, cudaMemcpyDeviceToDevice));
        CK(cudaEventRecord(a)); launch(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); t.push_back(ms * 1000.0F);
    }
    CK(cudaGetLastError());
    std::sort(t.begin(), t.end()); return t[t.size() / 2];
}

template <uint32_t W, bool UA>
static void run_case(const Plan &p, uint32_t V, const char *label) {
    const uint32_t n = p.n; const size_t count = size_t(n) * V, bytes = count * 4;
    std::vector<float> h(count); std::mt19937 rng(1234); std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    for (auto &x : h) x = dist(rng);
    float *d_src, *d_data, *d_l, *d_u, *d_d, *d_gf, *d_gb;
    CK(cudaMalloc(&d_src, bytes)); CK(cudaMalloc(&d_data, bytes));
    CK(cudaMalloc(&d_l, p.l.size() * 4)); CK(cudaMalloc(&d_u, p.u.size() * 4)); CK(cudaMalloc(&d_d, p.d.size() * 4));
    CK(cudaMemcpy(d_src, h.data(), bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_l, p.l.data(), p.l.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_u, p.u.data(), p.u.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_d, p.d.data(), p.d.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_gf, 64 * W * W * 4)); CK(cudaMalloc(&d_gb, 64 * W * W * 4));

    const unsigned T = 32; const unsigned G = (V + T - 1) / T;
    std::vector<float> base(count), out(count);
    const int reps = 200;
    const double t0 = time_us([&] { k0_baseline<W, UA><<<G, T>>>(V, n, d_l, d_u, d_d, d_data); }, d_data, d_src, bytes, reps);
    CK(cudaMemcpy(base.data(), d_data, bytes, cudaMemcpyDeviceToHost));

    // error of the baseline against a double reference (first 64 vectors)
    double base_err = 0.0;
    {
        std::vector<float> b(n); std::vector<double> x(n);
        for (uint32_t v = 0; v < std::min(V, 64U); ++v) {
            for (uint32_t i = 0; i < n; ++i) b[i] = h[size_t(i) * V + v];
            reference(p, b.data(), x.data());
            for (uint32_t i = 0; i < n; ++i) base_err = std::max(base_err, std::abs(double(base[size_t(i) * V + v]) - x[i]));
        }
    }
    auto compare = [&](double &maxdiff) { CK(cudaMemcpy(out.data(), d_data, bytes, cudaMemcpyDeviceToHost));
        maxdiff = 0.0; bool exact = true;
        for (size_t i = 0; i < count; ++i) { const double df = std::abs(double(out[i]) - double(base[i]));
            maxdiff = std::max(maxdiff, df); exact &= std::memcmp(&out[i], &base[i], 4) == 0; }
        return exact; };

    std::printf("%-22s n=%4u V=%5u w=%u | K0 baseline %8.1f us (vs f64 ref max %.2e)\n", label, n, V, W, t0, base_err);
    double md;
    {
        const double t = time_us([&] { k1_prefetch<W, UA, 16><<<G, T>>>(V, n, d_l, d_u, d_d, d_data); }, d_data, d_src, bytes, reps);
        const bool ex = compare(md);
        std::printf("    K1 prefetch U=16        %8.1f us  %5.1fx  bit-exact=%s\n", t, t0 / t, ex ? "yes" : "NO");
    }
    {
        const double t = time_us([&] { k1_prefetch<W, UA, 32><<<G, T>>>(V, n, d_l, d_u, d_d, d_data); }, d_data, d_src, bytes, reps);
        const bool ex = compare(md);
        std::printf("    K1 prefetch U=32        %8.1f us  %5.1fx  bit-exact=%s\n", t, t0 / t, ex ? "yes" : "NO");
    }
    auto chunked = [&](auto Ptag) {
        constexpr uint32_t P = decltype(Ptag)::value;
        const uint32_t C = (n + P - 1) / P;
        std::vector<float> gf, gb; transfer(p, P, C, gf, gb);
        float gmax = 0.0F;
        for (size_t i = W * W; i < gf.size(); ++i) gmax = std::max(gmax, std::abs(gf[i]));
        for (size_t i = 0; i + W * W < gb.size(); ++i) gmax = std::max(gmax, std::abs(gb[i]));
        CK(cudaMemcpy(d_gf, gf.data(), gf.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_gb, gb.data(), gb.size() * 4, cudaMemcpyHostToDevice));
        const dim3 block(32, P), grid((V + 31) / 32);
        const double t = time_us([&] { k2_chunked<W, UA, P, 8><<<grid, block>>>(V, n, C, d_l, d_u, d_d, d_gf, d_gb, d_data); },
                                 d_data, d_src, bytes, reps);
        const bool ex = compare(md);
        std::printf("    K2 chunked P=%-2u C=%-4u  %8.1f us  %5.1fx  max|diff vs K0|=%.2e max|G|=%.1e%s\n", P, C, t, t0 / t, md, gmax, ex ? " (bit-exact)" : "");
    };
    chunked(std::integral_constant<uint32_t, 4>{});
    chunked(std::integral_constant<uint32_t, 8>{});
    chunked(std::integral_constant<uint32_t, 16>{});
    chunked(std::integral_constant<uint32_t, 32>{});
    cudaFree(d_src); cudaFree(d_data); cudaFree(d_l); cudaFree(d_u); cudaFree(d_d); cudaFree(d_gf); cudaFree(d_gb);
}

static void dispatch(const Plan &p, uint32_t V, const char *label) {
    switch (p.w) {
        case 1: run_case<1, false>(p, V, label); break;
        case 3: run_case<3, true>(p, V, label); break;
        case 5: run_case<5, false>(p, V, label); break;
        case 7: run_case<7, false>(p, V, label); break;
        default: std::printf("%s: w=%u not benched\n", label, p.w);
    }
}

int main(int argc, char **argv) {
    // usage: kbench <plan> <V> [label]
    for (int a = 1; a + 1 < argc; a += 2) {
        const Plan p = load(argv[a]);
        const uint32_t V = uint32_t(std::atoi(argv[a + 1]));
        std::string label = argv[a]; label = label.substr(label.find_last_of('/') + 1);
        dispatch(p, V, label.c_str());
    }
}
