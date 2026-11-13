// Frame I/O benchmark host for a vapoursynth-classic (or any API 4) core, no Python.
// A source filter makes a fresh frame per request (as a decoder would), one dsmvc
// filter processes it, and frames are requested like vspipe does (`requests` in
// flight). Prints fps and, with --hash, an order-independent digest of all output
// so that two I/O modes can be compared exactly.
//
// usage: classic_frame_bench --plugin libdsmvc.so [--filter Debicubic]
//          [--format gray|yuv420p8|yuv420p16|rgb|grayh] [--width 1920] [--height 1080]
//          [--dst-width 1280] [--dst-height 720] [--backend cuda|vulkan|cpu|auto]
//          [--frames 600] [--warmup 32] [--threads N] [--requests 16] [--cache MB]
//          [--loop K] [--hash] [--cores N] [--backend2 B [--dst2-width W --dst2-height H]] [--arg key=value ...]
//          [--gpu] [--gpu-resident] [--exit-sleep S]
// --backend2 chains a second filter (default 80% of the first output size) on another
// backend in the same core, so both backends share the frame pool blocks.
// --cores N runs N independent cores at once, then frees the first one while the
// others run a second phase (create/free ordering and per-core state are checked
// by comparing digests of both phases).
// --arg passes extra filter arguments: an integer, a float or (otherwise) a string.
// --gpu-resident (R81+): the source patterns are uploaded once and looped on the
// GPU (std.Loop), and the dsmvc output stays on the GPU (no download, no hash):
// throughput of the GPU work alone.
// --gpu (VapourSynth R81+) wraps the dsmvc filters in std.GPUUpload/GPUDownload,
// so they run on GPU frames.
#include <VapourSynth4.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <csignal>
#ifdef __linux__
#include <execinfo.h>
#include <sys/prctl.h>
#include <unistd.h>
#endif

namespace {

const VSAPI *vsapi = nullptr;
constexpr int kPatterns = 8;

struct Source {
    VSVideoInfo vi;
    std::vector<std::vector<uint8_t>> planes[kPatterns];
    int loop = 0;
};

const VSFrame *VS_CC source_get_frame(
    int n, int reason, void *instance, void **, VSFrameContext *,
    VSCore *core, const VSAPI *api) {
    if (reason != arInitial) return nullptr;
    auto *s = static_cast<Source *>(instance);
    VSFrame *frame = api->newVideoFrame(
        &s->vi.format, s->vi.width, s->vi.height, nullptr, core);
    const int pattern = (s->loop > 0 ? n % s->loop : n) % kPatterns;
    for (int p = 0; p < s->vi.format.numPlanes; ++p) {
        const int row = api->getFrameWidth(frame, p) * s->vi.format.bytesPerSample;
        const int h = api->getFrameHeight(frame, p);
        uint8_t *dst = api->getWritePtr(frame, p);
        const ptrdiff_t stride = api->getStride(frame, p);
        const auto &src = s->planes[pattern][p];
        for (int y = 0; y < h; ++y) {
            std::memcpy(dst + y * stride, src.data() + static_cast<size_t>(y) * row, row);
        }
    }
    return frame;
}

void VS_CC source_free(void *instance, VSCore *, const VSAPI *) {
    delete static_cast<Source *>(instance);
}

struct Run {
    VSNode *node = nullptr;
    int total = 0;
    int warmup = 0;
    int requests = 1;
    bool hash = false;
    std::mutex mutex;
    std::condition_variable cv;
    int base = 0;   // first frame number of this phase
    int next = 0;
    int done = 0;
    int in_flight = 0;
    int failed = 0;
    std::string first_error;
    uint64_t digest = 0;
    std::chrono::steady_clock::time_point start;
    bool started = false;
};

uint64_t hash_frame(const VSFrame *frame, int n) {
    uint64_t h = 1469598103934665603ULL ^ static_cast<uint64_t>(n) * 0x9E3779B97F4A7C15ULL;
    const VSVideoFormat *fmt = vsapi->getVideoFrameFormat(frame);
    for (int p = 0; p < fmt->numPlanes; ++p) {
        const uint8_t *ptr = vsapi->getReadPtr(frame, p);
        const ptrdiff_t stride = vsapi->getStride(frame, p);
        const int row = vsapi->getFrameWidth(frame, p) * fmt->bytesPerSample;
        const int rows = vsapi->getFrameHeight(frame, p);
        for (int y = 0; y < rows; ++y) {
            const uint8_t *r = ptr + y * stride;
            for (int x = 0; x < row; ++x) h = (h ^ r[x]) * 1099511628211ULL;
        }
    }
    return h;
}

void request_more(Run &run);

void VS_CC on_frame(void *user, const VSFrame *frame, int n, VSNode *, const char *error) {
    auto &run = *static_cast<Run *>(user);
    uint64_t h = 0;
    if (frame && run.hash && n - run.base >= run.warmup) h = hash_frame(frame, n - run.base);
    if (frame) vsapi->freeFrame(frame);
    {
        std::lock_guard<std::mutex> lock(run.mutex);
        if (!frame) {
            ++run.failed;
            if (run.first_error.empty()) run.first_error = error ? error : "unknown";
        }
        run.digest ^= h;
        ++run.done;
        --run.in_flight;
        if (run.done == run.warmup && !run.started) {
            run.started = true;
            run.start = std::chrono::steady_clock::now();
        }
    }
    request_more(run);
    run.cv.notify_all();
}

void request_more(Run &run) {
    for (;;) {
        int n;
        {
            std::lock_guard<std::mutex> lock(run.mutex);
            if (run.next >= run.total || run.in_flight >= run.requests || run.failed) return;
            n = run.next++;
            ++run.in_flight;
        }
        vsapi->getFrameAsync(run.base + n, run.node, on_frame, &run);
    }
}

} // namespace

#ifdef __linux__
// BENCH_CRASH_TRACE=1: print the faulting thread's stack on SIGSEGV/SIGABRT,
// and let any process of this user attach a debugger (for hangs at exit).
void crash_trace(int signal) {
    void *frames[64];
    const int count = backtrace(frames, 64);
    char header[64];
    const int length = std::snprintf(header, sizeof(header), "signal %d, tid %ld\n",
                                     signal, static_cast<long>(gettid()));
    (void)!write(2, header, static_cast<size_t>(length));
    backtrace_symbols_fd(frames, count, 2);
    std::signal(signal, SIG_DFL);
    std::raise(signal);
}

void install_crash_trace() {
    if (!std::getenv("BENCH_CRASH_TRACE")) return;
    prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
    std::signal(SIGSEGV, crash_trace);
    std::signal(SIGABRT, crash_trace);
}
#else
void install_crash_trace() {}
#endif

int main(int argc, char **argv) {
    install_crash_trace();
    std::string plugin, filter = "Debicubic", format = "gray", backend;
    int width = 1920, height = 1080, dst_width = 1280, dst_height = 720;
    int frames = 600, warmup = 32, threads = 0, requests = 16, loop = 0;
    long long cache_mb = 0;
    bool hash = false;
    bool gpu = false;
    bool resident = false;
    double exit_sleep = 0.0;
    int cores = 1;
    std::string backend2;
    int dst2_width = 0, dst2_height = 0;
    std::vector<std::pair<std::string, std::string>> extra;
    for (int i = 1; i < argc; ++i) {
        const std::string o = argv[i];
        auto value = [&] { return i + 1 < argc ? argv[++i] : (std::exit(2), nullptr); };
        if (o == "--plugin") plugin = value();
        else if (o == "--filter") filter = value();
        else if (o == "--format") format = value();
        else if (o == "--backend") backend = value();
        else if (o == "--width") width = std::atoi(value());
        else if (o == "--height") height = std::atoi(value());
        else if (o == "--dst-width") dst_width = std::atoi(value());
        else if (o == "--dst-height") dst_height = std::atoi(value());
        else if (o == "--frames") frames = std::atoi(value());
        else if (o == "--warmup") warmup = std::atoi(value());
        else if (o == "--threads") threads = std::atoi(value());
        else if (o == "--requests") requests = std::atoi(value());
        else if (o == "--cache") cache_mb = std::atoll(value());
        else if (o == "--loop") loop = std::atoi(value());
        else if (o == "--hash") hash = true;
        else if (o == "--gpu") gpu = true;
        else if (o == "--gpu-resident") gpu = resident = true;
        else if (o == "--exit-sleep") exit_sleep = std::atof(value());
        else if (o == "--cores") cores = std::atoi(value());
        else if (o == "--backend2") backend2 = value();
        else if (o == "--dst2-width") dst2_width = std::atoi(value());
        else if (o == "--dst2-height") dst2_height = std::atoi(value());
        else if (o == "--arg") {
            const std::string kv = value();
            const auto eq = kv.find('=');
            if (eq == std::string::npos) { std::fprintf(stderr, "--arg needs key=value\n"); return 2; }
            extra.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
        }
        else { std::fprintf(stderr, "unknown option %s\n", o.c_str()); return 2; }
    }
    if (plugin.empty()) { std::fprintf(stderr, "--plugin is required\n"); return 2; }

    vsapi = getVapourSynthAPI(VAPOURSYNTH_API_VERSION);
    const int phase_frames = ((frames + warmup + 7) / 8) * 8; // multiple of the pattern count
    const int source_length = cores > 1 ? phase_frames * 2 : frames + warmup;

    unsigned family = cfGray, sample = stInteger, bits = 8, ssw = 0, ssh = 0;
    if (format == "gray") { family = cfGray; sample = stFloat; bits = 32; }
    else if (format == "grayh") { family = cfGray; sample = stInteger; bits = 16; }
    else if (format == "yuv420p8") { family = cfYUV; bits = 8; ssw = ssh = 1; }
    else if (format == "yuv420p16") { family = cfYUV; bits = 16; ssw = ssh = 1; }
    else if (format == "rgb") { family = cfRGB; sample = stFloat; bits = 32; }
    else { std::fprintf(stderr, "unknown format %s\n", format.c_str()); return 2; }

    struct Pipeline { VSCore *core = nullptr; VSNode *out = nullptr; };
    auto build = [&](Pipeline &pipeline) -> bool {
        VSCore *core = vsapi->createCore(0);
        if (threads > 0) vsapi->setThreadCount(threads, core);
        if (cache_mb > 0) vsapi->setMaxCacheSize(cache_mb << 20, core);

        VSPlugin *std_plugin = vsapi->getPluginByID("com.vapoursynth.std", core);
        VSMap *args = vsapi->createMap();
        vsapi->mapSetData(args, "path", plugin.c_str(), static_cast<int>(plugin.size()), dtUtf8, maAppend);
        VSMap *res = vsapi->invoke(std_plugin, "LoadPlugin", args);
        if (const char *e = vsapi->mapGetError(res)) { std::fprintf(stderr, "LoadPlugin: %s\n", e); return false; }
        vsapi->freeMap(res); vsapi->freeMap(args);

        auto *source = new Source{};
        source->loop = loop;
        if (!vsapi->queryVideoFormat(&source->vi.format, family, sample, bits, ssw, ssh, core)) {
            std::fprintf(stderr, "bad format\n"); return false;
        }
        source->vi.width = width; source->vi.height = height;
        source->vi.fpsNum = 24; source->vi.fpsDen = 1; source->vi.numFrames = source_length;
        for (int k = 0; k < kPatterns; ++k) {
            for (int p = 0; p < source->vi.format.numPlanes; ++p) {
                const int w = p ? width >> ssw : width, h = p ? height >> ssh : height;
                const int bps = source->vi.format.bytesPerSample;
                std::vector<uint8_t> plane(static_cast<size_t>(w) * h * bps);
                uint32_t state = 0x1234567u + k * 7919u + p * 104729u;
                for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
                    state = state * 1664525u + 1013904223u;
                    const float v = 0.25f + 0.5f * ((state >> 8) & 0xFFFF) / 65535.0f
                        + 0.1f * ((i % w) / static_cast<float>(w));
                    if (bps == 4) std::memcpy(&plane[i * 4], &v, 4);
                    else if (bps == 2) { const uint16_t s = static_cast<uint16_t>(v * 60000); std::memcpy(&plane[i * 2], &s, 2); }
                    else plane[i] = static_cast<uint8_t>(v * 200);
                }
                source->planes[k].push_back(std::move(plane));
            }
        }
        VSNode *src = vsapi->createVideoFilter2(
            "Source", &source->vi, source_get_frame, source_free, fmParallel, nullptr, 0, source, core);
        // std.GPUUpload / std.GPUDownload around the dsmvc filters with --gpu.
        auto wrap = [&](const char *name, VSNode *node) -> VSNode * {
            VSMap *in = vsapi->createMap();
            vsapi->mapConsumeNode(in, "clip", node, maAppend);
            VSMap *out = vsapi->invoke(std_plugin, name, in);
            vsapi->freeMap(in);
            if (const char *e = vsapi->mapGetError(out)) {
                std::fprintf(stderr, "%s: %s\n", name, e);
                vsapi->freeMap(out);
                return nullptr;
            }
            VSNode *result = vsapi->mapGetNode(out, "clip", 0, nullptr);
            vsapi->freeMap(out);
            return result;
        };
        if (resident) {
            // Upload the patterns once (the core caches them on the GPU), loop them.
            VSMap *in = vsapi->createMap();
            vsapi->mapConsumeNode(in, "clip", src, maAppend);
            vsapi->mapSetInt(in, "first", 0, maAppend);
            vsapi->mapSetInt(in, "length", kPatterns, maAppend);
            VSMap *out = vsapi->invoke(std_plugin, "Trim", in);
            vsapi->freeMap(in);
            if (const char *e = vsapi->mapGetError(out)) { std::fprintf(stderr, "Trim: %s\n", e); return false; }
            src = vsapi->mapGetNode(out, "clip", 0, nullptr);
            vsapi->freeMap(out);
            if (!(src = wrap("GPUUpload", src))) return false;
            in = vsapi->createMap();
            vsapi->mapConsumeNode(in, "clip", src, maAppend);
            vsapi->mapSetInt(in, "times", (source_length + kPatterns - 1) / kPatterns, maAppend);
            out = vsapi->invoke(std_plugin, "Loop", in);
            vsapi->freeMap(in);
            if (const char *e = vsapi->mapGetError(out)) { std::fprintf(stderr, "Loop: %s\n", e); return false; }
            src = vsapi->mapGetNode(out, "clip", 0, nullptr);
            vsapi->freeMap(out);
        } else if (gpu && !(src = wrap("GPUUpload", src))) {
            return false;
        }

        VSPlugin *dsmvc = vsapi->getPluginByNamespace("dsmvc", core);
        if (!dsmvc) { std::fprintf(stderr, "dsmvc namespace not found\n"); return false; }
        args = vsapi->createMap();
        vsapi->mapConsumeNode(args, "src", src, maAppend);
        vsapi->mapSetInt(args, "width", dst_width, maAppend);
        vsapi->mapSetInt(args, "height", dst_height, maAppend);
        if (!backend.empty()) {
            vsapi->mapSetData(args, "backend", backend.c_str(), static_cast<int>(backend.size()), dtUtf8, maAppend);
        }
        for (const auto &[key, text] : extra) {
            char *end = nullptr;
            const long long as_int = std::strtoll(text.c_str(), &end, 10);
            if (!text.empty() && *end == '\0') { vsapi->mapSetInt(args, key.c_str(), as_int, maAppend); continue; }
            const double as_float = std::strtod(text.c_str(), &end);
            if (!text.empty() && *end == '\0') { vsapi->mapSetFloat(args, key.c_str(), as_float, maAppend); continue; }
            vsapi->mapSetData(args, key.c_str(), text.c_str(), static_cast<int>(text.size()), dtUtf8, maAppend);
        }
        res = vsapi->invoke(dsmvc, filter.c_str(), args);
        if (const char *e = vsapi->mapGetError(res)) { std::fprintf(stderr, "%s: %s\n", filter.c_str(), e); return false; }
        pipeline.out = vsapi->mapGetNode(res, "clip", 0, nullptr);
        pipeline.core = core;
        vsapi->freeMap(res); vsapi->freeMap(args);
        if (!backend2.empty()) {
            args = vsapi->createMap();
            vsapi->mapConsumeNode(args, "src", pipeline.out, maAppend);
            vsapi->mapSetInt(args, "width", dst2_width > 0 ? dst2_width : dst_width * 4 / 5, maAppend);
            vsapi->mapSetInt(args, "height", dst2_height > 0 ? dst2_height : dst_height * 4 / 5, maAppend);
            vsapi->mapSetData(args, "backend", backend2.c_str(), static_cast<int>(backend2.size()), dtUtf8, maAppend);
            res = vsapi->invoke(dsmvc, filter.c_str(), args);
            if (const char *e = vsapi->mapGetError(res)) { std::fprintf(stderr, "stage 2 %s: %s\n", filter.c_str(), e); return false; }
            pipeline.out = vsapi->mapGetNode(res, "clip", 0, nullptr);
            vsapi->freeMap(res); vsapi->freeMap(args);
        }
        if (gpu && !resident && !(pipeline.out = wrap("GPUDownload", pipeline.out))) return false;
        return true;
    };

    std::vector<Pipeline> pipelines(static_cast<size_t>(cores));
    for (auto &pipeline : pipelines) {
        if (!build(pipeline)) return 1;
    }

    // Runs one phase on the given pipelines at the same time; returns false on failure.
    std::vector<uint64_t> digests;
    auto run_phase = [&](const std::vector<Pipeline *> &active, int base, const char *label) -> bool {
        std::vector<std::unique_ptr<Run>> runs;
        for (Pipeline *pipeline : active) {
            auto run = std::make_unique<Run>();
            run->node = pipeline->out; run->base = base;
            run->total = frames + warmup; run->warmup = warmup;
            run->requests = requests; run->hash = hash;
            if (warmup == 0) { run->started = true; run->start = std::chrono::steady_clock::now(); }
            runs.push_back(std::move(run));
        }
        for (auto &run : runs) request_more(*run);
        bool ok = true;
        digests.clear();
        for (size_t i = 0; i < runs.size(); ++i) {
            Run &run = *runs[i];
            {
                std::unique_lock<std::mutex> lock(run.mutex);
                run.cv.wait(lock, [&] { return run.done >= run.total || (run.failed && run.in_flight == 0); });
            }
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - run.start).count();
            if (run.failed) {
                std::fprintf(stderr, "FAILED (%d frames): %s\n", run.failed, run.first_error.c_str());
                ok = false;
                continue;
            }
            if (cores > 1) std::printf("%s core %zu: ", label, i);
            std::printf("frames=%d seconds=%.3f fps=%.1f", frames, seconds, frames / seconds);
            if (hash) std::printf(" digest=%016llx", static_cast<unsigned long long>(run.digest));
            std::printf("\n");
            std::fflush(stdout);
            digests.push_back(run.digest);
        }
        return ok;
    };

    std::vector<Pipeline *> all;
    for (auto &pipeline : pipelines) all.push_back(&pipeline);
    if (!run_phase(all, 0, "phase1")) return 1;
    const std::vector<uint64_t> first_digests = digests;
    if (hash) {
        for (uint64_t d : first_digests) {
            if (d != first_digests.front()) { std::fprintf(stderr, "MISMATCH between cores\n"); return 1; }
        }
    }
    if (cores > 1) {
        // Free the first core while the others stay alive, then keep working on them.
        vsapi->freeNode(pipelines[0].out);
        vsapi->freeCore(pipelines[0].core);
        pipelines[0] = Pipeline{};
        std::vector<Pipeline *> rest;
        for (size_t i = 1; i < pipelines.size(); ++i) rest.push_back(&pipelines[i]);
        if (!run_phase(rest, phase_frames, "phase2")) return 1;
        if (hash) {
            for (uint64_t d : digests) {
                if (d != first_digests.front()) { std::fprintf(stderr, "MISMATCH after freeing a core\n"); return 1; }
            }
        }
    }
    for (auto &pipeline : pipelines) {
        if (pipeline.out) vsapi->freeNode(pipeline.out);
        if (pipeline.core) vsapi->freeCore(pipeline.core);
    }
    if (exit_sleep > 0.0) std::this_thread::sleep_for(std::chrono::duration<double>(exit_sleep));
    return 0;
}
