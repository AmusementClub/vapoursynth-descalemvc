// Dumps the Float32 LDL^T factors of real dsmvc axis plans for cuda_solve_bench.
// Build: g++ -std=c++23 -O2 -I include cuda_solve_plans.cpp <build>/libdsmvc_engine.a
//        -lcudart -l:libvulkan.so.1 -lpthread -ldl   (as the engine was configured)
// Run: cuda_solve_plans <output dir>
#include "dsmvc/engine.hpp"
#include <cstdio>
#include <cstdint>
#include <string>
int main(int argc, char **argv) {
    struct Case { const char *name; int src, dst; dsmvc::KernelKind kind; int taps; double blur = 1.0; double b = 0.0, c = 0.5; };
    const Case cases[] = {
        {"bicubic_1080_720", 1080, 720, dsmvc::KernelKind::bicubic, 3},
        {"bicubic_1920_1280", 1920, 1280, dsmvc::KernelKind::bicubic, 3},
        {"bilinear_1920_1280", 1920, 1280, dsmvc::KernelKind::bilinear, 3},
        {"lanczos3_1920_1280", 1920, 1280, dsmvc::KernelKind::lanczos, 3},
        {"lanczos3_1080_720", 1080, 720, dsmvc::KernelKind::lanczos, 3},
        {"bicubic_1080_873", 1080, 873, dsmvc::KernelKind::bicubic, 3},
        {"bicubic_3840_2560", 3840, 2560, dsmvc::KernelKind::bicubic, 3},
        {"bicubic_1080_1040", 1080, 1040, dsmvc::KernelKind::bicubic, 3},
        {"bicubic_1080_1070", 1080, 1070, dsmvc::KernelKind::bicubic, 3},
        {"mitchell_1080_720", 1080, 720, dsmvc::KernelKind::bicubic, 3, 1.0, 1.0/3, 1.0/3},
        {"lanczos4_1080_720", 1080, 720, dsmvc::KernelKind::lanczos, 4},
        {"lanczos5_1080_720", 1080, 720, dsmvc::KernelKind::lanczos, 5},
        {"spline64_1080_720", 1080, 720, dsmvc::KernelKind::spline64, 3},
        {"lanczos3_blur08_1080_810", 1080, 810, dsmvc::KernelKind::lanczos, 3, 0.8},
        {"bicubic_blur12_1080_720", 1080, 720, dsmvc::KernelKind::bicubic, 3, 1.2},
        {"lanczos3_1080_1040", 1080, 1040, dsmvc::KernelKind::lanczos, 3},
    };
    for (const auto &c : cases) {
        dsmvc::AxisRequest r;
        r.source_size = c.src; r.destination_size = c.dst;
        r.active_length = c.dst; r.kernel.kind = c.kind; r.kernel.taps = c.taps; r.kernel.blur = c.blur; r.kernel.b = c.b; r.kernel.c = c.c;
        const auto plan = dsmvc::build_axis_plan(r);
        const std::uint32_t n = plan.destination_size, w = plan.half_bandwidth;
        std::printf("%s n=%u w=%u f64=%d rcond=%g\n", c.name, n, w,
                    plan.requires_float64() ? 1 : 0, plan.normal_rcond);
        FILE *f = std::fopen((std::string(argv[1]) + "/" + c.name + ".plan").c_str(), "wb");
        std::fwrite(&n, 4, 1, f); std::fwrite(&w, 4, 1, f);
        std::fwrite(plan.lower_ld.data(), 4, plan.lower_ld.size(), f);
        std::fwrite(plan.upper_l.data(), 4, plan.upper_l.size(), f);
        std::fwrite(plan.inverse_diagonal.data(), 4, plan.inverse_diagonal.size(), f);
        std::fclose(f);
    }
}
