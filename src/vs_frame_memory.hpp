#pragma once

#include "dsmvc/host_blocks.hpp"

#include <VapourSynth4.h>

// Glue between a VapourSynth core that exposes frame pool blocks through the
// vapoursynth-classic "frame memory" extension and dsmvc's host block registry.
// On any other core everything here reports "not available" and the plugin
// keeps staging frames as before.
namespace dsmvc::vs_frame_memory {

// Attaches the filter to direct block I/O for `domain` on `core`. Returns false
// (and does nothing) unless the backend enabled the domain and the core
// provides the extension. Each true return must be paired with detach().
[[nodiscard]] bool attach(
    VSCore *core, const VSAPI *vsapi, host_blocks::Domain domain);
void detach(VSCore *core, const VSAPI *vsapi, host_blocks::Domain domain);

// Reports the pool blocks behind the planes of a frame that is about to be
// transferred. The writable overload fetches each write pointer first, because
// a copy-on-write frame can move to a different block when it is written.
void note_frame(
    VSCore *core, const VSAPI *vsapi, host_blocks::Domain domain,
    const VSFrame *frame);
void note_frame(
    VSCore *core, const VSAPI *vsapi, host_blocks::Domain domain,
    VSFrame *frame);

} // namespace dsmvc::vs_frame_memory
