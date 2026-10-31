// MobileGlues - egl/framediag.h
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

#ifndef MOBILEGLUES_EGL_FRAMEDIAG_H
#define MOBILEGLUES_EGL_FRAMEDIAG_H

#include "../config/config.h"
#include "../gl/envvars.h"
#include <EGL/egl.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mutex>
#include <vector>

// Opt-in frame-pacing diagnostics.
//
// A frame counter that says 90 only says that eglSwapBuffers was called 90 times
// in a second. It cannot tell an even 11.1 ms cadence from bursts of fast frames
// followed by long ones, and the second feels like 30 fps. This records the
// cadence itself, once per second, into <MG dir>/framediag.log.
//
// Off by default. Turn it on with EITHER
//   - the environment variable MG_FRAME_DIAG=1, or
//   - "frameDiagnostics": 1 in MG's config.json.
// Off, every entry point below is one relaxed atomic load and a branch: no clock
// reads, no locks, no file. It changes nothing about how frames are presented; it
// only reads the clock around the swap call.
//
// What is written: timings, EGL surface lifecycle events, surface size, and which
// launcher environment variables are set (their numeric version codes). Nothing
// about the user, the world, or the device identity.
//
// Columns of a window line (all times in ms):
//   int   = time between the START of one swap call and the start of the next.
//           This is the delivery cadence: even = smooth, spiky = choppy.
//   swap  = time spent INSIDE the swap call. Large = the swap is blocking, i.e.
//           waiting on vsync or on a full buffer queue.
//   work  = time between the end of one swap and the start of the next. This is
//           everything the game and renderer do to build a frame (CPU side).
//   short = intervals under half the median (frames delivered in a burst)
//   long  = intervals over 1.5x the median (a frame took visibly longer)
//   stall = intervals over 50 ms
namespace mg_framediag {

namespace detail {

constexpr int64_t kWindowNs = 1000000000LL; // one summary line per second
constexpr int kMaxWindows = 1800;           // stop after ~30 minutes of play
constexpr uint32_t kSizeProbeFrames = 300;  // how often the caller re-reads the surface size

inline int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline double to_ms(int64_t ns) {
    return ns < 0 ? 0.0 : static_cast<double>(ns) / 1e6;
}

struct Stats {
    double mean = 0, sd = 0, min = 0, p50 = 0, p95 = 0, p99 = 0, max = 0;
};

// Sorts v in place; nearest-rank percentiles.
inline Stats compute(std::vector<double>& v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / v.size();
    double var = 0;
    for (double x : v) var += (x - s.mean) * (x - s.mean);
    s.sd = std::sqrt(var / v.size());
    auto pct = [&](double p) {
        size_t idx = static_cast<size_t>(std::ceil(p * v.size()));
        if (idx > 0) --idx;
        if (idx >= v.size()) idx = v.size() - 1;
        return v[idx];
    };
    s.min = v.front();
    s.max = v.back();
    s.p50 = pct(0.50);
    s.p95 = pct(0.95);
    s.p99 = pct(0.99);
    return s;
}

struct State {
    std::mutex mutex;
    FILE* file = nullptr;
    bool open_tried = false;
    bool capped = false;
    int64_t t0 = 0;
    int64_t window_start = 0;
    int64_t prev_begin = 0;
    int64_t prev_end = 0;
    bool have_prev = false;
    EGLDisplay last_dpy = EGL_NO_DISPLAY;
    EGLSurface last_surface = EGL_NO_SURFACE;
    uint32_t frames_since_probe = 0;
    int windows_written = 0;
    int last_w = -1;
    int last_h = -1;
    std::vector<double> interval, swap, work;
};

// Leaked on purpose: swaps can still arrive on another thread during process
// teardown, and a destroyed mutex there is worse than an unreleased FILE*.
inline State& state() {
    static State* s = new State();
    return *s;
}

inline int64_t& tls_begin() {
    static thread_local int64_t t = 0;
    return t;
}

inline void write_line(State& s, const char* fmt, ...) {
    if (!s.file) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(s.file, fmt, ap);
    va_end(ap);
    fputc('\n', s.file);
    fflush(s.file);
}

// Caller holds the lock.
inline void ensure_open(State& s) {
    if (s.open_tried) return;
    s.open_tried = true;
    s.t0 = now_ns();
    const char* dir = (mg_directory_path && mg_directory_path[0]) ? mg_directory_path : "/sdcard/MG";
    char path[512];
    snprintf(path, sizeof(path), "%s/framediag.log", dir);
    // "w": the file always describes the most recent launch.
    s.file = fopen(path, "w");
    if (!s.file) return;
    int plugin = 0, fcl = 0, zalith = 0, pgw = 0;
    GetEnvVarInt("MG_PLUGIN_STATUS", &plugin, 0);
    GetEnvVarInt("FCL_VERSION_CODE", &fcl, 0);
    GetEnvVarInt("ZALITH_VERSION_CODE", &zalith, 0);
    GetEnvVarInt("PGW_VERSION_CODE", &pgw, 0);
    write_line(s, "# MobileGlues frame diagnostics v1");
    write_line(s, "# launcher env: MG_PLUGIN_STATUS=%d FCL_VERSION_CODE=%d ZALITH_VERSION_CODE=%d PGW_VERSION_CODE=%d",
               plugin, fcl, zalith, pgw);
    write_line(s, "# int=swap-to-swap cadence, swap=time inside swap call, work=time between swaps (all ms)");
    write_line(s, "# short=int<0.5*median, long=int>1.5*median, stall=int>50ms");
}

inline void flush_window(State& s, int64_t t_end) {
    const size_t frames = s.swap.size();
    const double secs = static_cast<double>(t_end - s.window_start) / 1e9;
    const Stats in = compute(s.interval);
    const Stats sw = compute(s.swap);
    const Stats wk = compute(s.work);
    int shorts = 0, longs = 0, stalls = 0;
    for (double x : s.interval) {
        if (x < 0.5 * in.p50) ++shorts;
        if (x > 1.5 * in.p50) ++longs;
        if (x > 50.0) ++stalls;
    }
    write_line(s,
               "t=%7.1fs frames=%3zu fps=%6.1f | int mean=%6.2f sd=%5.2f p50=%6.2f p95=%6.2f p99=%6.2f max=%7.2f "
               "short=%d long=%d stall=%d | swap mean=%5.2f p95=%5.2f max=%6.2f | work mean=%5.2f p95=%5.2f max=%6.2f",
               to_ms(t_end - s.t0) / 1000.0, frames, secs > 0 ? frames / secs : 0.0, in.mean, in.sd, in.p50, in.p95,
               in.p99, in.max, shorts, longs, stalls, sw.mean, sw.p95, sw.max, wk.mean, wk.p95, wk.max);
    s.interval.clear();
    s.swap.clear();
    s.work.clear();
    s.window_start = t_end;
    if (++s.windows_written >= kMaxWindows) {
        s.capped = true;
        write_line(s, "# stopped after %d windows", kMaxWindows);
    }
}

} // namespace detail

// Cheap and safe to call from anywhere. The decision is made once config.json is
// loaded (or immediately if the environment variable is set).
inline bool enabled() {
    static std::atomic<int> cached{-1};
    const int c = cached.load(std::memory_order_relaxed);
    if (c >= 0) return c == 1;
    int v = 0;
    GetEnvVarInt("MG_FRAME_DIAG", &v, 0);
    if (v <= 0 && ::initialized) {
        const int cfg = config_get_int(const_cast<char*>("frameDiagnostics"));
        if (cfg > 0) v = cfg;
    }
    if (v > 0) {
        cached.store(1, std::memory_order_relaxed);
        return true;
    }
    // Only a settled "no" is cached; before the config is loaded it may still turn on.
    if (::initialized) cached.store(0, std::memory_order_relaxed);
    return false;
}

// Free-form lifecycle event (surface created/destroyed, swap interval set, ...).
inline void event(const char* fmt, ...) {
    if (!enabled()) return;
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    detail::ensure_open(s);
    detail::write_line(s, "t=%7.1fs EVENT %s", detail::to_ms(detail::now_ns() - s.t0) / 1000.0, msg);
}

// Call immediately before the swap. Returns true when the caller should query the
// surface size and pass it to note_surface_size (first frame, new surface, or
// every kSizeProbeFrames frames).
inline bool swap_begin(EGLDisplay dpy, EGLSurface surface) {
    if (!enabled()) return false;
    const int64_t t = detail::now_ns();
    detail::tls_begin() = t;
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.capped) return false;
    detail::ensure_open(s);
    bool probe = false;
    if (dpy != s.last_dpy || surface != s.last_surface) {
        detail::write_line(s, "t=%7.1fs EVENT presenting to surface %p (display %p)",
                           detail::to_ms(t - s.t0) / 1000.0, surface, dpy);
        s.last_dpy = dpy;
        s.last_surface = surface;
        s.have_prev = false; // do not count the gap between two surfaces as a frame interval
        s.frames_since_probe = 0;
        probe = true;
    } else if (++s.frames_since_probe >= detail::kSizeProbeFrames) {
        s.frames_since_probe = 0;
        probe = true;
    }
    if (s.have_prev) {
        s.interval.push_back(detail::to_ms(t - s.prev_begin));
        s.work.push_back(detail::to_ms(t - s.prev_end));
    }
    s.prev_begin = t;
    return probe;
}

// Call immediately after the swap returns.
inline void swap_end() {
    if (!enabled()) return;
    const int64_t t = detail::now_ns();
    const int64_t b = detail::tls_begin();
    if (b == 0) return;
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.capped) return;
    s.prev_end = t;
    s.have_prev = true;
    s.swap.push_back(detail::to_ms(t - b));
    if (s.window_start == 0) s.window_start = b;
    if (t - s.window_start >= detail::kWindowNs) detail::flush_window(s, t);
}

// Logged only when it changes, so a resize shows up as one line.
inline void note_surface_size(int w, int h) {
    if (!enabled()) return;
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (w == s.last_w && h == s.last_h) return;
    s.last_w = w;
    s.last_h = h;
    detail::ensure_open(s);
    detail::write_line(s, "t=%7.1fs EVENT surface size %dx%d", detail::to_ms(detail::now_ns() - s.t0) / 1000.0, w, h);
}

} // namespace mg_framediag

#endif // MOBILEGLUES_EGL_FRAMEDIAG_H
