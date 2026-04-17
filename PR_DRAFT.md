# Title

feat: ctimer-based CPU profiling fallback when perf_event_open is blocked

# Description

On ECS/Fargate (and similar environments where seccomp blocks `perf_event_open`), CPU profiling was silently broken. This adds an automatic fallback using per-thread CPU timers + userspace frame-pointer unwinding, ported from async-profiler's approach.

When `perf_event_open` fails with EACCES/EPERM/ENOSYS/EOPNOTSUPP, `PerfSampler` transparently switches to ctimer mode. The rest of the pipeline sees the same `Sample` type — symbolization, trace writing, and the viewer all work unchanged.

The core of the fallback lives in the new `fp_profiler` module inside `perf-self-profile`:
- **safe load**: tiny asm trampoline + SIGSEGV handler for fault-tolerant pointer reads (so a corrupt FP chain aborts the walk instead of crashing the process)
- **FP unwinder**: walks the rbp/x29 chain from the signal handler's ucontext, with dead zone checks, max frame size validation, and aarch64 PAC bit stripping
- **ctimer engine**: `timer_create(CLOCK_THREAD_CPUTIME_ID)` with `SIGEV_THREAD_ID` per thread, fires SIGPROF when CPU time is consumed. Avoids the two main itimer biases (arbitrary thread delivery + single pending signal)
- **lock-free sample buffer**: static ring buffer for signal handler -> flush thread communication, no allocation in the signal path

Tested both backends side by side in Docker — sample counts, callchain depths, and leaf frame distributions are comparable. ctimer stacks are 1 frame shallower (FP walker stops at userspace boundary, perf's kernel unwinder sees vDSO), which is expected and not a problem.

Also cleaned up the sampler internals: extracted a `SamplerBackend` trait so the perf and ctimer backends implement a shared interface, and split `sampler.rs` into a dispatch layer (`sampler.rs`) + perf backend (`perf_sampler.rs`) to mirror `ctimer_sampler.rs`.

`new_per_thread` now does a probe `perf_event_open` call so failure is detected at creation time rather than silently swallowed on every `track_current_thread()`. Sched event profiling (context switches) has no userspace fallback — it requires kernel support — so when perf is blocked it returns a clear error instead of silently producing zero data.

`DIAL9_FORCE_CTIMER=1` env var forces ctimer even when perf works, useful for testing.
