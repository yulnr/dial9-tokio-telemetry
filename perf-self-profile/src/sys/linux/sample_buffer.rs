//! Lock-free sample buffer for signal handler → flush thread communication.
//!
//! The SIGPROF signal handler (fired by ctimer) writes samples into this
//! fixed-size static buffer. The flush thread drains completed samples.
//!
//! Design:
//! - Static array of `SampleSlot` + `AtomicUsize` write index.
//! - Signal handler claims a unique slot via `fetch_add` on the write index.
//! - Per-slot `AtomicU8` state field ensures the flush thread only reads
//!   fully-written samples.
//! - No allocation, no locks — async-signal-safe on the write path.

use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

use super::fp_profiler::unwind::MAX_FRAMES;

/// Buffer capacity. Must be large enough to absorb bursts between drains.
/// At 100 Hz across 16 threads, drained every 100ms = ~160 samples/drain.
/// 16384 gives ~100x headroom.
const BUFFER_CAP: usize = 16384;

/// Slot states.
const SLOT_EMPTY: u8 = 0;
const SLOT_WRITING: u8 = 1;
const SLOT_READY: u8 = 2;

/// A single sample slot. Fixed-size, no heap allocation.
///
/// Layout is carefully ordered: `state` first so the flush thread can
/// skip-scan efficiently.
#[repr(C)]
struct SampleSlot {
    state: AtomicU8,
    pid: u32,
    tid: u32,
    time: u64,
    cpu: u32,
    period: u64,
    num_frames: u32,
    frames: [u64; MAX_FRAMES],
}

impl SampleSlot {
    const fn new() -> Self {
        Self {
            state: AtomicU8::new(SLOT_EMPTY),
            pid: 0,
            tid: 0,
            time: 0,
            cpu: 0,
            period: 0,
            num_frames: 0,
            frames: [0u64; MAX_FRAMES],
        }
    }
}

// SAFETY: SampleSlot is only accessed via atomic state coordination.
// Signal handlers write only to slots they've claimed (unique index),
// and the flush thread reads only READY slots.
unsafe impl Sync for SampleSlot {}

/// The global sample buffer. Static so signal handlers can access it
/// without any indirection.
static BUFFER: SampleBuffer = SampleBuffer::new();

struct SampleBuffer {
    slots: [SampleSlot; BUFFER_CAP],
    /// Monotonically increasing write cursor. Signal handlers `fetch_add`
    /// this to claim a unique slot index (index % BUFFER_CAP).
    write_idx: AtomicUsize,
    /// Read cursor for the drain path. Only advanced by the flush thread
    /// (single consumer).
    read_idx: AtomicUsize,
    /// Count of dropped samples (buffer full).
    dropped: AtomicUsize,
}

impl SampleBuffer {
    const fn new() -> Self {
        const EMPTY_SLOT: SampleSlot = SampleSlot::new();
        Self {
            slots: [EMPTY_SLOT; BUFFER_CAP],
            write_idx: AtomicUsize::new(0),
            read_idx: AtomicUsize::new(0),
            dropped: AtomicUsize::new(0),
        }
    }
}

/// Data extracted from a sample slot for the flush thread.
pub(crate) struct DrainedSample {
    pub pid: u32,
    pub tid: u32,
    pub time: u64,
    pub cpu: u32,
    /// Effective sampling period in nanoseconds, accounting for timer overruns.
    /// `interval_ns * (1 + overrun_count)`, used as sample weight in flamegraphs.
    pub period: u64,
    pub num_frames: u32,
    pub frames: [u64; MAX_FRAMES],
}

/// Claim a slot for writing from a signal handler. Returns a mutable
/// reference to the slot's data fields (excluding `state`).
///
/// # Safety
/// - Must be called from a signal handler context (async-signal-safe).
/// - The returned pointer is valid only until the caller calls [`commit_slot`].
/// - Only one signal handler invocation writes to a given slot (guaranteed
///   by the atomic `fetch_add`).
pub(crate) unsafe fn claim_slot() -> Option<SlotWriter> {
    let idx = BUFFER.write_idx.fetch_add(1, Ordering::Relaxed);
    let slot_idx = idx % BUFFER_CAP;
    let slot = &BUFFER.slots[slot_idx];

    // If the slot isn't empty, the buffer wrapped around and the flush
    // thread hasn't caught up. Drop this sample.
    if slot
        .state
        .compare_exchange(
            SLOT_EMPTY,
            SLOT_WRITING,
            Ordering::Acquire,
            Ordering::Relaxed,
        )
        .is_err()
    {
        BUFFER.dropped.fetch_add(1, Ordering::Relaxed);
        return None;
    }

    Some(SlotWriter {
        slot: slot as *const SampleSlot as *mut SampleSlot,
        committed: false,
    })
}

/// Handle for writing sample data into a claimed slot.
///
/// If dropped without calling [`commit`](Self::commit), the slot is
/// committed with `num_frames = 0` so the drain cursor can advance past it
/// (a permanently stuck SLOT_WRITING would block all subsequent drains).
pub(crate) struct SlotWriter {
    slot: *mut SampleSlot,
    committed: bool,
}

impl SlotWriter {
    #[inline]
    pub(crate) unsafe fn write(&mut self, pid: u32, tid: u32, time: u64, cpu: u32, period: u64) {
        let s = unsafe { &mut *self.slot };
        s.pid = pid;
        s.tid = tid;
        s.time = time;
        s.cpu = cpu;
        s.period = period;
    }

    #[inline]
    pub(crate) unsafe fn write_frames(&mut self, frames: &[u64], count: u32) {
        let s = unsafe { &mut *self.slot };
        s.num_frames = count;
        let copy_len = (count as usize).min(MAX_FRAMES);
        // SAFETY: both slices are within bounds (frames comes from unwind
        // which caps at MAX_FRAMES, and s.frames has MAX_FRAMES capacity).
        unsafe {
            core::ptr::copy_nonoverlapping(frames.as_ptr(), s.frames.as_mut_ptr(), copy_len);
        }
    }

    #[inline]
    pub(crate) unsafe fn commit(mut self) {
        self.committed = true;
        unsafe { (*self.slot).state.store(SLOT_READY, Ordering::Release) };
    }
}

impl Drop for SlotWriter {
    fn drop(&mut self) {
        if !self.committed {
            // Abandoned slot: commit with zero frames so drain can advance
            // past it. A permanently stuck SLOT_WRITING would block all
            // subsequent drains.
            unsafe {
                (*self.slot).num_frames = 0;
                (*self.slot).state.store(SLOT_READY, Ordering::Release);
            }
        }
    }
}

/// Drain all ready samples, calling `f` for each one.
///
/// Single-consumer: must be called from one thread only (the flush thread).
pub(crate) fn drain(mut f: impl FnMut(DrainedSample)) {
    let write = BUFFER.write_idx.load(Ordering::Acquire);
    let mut read = BUFFER.read_idx.load(Ordering::Relaxed);

    while read < write {
        let slot_idx = read % BUFFER_CAP;
        let slot = &BUFFER.slots[slot_idx];

        // If the slot isn't READY yet, the signal handler is still writing.
        // Since we process in order, stop here and retry next drain cycle.
        if slot.state.load(Ordering::Acquire) != SLOT_READY {
            break;
        }

        // Read out the sample data.
        let sample = DrainedSample {
            pid: slot.pid,
            tid: slot.tid,
            time: slot.time,
            cpu: slot.cpu,
            period: slot.period,
            num_frames: slot.num_frames,
            frames: slot.frames,
        };

        // Reset slot to empty.
        slot.state.store(SLOT_EMPTY, Ordering::Release);

        read += 1;
        f(sample);
    }

    BUFFER.read_idx.store(read, Ordering::Relaxed);
}

/// Returns true if there are pending samples to read.
pub(crate) fn has_pending() -> bool {
    let write = BUFFER.write_idx.load(Ordering::Acquire);
    let read = BUFFER.read_idx.load(Ordering::Relaxed);
    read < write
}

/// Returns and resets the count of dropped samples since last call.
pub(crate) fn take_dropped_count() -> usize {
    BUFFER.dropped.swap(0, Ordering::Relaxed)
}

#[cfg(test)]
fn reset_buffer() {
    // Reset all slots and cursors. Only safe in single-threaded tests.
    for slot in &BUFFER.slots {
        slot.state.store(SLOT_EMPTY, Ordering::Relaxed);
    }
    BUFFER.write_idx.store(0, Ordering::Relaxed);
    BUFFER.read_idx.store(0, Ordering::Relaxed);
    BUFFER.dropped.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests must run sequentially (shared static buffer). Use `cargo test
    // -- --test-threads=1` or accept that ordering is nondeterministic but
    // each test resets the buffer first.

    #[test]
    fn round_trip_single_sample() {
        reset_buffer();

        unsafe {
            let mut slot = claim_slot().expect("should claim slot");
            slot.write(1000, 42, 999_000_000, 3, 10_000_000);
            let frames = [0x1234u64, 0x5678, 0x9abc];
            slot.write_frames(&frames, 3);
            slot.commit();
        }

        assert!(has_pending());

        let mut got = Vec::new();
        drain(|s| got.push(s));
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].pid, 1000);
        assert_eq!(got[0].tid, 42);
        assert_eq!(got[0].time, 999_000_000);
        assert_eq!(got[0].cpu, 3);
        assert_eq!(got[0].period, 10_000_000);
        assert_eq!(got[0].num_frames, 3);
        assert_eq!(got[0].frames[0], 0x1234);
        assert_eq!(got[0].frames[1], 0x5678);
        assert_eq!(got[0].frames[2], 0x9abc);

        assert!(!has_pending());
    }

    #[test]
    fn dropped_slot_writer_commits_empty_so_drain_advances() {
        reset_buffer();

        // Write and commit one sample
        unsafe {
            let mut slot = claim_slot().unwrap();
            slot.write(1, 1, 100, 0, 0);
            slot.write_frames(&[0xAA], 1);
            slot.commit();
        }
        // Claim a second but drop without commit — Drop auto-commits with
        // num_frames=0 so drain isn't permanently blocked.
        unsafe {
            let mut slot = claim_slot().unwrap();
            slot.write(2, 2, 200, 0, 0);
            slot.write_frames(&[0xBB], 1);
            // no commit — Drop handles it
            drop(slot);
        }
        // Write and commit a third
        unsafe {
            let mut slot = claim_slot().unwrap();
            slot.write(3, 3, 300, 0, 0);
            slot.write_frames(&[0xCC], 1);
            slot.commit();
        }

        // Drain sees all three: real, abandoned (0 frames), real
        let mut got = Vec::new();
        drain(|s| got.push(s));
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].pid, 1);
        assert_eq!(got[1].pid, 2);
        assert_eq!(got[1].num_frames, 0); // abandoned slot
        assert_eq!(got[2].pid, 3);
    }

    #[test]
    fn take_dropped_count_resets() {
        reset_buffer();

        let count = take_dropped_count();
        assert_eq!(count, 0);

        // Artificially bump the dropped counter
        BUFFER.dropped.store(5, Ordering::Relaxed);
        assert_eq!(take_dropped_count(), 5);
        assert_eq!(take_dropped_count(), 0);
    }

    #[test]
    fn empty_drain_is_noop() {
        reset_buffer();

        assert!(!has_pending());
        let mut count = 0;
        drain(|_| count += 1);
        assert_eq!(count, 0);
    }
}
