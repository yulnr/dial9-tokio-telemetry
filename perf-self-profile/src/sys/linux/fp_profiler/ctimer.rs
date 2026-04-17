//! Per-thread CPU timer engine, equivalent to async-profiler's `-e ctimer`.
//!
//! Uses `timer_create(CLOCK_THREAD_CPUTIME_ID, ...)` with `SIGEV_THREAD_ID`
//! so each thread gets its own timer that fires SIGPROF *on that thread*
//! when it has consumed N nanoseconds of CPU time. This avoids the two main
//! itimer biases:
//!
//!   1. itimer delivers SIGPROF to the process; the kernel picks a thread,
//!      and that pick is not weighted by CPU usage. Busy threads get
//!      undersampled relative to their actual CPU consumption.
//!   2. Only one itimer signal can be pending per process at a time, so on
//!      multi-core workloads you systematically lose samples.
//!
//! ctimer fixes both because the timer is bound to a specific tid via the
//! sigevent's _tid field, and each thread accumulates its own CPU time.
//!
//! # Lifecycle
//!
//! 1. Call `start(interval_ns)` once from the main thread. This installs the
//!    SIGPROF handler.
//! 2. Each thread that wants to be profiled calls `register_thread()` from
//!    its own context. This creates the per-thread timer and arms it.
//! 3. On thread exit, call `unregister_thread()` to delete the timer. (Real
//!    profilers hook pthread_create / thread cleanup; we expose this manually.)
//! 4. Call `stop()` to disable sampling globally.

mod imp {
    use std::ptr;
    use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

    const SIGEV_THREAD_ID: libc::c_int = 4;

    #[repr(C)]
    union SigvalUnion {
        sival_int: libc::c_int,
        sival_ptr: *mut libc::c_void,
    }

    // Layout must match glibc's `struct sigevent` (64 bytes on LP64).
    // We split the _sigev_un union into an explicit _tid field (which
    // SIGEV_THREAD_ID needs) plus padding for the remaining union bytes.
    #[repr(C)]
    struct Sigevent {
        sigev_value: SigvalUnion,
        sigev_signo: libc::c_int,
        sigev_notify: libc::c_int,
        sigev_notify_thread_id: libc::c_int,
        _pad: [libc::c_int; 11],
    }

    const _: () = assert!(
        std::mem::size_of::<Sigevent>() == 64,
        "Sigevent layout must match kernel struct sigevent (64 bytes)"
    );

    unsafe extern "C" {
        fn timer_create(
            clockid: libc::clockid_t,
            sevp: *mut Sigevent,
            timerid: *mut libc::timer_t,
        ) -> libc::c_int;
        fn timer_settime(
            timerid: libc::timer_t,
            flags: libc::c_int,
            new_value: *const libc::itimerspec,
            old_value: *mut libc::itimerspec,
        ) -> libc::c_int;
        fn timer_delete(timerid: libc::timer_t) -> libc::c_int;
    }

    static INTERVAL_NS: AtomicI64 = AtomicI64::new(0);
    static RUNNING: AtomicBool = AtomicBool::new(false);

    thread_local! {
        static THREAD_TIMER: std::cell::Cell<Option<libc::timer_t>> =
            const { std::cell::Cell::new(None) };
    }

    /// Install SIGPROF handler and remember the sampling interval.
    ///
    /// # Safety
    /// Modifies process-global signal state. Call once.
    pub unsafe fn start(
        interval_ns: i64,
        handler: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void),
    ) -> Result<(), std::io::Error> {
        if interval_ns <= 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "interval must be positive",
            ));
        }

        let mut sa: libc::sigaction = unsafe { std::mem::zeroed() };
        sa.sa_sigaction = handler as usize;
        sa.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
        unsafe { libc::sigemptyset(&mut sa.sa_mask) };

        if unsafe { libc::sigaction(libc::SIGPROF, &sa, ptr::null_mut()) } != 0 {
            return Err(std::io::Error::last_os_error());
        }

        INTERVAL_NS.store(interval_ns, Ordering::Release);
        RUNNING.store(true, Ordering::Release);
        Ok(())
    }

    pub fn stop() {
        RUNNING.store(false, Ordering::Release);
    }

    /// Re-enable sampling after `stop()`. Per-thread timers keep firing;
    /// this just tells the signal handler to record samples again.
    pub fn resume() {
        RUNNING.store(true, Ordering::Release);
    }

    pub fn is_running() -> bool {
        RUNNING.load(Ordering::Acquire)
    }

    /// Returns the sampling interval in nanoseconds.
    pub fn interval_ns() -> i64 {
        INTERVAL_NS.load(Ordering::Relaxed)
    }

    /// Returns the calling thread's timer handle, or `None` if not registered.
    ///
    /// Called from the SIGPROF handler (same thread as the timer) to pass to
    /// `timer_getoverrun` for accurate sample weighting.
    pub fn current_thread_timer_id() -> Option<libc::timer_t> {
        THREAD_TIMER.with(|c| c.get())
    }

    /// Create and arm a per-thread CPU timer for the *calling* thread.
    ///
    /// # Safety
    /// Calls into the kernel timer API.
    pub fn register_thread() -> Result<(), std::io::Error> {
        if !RUNNING.load(Ordering::Acquire) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "ctimer is not running (call start first)",
            ));
        }
        let interval = INTERVAL_NS.load(Ordering::Acquire);

        let existing = THREAD_TIMER.with(|c| c.get());
        if existing.is_some() {
            return Ok(());
        }

        let tid = gettid();

        let mut sev: Sigevent = unsafe { std::mem::zeroed() };
        sev.sigev_notify = SIGEV_THREAD_ID;
        sev.sigev_signo = libc::SIGPROF;
        sev.sigev_notify_thread_id = tid;
        sev.sigev_value = SigvalUnion { sival_int: tid };

        let mut timerid: libc::timer_t = ptr::null_mut();
        if unsafe { timer_create(libc::CLOCK_THREAD_CPUTIME_ID, &mut sev, &mut timerid) } != 0 {
            return Err(std::io::Error::last_os_error());
        }

        let sec = interval / 1_000_000_000;
        let nsec = interval % 1_000_000_000;
        let spec = libc::itimerspec {
            it_interval: libc::timespec {
                tv_sec: sec,
                tv_nsec: nsec,
            },
            it_value: libc::timespec {
                tv_sec: sec,
                tv_nsec: nsec,
            },
        };

        if unsafe { timer_settime(timerid, 0, &spec, ptr::null_mut()) } != 0 {
            let err = std::io::Error::last_os_error();
            unsafe { timer_delete(timerid) };
            return Err(err);
        }

        THREAD_TIMER.with(|c| c.set(Some(timerid)));
        Ok(())
    }

    pub fn unregister_thread() {
        THREAD_TIMER.with(|c| {
            if let Some(t) = c.take() {
                let zero: libc::itimerspec = unsafe { std::mem::zeroed() };
                unsafe { timer_settime(t, 0, &zero, ptr::null_mut()) };
                unsafe { timer_delete(t) };
            }
        });
    }

    #[inline]
    fn gettid() -> libc::c_int {
        unsafe { libc::syscall(libc::SYS_gettid) as libc::c_int }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        extern "C" fn dummy_handler(
            _: libc::c_int,
            _: *mut libc::siginfo_t,
            _: *mut libc::c_void,
        ) {
        }

        #[test]
        fn start_rejects_zero_interval() {
            let err = unsafe { start(0, dummy_handler) }.unwrap_err();
            assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        }

        #[test]
        fn start_rejects_negative_interval() {
            let err = unsafe { start(-1, dummy_handler) }.unwrap_err();
            assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        }

        #[test]
        fn register_thread_fails_when_not_running() {
            RUNNING.store(false, Ordering::Release);
            let err = register_thread().unwrap_err();
            assert!(err.to_string().contains("not running"));
        }
    }
}

pub use imp::{
    current_thread_timer_id, interval_ns, is_running, register_thread, resume, start, stop,
    unregister_thread,
};
