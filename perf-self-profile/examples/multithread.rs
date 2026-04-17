//! Example: profile all threads spawned by the process.
//!
//! Build with frame pointers:
//!   RUSTFLAGS="-C force-frame-pointers=yes" cargo run --release --example multithread
//!
//! Writes collapsed stacks to a file (default: `multithread.folded`).
//! Use a CLI arg to change the output path:
//!   cargo run --release --example multithread -- ctimer.folded

use dial9_perf_self_profile::{
    EventSource, PerfSampler, SamplerConfig, SamplingMode, ctimer_register_thread,
    ctimer_unregister_thread, resolve_symbol,
};
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "multithread.folded".to_string());

    let mut sampler = match PerfSampler::start(SamplerConfig {
        sampling: SamplingMode::FrequencyHz(999),
        event_source: EventSource::SwCpuClock,
        include_kernel: false,
    }) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to start sampler: {e}");
            eprintln!("Try: echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid");
            std::process::exit(1);
        }
    };

    let stop = Arc::new(AtomicBool::new(false));

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let stop = stop.clone();
            thread::Builder::new()
                .name(format!("worker-{i}"))
                .spawn(move || {
                    let _ = ctimer_register_thread();
                    let result = cpu_work(&stop);
                    ctimer_unregister_thread();
                    result
                })
                .unwrap()
        })
        .collect();

    // Let threads run
    thread::sleep(std::time::Duration::from_secs(1));
    stop.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().unwrap();
    }

    sampler.disable();
    let samples = sampler.drain_samples();
    eprintln!("Collected {} samples", samples.len());

    // Write collapsed stacks (inferno/flamegraph format).
    // Each line: "sym1;sym2;sym3 weight\n" (deepest frame last).
    let mut file = std::fs::File::create(&out_path).expect("failed to create output file");
    for sample in &samples {
        let syms: Vec<String> = sample
            .callchain
            .iter()
            .rev()
            .map(|addr| {
                let info = resolve_symbol(*addr);
                info.name.unwrap_or_else(|| format!("{:#x}", addr))
            })
            .collect();
        if !syms.is_empty() {
            writeln!(file, "{} {}", syms.join(";"), sample.period).unwrap();
        }
    }
    eprintln!("Wrote collapsed stacks to {out_path}");

    // Show samples per thread
    let mut by_tid: HashMap<u32, usize> = HashMap::new();
    for s in &samples {
        *by_tid.entry(s.tid).or_default() += 1;
    }
    let mut tids: Vec<_> = by_tid.into_iter().collect();
    tids.sort_by_key(|b| std::cmp::Reverse(b.1));
    for (tid, count) in &tids {
        eprintln!("  tid={tid}: {count} samples");
    }
}

#[inline(never)]
fn cpu_work(stop: &AtomicBool) -> u64 {
    let mut sum = 0u64;
    let mut i = 0u64;
    while !stop.load(Ordering::Relaxed) {
        sum = sum.wrapping_add(i);
        std::hint::black_box(sum);
        i += 1;
    }
    sum
}
