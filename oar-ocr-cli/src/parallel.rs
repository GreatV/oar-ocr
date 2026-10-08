//! Page-level data parallelism across devices.
//!
//! One worker thread per requested device, each holding its own pipeline
//! replica built by a per-worker factory. Work is handed out as chunks (the
//! same chunk size the single-device path uses) so detection batching
//! survives, under a sliding window: a chunk may only be dispatched once the
//! ordered writer has progressed within `WINDOW_BASE × replicas` chunks of
//! it, so the reorder buffer and the rendered-but-unwritten work stay
//! bounded no matter the input size. Completed chunks are delivered to
//! `on_chunk` strictly in input order while the workers keep going. The
//! first failed chunk aborts the run once the writer reaches it, and
//! dispatch stops so no new chunks are taken. A worker whose replica
//! fails to build reports that error and leaves; if no replica could be
//! built at all, the first build error is returned.

use anyhow::Result;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};

/// In-flight window multiplier over the replica count.
const WINDOW_BASE: usize = 2;

/// The ordered writer's progress, gating dispatch.
struct WriteGate {
    next_to_write: Mutex<usize>,
    progressed: Condvar,
}

impl WriteGate {
    fn new() -> Self {
        Self {
            next_to_write: Mutex::new(0),
            progressed: Condvar::new(),
        }
    }

    /// Block until dispatching `index` is allowed (the writer has come
    /// within `limit` chunks) or the run is aborting.
    fn wait_admissible(&self, index: usize, limit: usize, aborting: &AtomicBool) -> bool {
        let mut next = self.next_to_write.lock().expect("gate poisoned");
        loop {
            if aborting.load(Ordering::SeqCst) {
                return false;
            }
            if *next + limit > index {
                return true;
            }
            next = self.progressed.wait(next).expect("gate poisoned");
        }
    }

    fn advance(&self, next: usize) {
        let mut guard = self.next_to_write.lock().expect("gate poisoned");
        *guard = next;
        self.progressed.notify_all();
    }

    /// Abort the run: set the flag and wake a dispatcher parked on the
    /// window. Storing under the gate mutex pairs with the check in
    /// `wait_admissible`, so a dispatcher about to park cannot miss it.
    fn abort(&self, aborting: &AtomicBool) {
        let _guard = self.next_to_write.lock().expect("gate poisoned");
        aborting.store(true, Ordering::SeqCst);
        self.progressed.notify_all();
    }
}

/// Run `process` concurrently, one replica per worker, streaming results in
/// input order through `on_chunk`.
///
/// When `replica_count` is at most one, or there is at most one job,
/// everything runs inline on the caller's replica — single-device behavior
/// is exactly today's. A job's failure is delivered to `on_chunk` at its
/// input position and the run stops there; if `on_chunk` itself fails, the
/// run aborts the same way. In both cases the offending error is returned.
/// A worker whose replica fails to build leaves the pool and the run
/// proceeds on the rest; if none could be built, that build error is
/// returned.
pub(crate) fn run_parallel<M, T, R, B, F, O>(
    replica_count: usize,
    jobs: Vec<T>,
    make_replica: B,
    process: F,
    mut on_chunk: O,
) -> Result<()>
where
    T: Send + 'static,
    R: Send,
    // Each replica is built and used on its own worker thread, so it need not
    // be `Send` (CUDA-graph-holding parsers are not).
    B: Fn() -> Result<M> + Send + Sync,
    F: Fn(&mut M, usize, T) -> Result<R> + Send + Sync + 'static,
    O: FnMut(usize, Result<R>) -> Result<()>,
{
    let job_count = jobs.len();
    if replica_count <= 1 || job_count <= 1 {
        let mut replica = first_replica(replica_count.max(1), &make_replica)?;
        for (index, job) in jobs.into_iter().enumerate() {
            on_chunk(index, process(&mut replica, index, job))?;
        }
        return Ok(());
    }
    let window_limit = WINDOW_BASE.max(1) * replica_count;
    let (work_tx, work_rx) = mpsc::channel::<(usize, T)>();
    let (result_tx, result_rx) = mpsc::sync_channel::<(usize, Result<R>)>(window_limit);
    let work_rx = Arc::new(Mutex::new(work_rx));
    let gate = Arc::new(WriteGate::new());
    let aborting = Arc::new(AtomicBool::new(false));
    let make_replica = Arc::new(make_replica);
    // Replicas are built one at a time: the first build populates the model
    // cache, so the others load from it instead of racing to download the
    // same files.
    let build_lock = Arc::new(Mutex::new(()));
    let process = Arc::new(process);

    std::thread::scope(|scope| -> Result<()> {
        // Owned here so the writer can drop it on exit: workers still
        // sending then fail fast instead of blocking the scope join.
        let result_rx = result_rx;
        for _ in 0..replica_count {
            let work_rx = Arc::clone(&work_rx);
            let result_tx = result_tx.clone();
            let make_replica = Arc::clone(&make_replica);
            let build_lock = Arc::clone(&build_lock);
            let process = Arc::clone(&process);
            let aborting = Arc::clone(&aborting);
            scope.spawn(move || {
                let built = {
                    let _guard = build_lock.lock().expect("build lock poisoned");
                    // A run that already failed needs no more replicas.
                    if aborting.load(Ordering::SeqCst) {
                        return;
                    }
                    make_replica()
                };
                let mut replica = match built {
                    Ok(replica) => replica,
                    Err(error) => {
                        // No replica, no chunks: report and let the writer
                        // mark the run short.
                        let _ = result_tx.send((usize::MAX, Err(error)));
                        return;
                    }
                };
                loop {
                    // Bind the recv so the queue mutex guard drops before
                    // processing — otherwise one worker holds the queue
                    // locked for its whole chunk and the rest serialize.
                    let job = work_rx.lock().expect("work queue poisoned").recv();
                    let Ok((index, job)) = job else {
                        break;
                    };
                    // Skip chunks still queued when the run aborts.
                    if aborting.load(Ordering::SeqCst) {
                        break;
                    }
                    let result = process(&mut replica, index, job);
                    if result_tx.send((index, result)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(result_tx);

        // Dispatcher: feeds the workers under the sliding window.
        {
            let work_tx = work_tx.clone();
            let gate = Arc::clone(&gate);
            let aborting = Arc::clone(&aborting);
            scope.spawn(move || {
                for (index, job) in jobs.into_iter().enumerate() {
                    if !gate.wait_admissible(index, window_limit, &aborting) {
                        break;
                    }
                    if work_tx.send((index, job)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(work_tx);

        // Ordered writer: deliver completed chunks in input order and
        // advance the window so dispatch keeps flowing. The reorder buffer
        // never holds more than the window limit because dispatch cannot
        // outrun the writer by more than that.
        let mut buffered: BTreeMap<usize, Result<R>> = BTreeMap::new();
        // Real chunk results only; build-failure reports do not count.
        let mut received = 0usize;
        let mut next_to_write = 0usize;
        let mut failure: Option<anyhow::Error> = None;
        let mut build_failure: Option<anyhow::Error> = None;
        while received < job_count && failure.is_none() {
            let Ok((index, result)) = result_rx.recv() else {
                break;
            };
            if index == usize::MAX {
                // A worker could not build its replica. If every worker
                // reports this, the channel disconnects below and the
                // first error surfaces; if some worker succeeded, the run
                // merely proceeds with fewer replicas.
                build_failure = build_failure.or(result.err());
                continue;
            }
            received += 1;
            buffered.insert(index, result);
            while let Some(result) = buffered.remove(&next_to_write) {
                if failure.is_none() {
                    // The failure is delivered at its ordered position, like
                    // the inline path's `?`, and stops the stream there.
                    let failed = result.is_err();
                    if let Err(error) = on_chunk(next_to_write, result) {
                        failure = Some(error);
                    } else if failed {
                        failure = Some(anyhow::anyhow!("chunk {next_to_write} failed"));
                    }
                    if failure.is_some() {
                        gate.abort(&aborting);
                    }
                }
                next_to_write += 1;
                gate.advance(next_to_write);
            }
        }
        // Stop receiving: any worker still sending a result (or a
        // build-failure report) now gets an error and exits, so a full
        // result channel cannot block the scope join.
        drop(result_rx);
        // Wake a dispatcher still parked on the window before joining it —
        // storing the flag alone would leave it asleep forever.
        gate.abort(&aborting);
        if failure.is_none() && next_to_write < job_count {
            failure = match build_failure {
                // Nothing was written, so the run never started: the build
                // error says why.
                Some(error) if next_to_write == 0 => Some(error),
                _ => Some(anyhow::anyhow!(
                    "a worker stopped before finishing its chunks"
                )),
            };
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    })
}

/// Builds replicas until one succeeds, trying each listed device once, so a
/// run that needs a single replica is not failed by one bad device entry.
fn first_replica<M>(attempts: usize, make_replica: impl Fn() -> Result<M>) -> Result<M> {
    let mut first_error = None;
    for _ in 0..attempts {
        match make_replica() {
            Ok(replica) => return Ok(replica),
            Err(error) => {
                first_error.get_or_insert(error);
            }
        }
    }
    Err(first_error.expect("at least one build attempt"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn streams_in_order_and_bounds_the_reorder_window() {
        let jobs: Vec<usize> = (0..64).collect();
        let replicas = 3;
        let max_concurrent = Arc::new(AtomicUsize::new(0));
        let active = Arc::new(AtomicUsize::new(0));
        let delivered = Arc::new(AtomicUsize::new(0));
        let inflight = Arc::new(AtomicUsize::new(0));
        let max_inflight = Arc::new(AtomicUsize::new(0));
        run_parallel(
            replicas,
            jobs,
            || Ok(Vec::<u8>::new()),
            {
                let max_concurrent = Arc::clone(&max_concurrent);
                let active = Arc::clone(&active);
                let inflight = Arc::clone(&inflight);
                let max_inflight = Arc::clone(&max_inflight);
                move |_replica, _index, job| {
                    let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                    max_concurrent.fetch_max(now, Ordering::SeqCst);
                    inflight.fetch_add(1, Ordering::SeqCst);
                    max_inflight.fetch_max(inflight.load(Ordering::SeqCst), Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    inflight.fetch_sub(1, Ordering::SeqCst);
                    active.fetch_sub(1, Ordering::SeqCst);
                    Ok(job * 2)
                }
            },
            {
                let delivered = Arc::clone(&delivered);
                move |index, result| {
                    assert_eq!(result.unwrap(), index * 2, "chunks stream in input order");
                    delivered.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
            },
        )
        .expect("run succeeds");
        assert_eq!(delivered.load(Ordering::SeqCst), 64);
        assert!(max_concurrent.load(Ordering::SeqCst) > 1, "workers overlap");
        // Dispatch cannot outrun the writer by more than the window.
        assert!(
            max_inflight.load(Ordering::SeqCst) <= WINDOW_BASE * replicas,
            "in-flight chunks {} exceeded the window {}",
            max_inflight.load(Ordering::SeqCst),
            WINDOW_BASE * replicas
        );
    }

    #[test]
    fn aborts_at_first_failed_chunk_in_order() {
        let jobs: Vec<usize> = (0..16).collect();
        let delivered = Arc::new(AtomicUsize::new(0));
        let failure_index = 4usize;
        let run = run_parallel(
            2,
            jobs,
            || Ok(()),
            move |_replica, index, _job| {
                std::thread::sleep(std::time::Duration::from_micros(100));
                if index == failure_index {
                    anyhow::bail!("chunk {index} failed");
                }
                Ok(index)
            },
            {
                let delivered = Arc::clone(&delivered);
                move |index, result| {
                    if index == failure_index {
                        assert!(result.is_err(), "the failure arrives at its position");
                    } else {
                        assert!(result.is_ok());
                    }
                    delivered.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
            },
        );
        assert!(run.is_err(), "the run reports the failure");
        // The failure is delivered at its position, so everything before it
        // was written and nothing after it.
        assert_eq!(delivered.load(Ordering::SeqCst), failure_index + 1);
    }

    #[test]
    fn reports_the_build_error_when_no_replica_could_be_built() {
        // More jobs than the window, so a dispatcher nobody wakes would
        // park forever: the run must return the build error promptly.
        let jobs: Vec<usize> = (0..32).collect();
        let written = Arc::new(AtomicUsize::new(0));
        let run = run_parallel(
            3,
            jobs,
            || Err(anyhow::anyhow!("no such device")),
            |_: &mut (), _index, _job| Ok(()),
            {
                let written = Arc::clone(&written);
                move |_index, _result| {
                    written.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
            },
        );
        let error = run.expect_err("the build failure is returned");
        assert_eq!(written.load(Ordering::SeqCst), 0, "nothing is written");
        assert!(
            error.to_string().contains("no such device"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn aborts_promptly_when_the_surviving_replica_fails() {
        // The first replica builds and its first chunk fails, so the writer
        // stops receiving; the other two builds fail only afterwards. Their
        // late reports plus the survivor's remaining results exceed the
        // result channel, which must not block the join.
        let builds = AtomicUsize::new(0);
        let run = run_parallel(
            3,
            (0..32).collect::<Vec<usize>>(),
            || {
                if builds.fetch_add(1, Ordering::SeqCst) == 0 {
                    Ok(())
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    Err(anyhow::anyhow!("no such device"))
                }
            },
            |_: &mut (), _index, _job| Err::<(), _>(anyhow::anyhow!("chunk failed")),
            |_index, result| result,
        );
        let error = run.expect_err("the chunk failure is returned");
        assert!(
            error.to_string().contains("chunk failed"),
            "unexpected error: {error}"
        );
    }
}
