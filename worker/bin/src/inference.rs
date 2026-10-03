//! Provider-neutral thread ownership and immutable observed runtime facts.

use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::exit::Exit;
use crate::msg::Readiness;
use crate::poll::{Timeout, poll_until};
use crate::seam::Clock;
use seeon_worker_runtime::cpu::Info;

pub mod cpu;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Runtime {
    TensorRt,
    OnnxRuntimeCpu(Info),
}

/// Facts are published before readiness. CPU actors latch fatal results here
/// before replying; queue capacity and reply consumption cannot erase that failure.
#[derive(Default)]
pub struct State {
    runtime: OnceLock<Runtime>,
    failure: OnceLock<Exit>,
}

impl State {
    pub fn runtime(&self) -> Option<&Runtime> {
        self.runtime.get()
    }
    pub fn failure(&self) -> Option<Exit> {
        self.failure.get().copied()
    }
    pub(crate) fn started(&self, runtime: Runtime) -> bool {
        self.runtime.set(runtime).is_ok()
    }
    pub(crate) fn fail(&self, exit: Exit) {
        let _ = self.failure.set(exit);
    }
}

/// Request channels never transfer a native model across its owning thread.
pub struct Owner<R> {
    pub thread: JoinHandle<()>,
    pub readiness: Receiver<Readiness>,
    pub requests: SyncSender<R>,
    pub state: Arc<State>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JoinError {
    Timeout(Timeout),
    Panicked,
}

/// Consumes the handle. Supervisors retain timed-out handles by polling before take.
pub fn join(
    thread: JoinHandle<()>,
    clock: &dyn Clock,
    deadline: Duration,
) -> Result<(), JoinError> {
    poll_until(clock, deadline, "inference owner exit", || {
        thread.is_finished()
    })
    .map_err(JoinError::Timeout)?;
    thread.join().map_err(|_| JoinError::Panicked)
}
