//! Owned process-group lifecycle. Exit is observed with waitid NOWAIT, so the
//! leader stays unreaped and its group ID stays owned until after the kill.
//! Only ECHILD revokes ownership. Descendants are not claimed to be reaped,
//! and a blocked kernel wait is not claimed to be preempted.

use std::os::fd::OwnedFd;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::Instant;

use rustix::io::Errno;
use rustix::process::{Pid, Signal, WaitId, WaitIdOptions, kill_process_group, waitid};

use super::super::LiveBuildError;
use super::READ_SLICE;

const OBSERVE: WaitIdOptions = WaitIdOptions::EXITED
    .union(WaitIdOptions::NOHANG)
    .union(WaitIdOptions::NOWAIT);

pub(in crate::engine_build::live::process) struct OwnedChild {
    child: Option<Child>,
    group: Option<Pid>,
}

impl OwnedChild {
    pub(in crate::engine_build::live::process) fn spawn(
        command: Command,
        directory: &Path,
        writer: OwnedFd,
    ) -> Result<Self, LiveBuildError> {
        let mut command = command;
        command
            .current_dir(directory)
            .stdin(Stdio::null())
            .stdout(stdio_of(&writer)?)
            .stderr(stdio_of(&writer)?)
            .process_group(0);
        drop(writer);
        let child = command.spawn().map_err(|_| LiveBuildError::Process)?;
        let group = Pid::from_child(&child);
        if group.is_init() {
            return Err(LiveBuildError::Process);
        }
        Ok(Self {
            child: Some(child),
            group: Some(group),
        })
    }

    pub(in crate::engine_build::live::process) fn pid(&self) -> Result<Pid, LiveBuildError> {
        self.group
            .filter(|pid| !pid.is_init())
            .ok_or(LiveBuildError::Process)
    }

    pub(in crate::engine_build::live::process) fn exited(
        &mut self,
        deadline: Instant,
    ) -> Result<bool, LiveBuildError> {
        let Some(pid) = self.owned_pid() else {
            return Err(LiveBuildError::Process);
        };
        loop {
            if Instant::now() >= deadline {
                return Ok(false);
            }
            match waitid(WaitId::Pid(pid), OBSERVE) {
                Ok(status) => return Ok(status.is_some()),
                Err(Errno::INTR) => continue,
                Err(Errno::CHILD) => {
                    self.revoke();
                    return Err(LiveBuildError::Process);
                }
                Err(_) => return Err(LiveBuildError::Process),
            }
        }
    }

    pub(in crate::engine_build::live::process) fn finish(
        &mut self,
        deadline: Instant,
    ) -> Result<i32, LiveBuildError> {
        loop {
            if Instant::now() >= deadline {
                self.abort()?;
                return Err(LiveBuildError::Process);
            }
            if self.exited(deadline)? {
                self.signal_owned()?;
                return self.reap_once();
            }
            std::thread::sleep(READ_SLICE.min(deadline.saturating_duration_since(Instant::now())));
        }
    }

    pub(in crate::engine_build::live::process) fn abort(&mut self) -> Result<(), LiveBuildError> {
        self.signal_owned()?;
        let _ = self.reap_once();
        if self.child.is_some() {
            Err(LiveBuildError::Process)
        } else {
            Ok(())
        }
    }

    fn signal_owned(&mut self) -> Result<(), LiveBuildError> {
        if !self.still_owned()? {
            return Err(LiveBuildError::Process);
        }
        let Some(group) = self.group.filter(|pid| !pid.is_init()) else {
            return Err(LiveBuildError::Process);
        };
        match kill_process_group(group, Signal::KILL) {
            Ok(()) => Ok(()),
            Err(Errno::CHILD) => {
                self.revoke();
                Err(LiveBuildError::Process)
            }
            Err(_) => Err(LiveBuildError::Process),
        }
    }

    fn still_owned(&mut self) -> Result<bool, LiveBuildError> {
        let Some(pid) = self.owned_pid() else {
            return Ok(false);
        };
        loop {
            match waitid(WaitId::Pid(pid), OBSERVE) {
                Ok(_) => return Ok(true),
                Err(Errno::INTR) => continue,
                Err(Errno::CHILD) => {
                    self.revoke();
                    return Ok(false);
                }
                Err(_) => return Err(LiveBuildError::Process),
            }
        }
    }

    fn reap_once(&mut self) -> Result<i32, LiveBuildError> {
        let Some(mut child) = self.child.take() else {
            return Err(LiveBuildError::Process);
        };
        match child.wait() {
            Ok(status) => {
                self.group = None;
                exit_code(status)
            }
            Err(error) if error.raw_os_error() == Some(Errno::CHILD.raw_os_error()) => {
                self.revoke();
                Err(LiveBuildError::Process)
            }
            Err(_) => {
                self.child = Some(child);
                Err(LiveBuildError::Process)
            }
        }
    }

    fn owned_pid(&self) -> Option<Pid> {
        self.child.as_ref()?;
        self.group.filter(|pid| !pid.is_init())
    }

    fn revoke(&mut self) {
        self.child = None;
        self.group = None;
    }
}

impl Drop for OwnedChild {
    fn drop(&mut self) {
        if self.owned_pid().is_none() {
            return;
        }
        if self.still_owned().unwrap_or(false) {
            let _ = self.signal_owned();
        }
        if let Some(mut child) = self.child.take() {
            let _ = child.wait();
        }
        self.group = None;
    }
}

fn stdio_of(writer: &OwnedFd) -> Result<Stdio, LiveBuildError> {
    Ok(Stdio::from(
        writer.try_clone().map_err(|_| LiveBuildError::Process)?,
    ))
}

fn exit_code(status: ExitStatus) -> Result<i32, LiveBuildError> {
    status.code().ok_or(LiveBuildError::Process)
}
