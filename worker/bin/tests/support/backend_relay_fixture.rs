//! Owned scratch and bounded control of the real backend/proxy child.

use std::fs;
use std::io::{self, Read, Write};
use std::os::fd::AsFd;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use super::worker::seam::{IdSource, RandomIds};
use rustix::fs::{OFlags, fcntl_getfl, fcntl_setfl};
use serde_json::{Value, json};

pub const RELAY_TOKEN: &str = "relay-token";
const MESSAGE_LIMIT: usize = 64 * 1024;
const POLL: Duration = Duration::from_millis(10);
const CONTROL_LIMIT: Duration = Duration::from_secs(15);
const KILL_LIMIT: Duration = Duration::from_secs(5);

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub struct OwnedDir(pub PathBuf);

impl OwnedDir {
    pub fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "seeon-backend-relay-{}",
            RandomIds.uuid4().expect("owned scratch UUID")
        ));
        fs::create_dir(&path).expect("exclusive owned scratch directory");
        Self(path)
    }
}

impl Drop for OwnedDir {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("owned backend-relay scratch cleanup failed: {error}");
        }
    }
}

fn nonblocking(pipe: &impl AsFd) {
    let flags = fcntl_getfl(pipe).expect("pipe flags");
    fcntl_setfl(pipe, flags | OFlags::NONBLOCK).expect("nonblocking child pipe");
}

fn timed_out() -> io::Error {
    io::Error::new(io::ErrorKind::TimedOut, "backend-relay child deadline")
}

pub struct Backend {
    child: Child,
    input: Option<ChildStdin>,
    output: Option<ChildStdout>,
    pending: Vec<u8>,
    serial: u64,
    base: String,
}

impl Backend {
    pub fn start(alert_fixture: &Path, entry_path: &Path) -> Self {
        let python = std::env::var_os("SEEON_TEST_PYTHON").expect("SEEON_TEST_PYTHON is required");
        assert!(
            !python.to_string_lossy().trim().is_empty(),
            "nonblank test Python path"
        );
        let dsn =
            std::env::var("SEEON_TEST_POSTGRES_DSN").expect("SEEON_TEST_POSTGRES_DSN is required");
        assert!(
            !dsn.trim().is_empty() && !dsn.contains('\0'),
            "explicit isolated test DSN"
        );
        let child = Command::new(python)
            .arg("-u")
            .arg(repo_root().join("worker/bin/tests/support/backend_relay_server.py"))
            .arg(alert_fixture)
            .arg(entry_path)
            .env("PYTHONPATH", repo_root())
            .current_dir(repo_root())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("owned Python backend child starts");
        // Establish ownership before any fallible pipe/readiness setup.
        let mut backend = Self {
            child,
            input: None,
            output: None,
            pending: Vec::new(),
            serial: 0,
            base: String::new(),
        };
        let input = backend.child.stdin.take().expect("child stdin");
        nonblocking(&input);
        backend.input = Some(input);
        let output = backend.child.stdout.take().expect("child stdout");
        nonblocking(&output);
        backend.output = Some(output);
        let ready = backend
            .receive(Instant::now() + Duration::from_secs(45))
            .expect("bounded real-backend readiness");
        assert_eq!(ready["kind"], "ready", "{ready}");
        let address: std::net::SocketAddr = ready["proxy_address"]
            .as_str()
            .expect("owned proxy address")
            .parse()
            .expect("loopback address");
        assert_eq!(address.ip(), std::net::Ipv4Addr::LOCALHOST);
        assert_ne!(address.port(), 0);
        backend.base = format!("http://{address}");
        backend
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    pub fn snapshot(&mut self) -> Value {
        self.command("snapshot", Instant::now() + CONTROL_LIMIT)
            .expect("bounded SQL snapshot")["snapshot"]
            .clone()
    }

    pub fn stop(mut self) {
        let deadline = Instant::now() + CONTROL_LIMIT;
        let reply = self
            .command("stop", deadline)
            .expect("bounded backend stop");
        assert_eq!(
            reply["stopped"], true,
            "sandbox is dropped before stop acknowledgement"
        );
        let status = self
            .wait_until(deadline)
            .expect("owned backend child exits and is reaped");
        assert!(status.success(), "backend helper exited unsuccessfully");
    }

    fn command(&mut self, op: &str, deadline: Instant) -> io::Result<Value> {
        self.serial += 1;
        let mut bytes =
            serde_json::to_vec(&json!({"id": self.serial, "op": op})).map_err(io::Error::other)?;
        bytes.push(b'\n');
        self.send(&bytes, deadline)?;
        let reply = self.receive(deadline)?;
        if reply["id"] != self.serial || reply.get("error").is_some() {
            return Err(io::Error::other(format!(
                "backend helper protocol refusal: {reply}"
            )));
        }
        Ok(reply)
    }

    fn send(&mut self, bytes: &[u8], deadline: Instant) -> io::Result<()> {
        let mut remaining = bytes;
        while !remaining.is_empty() {
            if Instant::now() >= deadline {
                return Err(timed_out());
            }
            let input = self
                .input
                .as_mut()
                .ok_or_else(|| io::Error::other("no child stdin"))?;
            match input.write(remaining) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "child stdin closed",
                    ));
                }
                Ok(count) => remaining = &remaining[count..],
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => thread::sleep(POLL),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    fn receive(&mut self, deadline: Instant) -> io::Result<Value> {
        loop {
            if let Some(end) = self.pending.iter().position(|&byte| byte == b'\n') {
                let line: Vec<u8> = self.pending.drain(..=end).collect();
                return serde_json::from_slice(&line).map_err(io::Error::other);
            }
            if Instant::now() >= deadline {
                return Err(timed_out());
            }
            let mut chunk = [0_u8; 4096];
            let output = self
                .output
                .as_mut()
                .ok_or_else(|| io::Error::other("no child stdout"))?;
            match output.read(&mut chunk) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "backend child closed",
                    ));
                }
                Ok(count) => {
                    if self.pending.len() + count > MESSAGE_LIMIT {
                        return Err(io::Error::other("backend protocol message exceeds cap"));
                    }
                    self.pending.extend_from_slice(&chunk[..count]);
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => thread::sleep(POLL),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
    }

    fn wait_until(&mut self, deadline: Instant) -> io::Result<ExitStatus> {
        loop {
            if let Some(status) = self.child.try_wait()? {
                return Ok(status);
            }
            if Instant::now() >= deadline {
                return Err(timed_out());
            }
            thread::sleep(POLL);
        }
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(Some(_))) {
            return;
        }
        // Also runs on assertion/readiness/protocol failures. No reader threads
        // or unbounded wait/join: give the sandbox its stop path, then kill/reap.
        self.serial += 1;
        let stop = format!("{{\"id\":{},\"op\":\"stop\"}}\n", self.serial);
        let _ = self.send(stop.as_bytes(), Instant::now() + Duration::from_secs(1));
        self.input.take();
        self.output.take();
        if self.wait_until(Instant::now() + CONTROL_LIMIT).is_err() {
            let _ = self.child.kill();
            if let Err(error) = self.wait_until(Instant::now() + KILL_LIMIT) {
                eprintln!("owned backend child kill/reap failed: {error}");
            }
        }
    }
}
