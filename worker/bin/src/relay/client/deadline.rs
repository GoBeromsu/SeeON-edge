//! A borrowed relay view whose per-call timeout is the minimum of the client's
//! baseline and the time left on a live [`ShutdownDeadline`].
//!
//! The effective timeout is computed immediately before send, after the
//! request is built: read the clock, then the deadline. A deadline published
//! by an earlier observation during that clock read is therefore respected.
//! `None` keeps the baseline. `now >= deadline` returns [`RequestError::Cutoff`]
//! and sends nothing.
//!
//! ureq's blocking send cannot be preempted. A deadline that becomes shorter
//! after send has started still bounds the call only by the timeout pinned
//! before send; this view does not detach a request thread to cancel it.
//! The live deadline is reread before a completed response is exposed. A
//! typed global timeout is [`RequestError::Cutoff`] only when that pinned
//! timeout was the shutdown remainder (`remaining <= baseline`). A genuine
//! baseline timeout, or a connection failure while the budget is still live,
//! stays [`RequestError::Transport`]. Local cutoff never synthesizes an HTTP
//! status, headers or body.

use std::error::Error;
use std::fmt;
use std::time::Duration;

use ureq::RequestBuilder;
use ureq::typestate::WithBody;

use crate::seam::Clock;
use crate::shutdown::ShutdownDeadline;

use super::error::TransportError;
use super::{ALERTS_PATH, JSON_CONTENT_TYPE, RelayClient, read_ureq};
use crate::relay::wire::Response;

/// Why a deadline-bounded relay call produced no usable response.
///
/// [`Self::Cutoff`] means the shared shutdown deadline left no time to send,
/// or a shutdown-limited call exhausted the time pinned before send. It is
/// not an HTTP result. [`Self::Transport`] is the existing redacted transport
/// failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RequestError {
    Cutoff,
    Transport(TransportError),
}

impl fmt::Display for RequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cutoff => formatter.write_str("relay call cut off by the shutdown deadline"),
            Self::Transport(error) => error.fmt(formatter),
        }
    }
}

impl Error for RequestError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Cutoff => None,
            Self::Transport(error) => Some(error),
        }
    }
}

/// Zero-allocation borrowed view of a [`RelayClient`].
///
/// The agent, token and request policy (token header only, no proxy, no
/// redirect, no retry, bounded body) are the client's. Each call pins
/// `min(baseline, remaining)` with ureq's per-request `timeout_global`.
pub struct DeadlineClient<'a> {
    client: &'a RelayClient,
    clock: &'a dyn Clock,
    deadline: &'a ShutdownDeadline,
}

impl<'a> DeadlineClient<'a> {
    pub(super) fn new(
        client: &'a RelayClient,
        clock: &'a dyn Clock,
        deadline: &'a ShutdownDeadline,
    ) -> Self {
        Self {
            client,
            clock,
            deadline,
        }
    }

    /// Sender-side mutation guard. `Ok` means the deadline is unset or still
    /// in the future. This never returns a transport error.
    pub fn check(&self) -> Result<(), RequestError> {
        self.budget().map(|_| ())
    }

    /// POST `body` as `application/json`.
    pub fn post_json(&self, path: &str, body: &[u8]) -> Result<Response, RequestError> {
        self.send(body, |client| {
            Ok(client
                .agent
                .post(client.uri(path, &[])?)
                .content_type(JSON_CONTENT_TYPE))
        })
    }

    /// POST the alert body to the relay alerts path.
    pub fn post_alert(&self, body: &[u8]) -> Result<Response, RequestError> {
        self.post_json(ALERTS_PATH, body)
    }

    /// PUT `body` as `application/json`.
    pub fn put_json(&self, path: &str, body: &[u8]) -> Result<Response, RequestError> {
        self.send(body, |client| {
            Ok(client
                .agent
                .put(client.uri(path, &[])?)
                .content_type(JSON_CONTENT_TYPE))
        })
    }

    fn send(
        &self,
        body: &[u8],
        build: impl FnOnce(&RelayClient) -> Result<RequestBuilder<WithBody>, TransportError>,
    ) -> Result<Response, RequestError> {
        self.check()?;
        let request = build(self.client).map_err(RequestError::Transport)?;
        let (timeout, shutdown_limited) = self.budget()?;
        let read = read_ureq(
            self.client
                .with_token(request)
                .config()
                .timeout_global(Some(timeout))
                .build()
                .send(body),
        );
        self.finish(read, shutdown_limited)
    }

    fn finish(
        &self,
        read: Result<Response, ureq::Error>,
        shutdown_limited: bool,
    ) -> Result<Response, RequestError> {
        self.check()?;
        match read {
            Ok(response) => Ok(response),
            Err(error) => Err(self.classify(error, shutdown_limited)),
        }
    }

    /// Clock first, then the deadline, so a publication made while reading
    /// the clock is visible. `Ok(baseline)` is normal running.
    fn budget(&self) -> Result<(Duration, bool), RequestError> {
        let now = self.clock.monotonic();
        match self.deadline.deadline() {
            None => Ok((self.client.timeout, false)),
            Some(deadline) if now >= deadline => Err(RequestError::Cutoff),
            Some(deadline) => {
                let remaining = deadline.saturating_sub(now);
                Ok((
                    self.client.timeout.min(remaining),
                    remaining <= self.client.timeout,
                ))
            }
        }
    }

    fn classify(&self, error: ureq::Error, shutdown_limited: bool) -> RequestError {
        let global = matches!(error, ureq::Error::Timeout(ureq::Timeout::Global));
        if global && shutdown_limited {
            return RequestError::Cutoff;
        }
        RequestError::Transport(self.client.transport(&error))
    }
}
