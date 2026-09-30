//! The request side of Python `shared/events/evidence_http_transport.py`
//! (`bounded_request`, `bounded_file_request`) and the relay endpoints of
//! `RelayEvidenceClient`: one ureq agent, one global deadline per call, no
//! retries, no redirects, no proxy. The relay token travels only in the
//! `X-Edge-Relay-Token` header; it is never logged, displayed or returned.

use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::time::Duration;

use ureq::http::{self, HeaderName, HeaderValue, Uri};
use ureq::{Agent, Body, RequestBuilder};

use super::wire::{MAX_RESPONSE_BYTES, Response};

mod error;
mod url;

pub use error::{ConfigError, TransportError};

/// Header that carries the relay token (Python `RelayEvidenceClient._headers`).
pub const TOKEN_HEADER: &str = "X-Edge-Relay-Token";
/// Relay capability probe endpoint, relative to the base URL.
pub const CAPABILITIES_PATH: &str = "api/v1/relay/capabilities";
/// Relay alert (event) endpoint, relative to the base URL.
pub const ALERTS_PATH: &str = "api/v1/relay/alerts";
/// Query key that carries the camera id on the capability probe.
pub const CAMERA_ID_QUERY_KEY: &str = "camera_id";
/// Python `ALERT_DELIVERY_TIMEOUT_SEC`: the default deadline of one call.
pub const ALERT_DELIVERY_TIMEOUT: Duration = Duration::from_secs(16);

const JSON_CONTENT_TYPE: &str = "application/json";

/// An HTTP client bound to one relay base URL, token and per-call deadline.
#[derive(Clone)]
pub struct RelayClient {
    agent: Agent,
    base: String,
    token: HeaderValue,
    timeout: Duration,
}

impl fmt::Debug for RelayClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RelayClient")
            .field("base", &self.base)
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

impl RelayClient {
    /// Validates `base_url` (Python `normalize_http_base`) and `token`.
    pub fn new(base_url: &str, token: &str, timeout: Duration) -> Result<Self, ConfigError> {
        let base = url::normalize_base(base_url)?;
        if token.is_empty() || !token.bytes().all(|byte| byte.is_ascii_graphic()) {
            return Err(ConfigError::Token);
        }
        let mut token = HeaderValue::from_str(token).map_err(|_| ConfigError::Token)?;
        token.set_sensitive(true);
        let config = Agent::config_builder()
            .http_status_as_error(false)
            .max_redirects(0)
            .timeout_global(Some(timeout))
            .proxy(None)
            .build();
        Ok(Self {
            agent: config.into(),
            base,
            token,
            timeout,
        })
    }

    /// The deadline applied to each call.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// GET `path` with a form-encoded query (Python `urlencode`).
    pub fn get(&self, path: &str, query: &[(&str, &str)]) -> Result<Response, TransportError> {
        let request = self.agent.get(self.uri(path, query)?);
        self.read(self.with_token(request).call())
    }

    /// The Python `RelayEvidenceClient.probe_capabilities` request.
    pub fn get_capabilities(&self, camera_id: &str) -> Result<Response, TransportError> {
        self.get(CAPABILITIES_PATH, &[(CAMERA_ID_QUERY_KEY, camera_id)])
    }

    /// POST `body` (Python `encode_json` bytes) as `application/json`.
    pub fn post_json(&self, path: &str, body: &[u8]) -> Result<Response, TransportError> {
        let request = self.agent.post(self.uri(path, &[])?);
        self.read(
            self.with_token(request)
                .content_type(JSON_CONTENT_TYPE)
                .send(body),
        )
    }

    /// The Python `RelayEvidenceClient.send_event` request.
    pub fn post_alert(&self, body: &[u8]) -> Result<Response, TransportError> {
        self.post_json(ALERTS_PATH, body)
    }

    /// PUT `body` (Python `encode_json` bytes) as `application/json`.
    pub fn put_json(&self, path: &str, body: &[u8]) -> Result<Response, TransportError> {
        let request = self.agent.put(self.uri(path, &[])?);
        self.read(
            self.with_token(request)
                .content_type(JSON_CONTENT_TYPE)
                .send(body),
        )
    }

    /// PUT raw `body` with the caller's headers (framing and token headers
    /// among them are ignored).
    pub fn put_bytes(
        &self,
        path: &str,
        headers: &[(HeaderName, HeaderValue)],
        body: &[u8],
    ) -> Result<Response, TransportError> {
        let request = self.with_headers(self.agent.put(self.uri(path, &[])?), headers);
        self.read(request.send(body))
    }

    /// PUT the whole of `file`, rewound first as Python `media.seek(0)`;
    /// Content-Length is the file's length.
    pub fn put_file(
        &self,
        path: &str,
        headers: &[(HeaderName, HeaderValue)],
        file: &File,
    ) -> Result<Response, TransportError> {
        let uri = self.uri(path, &[])?;
        let mut media = file;
        media
            .seek(SeekFrom::Start(0))
            .map_err(|error| self.transport(&error.into()))?;
        self.read(self.with_headers(self.agent.put(uri), headers).send(media))
    }

    /// Python `join_http_url(base, path)` plus `?query` when there is one.
    fn uri(&self, path: &str, query: &[(&str, &str)]) -> Result<Uri, TransportError> {
        Uri::try_from(url::join(&self.base, path, query))
            .map_err(|error| self.transport(&ureq::Error::BadUri(error.to_string())))
    }

    fn with_token<B>(&self, request: RequestBuilder<B>) -> RequestBuilder<B> {
        request.header(TOKEN_HEADER, self.token.clone())
    }

    fn with_headers<B>(
        &self,
        mut request: RequestBuilder<B>,
        headers: &[(HeaderName, HeaderValue)],
    ) -> RequestBuilder<B> {
        for (name, value) in headers {
            let reserved = ["content-length", "transfer-encoding", TOKEN_HEADER];
            if !reserved
                .iter()
                .any(|skip| name.as_str().eq_ignore_ascii_case(skip))
            {
                request = request.header(name.clone(), value.clone());
            }
        }
        self.with_token(request)
    }

    /// Status, headers (lower-case names, Latin-1 values, wire order per
    /// name) and at most `MAX_RESPONSE_BYTES + 1` body bytes.
    fn read(
        &self,
        sent: Result<http::Response<Body>, ureq::Error>,
    ) -> Result<Response, TransportError> {
        let response = sent.map_err(|error| self.transport(&error))?;
        let status = response.status().as_u16();
        let headers = response
            .headers()
            .iter()
            .map(|(name, value)| {
                (
                    name.as_str().to_owned(),
                    value.as_bytes().iter().map(|&b| char::from(b)).collect(),
                )
            })
            .collect();
        let mut body = Vec::new();
        let limit = u64::try_from(MAX_RESPONSE_BYTES)
            .unwrap_or(u64::MAX)
            .saturating_add(1);
        let mut reader = response.into_body().into_reader().take(limit);
        reader
            .read_to_end(&mut body)
            .map_err(|error| self.transport(&error.into()))?;
        Ok(Response {
            status,
            headers,
            body,
        })
    }

    fn transport(&self, error: &ureq::Error) -> TransportError {
        TransportError::new(error, self.token.to_str().unwrap_or_default())
    }
}
