//! Typed refusals of the relay client: a bad configuration, or no HTTP
//! response at all. Neither ever carries the relay token.

use std::error::Error;
use std::fmt;

use crate::relay::wire::{DeliveryFailure, describe_transport_error};

const REDACTED: &str = "<relay-token>";

/// Why a relay base URL or token was refused (Python raises
/// `EvidenceClientConfigurationError`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigError {
    /// The scheme is not `http` or `https`.
    Scheme,
    /// The URL names no host.
    Host,
    /// The URL carries a user name or password.
    Credentials,
    /// The URL has a query, fragment, whitespace or control character, or does not parse.
    Base,
    /// The token is empty or holds a byte outside printable ASCII.
    Token,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Scheme => "relay URL scheme must be http or https",
            Self::Host => "relay URL names no host",
            Self::Credentials => "relay URL must not carry credentials",
            Self::Base => "relay URL must be a plain absolute http(s) URL",
            Self::Token => "relay token must be non-empty printable ASCII",
        })
    }
}

impl Error for ConfigError {}

/// No HTTP response: the request was not sent, or the response was not read
/// before the deadline. `kind` names the ureq error variant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransportError {
    kind: &'static str,
    message: String,
}

impl TransportError {
    /// Wraps `error`, with every occurrence of `token` replaced by a placeholder.
    pub(super) fn new(error: &ureq::Error, token: &str) -> Self {
        let message = error.to_string();
        let message = if token.is_empty() {
            message
        } else {
            message.replace(token, REDACTED)
        };
        Self {
            kind: kind_of(error),
            message,
        }
    }

    pub fn kind(&self) -> &'static str {
        self.kind
    }

    /// True when the call's deadline expired.
    pub fn is_timeout(&self) -> bool {
        self.kind == "Timeout"
    }
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&describe_transport_error(self.kind, &self.message))
    }
}

impl Error for TransportError {}

/// Python maps every transport exception to `DeliveryFailure(RETRY, "NETWORK")`.
impl From<TransportError> for DeliveryFailure {
    fn from(error: TransportError) -> Self {
        DeliveryFailure::network(error.kind, &error.message)
    }
}

fn kind_of(error: &ureq::Error) -> &'static str {
    match error {
        ureq::Error::Timeout(_) => "Timeout",
        ureq::Error::Io(_) => "Io",
        ureq::Error::HostNotFound => "HostNotFound",
        ureq::Error::ConnectionFailed => "ConnectionFailed",
        ureq::Error::Protocol(_) => "Protocol",
        ureq::Error::BadUri(_) => "BadUri",
        ureq::Error::Http(_) => "Http",
        ureq::Error::Tls(_) | ureq::Error::Rustls(_) | ureq::Error::Pem(_) => "Tls",
        ureq::Error::LargeResponseHeader(..) => "LargeResponseHeader",
        _ => "Error",
    }
}
