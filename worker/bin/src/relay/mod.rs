//! Relay HTTP delivery (Python `shared/events/evidence_http_transport.py` and
//! `evidence_export_client.py`): `client` sends requests, `wire` turns their
//! results into receipts or `DeliveryFailure`s.

pub mod cameras;
pub mod capabilities;
pub mod client;
pub mod wire;

pub use client::{ConfigError, RelayClient, TransportError};
pub use wire::{DeliveryFailure, HttpResult, Response};
