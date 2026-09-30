//! The domain resolvers of `BackendWorkerConfigPayload`
//! (`resolved_detection_windows`, `resolved_domain_enabled`) and the
//! `DomainsConfig` choice `to_worker_config` makes from them.

use std::collections::{BTreeMap, BTreeSet};

use super::{DetectionWindow, WorkerConfigPayload};
use crate::config::lookup;
use crate::json::Json;

/// `KNOWN_DOMAIN_NAMES` (`worker/runtime/config/domain_models.py`), the
/// module ids of `DOMAIN_REGISTRY`.
pub const KNOWN_DOMAINS: [&str; 2] = ["bed_exit", "fall"];

/// The domain signal `to_worker_config` hands to `DomainsConfig`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DomainSelection {
    /// `domains` was present: enable overrides for the named domains only.
    Override {
        fall: Option<bool>,
        bed_exit: Option<bool>,
    },
    /// No override: the sorted union of the cameras' domains, `None` when
    /// no camera declared any.
    Cameras(Option<Vec<String>>),
}

impl WorkerConfigPayload {
    /// `resolved_detection_windows`: `detection_windows` when present (null,
    /// misshapen and invalid members are dropped), else a valid
    /// `night_window` under `bed_exit`.
    pub fn detection_windows(&self) -> BTreeMap<String, DetectionWindow> {
        match &self.detection_windows {
            Some(members) => members
                .iter()
                .filter_map(|(domain, value)| {
                    DetectionWindow::shape(value)
                        .filter(DetectionWindow::is_valid)
                        .map(|window| (domain.clone(), window))
                })
                .collect(),
            None => self
                .night_window
                .iter()
                .filter(|window| window.is_valid())
                .map(|window| ("bed_exit".to_owned(), window.clone()))
                .collect(),
        }
    }

    /// `resolved_domain_enabled`: known domains whose value is an object
    /// with a boolean `enabled`.
    pub fn domain_enabled(&self) -> BTreeMap<String, bool> {
        let Some(members) = &self.domains else {
            return BTreeMap::new();
        };
        members
            .iter()
            .filter(|(domain, _)| KNOWN_DOMAINS.contains(&domain.as_str()))
            .filter_map(|(domain, value)| match value {
                Json::Object(fields) => match lookup(fields, "enabled") {
                    Some(Json::Bool(enabled)) => Some((domain.clone(), *enabled)),
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    /// The `DomainsConfig` choice in `to_worker_config`.
    pub fn domain_selection(&self) -> DomainSelection {
        if self.domains.is_some() {
            let enabled = self.domain_enabled();
            return DomainSelection::Override {
                fall: enabled.get("fall").copied(),
                bed_exit: enabled.get("bed_exit").copied(),
            };
        }
        let cameras = self.cameras();
        if cameras.iter().all(|camera| camera.domains.is_none()) {
            return DomainSelection::Cameras(None);
        }
        let union: BTreeSet<&String> = cameras
            .iter()
            .flat_map(|c| c.domains.iter().flatten())
            .collect();
        DomainSelection::Cameras(Some(union.into_iter().cloned().collect()))
    }
}
