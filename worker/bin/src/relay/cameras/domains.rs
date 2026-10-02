//! The domain resolvers of `BackendWorkerConfigPayload`
//! (`resolved_detection_windows`, `resolved_domain_enabled`) and the
//! `DomainsConfig` choice `to_worker_config` makes from them.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    CameraConfigError, DetectionWindow, WindowCandidate, WorkerConfigPayload, window_candidates,
};
use crate::config::lookup;
use crate::json::Json;

/// `KNOWN_DOMAIN_NAMES` (`worker/runtime/config/domain_models.py`), the
/// module ids of `DOMAIN_REGISTRY`.
pub const KNOWN_DOMAINS: [&str; 2] = ["bed_exit", "fall"];
/// Registry default for both known domains (`DetectionModuleDefinition.enabled`
/// and `DomainRegistration.enabled`, neither overridden by a definition).
const REGISTRY_DOMAIN_DEFAULT: bool = true;

/// Effective domain enablement: registry defaults overlaid by a global
/// override, or replaced outright by a camera-domain union.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedDomains {
    pub fall: bool,
    pub bed_exit: bool,
}

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

impl DomainSelection {
    /// `WorkerConfig.enabled_domains` for this selection.
    ///
    /// `Override` names only the domains the global map actually enabled;
    /// an unnamed domain keeps `REGISTRY_DOMAIN_DEFAULT`. `Cameras(None)`
    /// is that default for both. `Cameras(Some)` is the legacy replace-list:
    /// only listed names are on, and a name outside `KNOWN_DOMAINS` refuses.
    pub fn resolve(&self) -> Result<ResolvedDomains, CameraConfigError> {
        let (fall, bed_exit) = match self {
            Self::Override { fall, bed_exit } => (
                fall.unwrap_or(REGISTRY_DOMAIN_DEFAULT),
                bed_exit.unwrap_or(REGISTRY_DOMAIN_DEFAULT),
            ),
            Self::Cameras(None) => (REGISTRY_DOMAIN_DEFAULT, REGISTRY_DOMAIN_DEFAULT),
            Self::Cameras(Some(names)) => {
                if names
                    .iter()
                    .any(|name| !KNOWN_DOMAINS.contains(&name.as_str()))
                {
                    return Err(CameraConfigError::UnknownDomain);
                }
                (
                    names.iter().any(|name| name == "fall"),
                    names.iter().any(|name| name == "bed_exit"),
                )
            }
        };
        Ok(ResolvedDomains { fall, bed_exit })
    }
}

impl WorkerConfigPayload {
    /// `resolved_detection_windows`: `detection_windows` when present (null,
    /// misshapen and invalid members are dropped), else a valid
    /// `night_window` under `bed_exit`.
    pub fn detection_windows(&self) -> BTreeMap<String, DetectionWindow> {
        let mut windows = BTreeMap::new();
        for (domain, candidate) in window_candidates(self) {
            let WindowCandidate::Window(window) = candidate else {
                continue;
            };
            if !window.is_valid() {
                continue;
            }
            windows.insert(domain.to_owned(), window);
        }
        windows
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
