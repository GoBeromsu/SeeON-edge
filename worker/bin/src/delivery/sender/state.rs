//! In-memory sender bookkeeping: Python `EvidenceSender`'s `_attempts`,
//! `_deferred`, `_blocked_until` and `_select`, plus the accepted_local
//! event ids whose snapshot requests are skipped.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Duration;

use serde_json::Value;

use crate::delivery::MAX_ACCEPTED_ENTRIES;

/// Python `MAX_ENTRY_ATTEMPTS`: counted attempts before an entry is retired.
pub const MAX_ENTRY_ATTEMPTS: u32 = 10;

/// Sender state carried between drain passes; lost on restart, as in Python.
#[derive(Debug, Default)]
pub struct SenderState {
    attempts: HashMap<String, u32>,
    deferred: HashSet<String>,
    blocked_until: HashMap<String, Duration>,
    accepted_local: HashSet<String>,
    accepted_local_order: VecDeque<String>,
}

impl SenderState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Counted attempts for `entry_id`.
    pub fn attempts(&self, entry_id: &str) -> u32 {
        self.attempts.get(entry_id).copied().unwrap_or(0)
    }

    pub fn is_deferred(&self, entry_id: &str) -> bool {
        self.deferred.contains(entry_id)
    }

    /// Monotonic instant before which `entry_id` is not selected.
    pub fn blocked_until(&self, entry_id: &str) -> Option<Duration> {
        self.blocked_until.get(entry_id).copied()
    }

    /// True when the relay answered `accepted_local` for this edge event.
    pub fn is_accepted_local(&self, edge_event_id: &str) -> bool {
        self.accepted_local.contains(edge_event_id)
    }

    /// Python `_select`: prefer live entries (fewer than
    /// `MAX_ENTRY_ATTEMPTS`), keep only due ones, skip deferred ones until
    /// every due entry is deferred, then take the first EVENT or else the
    /// first entry in file-name order.
    pub(super) fn select<'a>(&mut self, entries: &[&'a Value], now: Duration) -> Option<&'a Value> {
        let live: Vec<&'a Value> = entries
            .iter()
            .copied()
            .filter(|entry| self.attempts(entry_id(entry)) < MAX_ENTRY_ATTEMPTS)
            .collect();
        let candidates = if live.is_empty() {
            entries.to_vec()
        } else {
            live
        };
        let due: Vec<&'a Value> = candidates
            .into_iter()
            .filter(|entry| {
                self.blocked_until(entry_id(entry))
                    .is_none_or(|until| until <= now)
            })
            .collect();
        if due.is_empty() {
            return None;
        }
        let mut undeferred: Vec<&'a Value> = due
            .iter()
            .copied()
            .filter(|entry| !self.is_deferred(entry_id(entry)))
            .collect();
        if undeferred.is_empty() {
            self.deferred.clear();
            undeferred = due;
        }
        undeferred
            .iter()
            .copied()
            .find(|entry| entry.get("kind").and_then(Value::as_str) == Some("EVENT"))
            .or_else(|| undeferred.first().copied())
    }

    pub(super) fn defer(&mut self, entry_id: &str) {
        self.deferred.insert(entry_id.to_owned());
    }

    pub(super) fn undefer(&mut self, entry_id: &str) {
        self.deferred.remove(entry_id);
    }

    pub(super) fn count_attempt(&mut self, entry_id: &str) {
        *self.attempts.entry(entry_id.to_owned()).or_insert(0) += 1;
    }

    pub(super) fn forget_attempts(&mut self, entry_id: &str) {
        self.attempts.remove(entry_id);
    }

    pub(super) fn block(&mut self, entry_id: &str, until: Duration) {
        self.blocked_until.insert(entry_id.to_owned(), until);
    }

    pub(super) fn unblock(&mut self, entry_id: &str) {
        self.blocked_until.remove(entry_id);
    }

    /// Remember an accepted_local edge event, oldest forgotten first once
    /// the queue's own entry bound is exceeded.
    pub(super) fn record_accepted_local(&mut self, edge_event_id: &str) {
        if !self.accepted_local.insert(edge_event_id.to_owned()) {
            return;
        }
        self.accepted_local_order
            .push_back(edge_event_id.to_owned());
        while self.accepted_local_order.len() > MAX_ACCEPTED_ENTRIES {
            if let Some(oldest) = self.accepted_local_order.pop_front() {
                self.accepted_local.remove(&oldest);
            }
        }
    }
}

/// The entry id of a queue entry; entries without one are filtered earlier.
pub(super) fn entry_id(entry: &Value) -> &str {
    entry.get("entry_id").and_then(Value::as_str).unwrap_or("")
}
