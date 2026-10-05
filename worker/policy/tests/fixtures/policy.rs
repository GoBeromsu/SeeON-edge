//! Scored-policy replay shared by the fall, bed and detection-window fixtures.
//! Requests are the rows the Python parity tests built; expected rows are what
//! the Python owner rendered through the same test's `_render`. Replays render
//! the public API's answers in the probes' TSV rows so both compare line by line.
//! A malformed request row is a fixture defect and panics; a domain refusal is `Err`.
#![allow(dead_code)]

use crate::support;
use seeon_worker::episode::BusinessEvent;
use seeon_worker::trace::{DecisionTraceSnapshot, NumericTraceValue};
use serde_json::Value;
use std::fmt::Write;
use std::str::{FromStr, Split};

/// The probes' per-collection bound and every library capacity they configure.
pub const MAX_ITEMS: usize = 64;

pub struct Fields<'a> {
    line: &'a str,
    fields: Split<'a, char>,
}

impl<'a> Fields<'a> {
    pub fn new(line: &'a str) -> Self {
        Self {
            line,
            fields: line.split('\t'),
        }
    }

    fn defect(&self, what: &str) -> ! {
        panic!("fixture request row has malformed {what}: {:?}", self.line)
    }

    pub fn next(&mut self) -> &'a str {
        match self.fields.next() {
            Some(token) => token,
            None => self.defect("arity"),
        }
    }

    pub fn number<T: FromStr>(&mut self) -> T {
        let token = self.next();
        token.parse().unwrap_or_else(|_| self.defect(token))
    }

    pub fn count(&mut self) -> usize {
        let count: usize = self.number();
        if count > MAX_ITEMS {
            self.defect("collection count");
        }
        count
    }

    pub fn float(&mut self) -> f64 {
        let token = self.next();
        float_token(token).unwrap_or_else(|| self.defect(token))
    }

    pub fn text(&mut self) -> String {
        let token = self.next();
        if token.len() > 2048
            || !token.len().is_multiple_of(2)
            || !token.bytes().all(|b| b.is_ascii_hexdigit())
        {
            self.defect(token);
        }
        let bytes = (0..token.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&token[i..i + 2], 16).expect("hex digits"))
            .collect();
        String::from_utf8(bytes).unwrap_or_else(|_| self.defect(token))
    }

    pub fn optional_id(&mut self) -> Option<u64> {
        match self.next() {
            "-" => None,
            token => Some(token.parse().unwrap_or_else(|_| self.defect(token))),
        }
    }

    pub fn optional_float(&mut self) -> Option<f64> {
        match self.next() {
            "-" => None,
            token => Some(float_token(token).unwrap_or_else(|| self.defect(token))),
        }
    }

    pub fn end(&mut self) {
        if self.fields.next().is_some() {
            self.defect("trailing fields");
        }
    }
}

/// Exactly 16 hex digits of finite IEEE-754 binary64 bits.
pub fn float_token(token: &str) -> Option<f64> {
    if token.len() != 16 || !token.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let value = f64::from_bits(u64::from_str_radix(token, 16).ok()?);
    value.is_finite().then_some(value)
}

pub fn hex(text: &str) -> String {
    support::hex(text.as_bytes())
}

pub fn bits(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

pub fn optional<T: ToString>(value: Option<T>) -> String {
    value.map_or_else(|| "-".into(), |value| value.to_string())
}

pub fn event(fields: &mut Fields<'_>) -> BusinessEvent {
    BusinessEvent {
        domain: fields.text(),
        event_type: fields.text(),
        identity: fields.text(),
        camera_id: fields.text(),
        facility_id: fields.text(),
        time_sec: fields.float(),
        probability: fields.optional_float(),
        person_id: fields.optional_id(),
        bed_id: fields.optional_id(),
    }
}

pub fn events(out: &mut String, events: &[BusinessEvent]) {
    for event in events {
        writeln!(
            out,
            "E\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            hex(&event.domain),
            hex(&event.event_type),
            hex(&event.identity),
            hex(&event.camera_id),
            hex(&event.facility_id),
            bits(event.time_sec),
            optional(event.probability.map(bits)),
            optional(event.person_id),
            optional(event.bed_id)
        )
        .expect("writing to String");
    }
}

pub fn traces(out: &mut String, traces: &[DecisionTraceSnapshot]) {
    for trace in traces {
        writeln!(
            out,
            "T\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            hex(trace.reason.as_str()),
            hex(trace.previous_state.as_str()),
            hex(trace.current_state.as_str()),
            u8::from(trace.triggered),
            optional(trace.track_id),
            optional(trace.bed_id),
            trace.values().len(),
            trace.missing_values().len()
        )
        .expect("writing to String");
        let mut values: Vec<_> = trace.values().iter().collect();
        values.sort_by_key(|(name, _)| name.as_str());
        for (name, value) in values {
            let (kind, encoded) = match value {
                NumericTraceValue::Integer(value) => ("I", value.to_string()),
                NumericTraceValue::Float(value) => ("F", bits(value.get())),
            };
            writeln!(out, "V\t{}\t{kind}\t{encoded}", hex(name.as_str()))
                .expect("writing to String");
        }
        let mut missing: Vec<_> = trace.missing_values().iter().collect();
        missing.sort_by_key(|(name, _)| name.as_str());
        for (name, reason) in missing {
            writeln!(out, "M\t{}\t{}", hex(name.as_str()), hex(reason.as_str()))
                .expect("writing to String");
        }
    }
}

/// One recorded probe exchange; `expected` is `None` when Python demanded rejection.
pub struct Transcript {
    pub context: String,
    pub request: Vec<String>,
    pub expected: Option<Vec<String>>,
}

impl Transcript {
    /// The request rows as the replays take them.
    pub fn request(&self) -> Vec<&str> {
        self.request.iter().map(String::as_str).collect()
    }
}

/// Recorded probe rows. The recorder stores a row longer than its line limit as
/// that row's tab-separated fields, so the row is those fields joined by tabs.
pub fn rows(value: &Value) -> Vec<String> {
    support::array(value)
        .iter()
        .map(|row| match row {
            Value::String(row) => row.clone(),
            fields => {
                let fields: Vec<&str> = support::array(fields).iter().map(support::text).collect();
                assert!(fields.len() > 1, "split row has fields: {fields:?}");
                fields.join("\t")
            }
        })
        .collect()
}

/// Recorded request rows after their sha256 check against the probe payload.
pub fn request(recorded: &Value, context: &str) -> Vec<String> {
    let request = rows(&recorded["request"]);
    assert_eq!(
        support::sha256(format!("{}\n", request.join("\n")).as_bytes()),
        support::text(&recorded["request_sha256"]),
        "{context}: request sha256"
    );
    request
}

/// Every exchange the named Python test made, after checking the recorded case and
/// exchange counts (so an emptied fixture cannot pass) and each request's sha256.
pub fn transcripts(fixture: &Value, test: &str, cases: usize, exchanges: usize) -> Vec<Transcript> {
    let recorded = support::array(&fixture["tests"][test]);
    assert_eq!(recorded.len(), cases, "{test}: parameter cases");
    let mut out = Vec::new();
    for case in recorded {
        let params = support::text(&case["params"]);
        for (index, transcript) in support::array(&case["transcripts"]).iter().enumerate() {
            let context = format!("{test} {params} exchange {index}");
            let request = request(transcript, &context);
            let expected = match &transcript["expected"] {
                Value::String(tag) if tag == "rejected" => None,
                expected => Some(rows(expected)),
            };
            out.push(Transcript {
                context,
                request,
                expected,
            });
        }
    }
    assert_eq!(out.len(), exchanges, "{test}: probe exchanges");
    out
}

/// Rendered rows against the Python rows; the first differing row is named.
pub fn assert_lines(context: &str, actual: &str, expected: &[String]) {
    let actual: Vec<&str> = actual
        .strip_suffix('\n')
        .unwrap_or_else(|| panic!("{context}: output is not LF-terminated"))
        .split('\n')
        .collect();
    if let Some(index) =
        (0..actual.len().min(expected.len())).find(|&index| actual[index] != expected[index])
    {
        panic!(
            "{context}: row {index}\n  rust:   {:?}\n  python: {:?}",
            actual[index], expected[index]
        );
    }
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: row count (rows agree up to the shorter)"
    );
}

/// A replay against the recorded exchange: rows must match, or Python demanded
/// rejection and the public API must refuse the same request.
pub fn assert_transcript(transcript: &Transcript, replay: Result<String, String>) {
    let context = &transcript.context;
    match (&transcript.expected, replay) {
        (Some(expected), Ok(actual)) => assert_lines(context, &actual, expected),
        (Some(_), Err(cause)) => {
            panic!("{context}: Rust refused a Python-accepted request: {cause}")
        }
        (None, Ok(actual)) => panic!(
            "{context}: Rust accepted a Python-rejected request ({} rows)",
            actual.lines().count()
        ),
        (None, Err(_)) => {}
    }
}
