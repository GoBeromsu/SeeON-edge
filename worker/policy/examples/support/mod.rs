//! Test-only bounded TSV transport shared by the scored-policy examples.
use seeon_worker::episode::BusinessEvent;
use seeon_worker::trace::{DecisionTraceSnapshot, NumericTraceValue};
use std::fmt::Write;
use std::io::{Read, Write as IoWrite};

pub const MAX_INPUT: usize = 128 * 1024;
pub const MAX_OUTPUT: usize = 4 * 1024 * 1024;
pub const MAX_CALLS: usize = 128;
pub const MAX_ITEMS: usize = 64;
pub type Result<T> = std::result::Result<T, ()>;

pub mod wire {
    use super::{MAX_ITEMS, Result};
    use std::fmt::Write;
    use std::str::{FromStr, Split};

    pub struct Fields<'a>(Split<'a, char>);
    impl<'a> Fields<'a> {
        pub fn new(line: &'a str) -> Self {
            Self(line.split('\t'))
        }
        pub fn next(&mut self) -> Result<&'a str> {
            self.0.next().ok_or(())
        }
        pub fn number<T: FromStr>(&mut self) -> Result<T> {
            self.next()?.parse().map_err(|_| ())
        }
        pub fn count(&mut self) -> Result<usize> {
            let count = self.number()?;
            if count > MAX_ITEMS {
                Err(())
            } else {
                Ok(count)
            }
        }
        pub fn float(&mut self) -> Result<f64> {
            let token = self.next()?;
            if token.len() != 16 || !token.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(());
            }
            let value = f64::from_bits(u64::from_str_radix(token, 16).map_err(|_| ())?);
            if value.is_finite() {
                Ok(value)
            } else {
                Err(())
            }
        }
        pub fn text(&mut self) -> Result<String> {
            let token = self.next()?;
            if token.len() > 2048
                || !token.len().is_multiple_of(2)
                || !token.bytes().all(|b| b.is_ascii_hexdigit())
            {
                return Err(());
            }
            let bytes = (0..token.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&token[i..i + 2], 16).map_err(|_| ()))
                .collect::<Result<Vec<_>>>()?;
            String::from_utf8(bytes).map_err(|_| ())
        }
        pub fn optional_id(&mut self) -> Result<Option<u64>> {
            match self.next()? {
                "-" => Ok(None),
                token => token.parse().map(Some).map_err(|_| ()),
            }
        }
        pub fn optional_float(&mut self) -> Result<Option<f64>> {
            let token = self.next()?;
            if token == "-" {
                Ok(None)
            } else {
                Self::new(token).float().map(Some)
            }
        }
        pub fn end(&mut self) -> Result<()> {
            if self.0.next().is_none() {
                Ok(())
            } else {
                Err(())
            }
        }
    }
    pub fn hex(text: &str) -> String {
        let mut encoded = String::with_capacity(text.len() * 2);
        for byte in text.bytes() {
            write!(encoded, "{byte:02x}").expect("writing to String");
        }
        encoded
    }
    pub fn bits(value: f64) -> String {
        format!("{:016x}", value.to_bits())
    }
    pub fn optional<T: ToString>(value: Option<T>) -> String {
        value.map_or_else(|| "-".into(), |value| value.to_string())
    }
}

pub fn event(fields: &mut wire::Fields<'_>) -> Result<BusinessEvent> {
    Ok(BusinessEvent {
        domain: fields.text()?,
        event_type: fields.text()?,
        identity: fields.text()?,
        camera_id: fields.text()?,
        facility_id: fields.text()?,
        time_sec: fields.float()?,
        probability: fields.optional_float()?,
        person_id: fields.optional_id()?,
        bed_id: fields.optional_id()?,
    })
}

pub fn events(out: &mut String, events: &[BusinessEvent]) -> Result<()> {
    use wire::{bits, hex, optional};
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
        .map_err(|_| ())?;
    }
    Ok(())
}

pub fn traces(out: &mut String, traces: &[DecisionTraceSnapshot]) -> Result<()> {
    use wire::{bits, hex, optional};
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
        .map_err(|_| ())?;
        let mut values: Vec<_> = trace.values().iter().collect();
        values.sort_by_key(|(name, _)| name.as_str());
        for (name, value) in values {
            let (kind, encoded) = match value {
                NumericTraceValue::Integer(value) => ("I", value.to_string()),
                NumericTraceValue::Float(value) => ("F", bits(value.get())),
            };
            writeln!(out, "V\t{}\t{kind}\t{encoded}", hex(name.as_str())).map_err(|_| ())?;
        }
        let mut missing: Vec<_> = trace.missing_values().iter().collect();
        missing.sort_by_key(|(name, _)| name.as_str());
        for (name, reason) in missing {
            writeln!(out, "M\t{}\t{}", hex(name.as_str()), hex(reason.as_str())).map_err(|_| ())?;
        }
    }
    Ok(())
}

pub fn lines(input: &str) -> Result<std::str::SplitTerminator<'_, char>> {
    if input.len() > MAX_INPUT || !input.is_ascii() || !input.ends_with('\n') {
        Err(())
    } else {
        Ok(input.split_terminator('\n'))
    }
}

pub fn bounded(out: &str) -> Result<()> {
    if out.len() > MAX_OUTPUT {
        Err(())
    } else {
        Ok(())
    }
}

/// `false` preserves a bounded domain-error transcript but exits nonzero.
/// Malformed transport returns Err and writes no stdout (including prior calls).
pub fn main(name: &'static str, run: fn(&str) -> Result<(String, bool)>) {
    std::panic::set_hook(Box::new(move |_| eprintln!("{name}: internal failure")));
    let result = (|| {
        if std::env::args_os().count() != 1 {
            return Err(());
        }
        let mut input = String::new();
        std::io::stdin()
            .lock()
            .take((MAX_INPUT + 1) as u64)
            .read_to_string(&mut input)
            .map_err(|_| ())?;
        let (output, success) = run(&input)?;
        bounded(&output)?;
        std::io::stdout()
            .lock()
            .write_all(output.as_bytes())
            .map_err(|_| ())?;
        if success { Ok(()) } else { Err(()) }
    })();
    if result.is_err() {
        eprintln!("{name}: rejected");
        std::process::exit(2);
    }
}
