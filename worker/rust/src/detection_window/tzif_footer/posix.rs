//! Admission grammar from CPython 3.12.3 Modules/_zoneinfo.c:
//! parse_tz_str, parse_abbr, parse_tz_delta, parse_transition_rule/time.
//! This recognizes a footer; it never constructs or substitutes timezone rules.
//! Unlike the POSIX specification, that C parser permits short abbreviations
//! and two-digit minute/second values above 59. Its input is a C string.

pub(super) fn accepts(bytes: &[u8]) -> bool {
    let end = bytes
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(bytes.len());
    Parser {
        rest: &bytes[..end],
    }
    .zone()
    .is_some()
}

struct Parser<'a> {
    rest: &'a [u8],
}

impl Parser<'_> {
    fn eat(&mut self, byte: u8) -> bool {
        if self.rest.first() != Some(&byte) {
            return false;
        }
        self.rest = &self.rest[1..];
        true
    }

    fn digits(&mut self, min: usize, max: usize) -> Option<u16> {
        let length = self
            .rest
            .iter()
            .take(max)
            .take_while(|b| b.is_ascii_digit())
            .count();
        if length < min {
            return None;
        }
        let number = self.rest[..length]
            .iter()
            .fold(0, |n, b| n * 10 + u16::from(b - b'0'));
        self.rest = &self.rest[length..];
        Some(number)
    }

    fn abbreviation(&mut self) -> Option<()> {
        if self.eat(b'<') {
            let length = self
                .rest
                .iter()
                .take_while(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-'))
                .count();
            self.rest = &self.rest[length..];
            return self.eat(b'>').then_some(());
        }
        let length = self
            .rest
            .iter()
            .take_while(|b| b.is_ascii_alphabetic())
            .count();
        if length == 0 {
            return None;
        }
        self.rest = &self.rest[length..];
        Some(())
    }

    fn time(&mut self, max_hour: u16) -> Option<()> {
        if matches!(self.rest.first(), Some(b'+' | b'-')) {
            self.rest = &self.rest[1..];
        }
        if self.digits(1, 3)? > max_hour {
            return None;
        }
        if self.eat(b':') {
            self.digits(2, 2)?;
            if self.eat(b':') {
                self.digits(2, 2)?;
            }
        }
        Some(())
    }

    fn rule(&mut self) -> Option<()> {
        if self.eat(b'M') {
            let month = self.digits(1, 2)?;
            if !(1..=12).contains(&month) || !self.eat(b'.') {
                return None;
            }
            let week = self.digits(1, 1)?;
            if !(1..=5).contains(&week) || !self.eat(b'.') || self.digits(1, 1)? > 6 {
                return None;
            }
        } else {
            let minimum = u16::from(self.eat(b'J'));
            if !(minimum..=365).contains(&self.digits(1, 3)?) {
                return None;
            }
        }
        if self.eat(b'/') {
            self.time(167)?;
        }
        Some(())
    }

    fn zone(&mut self) -> Option<()> {
        self.abbreviation()?;
        self.time(24)?;
        if self.rest.is_empty() {
            return Some(());
        }
        self.abbreviation()?;
        if self.rest.first() != Some(&b',') {
            self.time(24)?;
        }
        for _ in 0..2 {
            if !self.eat(b',') {
                return None;
            }
            self.rule()?;
        }
        self.rest.is_empty().then_some(())
    }
}

#[cfg(test)]
mod tests {
    use super::accepts;

    #[test]
    fn preserves_c_parser_acceptance_instead_of_jiff_or_python_alternate_grammar() {
        for text in [
            "A0",
            "AB0",
            "<A>0",
            "<>0",
            "ABC0:60",
            "ABC0:00:60",
            "ABC24:99:99",
            "ABC-24:99:99",
            "AAA0BBB,M3.2.0/-167:99:99,M11.1.0/167",
            "AAA0BBB,J1/0,J365/0",
            "AAA0BBB,0,365",
            "EST5\0garbage",
        ] {
            assert!(accepts(text.as_bytes()), "{text:?}");
        }
    }

    #[test]
    fn rejects_missing_offsets_and_malformed_or_out_of_range_rules() {
        for text in [
            "",
            "INVALID!!!",
            "ABC",
            "UT!0",
            "UTC25",
            "ABC0:1",
            "ABC0:00:1",
            "ABC0DEF",
            "ABC0DEF25,M3.2.0,M11.1.0",
            "ABC0DEF,J0,J365",
            "ABC0DEF,0,366",
            "ABC0DEF,M0.1.0,M11.1.0",
            "ABC0DEF,M3.6.0,M11.1.0",
            "ABC0DEF,M3.2.7,M11.1.0",
            "ABC0DEF,M3.2.0/168,M11.1.0",
            "ABC0DEF,M3.2.0,M11.1.0extra",
            "<AB!>0",
            "\0UTC0",
        ] {
            assert!(!accepts(text.as_bytes()), "{text:?}");
        }
    }
}
