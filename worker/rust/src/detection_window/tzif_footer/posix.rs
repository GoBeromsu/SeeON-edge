//! Admission grammar from CPython 3.12.3 Modules/_zoneinfo.c:
//! parse_tz_str, parse_abbr, parse_tz_delta, parse_transition_rule/time.
//! Unlike the POSIX specification, that C parser permits short abbreviations
//! and two-digit minute/second values above 59. Its input is a C string.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FooterRule {
    Fixed {
        std_seconds: i32,
    },
    Alternate {
        std_seconds: i32,
        dst_seconds: i32,
        start: TransitionRule,
        end: TransitionRule,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TransitionRule {
    pub(super) day: RuleDay,
    pub(super) time_seconds: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RuleDay {
    MonthWeekDay { month: u8, week: u8, weekday: u8 },
    Day { day: u16, julian: bool },
}

impl FooterRule {
    pub(super) fn requires_python_evaluator(self) -> bool {
        // Jiff accepts both day forms, but uses different day indexing and
        // leap-day semantics from the pinned CPython C evaluator.
        match self {
            Self::Fixed { .. } => false,
            Self::Alternate { start, end, .. } => {
                matches!(start.day, RuleDay::Day { .. }) || matches!(end.day, RuleDay::Day { .. })
            }
        }
    }
}

pub(super) fn parse(bytes: &[u8]) -> Option<FooterRule> {
    let end = bytes
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(bytes.len());
    Parser {
        rest: &bytes[..end],
    }
    .zone()
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

    fn time(&mut self, max_hour: u16) -> Option<i32> {
        let sign = if self.eat(b'-') {
            -1
        } else {
            self.eat(b'+');
            1
        };
        let hour = self.digits(1, 3)?;
        if hour > max_hour {
            return None;
        }
        let mut minute = 0;
        let mut second = 0;
        if self.eat(b':') {
            minute = self.digits(2, 2)?;
            if self.eat(b':') {
                second = self.digits(2, 2)?;
            }
        }
        Some(sign * (i32::from(hour) * 3600 + i32::from(minute) * 60 + i32::from(second)))
    }

    fn rule(&mut self) -> Option<TransitionRule> {
        let day = if self.eat(b'M') {
            let month = self.digits(1, 2)?;
            if !(1..=12).contains(&month) || !self.eat(b'.') {
                return None;
            }
            let week = self.digits(1, 1)?;
            if !(1..=5).contains(&week) || !self.eat(b'.') {
                return None;
            }
            let weekday = self.digits(1, 1)?;
            if weekday > 6 {
                return None;
            }
            RuleDay::MonthWeekDay {
                month: month as u8,
                week: week as u8,
                weekday: weekday as u8,
            }
        } else {
            let julian = self.eat(b'J');
            let day = self.digits(1, 3)?;
            if !(u16::from(julian)..=365).contains(&day) {
                return None;
            }
            RuleDay::Day { day, julian }
        };
        let time_seconds = if self.eat(b'/') {
            self.time(167)?
        } else {
            7200
        };
        Some(TransitionRule { day, time_seconds })
    }

    fn zone(&mut self) -> Option<FooterRule> {
        self.abbreviation()?;
        let std_seconds = -self.time(24)?;
        if self.rest.is_empty() {
            return Some(FooterRule::Fixed { std_seconds });
        }
        self.abbreviation()?;
        let dst_seconds = if self.rest.first() == Some(&b',') {
            std_seconds + 3600
        } else {
            -self.time(24)?
        };
        if !self.eat(b',') {
            return None;
        }
        let start = self.rule()?;
        if !self.eat(b',') {
            return None;
        }
        let end = self.rule()?;
        self.rest.is_empty().then_some(FooterRule::Alternate {
            std_seconds,
            dst_seconds,
            start,
            end,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            assert!(parse(text.as_bytes()).is_some(), "{text:?}");
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
            assert!(parse(text.as_bytes()).is_none(), "{text:?}");
        }
    }

    #[test]
    fn retains_signed_carries_posix_sign_and_default_dst() {
        for (text, std_seconds) in [
            ("A-0:00:60", 60),
            ("A+0:99:99", -6039),
            ("A24:99:99", -92439),
        ] {
            assert_eq!(
                parse(text.as_bytes()),
                Some(FooterRule::Fixed { std_seconds })
            );
        }
        assert_eq!(
            parse(b"A-24:99:99B,J59/-0:99:99,0"),
            Some(FooterRule::Alternate {
                std_seconds: 92439,
                dst_seconds: 96039,
                start: TransitionRule {
                    day: RuleDay::Day {
                        day: 59,
                        julian: true
                    },
                    time_seconds: -6039
                },
                end: TransitionRule {
                    day: RuleDay::Day {
                        day: 0,
                        julian: false
                    },
                    time_seconds: 7200
                },
            })
        );
    }
}
