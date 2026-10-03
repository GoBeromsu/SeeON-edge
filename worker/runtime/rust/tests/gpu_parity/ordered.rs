use std::fmt;

use super::spacing;

pub(super) struct Report {
    surface: String,
    frame: String,
    actual_count: usize,
    reference_count: usize,
    max_abs: Option<f64>,
    worst_index: Option<usize>,
    actual_at_worst: Option<f32>,
    reference_at_worst: Option<f32>,
    tolerance_at_worst: Option<f64>,
    over_limit_count: usize,
    nonfinite_count: usize,
    invalid_bound_count: usize,
    length_mismatch: bool,
    empty: bool,
    bound_error: Option<String>,
}

impl Report {
    pub(super) fn passes(&self) -> bool {
        !self.length_mismatch
            && !self.empty
            && self.nonfinite_count == 0
            && self.invalid_bound_count == 0
            && self.over_limit_count == 0
            && self.bound_error.is_none()
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let number = |value: Option<f64>| {
            value.map_or_else(|| "n/a".to_owned(), |value| format!("{value:e}"))
        };
        write!(
            f,
            "ordered surface={} frame={} count={}/{} max_abs={} worst_index={:?} \
             actual={:?} reference={:?} tolerance={} over_limit_count={} \
             nonfinite_count={} invalid_bound_count={} status={}",
            self.surface,
            self.frame,
            self.actual_count,
            self.reference_count,
            number(self.max_abs),
            self.worst_index,
            self.actual_at_worst,
            self.reference_at_worst,
            number(self.tolerance_at_worst),
            self.over_limit_count,
            self.nonfinite_count,
            self.invalid_bound_count,
            if self.passes() { "pass" } else { "fail" },
        )?;
        if self.length_mismatch {
            write!(f, " length_mismatch=true")?;
        }
        if self.empty {
            write!(f, " empty=true")?;
        }
        if let Some(error) = &self.bound_error {
            write!(f, " bound_error={error}")?;
        }
        Ok(())
    }
}

/// Compare each stored element to the oracle element at the same index.
pub(super) fn compare(
    surface: &str,
    frame: &str,
    actual: &[f32],
    reference: &[f32],
    max_abs: f64,
    oracle_ulps: Option<f64>,
) -> Report {
    let mut report = Report {
        surface: surface.to_owned(),
        frame: frame.to_owned(),
        actual_count: actual.len(),
        reference_count: reference.len(),
        max_abs: None,
        worst_index: None,
        actual_at_worst: None,
        reference_at_worst: None,
        tolerance_at_worst: None,
        over_limit_count: 0,
        nonfinite_count: actual.iter().filter(|value| !value.is_finite()).count()
            + reference.iter().filter(|value| !value.is_finite()).count(),
        invalid_bound_count: 0,
        length_mismatch: actual.len() != reference.len(),
        empty: actual.is_empty() || reference.is_empty(),
        bound_error: None,
    };

    if !max_abs.is_finite() {
        report.bound_error = Some("max_abs is non-finite".to_owned());
    } else if max_abs < 0.0 {
        report.bound_error = Some("max_abs is negative".to_owned());
    }
    if let Some(ulps) = oracle_ulps {
        if !ulps.is_finite() {
            report
                .bound_error
                .get_or_insert_with(|| "oracle ULP count is non-finite".to_owned());
        } else if ulps < 0.0 {
            report
                .bound_error
                .get_or_insert_with(|| "oracle ULP count is negative".to_owned());
        }
    }

    let invalid_parameters = report.bound_error.is_some();
    for (index, (&actual_value, &reference_value)) in actual.iter().zip(reference).enumerate() {
        if !actual_value.is_finite() || !reference_value.is_finite() {
            continue;
        }
        let difference = (f64::from(actual_value) - f64::from(reference_value)).abs();
        let tolerance = if invalid_parameters {
            report.invalid_bound_count += 1;
            None
        } else if let Some(ulps) = oracle_ulps {
            let spacing = spacing(reference_value);
            let ulp_bound = ulps * spacing;
            if !spacing.is_finite() || !ulp_bound.is_finite() {
                report.invalid_bound_count += 1;
                report
                    .bound_error
                    .get_or_insert_with(|| "oracle ULP bound is non-finite".to_owned());
                None
            } else {
                Some(max_abs.max(ulp_bound))
            }
        } else {
            Some(max_abs)
        };
        if let Some(tolerance) = tolerance
            && difference > tolerance
        {
            report.over_limit_count += 1;
        }
        if report.max_abs.is_none_or(|worst| difference > worst) {
            report.max_abs = Some(difference);
            report.worst_index = Some(index);
            report.actual_at_worst = Some(actual_value);
            report.reference_at_worst = Some(reference_value);
            report.tolerance_at_worst = tolerance;
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::super::{MAX_ABS, ORACLE_ULP};
    use super::compare;
    use crate::{BED_ROW, ROWS_FROM_SCORE, SCORE_COLUMN};

    fn strict(actual: &[f32], reference: &[f32]) -> super::Report {
        compare("pose", "frame", actual, reference, MAX_ABS, None)
    }

    fn bed(actual: &[f32], reference: &[f32]) -> super::Report {
        compare(
            "bed",
            "frame",
            actual,
            reference,
            MAX_ABS,
            Some(f64::from(ORACLE_ULP)),
        )
    }

    #[test]
    fn ordered_comparison_rejects_a_row_permutation() {
        let report = strict(&[1.0, 10.0, 2.0, 20.0], &[2.0, 20.0, 1.0, 10.0]);
        assert!(!report.passes());
        assert_eq!(report.worst_index, Some(1));
        assert_eq!(report.over_limit_count, 4);
    }

    #[test]
    fn raw_comparison_checks_errors_below_the_detection_gate() {
        let mut actual = vec![0.0; BED_ROW];
        actual[SCORE_COLUMN] = 0.1;
        let mut reference = actual.clone();
        reference[0] = 2e-4;
        assert!(f64::from(actual[SCORE_COLUMN]) < ROWS_FROM_SCORE);
        let report = bed(&actual, &reference);
        assert!(!report.passes());
        assert_eq!(report.over_limit_count, 1);
    }

    #[test]
    fn pose_is_strict_while_bed_admits_two_oracle_ulps() {
        let reference = 700.0_f32;
        let two_ulps = f32::from_bits(reference.to_bits() + ORACLE_ULP);
        let three_ulps = f32::from_bits(reference.to_bits() + ORACLE_ULP + 1);
        assert!(bed(&[two_ulps], &[reference]).passes());
        assert!(!strict(&[two_ulps], &[reference]).passes());
        assert!(!bed(&[three_ulps], &[reference]).passes());
    }

    #[test]
    fn bed_ulp_bound_uses_the_magnitude_of_negative_oracle_values() {
        let reference = -700.0_f32;
        let actual = f32::from_bits(reference.to_bits() + ORACLE_ULP);
        assert!(bed(&[actual], &[reference]).passes());
        assert_eq!(
            super::super::spacing(reference),
            super::super::spacing(700.0)
        );
    }

    #[test]
    fn nonfinite_values_fail_without_masking_their_count() {
        let report = strict(
            &[f32::NAN, f32::INFINITY, 1.0],
            &[0.0, 1.0, f32::NEG_INFINITY],
        );
        assert!(!report.passes());
        assert_eq!(report.nonfinite_count, 3);
    }

    #[test]
    fn nonfinite_and_overflowed_bounds_fail_closed() {
        for max_abs in [f64::NAN, f64::INFINITY] {
            let report = compare("pose", "frame", &[0.0], &[0.0], max_abs, None);
            assert!(!report.passes());
            assert!(report.bound_error.is_some());
        }
        let report = compare("bed", "frame", &[0.0], &[0.0], MAX_ABS, Some(f64::INFINITY));
        assert!(!report.passes());
        assert!(report.bound_error.is_some());
        let report = bed(&[f32::MAX], &[f32::MAX]);
        assert!(!report.passes());
        assert!(report.bound_error.is_some());
        let report = bed(&[f32::MAX, 1.0], &[f32::MAX, 0.0]);
        assert_eq!(report.invalid_bound_count, 1);
        assert_eq!(report.over_limit_count, 1);
    }

    #[test]
    fn unequal_and_empty_slices_fail_closed() {
        let unequal = strict(&[1.0, 2.0], &[1.0]);
        assert!(!unequal.passes());
        assert!(unequal.length_mismatch);
        assert_eq!(unequal.actual_count, 2);
        assert_eq!(unequal.reference_count, 1);
        let empty = strict(&[], &[]);
        assert!(!empty.passes());
        assert!(empty.empty);
    }

    #[test]
    fn absolute_boundary_is_inclusive_and_the_next_float_is_over_limit() {
        let boundary = MAX_ABS as f32;
        let over = f32::from_bits(boundary.to_bits() + 1);
        assert!(strict(&[boundary], &[0.0]).passes());
        let report = strict(&[over], &[0.0]);
        assert!(!report.passes());
        assert_eq!(report.over_limit_count, 1);
    }
}
