//! Test-only scored CPU policy/window differential, never geometry or inference.
//! Build (parent-owned): cargo build -p seeon-worker --example bed_policy_probe --offline --locked
//!
//! ASCII LF TSV v1; shared support encoding: UTF-8 text as hex, finite f64 bits,
//! decimal integers, 0/1 bool, '-' absent. No CLI args. 128 KiB input, 4 MiB
//! output, 128 calls, 64 per collection, 1024-byte text. Library capacities 64
//! except explicit episode capacity 1..64. These are narrower than Python.
//! BEDPROBE\t1 then N camera facility boot epoch generation min hold grace
//! in_dwell outside_dwell stale_after episode_capacity window watch_count
//! [track bed]... . window is '-' or start end zone (hex). N counts as a call.
//! U frame optional_pts observed_at snapshot_at wall region cycles persons beds
//! ids live features containments prior_rows. All collections are count-prefixed:
//! box = x1 y1 x2 y2 confidence polygon ('-' or count [x y]...);
//! features = all BedPoseFeatures fields in declaration order;
//! containments = count [column_count floats...]...;
//! prior_rows = count [previous_track box ratio_count floats...]... .
//! wall = '-' or S|E + hex civil-ISO datetime + decimal offset-seconds
//! ('-' offset = naive). S means identical Python tzinfo to THIS target; E means
//! external, including no active window. Recompute the tag after window changes.
//! Present clocks without an explicit S/E tag are rejected.
//! C optional_frame snapshot_at; W window; R all nine BusinessEvent fields.
//! WINDOWPROBE\t1 instead accepts W start end zone wall per call (no N).
//! Every zone read requires SEEON_TEST_ZONEINFO_DIR, with no default/database.
//!
//! Bed output O index switches shadow complete scoring_present events traces
//! assignments recoveries lost debug_present disposition; then E/T/V/M as fall,
//! G cumulative scoring (max containment, transitions, assignments), optional Q
//! scoring observation, A track bed candidate frames in_dwell out_dwell armed
//! last_pts last_box ('-' or '1' + box), H recovery(person bed), L lost(track),
//! D frame stale age region ('-' or source cycles) person_count bed_count
//! status_count raw_event_count, P/B boxes, S bed occupancy person box, J raw
//! events, Z watched track bed episode_state. Assignment/map order is numeric;
//! traces/events/observations/statuses retain order. Config is supplied in N.
//! Window output W index contains; END call_count always delimits transcripts.
//! Domain failures emit X index rejected|fatal|poisoned hex(Debug(cause))
//! optional_phase optional_track prefix_count followed by the normal observation
//! (E rows conserve the fatal accepted prefix). Any X exits 2, even after release
//! or inspection; malformed transport/config/window failure exits 2, no stdout.
//! Rust-only complete/error flags have no Python analogue. Watch pairs bound
//! episode inspection; private authority internals are not a serialized API.

mod support;
use seeon_worker::bed_exit::*;
use seeon_worker::detection_window::{AwareDateTime, ClockRelation, DateTime, DetectionWindow};
use std::fmt::Write;
use support::wire::{Fields, bits, hex, optional};
use support::{MAX_CALLS, MAX_ITEMS, Result};

mod transport {
    use super::*;

    pub fn collect<T>(
        f: &mut Fields<'_>,
        read: fn(&mut Fields<'_>) -> Result<T>,
    ) -> Result<Vec<T>> {
        (0..f.count()?).map(|_| read(f)).collect()
    }
    pub fn boolean(f: &mut Fields<'_>) -> Result<bool> {
        match f.next()? {
            "0" => Ok(false),
            "1" => Ok(true),
            _ => Err(()),
        }
    }
    pub fn optional_frame(f: &mut Fields<'_>) -> Result<Option<i64>> {
        match f.next()? {
            "-" => Ok(None),
            v => v.parse().map(Some).map_err(|_| ()),
        }
    }
    pub fn box_value(f: &mut Fields<'_>) -> Result<BoundingBox> {
        let (x1, y1, x2, y2, confidence) = (
            f.number()?,
            f.number()?,
            f.number()?,
            f.number()?,
            f.float()?,
        );
        let token = f.next()?;
        let polygon = if token == "-" {
            None
        } else {
            Some(
                (0..Fields::new(token).count()?)
                    .map(|_| Ok((f.number()?, f.number()?)))
                    .collect::<Result<_>>()?,
            )
        };
        Ok(BoundingBox {
            x1,
            y1,
            x2,
            y2,
            confidence,
            polygon,
        })
    }
    fn pose(f: &mut Fields<'_>) -> Result<BedPoseFeatures> {
        Ok(BedPoseFeatures {
            track_id: f.number()?,
            bed_id: f.optional_id()?,
            torso_in_frac: f.float()?,
            lower_in_frac: f.float()?,
            keypoint_in_frac: f.float()?,
            hip_depth: f.float()?,
            torso_angle: f.float()?,
            centroid_displacement: f.float()?,
            hip_x_rel: f.float()?,
            hip_y_rel: f.float()?,
            observability: f.float()?,
            bed_polygon_valid: boolean(f)?,
        })
    }
    pub fn wall(f: &mut Fields<'_>) -> Result<Option<AwareDateTime>> {
        let relation = match f.next()? {
            "-" => return Ok(None),
            "S" => ClockRelation::SameTargetTzinfo,
            "E" => ClockRelation::DifferentTzinfo,
            _ => return Err(()),
        };
        let local: DateTime = f.text()?.parse().map_err(|_| ())?;
        let offset = match f.next()? {
            "-" => None,
            v => Some(v.parse().map_err(|_| ())?),
        };
        AwareDateTime::new(local, offset, relation)
            .map(Some)
            .map_err(|_| ())
    }
    pub fn window(f: &mut Fields<'_>) -> Result<Option<DetectionWindow>> {
        let token = f.next()?;
        if token == "-" {
            return Ok(None);
        }
        let start = Fields::new(token).text()?;
        let (end, zone) = (f.text()?, f.text()?);
        let root = std::env::var_os("SEEON_TEST_ZONEINFO_DIR").ok_or(())?;
        DetectionWindow::from_zoneinfo_dir(&start, &end, &zone, std::path::Path::new(&root))
            .map(Some)
            .map_err(|_| ())
    }
    pub fn construct(f: &mut Fields<'_>) -> Result<(BedExitMonitor, Vec<(u64, usize)>)> {
        if f.next()? != "N" {
            return Err(());
        }
        let (camera_id, facility_id, boot, epoch) = (f.text()?, f.text()?, f.text()?, f.text()?);
        let generation = f.number()?;
        let mut config = BedExitConfig {
            camera_id,
            facility_id,
            min_containment: f.float()?,
            hold_frames: f.number()?,
            grace_frames: f.number()?,
            in_bed_dwell_sec: f.float()?,
            outside_dwell_sec: f.float()?,
            night_window: None,
        };
        let stale_after = f.float()?;
        let episodes = f.count()?;
        config.night_window = window(f)?;
        let watch = collect(f, |f| Ok((f.number()?, f.number()?)))?;
        if watch
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != watch.len()
        {
            return Err(());
        }
        f.end()?;
        let capacities = BedExitCapacities {
            tracks: MAX_ITEMS,
            observations: MAX_ITEMS,
            beds: MAX_ITEMS,
            pose_features: MAX_ITEMS,
            polygon_points: MAX_ITEMS,
            episodes,
        };
        let monitor = BedExitMonitor::new(config, boot, epoch, generation, capacities, stale_after)
            .map_err(|_| ())?;
        Ok((monitor, watch))
    }
    pub fn update(f: &mut Fields<'_>) -> Result<(BedExitInput, BedExitClocks)> {
        let (frame_index, time_sec) = (f.number()?, f.optional_float()?);
        let clocks = BedExitClocks {
            observed_at: f.float()?,
            snapshot_at: f.float()?,
            wall_time: wall(f)?,
        };
        let source = match f.text()?.as_str() {
            "fresh" => BedRegionCacheState::Fresh,
            "cached" => BedRegionCacheState::Cached,
            "empty" => BedRegionCacheState::Empty,
            "expired" => BedRegionCacheState::Expired,
            _ => return Err(()),
        };
        let input = BedExitInput {
            frame_index,
            time_sec,
            bed_region: BedRegionDebugSnapshot {
                source,
                empty_cycles: f.number()?,
            },
            person_boxes: collect(f, box_value)?,
            bed_boxes: collect(f, box_value)?,
            track_ids: collect(f, |f| f.optional_id())?,
            live_track_ids: collect(f, |f| f.number())?,
            bed_pose_features: collect(f, pose)?,
            containments: collect(f, |f| collect(f, |f| f.float()))?,
            prior_box_containments: collect(f, |f| {
                Ok(PriorBoxContainments {
                    previous_track_id: f.number()?,
                    previous_box: box_value(f)?,
                    ratios: collect(f, |f| f.float())?,
                })
            })?,
        };
        f.end()?;
        Ok((input, clocks))
    }
}

mod output {
    use super::*;
    use seeon_worker::episode::BusinessEvent;

    fn box_fields(b: &BoundingBox) -> String {
        let mut text = format!(
            "{}\t{}\t{}\t{}\t{}",
            b.x1,
            b.y1,
            b.x2,
            b.y2,
            bits(b.confidence)
        );
        match &b.polygon {
            None => text.push_str("\t-"),
            Some(points) => {
                write!(text, "\t{}", points.len()).expect("String");
                for (x, y) in points {
                    write!(text, "\t{x}\t{y}").expect("String");
                }
            }
        }
        text
    }
    fn scoring(out: &mut String, tag: &str, s: BedExitScoring) -> Result<()> {
        writeln!(
            out,
            "{tag}\t{}\t{}\t{}",
            bits(s.max_containment_observed),
            s.grace_positive_transitions,
            s.assignments_made
        )
        .map_err(|_| ())
    }
    pub fn observe(
        out: &mut String,
        index: usize,
        m: &BedExitMonitor,
        events: &[BusinessEvent],
        score: Option<BedExitScoring>,
        watch: &[(u64, usize)],
    ) -> Result<()> {
        writeln!(
            out,
            "O\t{index}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            m.track_id_switch_absorbed_total(),
            m.last_shadow_trace_count(),
            u8::from(m.last_update_complete()),
            u8::from(score.is_some()),
            events.len(),
            m.last_trace_snapshots().len(),
            m.assignments().len(),
            m.last_recovery_events().len(),
            m.last_lost_track_ids().len(),
            u8::from(m.last_debug_snapshot().is_some()),
            optional(m.last_episode_disposition().map(|d| hex(d.as_str())))
        )
        .map_err(|_| ())?;
        support::events(out, events)?;
        support::traces(out, m.last_trace_snapshots())?;
        scoring(out, "G", m.scoring())?;
        if let Some(score) = score {
            scoring(out, "Q", score)?;
        }
        for (track, a) in m.assignments() {
            writeln!(
                out,
                "A\t{track}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                optional(a.bed_id),
                optional(a.candidate_bed_id),
                a.candidate_frames,
                bits(a.in_bed_dwell_sec),
                bits(a.outside_dwell_sec),
                u8::from(a.armed),
                optional(a.last_time_sec.map(bits)),
                a.last_box
                    .as_ref()
                    .map_or_else(|| "-".into(), |b| format!("1\t{}", box_fields(b)))
            )
            .map_err(|_| ())?;
        }
        for event in m.last_recovery_events() {
            writeln!(out, "H\t{}\t{}", event.person_id, event.bed_id).map_err(|_| ())?;
        }
        for track in m.last_lost_track_ids() {
            writeln!(out, "L\t{track}").map_err(|_| ())?;
        }
        if let Some(d) = m.last_debug_snapshot() {
            let region = d.bed_region.map_or_else(
                || "-".into(),
                |r| format!("{}\t{}", hex(r.source.as_str()), r.empty_cycles),
            );
            writeln!(
                out,
                "D\t{}\t{}\t{}\t{region}\t{}\t{}\t{}\t{}",
                optional(d.frame_index),
                u8::from(d.stale),
                optional(d.observation_age_sec.map(bits)),
                d.person_boxes.len(),
                d.bed_boxes.len(),
                d.statuses.len(),
                d.events.len()
            )
            .map_err(|_| ())?;
            for b in &d.person_boxes {
                writeln!(out, "P\t{}", box_fields(b)).map_err(|_| ())?;
            }
            for b in &d.bed_boxes {
                writeln!(out, "B\t{}", box_fields(b)).map_err(|_| ())?;
            }
            for s in &d.statuses {
                writeln!(
                    out,
                    "S\t{}\t{}\t{}\t{}",
                    s.bed_id,
                    hex(s.occupancy.as_str()),
                    optional(s.person_id),
                    box_fields(&s.box_value)
                )
                .map_err(|_| ())?;
            }
            for e in &d.events {
                writeln!(out, "J\t{}\t{}", e.person_id, e.bed_id).map_err(|_| ())?;
            }
        }
        for &(track, bed) in watch {
            writeln!(
                out,
                "Z\t{track}\t{bed}\t{}",
                hex(m.episode_state(track, bed).as_str())
            )
            .map_err(|_| ())?;
        }
        support::bounded(out)
    }
    pub fn failure(
        out: &mut String,
        index: usize,
        error: BedExitError,
    ) -> Result<Vec<BusinessEvent>> {
        let (kind, cause, phase, track, events) = match error {
            BedExitError::Rejected(cause) => ("rejected", cause, None, None, Vec::new()),
            BedExitError::Poisoned(cause) => ("poisoned", cause, None, None, Vec::new()),
            BedExitError::FatalPartialState {
                phase,
                track_id,
                cause,
                emitted_events,
            } => ("fatal", cause, Some(phase), track_id, emitted_events),
        };
        writeln!(
            out,
            "X\t{index}\t{kind}\t{}\t{}\t{}\t{}",
            hex(&format!("{cause:?}")),
            optional(phase.map(|p| hex(&format!("{p:?}")))),
            optional(track),
            events.len()
        )
        .map_err(|_| ())?;
        Ok(events)
    }
}

fn run(input: &str) -> Result<(String, bool)> {
    let mut lines = support::lines(input)?;
    let header = lines.next().ok_or(())?;
    let mut out = format!("{header}\n");
    let mut calls = 0;
    let mut success = true;
    if header == "WINDOWPROBE\t1" {
        for line in lines {
            if calls == MAX_CALLS {
                return Err(());
            }
            let mut f = Fields::new(line);
            if f.next()? != "W" {
                return Err(());
            }
            let window = transport::window(&mut f)?.ok_or(())?;
            let now = transport::wall(&mut f)?.ok_or(())?;
            f.end()?;
            let contains = window.contains(now).map_err(|_| ())?;
            writeln!(out, "W\t{calls}\t{}", u8::from(contains)).map_err(|_| ())?;
            support::bounded(&out)?;
            calls += 1;
        }
        if calls == 0 {
            return Err(());
        }
    } else if header == "BEDPROBE\t1" {
        let (mut m, watch) = transport::construct(&mut Fields::new(lines.next().ok_or(())?))?;
        output::observe(&mut out, 0, &m, &[], None, &watch)?;
        calls = 1;
        for line in lines {
            if calls == MAX_CALLS {
                return Err(());
            }
            let mut f = Fields::new(line);
            let outcome = match f.next()? {
                "U" => {
                    let (input, clocks) = transport::update(&mut f)?;
                    m.update(&input, clocks)
                }
                "C" => {
                    let (frame, now) = (transport::optional_frame(&mut f)?, f.float()?);
                    f.end()?;
                    m.coast(frame, now)
                }
                "W" => {
                    let window = transport::window(&mut f)?;
                    f.end()?;
                    m.update_night_window(window).map(|()| BedExitOutcome {
                        events: Vec::new(),
                        scoring_observation: None,
                    })
                }
                "R" => {
                    let event = support::event(&mut f)?;
                    f.end()?;
                    m.release_onset(&event.identity);
                    Ok(BedExitOutcome {
                        events: Vec::new(),
                        scoring_observation: None,
                    })
                }
                _ => return Err(()),
            };
            let (events, score) = match outcome {
                Ok(outcome) => (outcome.events, outcome.scoring_observation),
                Err(error) => {
                    success = false;
                    (output::failure(&mut out, calls, error)?, None)
                }
            };
            output::observe(&mut out, calls, &m, &events, score, &watch)?;
            calls += 1;
        }
    } else {
        return Err(());
    }
    writeln!(out, "END\t{calls}").map_err(|_| ())?;
    support::bounded(&out)?;
    Ok((out, success))
}

fn main() {
    support::main("bed-policy-probe", run);
}
