//! Test-only scored-policy transport; never inference or a production replay CLI.
//! Build: cargo build -p seeon-worker --example fall_policy_probe --offline --locked
//!
//! Protocol v1: ASCII TSV, LF-terminated, header `FALLPROBE\t1`. All UTF-8 text
//! (including vocabulary) is lowercase hex; f64 is exactly 16 hex IEEE-754 bits.
//! Decimal integers, booleans 0/1, and `-` for absent numeric fields are distinct.
//! N camera facility boot epoch source_generation [all ten FallPolicyParameters
//! fields in declaration order] watch_count [track_ids...] constructs one owner.
//! U frame time live_count [ids...] score_count [id background transition fallen]...
//! missing_count [id reason]... updates it (`-` count means None, 0 an empty map).
//! C coasts. R takes all nine BusinessEvent fields in declaration order and
//! releases by exact identity, not other metadata.
//! N must occur exactly once, first; each following line is one call.
//!
//! Each call emits O index fresh switches event_count trace_count, then ordered E
//! event fields, ordered T reason previous current triggered track bed known_count
//! missing_count with V name I|F value / M name reason rows (maps sorted by token),
//! and S track generation fallen for every watched ID in supplied order.
//! END call_count terminates a successful response; there is no PASS claim.
//! Limits: 128 calls including N, 64 entries per counted collection, 128 KiB input,
//! 4 MiB output, 1024 UTF-8 bytes per text field. Library capacities are all 64.
//! These transport/library admission bounds are NOT Python-domain equivalence.

use seeon_worker::episode::BusinessEvent;
use seeon_worker::fall::FallPolicyDecider;

mod support;
use support::{MAX_CALLS, MAX_ITEMS, Result, wire};

mod transport {
    use super::{BusinessEvent, FallPolicyDecider, MAX_ITEMS, Result, wire::Fields};
    use seeon_worker::fall::{FallCapacities, FallPolicy, FallPolicyParameters, FallProbabilities};
    use seeon_worker::trace::DecisionTraceMissingReason;
    use std::collections::{BTreeMap, BTreeSet};

    pub fn construct(fields: &mut Fields<'_>) -> Result<(FallPolicyDecider, Vec<u64>)> {
        if fields.next()? != "N" {
            return Err(());
        }
        let (camera, facility, boot, epoch) = (
            fields.text()?,
            fields.text()?,
            fields.text()?,
            fields.text()?,
        );
        let source_generation = fields.number()?;
        let policy = FallPolicy::new(FallPolicyParameters {
            transition_threshold: fields.float()?,
            transition_votes: fields.number()?,
            transition_window: fields.number()?,
            fallen_threshold: fields.float()?,
            fallen_consecutive: fields.number()?,
            recovery_transition_max: fields.float()?,
            recovery_fallen_max: fields.float()?,
            recovery_consecutive: fields.number()?,
            track_ttl_frames: fields.number()?,
            cooldown_frames: fields.number()?,
        })
        .map_err(|_| ())?;
        let watch = (0..fields.count()?)
            .map(|_| fields.number())
            .collect::<Result<Vec<u64>>>()?;
        if watch.iter().copied().collect::<BTreeSet<_>>().len() != watch.len() {
            return Err(());
        }
        fields.end()?;
        let capacities = FallCapacities {
            retained_tracks: MAX_ITEMS,
            generation_identities: MAX_ITEMS,
            episodes: MAX_ITEMS,
            vote_window: MAX_ITEMS,
        };
        let decider = FallPolicyDecider::new(
            camera,
            facility,
            boot,
            epoch,
            source_generation,
            policy,
            capacities,
        )
        .map_err(|_| ())?;
        Ok((decider, watch))
    }

    pub fn update(
        d: &mut FallPolicyDecider,
        fields: &mut Fields<'_>,
    ) -> Result<Vec<BusinessEvent>> {
        let frame = fields.number()?;
        let time = fields.float()?;
        let live = (0..fields.count()?)
            .map(|_| fields.number())
            .collect::<Result<Vec<u64>>>()?;
        let mut scores = BTreeMap::new();
        for _ in 0..fields.count()? {
            let id = fields.number()?;
            let score = FallProbabilities::new(fields.float()?, fields.float()?, fields.float()?)
                .map_err(|_| ())?;
            if scores.insert(id, score).is_some() {
                return Err(());
            }
        }
        let missing_count = fields.next()?;
        let missing = if missing_count == "-" {
            None
        } else {
            let mut reasons = BTreeMap::new();
            for _ in 0..Fields::new(missing_count).count()? {
                let id = fields.number()?;
                let reason = DecisionTraceMissingReason::from_token(&fields.text()?).ok_or(())?;
                if reasons.insert(id, reason).is_some() {
                    return Err(());
                }
            }
            Some(reasons)
        };
        fields.end()?;
        d.update(frame, time, &scores, live, missing.as_ref())
            .map_err(|_| ())
    }

    pub fn release(
        d: &mut FallPolicyDecider,
        fields: &mut Fields<'_>,
    ) -> Result<Vec<BusinessEvent>> {
        let event = super::support::event(fields)?;
        fields.end()?;
        // Python returns None here. Compare the subsequent state/identity effects,
        // not Rust's extra boolean return value, and leave stale traces untouched.
        d.release_onset(&event).map_err(|_| ())?;
        Ok(Vec::new())
    }
}

mod output {
    use super::wire::optional;
    use super::{BusinessEvent, FallPolicyDecider, Result, support};
    use std::fmt::Write;

    pub fn observe(
        out: &mut String,
        index: usize,
        d: &FallPolicyDecider,
        events: &[BusinessEvent],
        watch: &[u64],
    ) -> Result<()> {
        writeln!(
            out,
            "O\t{index}\t{}\t{}\t{}\t{}",
            u8::from(d.last_update_evaluated()),
            d.track_id_switch_absorbed_total(),
            events.len(),
            d.last_trace_snapshots().len()
        )
        .map_err(|_| ())?;
        support::events(out, events)?;
        support::traces(out, d.last_trace_snapshots())?;
        for &track in watch {
            writeln!(
                out,
                "S\t{track}\t{}\t{}",
                optional(d.generation_for(track)),
                u8::from(d.is_fallen(track))
            )
            .map_err(|_| ())?;
        }
        support::bounded(out)
    }
}

fn run(input: &str) -> Result<String> {
    let mut lines = support::lines(input)?;
    if lines.next() != Some("FALLPROBE\t1") {
        return Err(());
    }
    let (mut d, watch) = transport::construct(&mut wire::Fields::new(lines.next().ok_or(())?))?;
    let mut out = String::from("FALLPROBE\t1\n");
    output::observe(&mut out, 0, &d, &[], &watch)?;
    let mut calls = 1;
    for line in lines {
        if calls == MAX_CALLS {
            return Err(());
        }
        let mut fields = wire::Fields::new(line);
        let events = match fields.next()? {
            "U" => transport::update(&mut d, &mut fields)?,
            "C" => {
                fields.end()?;
                d.coast().map_err(|_| ())?
            }
            "R" => transport::release(&mut d, &mut fields)?,
            _ => return Err(()),
        };
        output::observe(&mut out, calls, &d, &events, &watch)?;
        calls += 1;
    }
    out.push_str(&format!("END\t{calls}\n"));
    support::bounded(&out)?;
    Ok(out)
}

fn main() {
    support::main("fall-policy-probe", |input| Ok((run(input)?, true)));
}
