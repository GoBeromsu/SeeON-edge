//! Genuine lifecycle child, registered only through the private test root.

use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use super::super::super::{RELAY_URL, output, production_policy, runtime};
use super::super::shutdown;
use super::control::{Directory, nonce, ns};
use super::controller;
use super::supervision::{Receipt, diagnostics};
use crate::config::{env::Env, pull};
use crate::run::{boot::boot, media_config, settings::Settings, status::BootStatusContext};
use crate::seam::{Clock, IdSource, RandomIds, SystemClock};
use crate::shutdown::{MAX_SHUTDOWN_BUDGET, ShutdownDeadline};

pub(super) fn child(
    state_path: PathBuf,
    auxiliary: OsString,
    state: Directory,
    record: Directory,
    run: [u64; 2],
) {
    let env: Env = std::env::vars().collect();
    let mut settings = Settings::parse(
        env,
        &[
            "--state-dir".into(),
            state_path.into_os_string(),
            "--auxiliary-runtime".into(),
            auxiliary,
        ],
        production_policy(),
    )
    .expect("approved real fixture settings");
    let clock = Arc::new(SystemClock::new());
    let deadline = Arc::new(ShutdownDeadline::new(MAX_SHUTDOWN_BUDGET).unwrap());
    let boot_id = RandomIds.uuid4().expect("real boot identity");
    let token = settings.relay_token().expect("real relay token");
    pull::check_release_identity(RELAY_URL).expect("approved relay release identity");
    let config = pull::pull_startup_config(
        RELAY_URL,
        token,
        settings.state_dir(),
        std::path::Path::new(crate::config::windows::ZONEINFO_DIR),
    )
    .expect("approved real startup config");
    assert!(
        !config.cameras.is_empty() && !config.stale && config.source == pull::ConfigSource::Pulled
    );
    let report = BootStatusContext::for_config(
        RELAY_URL,
        token,
        &config,
        super::super::super::unix_seconds(clock.wall()).expect("actual wall time"),
    )
    .expect("real boot report context");
    settings.policy.deployed_batch = super::super::super::deployed_batch(&config);
    let booted =
        boot(settings, report, clock.as_ref(), Arc::clone(&deadline)).expect("genuine warmed boot");
    let lease = state.lease_identity();
    let model_states = [
        Arc::clone(&booted.models.fall().expect("actual fall owner").state),
        Arc::clone(&booted.models.bed().expect("actual bed owner").state),
        Arc::clone(
            &booted
                .models
                .stored_pose()
                .expect("actual stored-pose owner")
                .state,
        ),
    ];
    assert_eq!(
        Directory::open(&booted.admitted.flow.record_dir, false).identity(),
        record.identity()
    );
    let source = match media_config::assemble(&booted.admitted.flow, &config.cameras, &boot_id)
        .expect("same real media assembly")
    {
        media_config::MediaAssembly::Configured(config) => {
            config.sources.into_iter().next().unwrap()
        }
        media_config::MediaAssembly::Idle => panic!("N2 requires genuine nonempty media"),
    };
    let mut session = output::prepare(
        &booted,
        &config,
        &boot_id,
        Arc::clone(&clock) as Arc<dyn Clock>,
        Arc::clone(&deadline),
    )
    .unwrap_or_else(|failure| panic!("genuine output preparation: {}", failure.error));
    let media = session
        .media
        .as_ref()
        .expect("actual prepared media session");
    let diagnostics_owner = Arc::clone(&media.diagnostics);
    let commands = media.commands.clone();
    let observation = Arc::clone(&diagnostics_owner);
    let controller_clock = Arc::clone(&clock);
    let controller_deadline = Arc::clone(&deadline);
    let controls = record.copy();
    let runtime_start = ns(clock.monotonic());
    let controller = std::thread::spawn(move || {
        let result = std::panic::catch_unwind(|| {
            controller(
                controls,
                run,
                source,
                observation,
                commands,
                Arc::clone(&controller_clock),
                Arc::clone(&controller_deadline),
            )
        });
        if result.is_err() {
            let _ = controller_deadline.request_at(controller_clock.monotonic());
        }
        result
    });
    let (outcome, sink) = runtime::run(
        &mut session,
        &booted,
        clock.as_ref(),
        &deadline,
        config.directive,
    );
    let runtime_return = ns(clock.monotonic());
    let entered = controller
        .join()
        .expect("N2 controller thread")
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
    match &outcome {
        runtime::RunOutcome::Stopped => {}
        runtime::RunOutcome::Failed(error) => panic!("unrelated real runtime failure: {error}"),
        runtime::RunOutcome::Restart(_) => panic!("unrelated real runtime restart"),
    }
    let shutdown_call = ns(clock.monotonic());
    let actual = shutdown(session, booted, clock.as_ref(), &deadline, outcome, sink);
    let shutdown_return = ns(clock.monotonic());
    let returned = diagnostics(&diagnostics_owner.snapshot());
    let end = deadline
        .deadline()
        .expect("actual requested shutdown deadline");
    while {
        let now = clock.monotonic();
        let shared = deadline.deadline().expect("live shared deadline");
        assert_eq!(
            shared, end,
            "fixture must not shorten or replace the shared deadline"
        );
        now < shared
    } {
        clock.pause(Duration::from_millis(20));
    }
    crate::poll::poll_until(
        clock.as_ref(),
        clock.monotonic() + Duration::from_secs(8),
        "actual completed finalization attempt",
        || diagnostics_owner.snapshot().finalization_complete,
    )
    .expect("actual completed media finalization attempt");
    let query = nonce();
    assert_ne!(
        query, run,
        "query nonce must be newly undisclosed until post-return expiry"
    );
    record.publish(
        ".n2-query",
        b"N2QRY001",
        &[run[0], run[1], query[0], query[1]],
    );
    let query_publication = ns(clock.monotonic());
    let witness: [u64; 25] = record.wait(".n2-witness", b"N2WIT002", Duration::from_secs(8));
    let witness_collection = ns(clock.monotonic());
    assert_eq!(&witness[..13], &entered.entry);
    assert_eq!(&witness[13..15], &query);
    let collected = diagnostics(&diagnostics_owner.snapshot());
    let receipt = Receipt {
        actual,
        mapping: actual.exit().code(),
        lease,
        times: [
            runtime_start,
            runtime_return,
            entered.observed_ns,
            ns(deadline.requested_at().unwrap()),
            ns(end),
            shutdown_call,
            shutdown_return,
            query_publication,
            witness_collection,
            ns(clock.monotonic()),
        ],
        ack: [
            entered.ticket.session_id.into(),
            entered.ticket.session_valid.into(),
            entered.ticket.coalesced.into(),
        ],
        returned,
        collected,
        witness,
        model_failures: model_states
            .map(|state| state.failure().map_or(0, |exit| u64::from(exit.code()) + 1)),
    };
    record.publish(".n2-return", b"N2RET002", &receipt.fields());
    let parent: [u64; 6] = record.wait(".n2-parent", b"N2ACK001", Duration::from_secs(15));
    assert_eq!(
        parent,
        [run[0], run[1], query[0], query[1], lease[0], lease[1]]
    );
    assert_eq!(state.lease_identity(), lease);
    std::process::exit(i32::from(actual.exit().code()));
}
