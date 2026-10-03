//! Resolve an asynchronous admission ticket only from the owner's actual
//! active-record status. A requested stop must never invent an SDK session.

use seeon_deepstream_native::{
    MediaArgument, MediaCallStatus, MediaError, MediaOwner, MediaResult, RecordTicket,
};

pub(super) fn stop(
    owner: &mut MediaOwner,
    admitted: &RecordTicket,
) -> Result<MediaCallStatus, MediaError> {
    if admitted.session_valid != 0 {
        return owner.record_stop(admitted);
    }
    let status = owner.read_status()?;
    if status.result != MediaResult::Ok {
        return Err(MediaError::Native(MediaCallStatus {
            result: status.result,
            fatal: (status.fatal.code != 0).then_some(status.fatal),
            warning: (status.warning.code != 0).then_some(status.warning),
            required_bytes: None,
        }));
    }
    let active = status
        .sources
        .iter()
        .map(|source| source.active_record)
        .find(|active| matches_request(admitted, active))
        .ok_or(MediaError::InvalidArgument(MediaArgument::RecordSession))?;
    // Pass the observed ticket unchanged; native revalidates it against the
    // current slot before emitting stop-sr, including races after read_status.
    owner.record_stop(&active)
}

fn matches_request(admitted: &RecordTicket, active: &RecordTicket) -> bool {
    admitted.request_id != 0
        && active.source_id == admitted.source_id
        && active.binding == admitted.binding
        && active.request_id == admitted.request_id
        && active.session_valid == 1
        && active.session_id != u32::MAX
}

#[cfg(test)]
mod tests {
    use super::*;
    use seeon_deepstream_native::MediaBinding;

    fn admitted() -> RecordTicket {
        RecordTicket {
            binding: MediaBinding {
                token: 73,
                generation: 7,
                epoch: 11,
            },
            request_id: 5,
            source_id: 2,
            session_id: 0,
            session_valid: 0,
            coalesced: 0,
        }
    }

    #[test]
    fn actual_session_assignment_may_differ_from_unknown_admission() {
        let request = admitted();
        let active = RecordTicket {
            session_id: 19,
            session_valid: 1,
            ..request
        };
        assert!(matches_request(&request, &active));
        assert_eq!(request.session_valid, 0);
        assert_eq!(active.session_id, 19);
    }

    #[test]
    fn reused_vendor_session_does_not_authorize_another_request() {
        let request = admitted();
        let active = RecordTicket {
            request_id: 6,
            session_valid: 1,
            ..request
        };
        assert!(!matches_request(&request, &active));
    }

    #[test]
    fn mismatched_source_binding_and_missing_session_are_refused() {
        let request = admitted();
        let active = RecordTicket {
            session_valid: 1,
            ..request
        };
        let cases = [
            RecordTicket {
                source_id: 3,
                ..active
            },
            RecordTicket {
                binding: MediaBinding {
                    token: 74,
                    ..active.binding
                },
                ..active
            },
            RecordTicket {
                binding: MediaBinding {
                    generation: 8,
                    ..active.binding
                },
                ..active
            },
            RecordTicket {
                binding: MediaBinding {
                    epoch: 12,
                    ..active.binding
                },
                ..active
            },
            RecordTicket {
                session_valid: 0,
                ..active
            },
            RecordTicket {
                session_valid: 2,
                ..active
            },
            RecordTicket {
                session_id: u32::MAX,
                ..active
            },
        ];
        for candidate in cases {
            assert!(!matches_request(&request, &candidate), "{candidate:?}");
        }
        assert!(!matches_request(
            &RecordTicket {
                request_id: 0,
                ..request
            },
            &active
        ));
    }
}
