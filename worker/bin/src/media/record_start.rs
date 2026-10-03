//! Allocate recording request IDs at the media-owner call boundary.

use seeon_deepstream_native::{MediaArgument, MediaBinding, MediaError, MediaOwner};

use super::RecordStartReply;

/// One sequence for the media owner's complete serve loop.
pub(super) struct Sequence {
    last: u64,
}

impl Sequence {
    pub(super) const fn new() -> Self {
        Self { last: 0 }
    }

    fn allocate(&mut self) -> Result<u64, MediaError> {
        let request_id = self
            .last
            .checked_add(1)
            .ok_or(MediaError::InvalidArgument(MediaArgument::RequestId))?;
        self.last = request_id;
        Ok(request_id)
    }
}

pub(super) fn start(
    owner: &mut MediaOwner,
    sequence: &mut Sequence,
    source_id: u32,
    binding: MediaBinding,
    lookback_seconds: u32,
    forward_seconds: u32,
) -> RecordStartReply {
    with_request_id(sequence, |request_id| {
        owner.record_start(
            source_id,
            binding,
            request_id,
            lookback_seconds,
            forward_seconds,
        )
    })
}

fn with_request_id<T>(
    sequence: &mut Sequence,
    native_start: impl FnOnce(u64) -> Result<T, MediaError>,
) -> Result<T, MediaError> {
    let request_id = sequence.allocate()?;
    native_start(request_id)
}

#[cfg(test)]
mod tests {
    use seeon_deepstream_native::{MediaArgument, MediaError};

    use super::{Sequence, with_request_id};

    /// This is only a model of the local call boundary, not SDK acceptance.
    #[test]
    fn one_sequence_follows_interleaved_camera_attempts() {
        let mut sequence = Sequence::new();
        let mut calls = Vec::new();

        for source_id in [4, 9, 4, 2] {
            let request_id = with_request_id(&mut sequence, |request_id| {
                calls.push((source_id, request_id));
                Ok(request_id)
            })
            .expect("request ID allocated");
            assert_eq!(calls.last().map(|call| call.1), Some(request_id));
        }

        assert_eq!(calls, vec![(4, 1), (9, 2), (4, 3), (2, 4)]);
    }

    /// A refused native attempt consumes its ID; later calls never reuse it.
    #[test]
    fn refusal_leaves_a_gap_instead_of_reusing_the_request_id() {
        let mut sequence = Sequence::new();
        let mut calls = Vec::new();
        let refused = with_request_id::<()>(&mut sequence, |request_id| {
            calls.push(request_id);
            Err(MediaError::NativeContract)
        });
        assert_eq!(refused, Err(MediaError::NativeContract));

        let next = with_request_id(&mut sequence, |request_id| {
            calls.push(request_id);
            Ok(request_id)
        })
        .expect("next request ID allocated");

        assert_eq!(calls, vec![1, 2]);
        assert_eq!(next, 2);
    }

    /// Exhaustion is local: no native call is made and the sequence stays spent.
    #[test]
    fn exhaustion_does_not_call_native_or_wrap() {
        let mut sequence = Sequence { last: u64::MAX - 1 };
        let final_id = with_request_id(&mut sequence, Ok).expect("maximum ID is usable once");
        assert_eq!(final_id, u64::MAX);

        let mut native_called = false;
        let exhausted = with_request_id(&mut sequence, |request_id| {
            native_called = true;
            Ok(request_id)
        });

        assert_eq!(
            exhausted,
            Err(MediaError::InvalidArgument(MediaArgument::RequestId))
        );
        assert!(!native_called);
        assert_eq!(sequence.last, u64::MAX);
    }
}
