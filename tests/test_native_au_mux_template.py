def test_repeated_parameter_sets_do_not_change_the_configuration_signature() -> None:
    """Regression for the rebuild storm (#424).

    Cameras retransmit VPS/SPS/PPS periodically and an access unit sometimes
    carries the same set twice, so the blob alternates between two byte strings
    describing the identical configuration. Measured on the live fleet as
    codec_data_len flipping 210 to 105 -- exactly two copies versus one -- with
    every other signature input unchanged. Hashing the raw bytes turned that
    into a configuration change and requested a source rebuild 241 times in
    four minutes.
    """
    from fractions import Fraction

    from worker.adapters.decode.native_au_mux_template import (
        native_configuration_signature,
    )

    once = b"\x00\x00\x01VPS\x00\x00\x01SPS\x00\x00\x01PPS"
    twice = once + once
    args = ("caps", 640, 360, Fraction(1, 90_000))
    assert native_configuration_signature(1, 0, args[0], once, *args[1:]) == (
        native_configuration_signature(1, 0, args[0], twice, *args[1:])
    )


def test_a_real_parameter_change_still_changes_the_signature() -> None:
    """Deduplication must not blind the receiver to an actual reconfiguration."""
    from fractions import Fraction

    from worker.adapters.decode.native_au_mux_template import (
        native_configuration_signature,
    )

    baseline = b"\x00\x00\x01VPS\x00\x00\x01SPS\x00\x00\x01PPS"
    altered = b"\x00\x00\x01VPS\x00\x00\x01SPS-DIFFERENT\x00\x00\x01PPS"
    args = ("caps", 640, 360, Fraction(1, 90_000))
    assert native_configuration_signature(1, 0, args[0], baseline, *args[1:]) != (
        native_configuration_signature(1, 0, args[0], altered, *args[1:])
    )
    assert native_configuration_signature(1, 0, args[0], baseline, *args[1:]) != (
        native_configuration_signature(1, 0, args[0], baseline, 1280, 720, args[3])
    )


def test_four_byte_start_codes_deduplicate_too() -> None:
    """The four-byte start code's leading zero lands on the previous unit.

    Splitting on the three-byte pattern leaves ``PPS\\x00`` where a four-byte
    code follows, so identical parameter sets compared unequal and 49 of these
    gaps kept firing on cameras that emit four-byte codes even after the first
    deduplication landed.
    """
    from fractions import Fraction

    from worker.adapters.decode.native_au_mux_template import (
        native_configuration_signature,
    )

    args = ("caps", 640, 360, Fraction(1, 90_000))
    for once in (
        b"\x00\x00\x00\x01VPS\x00\x00\x00\x01SPS\x00\x00\x00\x01PPS",
        b"\x00\x00\x01VPS\x00\x00\x01SPS\x00\x00\x01PPS",
        b"\x00\x00\x00\x01VPS\x00\x00\x01SPS\x00\x00\x00\x01PPS",
    ):
        assert native_configuration_signature(1, 0, args[0], once, *args[1:]) == (
            native_configuration_signature(1, 0, args[0], once + once, *args[1:])
        ), f"failed to deduplicate {once!r}"
