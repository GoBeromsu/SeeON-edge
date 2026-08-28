#include "au_transport.hpp"

#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <utility>
#include <vector>

namespace {
void check(bool condition, const std::string& message) {
  if (!condition) { std::cerr << message << '\n'; std::abort(); }
}
std::vector<std::uint8_t> receive_exact(int descriptor, std::size_t size) {
  std::vector<std::uint8_t> bytes(size);
  std::size_t offset = 0;
  while (offset < size) {
    const auto count = recv(descriptor, bytes.data() + offset, size - offset, 0);
    check(count > 0, "AU sender stream ended early");
    offset += static_cast<std::size_t>(count);
  }
  return bytes;
}
seeon::AuEnvelope envelope(std::uint64_t sequence, std::size_t size) {
  return {"camera-a", 1, 1, sequence,
          {seeon::AuCodec::kH264, seeon::AuFraming::kAnnexB, 1, 1, 1, 1, 90000,
           640, 360, sequence == 1, "caps", {}, std::vector<std::uint8_t>(size, 0x41)}};
}
constexpr std::size_t kAuHeaderBytes = 84;

// Packed header: magic[4], body_size u32, kind u8 at 8, codec, framing,
// keyframe, generation u32, epoch u64 at offset 16.
std::size_t body_size_of(const std::vector<std::uint8_t>& header) {
  return static_cast<std::size_t>(header[4]) |
         static_cast<std::size_t>(header[5]) << 8U |
         static_cast<std::size_t>(header[6]) << 16U |
         static_cast<std::size_t>(header[7]) << 24U;
}

std::uint64_t decode_epoch(const std::vector<std::uint8_t>& header) {
  std::uint64_t epoch = 0;
  for (std::size_t index = 0; index < 8; ++index) {
    epoch |= static_cast<std::uint64_t>(header[16 + index]) << (8U * index);
  }
  return epoch;
}

std::uint64_t decode_sequence(const std::vector<std::uint8_t>& header) {
  std::uint64_t sequence = 0;
  for (std::size_t index = 0; index < 8; ++index) {
    sequence |= static_cast<std::uint64_t>(header[24 + index]) << (8U * index);
  }
  return sequence;
}

seeon::AuEnvelope envelope_for(std::uint64_t epoch, std::uint64_t sequence,
                                std::size_t size) {
  return {"camera-a", 1, epoch, sequence,
          {seeon::AuCodec::kH264, seeon::AuFraming::kAnnexB, 1, 1, 1, 1, 90000,
           640, 360, sequence == 1, "caps", {}, std::vector<std::uint8_t>(size, 0x41)}};
}
}  // namespace

int main() {
  int descriptors[2]{};
  check(socketpair(AF_UNIX, SOCK_STREAM, 0, descriptors) == 0, "socketpair failed");
  int send_buffer = 4096;
  setsockopt(descriptors[0], SOL_SOCKET, SO_SNDBUF, &send_buffer, sizeof(send_buffer));
  seeon::AuSender sender(descriptors[0], 2, seeon::kMaxAuFrameBytes);
  std::uint64_t sequence = 1;
  while (sequence < 100 && sender.enqueue(envelope(sequence, 1024 * 1024))) ++sequence;
  check(sequence < 100, "bounded sender never declared overflow");
  for (std::uint64_t expected = 1; expected <= sequence; ++expected) {
    const auto header = receive_exact(descriptors[1], 84);
    const auto body_size = static_cast<std::size_t>(header[4]) |
                           static_cast<std::size_t>(header[5]) << 8U |
                           static_cast<std::size_t>(header[6]) << 16U |
                           static_cast<std::size_t>(header[7]) << 24U;
    check(header[8] == (expected == sequence ? 2 : 1), "gap cut ahead of accepted AUs");
    check(receive_exact(descriptors[1], body_size).size() == body_size, "AU body truncated");
  }
  check(!sender.enqueue(envelope(999, seeon::kMaxAuFrameBytes + 1)),
        "oversized AU crossed the 32 MiB boundary");
  sender.stop();
  close(descriptors[0]);
  close(descriptors[1]);

  {
    int pair[2]{};
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "socketpair failed");
    seeon::AuSender clean(pair[0], 4, seeon::kMaxAuFrameBytes);
    check(clean.enqueue(envelope_for(2, 1, 64)),
          "control: a 64-byte epoch-transition envelope must be admitted by a "
          "sender whose gap slot is free");
    clean.stop();
    close(pair[1]);
  }

  // A new stream epoch begins at sequence 1 because PipelineBinding is
  // recreated on a source rebuild. The receiver treats the first unit of an
  // epoch as the carrier of the transition, and rejects the epoch as
  // discontinuous if it never arrives.
  //
  // But the gap slot holds exactly one envelope, and while it is occupied
  // EVERY enqueue is refused - including that first unit. A rebuild burst
  // fills the queue, the overflow claims the single gap slot, and the very
  // next thing produced is epoch N+1 sequence 1, which is discarded. The
  // receiver then sees the epoch beginning at sequence 2 and reports a
  // discontinuity, which requests another rebuild, which fills the queue
  // again.
  //
  // Observed on the live 13-camera stack: rings frozen at active_epoch=1
  // while triggers reached 5, and clip selection failing with 'trigger
  // stream epoch is no longer active'. This test pins the mechanism without
  // asserting a remedy, because the remedy is a contract decision about how
  // an epoch transition survives backpressure.
  {
    int pair[2]{};
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "socketpair failed");
    int small = 4096;
    setsockopt(pair[0], SOL_SOCKET, SO_SNDBUF, &small, sizeof(small));
    seeon::AuSender gapped(pair[0], 1, seeon::kMaxAuFrameBytes);

    // Fill the queue and then overflow it, so the gap slot is claimed.
    std::uint64_t filled = 0;
    for (std::uint64_t index = 1; index <= 64; ++index) {
      if (!gapped.enqueue(envelope_for(1, index, 8192))) {
        filled = index;
        break;
      }
    }
    check(filled != 0, "sender never overflowed, cannot exercise the gap slot");
    const auto dropped_after_overflow = gapped.dropped();
    check(dropped_after_overflow > 0, "overflow did not record a drop");

    // The source rebuilds: epoch 2 starts its sequence at 1. This envelope is
    // 64 bytes, far below every size limit, so its rejection can only be the
    // occupied gap slot -- a cumulative dropped() count alone would not
    // establish that, which is why the size is chosen to rule out the other
    // refusal conditions.
    // Still refused - backpressure is real and the unit is not queued.
    const bool transition_admitted = gapped.enqueue(envelope_for(2, 1, 64));
    check(!transition_admitted, "an overflowing sender must still refuse");
    check(gapped.dropped() == dropped_after_overflow + 1,
          "the epoch transition was refused without being counted as dropped");

    // The transition is RESERVED, not destroyed: refused for admission to the
    // queue, but held per camera and drained ahead of the gap as an ordinary
    // unit so the receiver can adopt the epoch.
    timeval timeout{};
    timeout.tv_sec = 2;
    setsockopt(pair[1], SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    bool saw_transition_as_normal_unit = false;
    bool saw_gap_before_transition = false;
    for (int index = 0; index < 512 && !saw_transition_as_normal_unit; ++index) {
      std::vector<std::uint8_t> header(kAuHeaderBytes);
      std::size_t offset = 0;
      bool complete = true;
      while (offset < kAuHeaderBytes) {
        const auto count =
            recv(pair[1], header.data() + offset, kAuHeaderBytes - offset, 0);
        if (count <= 0) { complete = false; break; }
        offset += static_cast<std::size_t>(count);
      }
      if (!complete) break;
      const auto body = body_size_of(header);
      std::vector<std::uint8_t> discard(body);
      std::size_t body_offset = 0;
      while (body_offset < body) {
        const auto count =
            recv(pair[1], discard.data() + body_offset, body - body_offset, 0);
        if (count <= 0) break;
        body_offset += static_cast<std::size_t>(count);
      }
      const bool is_gap = header[8] == 2;
      if (is_gap) saw_gap_before_transition = true;
      if (!is_gap && decode_epoch(header) == 2) saw_transition_as_normal_unit = true;
    }
    gapped.stop();
    check(saw_transition_as_normal_unit,
          "the unit opening epoch 2 must reach the wire as an ORDINARY access "
          "unit; a gap marker is discarded by NativeAuReceiver before the "
          "epoch is adopted, so delivering it as a gap fixes nothing (#429)");
    check(!saw_gap_before_transition,
          "the reserved transition must drain AHEAD of the gap marker, "
          "otherwise the receiver rebuilds before it can adopt the epoch");
  }

  // Cross-camera isolation: this sender is shared by every camera, so one
  // camera's transition must never suppress another's, and BOTH must reach the
  // wire. Asserting only that dropped() advanced would pass even if camera B
  // were silently suppressed.
  {
    int pair[2]{};
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "socketpair failed");
    int small = 4096;
    setsockopt(pair[0], SOL_SOCKET, SO_SNDBUF, &small, sizeof(small));
    seeon::AuSender shared(pair[0], 1, seeon::kMaxAuFrameBytes);
    for (std::uint64_t index = 1; index <= 64; ++index) {
      if (!shared.enqueue(envelope_for(1, index, 8192))) break;
    }
    auto high = envelope_for(7, 1, 64);
    high.camera = "camera-a";
    auto low = envelope_for(2, 1, 64);
    low.camera = "camera-b";
    // Both must be REFUSED, i.e. reserved. If either slipped into the ordinary
    // queue the test would pass without exercising reservation at all.
    check(!shared.enqueue(std::move(high)), "camera-a transition must be reserved");
    check(!shared.enqueue(std::move(low)), "camera-b transition must be reserved");

    timeval timeout{};
    timeout.tv_sec = 2;
    setsockopt(pair[1], SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    bool saw_a = false;
    bool saw_b = false;
    for (int index = 0; index < 512 && !(saw_a && saw_b); ++index) {
      std::vector<std::uint8_t> header(kAuHeaderBytes);
      std::size_t offset = 0;
      bool complete = true;
      while (offset < kAuHeaderBytes) {
        const auto count =
            recv(pair[1], header.data() + offset, kAuHeaderBytes - offset, 0);
        if (count <= 0) { complete = false; break; }
        offset += static_cast<std::size_t>(count);
      }
      if (!complete) break;
      const auto body = body_size_of(header);
      std::vector<std::uint8_t> payload(body);
      std::size_t body_offset = 0;
      while (body_offset < body) {
        const auto count =
            recv(pair[1], payload.data() + body_offset, body - body_offset, 0);
        if (count <= 0) break;
        body_offset += static_cast<std::size_t>(count);
      }
      if (header[8] == 2) continue;  // gap marker
      const std::string camera(payload.begin(),
                               payload.begin() + std::min<std::size_t>(8, payload.size()));
      if (decode_epoch(header) == 7 && camera.rfind("camera-a", 0) == 0) saw_a = true;
      if (decode_epoch(header) == 2 && camera.rfind("camera-b", 0) == 0) saw_b = true;
    }
    shared.stop();
    close(pair[1]);
    check(saw_a, "camera-a's transition must reach the wire");
    check(saw_b,
          "camera-b's transition must reach the wire too: a shared sender must "
          "not let one camera's higher epoch suppress another's");
  }

  // Generation reset: a higher generation restarts the epoch at 1, so it must
  // REPLACE a reservation holding an older generation's higher epoch.
  {
    int pair[2]{};
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "socketpair failed");
    int small = 4096;
    setsockopt(pair[0], SOL_SOCKET, SO_SNDBUF, &small, sizeof(small));
    seeon::AuSender rolling(pair[0], 1, seeon::kMaxAuFrameBytes);
    for (std::uint64_t index = 1; index <= 64; ++index) {
      if (!rolling.enqueue(envelope_for(1, index, 8192))) break;
    }
    auto stale = envelope_for(7, 1, 64);
    stale.generation = 3;
    auto fresh = envelope_for(1, 1, 64);
    fresh.generation = 4;  // higher generation, epoch restarted at 1
    check(!rolling.enqueue(std::move(stale)), "stale transition must be reserved");
    check(!rolling.enqueue(std::move(fresh)), "fresh transition must be reserved");

    timeval timeout{};
    timeout.tv_sec = 2;
    setsockopt(pair[1], SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    bool saw_fresh_generation = false;
    for (int index = 0; index < 512 && !saw_fresh_generation; ++index) {
      std::vector<std::uint8_t> header(kAuHeaderBytes);
      std::size_t offset = 0;
      bool complete = true;
      while (offset < kAuHeaderBytes) {
        const auto count =
            recv(pair[1], header.data() + offset, kAuHeaderBytes - offset, 0);
        if (count <= 0) { complete = false; break; }
        offset += static_cast<std::size_t>(count);
      }
      if (!complete) break;
      const auto body = body_size_of(header);
      std::vector<std::uint8_t> discard(body);
      std::size_t body_offset = 0;
      while (body_offset < body) {
        const auto count =
            recv(pair[1], discard.data() + body_offset, body - body_offset, 0);
        if (count <= 0) break;
        body_offset += static_cast<std::size_t>(count);
      }
      const std::uint32_t generation = static_cast<std::uint32_t>(header[12]) |
                                       static_cast<std::uint32_t>(header[13]) << 8U |
                                       static_cast<std::uint32_t>(header[14]) << 16U |
                                       static_cast<std::uint32_t>(header[15]) << 24U;
      if (header[8] != 2 && generation == 4) saw_fresh_generation = true;
    }
    rolling.stop();
    close(pair[1]);
    check(saw_fresh_generation,
          "a higher generation restarts the epoch at 1 and must replace a "
          "reservation holding an older generation's higher epoch");
  }

  // An envelope refused for a VALIDITY limit must never be reserved: a
  // reservation is later emitted as an ordinary unit and would otherwise
  // bypass the limit that rejected it.
  {
    int pair[2]{};
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "socketpair failed");
    seeon::AuSender guard(pair[0], 4, seeon::kMaxAuFrameBytes);
    check(!guard.enqueue(envelope_for(9, 1, seeon::kMaxAuFrameBytes + 1)),
          "an oversized envelope must be refused");

    // Deterministic, not timed: the refused envelope becomes the gap marker,
    // so read the frame that IS expected and require it to be a GAP. If the
    // validity guard were removed the envelope would be reserved and replayed
    // as an ordinary ACCESS_UNIT, and this assertion fails on kind rather than
    // on a timeout that could just be scheduling.
    timeval timeout{};
    timeout.tv_sec = 3;
    setsockopt(pair[1], SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    std::vector<std::uint8_t> header(kAuHeaderBytes);
    std::size_t got = 0;
    while (got < kAuHeaderBytes) {
      const auto count = recv(pair[1], header.data() + got, kAuHeaderBytes - got, 0);
      check(count > 0, "no frame arrived for the refused oversized envelope");
      got += static_cast<std::size_t>(count);
    }
    const auto body = body_size_of(header);
    std::vector<std::uint8_t> discard(body);
    std::size_t body_offset = 0;
    while (body_offset < body) {
      const auto count =
          recv(pair[1], discard.data() + body_offset, body - body_offset, 0);
      check(count > 0, "gap body truncated");
      body_offset += static_cast<std::size_t>(count);
    }
    guard.stop();
    close(pair[1]);
    check(header[8] == 2,
          "an oversized sequence-1 envelope must become a GAP, never a reserved "
          "unit replayed as an ordinary one - that would bypass "
          "kMaxAuFrameBytes");
  }


  // Ordering, proven on the wire through the public surface.
  //
  // run() drains a reserved transition ahead of queue_, re-checking before
  // every send, so a unit admitted after a reservation can never reach the
  // wire ahead of the epoch-opening unit. The construction: pin the sender
  // on one unit, reserve the transition while it is blocked, then admit a
  // later unit of the same camera and read the wire order.
  {
    int pair[2]{};
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "socketpair failed");
    int tiny = 4096;
    setsockopt(pair[0], SOL_SOCKET, SO_SNDBUF, &tiny, sizeof(tiny));
    setsockopt(pair[1], SOL_SOCKET, SO_RCVBUF, &tiny, sizeof(tiny));
    const std::size_t budget = 8 * 1024 * 1024;
    seeon::AuSender ordered(pair[0], 8, budget);
    timeval timeout{};
    timeout.tv_sec = 3;
    setsockopt(pair[1], SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    const auto read_header = [&]() -> std::vector<std::uint8_t> {
      std::vector<std::uint8_t> header(kAuHeaderBytes);
      std::size_t offset = 0;
      while (offset < kAuHeaderBytes) {
        const auto count =
            recv(pair[1], header.data() + offset, kAuHeaderBytes - offset, 0);
        check(count > 0, "sender stopped writing unexpectedly");
        offset += static_cast<std::size_t>(count);
      }
      return header;
    };
    const auto read_body = [&](std::size_t size) {
      std::vector<std::uint8_t> discard(size);
      std::size_t offset = 0;
      while (offset < size) {
        const auto count = recv(pair[1], discard.data() + offset, size - offset, 0);
        check(count > 0, "sender stopped writing a body unexpectedly");
        offset += static_cast<std::size_t>(count);
      }
    };

    // 1. Pin the sender on a unit the reader has not consumed.
    auto pinned = envelope_for(1, 1, 400 * 1024);
    check(ordered.enqueue(std::move(pinned)), "pinning unit must be admitted");
    const auto pinned_header = read_header();
    const auto pinned_body = body_size_of(pinned_header);

    // 2. Ballast consumes nearly the whole budget, so the transition cannot be
    //    queued and is reserved instead.
    check(ordered.enqueue(envelope_for(1, 2, 7800 * 1024)), "ballast must be admitted");
    check(!ordered.enqueue(envelope_for(2, 1, 512 * 1024)),
          "the epoch-opening unit must be refused and reserved");

    // 3. A later unit of the same camera is admitted (64 bytes fit the
    //    remaining budget) while the reservation is still pending.
    check(ordered.enqueue(envelope_for(2, 2, 64)),
          "a later unit must be admitted; nothing but congestion refuses it");

    // 4. On the wire: pinned first (it was mid-send), then the reserved
    //    transition ahead of everything queued behind it.
    read_body(pinned_body);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> order;
    for (int index = 0; index < 3; ++index) {
      const auto header = read_header();
      order.emplace_back(decode_epoch(header), decode_sequence(header));
      read_body(body_size_of(header));
    }
    ordered.stop();
    check(order[0] == std::make_pair(std::uint64_t{2}, std::uint64_t{1}),
          "the reserved epoch-opening unit must be the next thing on the wire, "
          "ahead of the queue (#429)");
    check(order[1] == std::make_pair(std::uint64_t{1}, std::uint64_t{2}) &&
              order[2] == std::make_pair(std::uint64_t{2}, std::uint64_t{2}),
          "queued units must follow the transition in admission order");
    close(pair[1]);
  }

  // The reservation allowance is a real cap, and a replacement credits the
  // reservation it displaces. Both are asserted on the wire: a reserved
  // transition eventually arrives as an ordinary unit, one that was refused
  // for the allowance never does.
  {
    int pair[2]{};
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "socketpair failed");
    int tiny = 4096;
    setsockopt(pair[0], SOL_SOCKET, SO_SNDBUF, &tiny, sizeof(tiny));
    setsockopt(pair[1], SOL_SOCKET, SO_RCVBUF, &tiny, sizeof(tiny));
    timeval timeout{};
    timeout.tv_sec = 3;
    setsockopt(pair[1], SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    const std::size_t budget = 8 * 1024 * 1024;  // allowance = 1 MiB
    seeon::AuSender capped(pair[0], 8, budget);

    auto pinned = envelope_for(1, 1, 400 * 1024);
    pinned.camera = "camera-p";
    check(capped.enqueue(std::move(pinned)), "pinning unit admitted");
    std::vector<std::uint8_t> head(kAuHeaderBytes);
    std::size_t got = 0;
    while (got < kAuHeaderBytes) {
      const auto count = recv(pair[1], head.data() + got, kAuHeaderBytes - got, 0);
      check(count > 0, "sender never began writing");
      got += static_cast<std::size_t>(count);
    }
    const auto pinned_body = body_size_of(head);
    auto ballast = envelope_for(1, 2, 7800 * 1024);
    ballast.camera = "camera-p";
    check(capped.enqueue(std::move(ballast)), "ballast admitted");

    // 900 KiB fits the 1 MiB allowance and is reserved.
    auto first = envelope_for(2, 1, 900 * 1024);
    first.camera = "camera-x";
    check(!capped.enqueue(std::move(first)), "first transition reserved");

    // A newer transition for the SAME camera must be accepted: crediting the
    // one it replaces keeps the projected total at 900 KiB, not 1.8 MiB.
    auto replacement = envelope_for(3, 1, 900 * 1024);
    replacement.camera = "camera-x";
    check(!capped.enqueue(std::move(replacement)), "replacement reserved");

    // A DIFFERENT camera's transition of the same size would take the total to
    // 1.8 MiB, beyond the allowance, so it must not be reserved.
    // Named to sort BEFORE camera-x: transitions_ is a std::map keyed by
    // camera, so if this one were reserved it would drain first. Seeing
    // camera-x without having seen it is therefore proof it was never
    // reserved, and needs no timeout.
    auto over = envelope_for(2, 1, 900 * 1024);
    over.camera = "camera-a";
    check(!capped.enqueue(std::move(over)), "over-allowance transition refused");

    // Drain and observe: camera-x epoch 3 must arrive as an ordinary unit,
    // camera-x epoch 2 must not (it was replaced), and camera-z must not (it
    // exceeded the allowance and became a gap at best).
    std::vector<std::uint8_t> discard(pinned_body);
    std::size_t off = 0;
    while (off < pinned_body) {
      const auto count = recv(pair[1], discard.data() + off, pinned_body - off, 0);
      check(count > 0, "pinned body truncated");
      off += static_cast<std::size_t>(count);
    }
    // The fence is transitions_' own ordering: camera-a would drain before
    // camera-x. Reading until camera-x arrives therefore settles both
    // negatives without relying on a timeout.
    bool saw_replacement = false;
    bool saw_replaced = false;
    bool saw_over = false;
    for (int index = 0; index < 256 && !saw_replacement; ++index) {
      std::vector<std::uint8_t> header(kAuHeaderBytes);
      std::size_t offset = 0;
      bool complete = true;
      while (offset < kAuHeaderBytes) {
        const auto count =
            recv(pair[1], header.data() + offset, kAuHeaderBytes - offset, 0);
        if (count <= 0) { complete = false; break; }
        offset += static_cast<std::size_t>(count);
      }
      if (!complete) break;
      const auto body = body_size_of(header);
      std::vector<std::uint8_t> payload(body);
      std::size_t body_offset = 0;
      while (body_offset < body) {
        const auto count =
            recv(pair[1], payload.data() + body_offset, body - body_offset, 0);
        if (count <= 0) break;
        body_offset += static_cast<std::size_t>(count);
      }
      if (header[8] == 2 || payload.size() < 8) continue;
      const std::string camera(payload.begin(), payload.begin() + 8);
      const auto epoch = decode_epoch(header);
      if (camera == "camera-x" && epoch == 3) saw_replacement = true;
      if (camera == "camera-x" && epoch == 2) saw_replaced = true;
      if (camera == "camera-a") saw_over = true;
    }
    capped.stop();
    close(pair[1]);
    check(saw_replacement,
          "the replacing transition must reach the wire; crediting the "
          "reservation it displaces is what keeps it inside the allowance. "
          "Its arrival is also the fence for the two negatives below");
    check(!saw_replaced,
          "the replaced transition must not also reach the wire");
    check(!saw_over,
          "a transition beyond the reservation allowance must not be reserved "
          "and replayed as an ordinary unit");
  }

  // An envelope whose identity fields overflow the wire's uint16 lengths can
  // be encoded neither as an ordinary unit nor as a gap marker, so it must
  // produce nothing at all rather than a silently truncated frame.
  {
    int pair[2]{};
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "socketpair failed");
    seeon::AuSender guard(pair[0], 4, seeon::kMaxAuFrameBytes);
    auto wide = envelope_for(4, 1, 64);
    wide.camera = std::string(static_cast<std::size_t>(UINT16_MAX) + 1, 'c');
    check(!guard.enqueue(std::move(wide)),
          "an over-width camera identity must be refused");
    guard.stop();
    close(pair[1]);
  }

  // Over-width identity must be refused AND must not consume the reservation
  // allowance.
  //
  // The absence of a reservation cannot be observed directly: stop() shuts the
  // write side instead of draining, reservations drain after the whole queue so
  // no sentinel can bracket them, and dropped() increments on both paths. So
  // this asserts a POSITIVE consequence instead. The allowance is aggregate, so
  // a wrongly reserved over-width envelope eats part of it; sizing a legitimate
  // transition to fit the allowance alone but not alongside that envelope turns
  // "was it reserved?" into "did the legitimate one reach the wire?", which is
  // bounded and deterministic.
  {
    int pair[2]{};
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "socketpair failed");
    int tiny = 4096;
    setsockopt(pair[0], SOL_SOCKET, SO_SNDBUF, &tiny, sizeof(tiny));
    setsockopt(pair[1], SOL_SOCKET, SO_RCVBUF, &tiny, sizeof(tiny));
    timeval timeout{};
    timeout.tv_sec = 3;
    setsockopt(pair[1], SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    const std::size_t budget = 8 * 1024 * 1024;  // allowance = 1 MiB
    seeon::AuSender width(pair[0], 8, budget);

    // Congest, so transitions are reserved rather than queued.
    auto pinned = envelope_for(1, 1, 400 * 1024);
    pinned.camera = "camera-w";
    check(width.enqueue(std::move(pinned)), "pinning unit admitted");
    std::vector<std::uint8_t> head(kAuHeaderBytes);
    std::size_t got = 0;
    while (got < kAuHeaderBytes) {
      const auto count = recv(pair[1], head.data() + got, kAuHeaderBytes - got, 0);
      check(count > 0, "sender never began writing");
      got += static_cast<std::size_t>(count);
    }
    const auto pinned_body = body_size_of(head);
    auto ballast = envelope_for(1, 2, 7800 * 1024);
    ballast.camera = "camera-w";
    check(width.enqueue(std::move(ballast)), "ballast admitted");

    // ~64 KiB of over-width identity. It must be refused and must NOT be
    // reserved; if it were, it would consume that much of the 1 MiB allowance.
    auto wide = envelope_for(4, 1, 64);
    wide.camera = std::string(static_cast<std::size_t>(UINT16_MAX) + 1, 'c');
    const auto before = width.dropped();
    check(!width.enqueue(std::move(wide)),
          "an over-width camera identity must be refused");
    check(width.dropped() == before + 1, "the refusal must be counted once");

    // 1000 KiB fits the allowance alone, but not alongside a wrongly reserved
    // 64 KiB over-width envelope.
    auto legitimate = envelope_for(2, 1, 1000 * 1024);
    legitimate.camera = "camera-v";
    check(!width.enqueue(std::move(legitimate)), "legitimate transition reserved");

    std::vector<std::uint8_t> discard(pinned_body);
    std::size_t off = 0;
    while (off < pinned_body) {
      const auto count = recv(pair[1], discard.data() + off, pinned_body - off, 0);
      check(count > 0, "pinned body truncated");
      off += static_cast<std::size_t>(count);
    }

    bool saw_legitimate = false;
    for (int index = 0; index < 64 && !saw_legitimate; ++index) {
      std::vector<std::uint8_t> header(kAuHeaderBytes);
      std::size_t offset = 0;
      bool complete = true;
      while (offset < kAuHeaderBytes) {
        const auto count =
            recv(pair[1], header.data() + offset, kAuHeaderBytes - offset, 0);
        if (count <= 0) { complete = false; break; }
        offset += static_cast<std::size_t>(count);
      }
      if (!complete) break;
      const auto body = body_size_of(header);
      std::vector<std::uint8_t> payload(body);
      std::size_t body_offset = 0;
      while (body_offset < body) {
        const auto count =
            recv(pair[1], payload.data() + body_offset, body - body_offset, 0);
        if (count <= 0) break;
        body_offset += static_cast<std::size_t>(count);
      }
      if (header[8] != 2 && payload.size() >= 8 &&
          std::string(payload.begin(), payload.begin() + 8) == "camera-v") {
        saw_legitimate = true;
      }
    }
    width.stop();
    close(pair[1]);
    check(saw_legitimate,
          "the legitimate transition never reached the wire, which means the "
          "over-width envelope consumed the reservation allowance: an invalid "
          "envelope must be refused outright, never reserved");
  }

  // Draining a reservation must CREDIT its bytes. Without that the allowance
  // is consumed permanently and a camera can never reserve again, so every
  // subsequent epoch roll loses its transition - the original defect, returned
  // by a slow leak instead of an instant one.
  {
    int pair[2]{};
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "socketpair failed");
    int tiny = 4096;
    setsockopt(pair[0], SOL_SOCKET, SO_SNDBUF, &tiny, sizeof(tiny));
    setsockopt(pair[1], SOL_SOCKET, SO_RCVBUF, &tiny, sizeof(tiny));
    timeval timeout{};
    timeout.tv_sec = 3;
    setsockopt(pair[1], SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    const std::size_t budget = 8 * 1024 * 1024;  // allowance = 1 MiB
    seeon::AuSender cycling(pair[0], 8, budget);

    const auto drain_until = [&](const std::string& camera, std::uint64_t epoch) {
      for (int index = 0; index < 256; ++index) {
        std::vector<std::uint8_t> header(kAuHeaderBytes);
        std::size_t offset = 0;
        while (offset < kAuHeaderBytes) {
          const auto count =
              recv(pair[1], header.data() + offset, kAuHeaderBytes - offset, 0);
          check(count > 0, "wire ended before the expected frame");
          offset += static_cast<std::size_t>(count);
        }
        const auto body = body_size_of(header);
        std::vector<std::uint8_t> payload(body);
        std::size_t body_offset = 0;
        while (body_offset < body) {
          const auto count =
              recv(pair[1], payload.data() + body_offset, body - body_offset, 0);
          check(count > 0, "body truncated");
          body_offset += static_cast<std::size_t>(count);
        }
        if (header[8] == 2 || payload.size() < camera.size()) continue;
        if (std::string(payload.begin(),
                        payload.begin() + static_cast<long>(camera.size())) == camera &&
            decode_epoch(header) == epoch) {
          return;
        }
      }
      check(false, "the expected frame never arrived");
    };

    // Round one: congest, reserve 900 KiB, then let everything drain.
    auto pinned = envelope_for(1, 1, 400 * 1024);
    pinned.camera = "camera-r";
    check(cycling.enqueue(std::move(pinned)), "pinning unit admitted");
    std::vector<std::uint8_t> head(kAuHeaderBytes);
    std::size_t got = 0;
    while (got < kAuHeaderBytes) {
      const auto count = recv(pair[1], head.data() + got, kAuHeaderBytes - got, 0);
      check(count > 0, "sender never began writing");
      got += static_cast<std::size_t>(count);
    }
    const auto pinned_body = body_size_of(head);
    auto ballast = envelope_for(1, 2, 7800 * 1024);
    ballast.camera = "camera-r";
    check(cycling.enqueue(std::move(ballast)), "ballast admitted");
    auto first = envelope_for(2, 1, 900 * 1024);
    first.camera = "camera-r";
    check(!cycling.enqueue(std::move(first)), "first transition reserved");
    std::vector<std::uint8_t> discard(pinned_body);
    std::size_t off = 0;
    while (off < pinned_body) {
      const auto count = recv(pair[1], discard.data() + off, pinned_body - off, 0);
      check(count > 0, "pinned body truncated");
      off += static_cast<std::size_t>(count);
    }
    drain_until("camera-r", 2);
    // The reservation drained ahead of the ballast; drain the ballast too so
    // round two starts from an empty queue.
    drain_until("camera-r", 1);

    // Round two: congest again and reserve for a DIFFERENT camera. This only
    // fits if round one's reservation was credited when it drained.
    auto pinned2 = envelope_for(1, 1, 400 * 1024);
    pinned2.camera = "camera-r";
    check(cycling.enqueue(std::move(pinned2)), "second pinning unit admitted");
    std::size_t got2 = 0;
    while (got2 < kAuHeaderBytes) {
      const auto count = recv(pair[1], head.data() + got2, kAuHeaderBytes - got2, 0);
      check(count > 0, "sender never began the second round");
      got2 += static_cast<std::size_t>(count);
    }
    const auto pinned2_body = body_size_of(head);
    auto ballast2 = envelope_for(1, 2, 7800 * 1024);
    ballast2.camera = "camera-r";
    check(cycling.enqueue(std::move(ballast2)), "second ballast admitted");
    auto second = envelope_for(3, 1, 900 * 1024);
    second.camera = "camera-q";
    check(!cycling.enqueue(std::move(second)), "second transition refused");
    std::vector<std::uint8_t> discard2(pinned2_body);
    std::size_t off2 = 0;
    while (off2 < pinned2_body) {
      const auto count = recv(pair[1], discard2.data() + off2, pinned2_body - off2, 0);
      check(count > 0, "second pinned body truncated");
      off2 += static_cast<std::size_t>(count);
    }
    drain_until("camera-q", 3);
    cycling.stop();
    close(pair[1]);
  }

  return 0;
}
