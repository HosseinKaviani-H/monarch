# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Fault-Tolerant Counter Service — Monarch Demo
==============================================

Demonstrates Monarch's fault tolerance with the simplest possible
stateful actor: a counter.  The full lifecycle is exercised:

    1. Spawn a ``CounterSupervisor`` that manages a mesh of ``CounterActor`` instances
    2. Increment counters via supervisor endpoint RPCs
    3. Checkpoint all actor state through the supervisor
    4. Inject a fault — kill one actor's process with SIGKILL
    5. Supervisor detects the failure via ``__supervise__``
    6. Supervisor respawns actors on a **fresh** ProcMesh
    7. State is restored from the last checkpoint
    8. Counting resumes — no data lost, no job crash

Key Monarch concepts demonstrated:
    - ``Actor`` / ``@endpoint`` for defining stateful services
    - ``ProcMesh`` / ``ActorMesh`` for spawning actors across processes
    - **Supervision tree**: ``__supervise__`` for catching mesh failures
    - **Checkpoint / restore** via endpoint RPCs
    - **Mesh ownership**: the supervisor owns the counter mesh, so failures
      propagate to it automatically

Usage::

    python examples/fault_tolerance_demo.py
"""

import logging
import os
import signal
import time

from monarch.actor import Actor, endpoint, MeshFailure, this_host, this_proc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("fault_tolerance_demo")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_ACTORS = 4
INCREMENTS_BEFORE_CRASH = 10
INCREMENTS_AFTER_RECOVERY = 10
CRASH_RANK = 2  # which actor's process to kill


# ---------------------------------------------------------------------------
# CounterActor — the simplest stateful actor
# ---------------------------------------------------------------------------


class CounterActor(Actor):
    """A minimal stateful actor that counts requests.

    Each instance maintains an integer count that can be incremented,
    queried, checkpointed, and restored.
    """

    def __init__(self):
        self.count = 0

    @endpoint
    def increment(self, n: int = 1) -> int:
        """Add *n* to the counter and return the new value."""
        self.count += n
        return self.count

    @endpoint
    def get_count(self) -> int:
        """Return the current count."""
        return self.count

    @endpoint
    def checkpoint(self) -> dict:
        """Return a serializable snapshot of this actor's state."""
        return {"count": self.count}

    @endpoint
    def restore(self, state: dict) -> None:
        """Restore state from a previous checkpoint."""
        self.count = state["count"]

    @endpoint
    def get_pid(self) -> int:
        """Return the OS process ID hosting this actor."""
        return os.getpid()


# ---------------------------------------------------------------------------
# CounterSupervisor — owns and supervises the CounterActor mesh
# ---------------------------------------------------------------------------


class CounterSupervisor(Actor):
    """Manages a mesh of CounterActors with checkpoint-based fault recovery.

    **Ownership.**  This actor spawns a ProcMesh and an ActorMesh of counters
    on top of it.  Because Monarch actors *own* the meshes they create, any
    failure in the counter mesh is reported here via ``__supervise__``.

    **Recovery strategy:**

    1. ``__supervise__`` records the failure and marks it as *handled*
       (returns ``True``), preventing the failure from crashing the client.
    2. The orchestrating script calls ``recover()``.
    3. ``recover()`` spawns a *new* ProcMesh with fresh CounterActors and
       restores each actor's state from the last checkpoint.
    """

    def __init__(self):
        self.checkpoints: dict = {}
        self.failure_detected: bool = False
        self.recovery_count: int = 0
        self._spawn_counters()

    # -- internal helpers --------------------------------------------------

    def _spawn_counters(self):
        """Create a ProcMesh with NUM_ACTORS processes and spawn counters."""
        self.counter_procs = this_host().spawn_procs(per_host={"workers": NUM_ACTORS})
        self.counters = self.counter_procs.spawn("counters", CounterActor)

    # -- supervision -------------------------------------------------------

    def __supervise__(self, failure: MeshFailure):
        """Called by Monarch when an owned mesh experiences a failure.

        Returning ``True`` tells Monarch the event is handled, preventing it
        from propagating further up the supervision tree (which would crash
        the client by default).
        """
        log.warning(
            "[Supervisor] Failure detected (recovery #%d): %s",
            self.recovery_count + 1,
            failure,
        )
        self.failure_detected = True
        self.recovery_count += 1
        return True

    # -- public endpoints --------------------------------------------------

    @endpoint
    def increment_all(self, n: int = 1) -> list:
        """Broadcast an increment to every counter and collect results."""
        return self.counters.increment.call(n).get()

    @endpoint
    def get_all_counts(self) -> list:
        """Collect the current count from every counter."""
        return self.counters.get_count.call().get()

    @endpoint
    def checkpoint_all(self) -> dict:
        """Take a checkpoint of all counter states."""
        results = self.counters.checkpoint.call().get().values()
        for rank, snapshot in enumerate(results):
            self.checkpoints[rank] = snapshot
        log.info(
            "[Supervisor] Checkpointed %d actors: %s",
            len(results),
            self.checkpoints,
        )
        return dict(self.checkpoints)

    @endpoint
    def inject_fault(self, rank: int) -> int:
        """Kill the process hosting the counter at *rank* (SIGKILL).

        This simulates an unrecoverable hardware or OS-level failure —
        the process is gone and cannot respond to any further messages.
        """
        pid = self.counters.slice(workers=rank).get_pid.call_one().get()
        log.warning("[Supervisor] Killing PID %d at rank %d", pid, rank)
        os.kill(pid, signal.SIGKILL)
        return pid

    @endpoint
    def is_failure_detected(self) -> bool:
        """Check whether ``__supervise__`` has fired since the last recovery."""
        return self.failure_detected

    @endpoint
    def recover(self) -> bool:
        """Respawn counters on a fresh ProcMesh and restore from checkpoint.

        Returns ``True`` if recovery was performed, ``False`` if no failure
        was pending.
        """
        if not self.failure_detected:
            return False

        log.info("[Supervisor] Starting recovery — spawning fresh ProcMesh...")
        self._spawn_counters()

        for rank, ckpt in self.checkpoints.items():
            self.counters.slice(workers=rank).restore.call_one(ckpt).get()
            log.info(
                "[Supervisor] Rank %d restored to count=%d",
                rank,
                ckpt["count"],
            )

        self.failure_detected = False
        log.info(
            "[Supervisor] Recovery complete (total recoveries: %d)",
            self.recovery_count,
        )
        return True

    @endpoint
    def get_recovery_count(self) -> int:
        """Return how many times recovery has been performed."""
        return self.recovery_count


# ---------------------------------------------------------------------------
# Demo driver
# ---------------------------------------------------------------------------


def main():
    # --- Phase 1: Spawn supervisor and counter actors ---------------------
    log.info("=" * 60)
    log.info("PHASE 1: Spawning supervisor with %d counter actors", NUM_ACTORS)
    log.info("=" * 60)

    supervisor = this_proc().spawn("supervisor", CounterSupervisor)

    # --- Phase 2: Increment counters --------------------------------------
    log.info("=" * 60)
    log.info("PHASE 2: Incrementing all counters %d times", INCREMENTS_BEFORE_CRASH)
    log.info("=" * 60)

    for _ in range(INCREMENTS_BEFORE_CRASH):
        supervisor.increment_all.call_one(1).get()

    counts_before = supervisor.get_all_counts.call_one().get().values()
    log.info("Counts before crash: %s", counts_before)
    assert all(
        c == INCREMENTS_BEFORE_CRASH for c in counts_before
    ), f"Expected all counts = {INCREMENTS_BEFORE_CRASH}, got {counts_before}"

    # --- Phase 3: Checkpoint ----------------------------------------------
    log.info("=" * 60)
    log.info("PHASE 3: Checkpointing all counter states")
    log.info("=" * 60)

    supervisor.checkpoint_all.call_one().get()

    # --- Phase 4: Inject fault (SIGKILL one process) ----------------------
    log.info("=" * 60)
    log.info("PHASE 4: Injecting fault at rank %d (SIGKILL)", CRASH_RANK)
    log.info("=" * 60)

    killed_pid = supervisor.inject_fault.call_one(CRASH_RANK).get()
    log.info("Killed process %d — waiting for supervisor to detect failure", killed_pid)

    deadline = time.monotonic() + 30
    while not supervisor.is_failure_detected.call_one().get():
        if time.monotonic() > deadline:
            raise TimeoutError("Supervisor did not detect failure within 30 seconds")
        time.sleep(0.5)
    log.info("Supervisor detected the failure via __supervise__")

    # --- Phase 5: Recover — respawn + restore from checkpoint -------------
    log.info("=" * 60)
    log.info("PHASE 5: Recovering — respawn + restore from checkpoint")
    log.info("=" * 60)

    recovered = supervisor.recover.call_one().get()
    assert recovered, "Expected recovery to succeed"

    counts_after_recovery = supervisor.get_all_counts.call_one().get().values()
    log.info("Counts after recovery: %s", counts_after_recovery)
    assert all(c == INCREMENTS_BEFORE_CRASH for c in counts_after_recovery), (
        f"Expected restored counts = {INCREMENTS_BEFORE_CRASH}, "
        f"got {counts_after_recovery}"
    )

    # --- Phase 6: Continue counting after recovery ------------------------
    log.info("=" * 60)
    log.info("PHASE 6: Continuing — %d more increments", INCREMENTS_AFTER_RECOVERY)
    log.info("=" * 60)

    for _ in range(INCREMENTS_AFTER_RECOVERY):
        supervisor.increment_all.call_one(1).get()

    expected_total = INCREMENTS_BEFORE_CRASH + INCREMENTS_AFTER_RECOVERY
    counts_final = supervisor.get_all_counts.call_one().get().values()
    log.info("Final counts: %s", counts_final)
    assert all(
        c == expected_total for c in counts_final
    ), f"Expected all counts = {expected_total}, got {counts_final}"

    # --- Summary ----------------------------------------------------------
    recovery_count = supervisor.get_recovery_count.call_one().get()

    log.info("=" * 60)
    log.info("ALL ASSERTIONS PASSED")
    log.info("=" * 60)
    log.info("  Actors:               %d", NUM_ACTORS)
    log.info("  Increments before:    %d", INCREMENTS_BEFORE_CRASH)
    log.info("  Crash injected at:    rank %d (PID %d)", CRASH_RANK, killed_pid)
    log.info("  Recovered count:      %d (matches checkpoint)", INCREMENTS_BEFORE_CRASH)
    log.info("  Increments after:     %d", INCREMENTS_AFTER_RECOVERY)
    log.info("  Final count (all):    %d", expected_total)
    log.info("  Total recoveries:     %d", recovery_count)
    log.info("  Data lost:            0")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
