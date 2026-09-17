"""The layer gate under concurrent storage streams.

A batch carries one consumer index, so once several storage transactions stream
at once the gate has to track the rest itself -- and it has to hand each of them
a slot nobody is still using. Both are what these cases pin.
"""

import unittest
from unittest import mock

from sglang.test.test_utils import CustomTestCase

NUM_LAYERS = 4


class _FakeEvent:
    """Stands in for a device event: `query` is the only thing the gate reads."""

    def __init__(self):
        self.ready = True

    def query(self):
        return self.ready

    def record(self):
        pass


class _FakeLayerLoadingEvent:
    def __init__(self, num_layers):
        self._num_layers = num_layers
        self.load_events = [_FakeEvent() for _ in range(num_layers)]
        self.start_event = _FakeEvent()
        self.generation = 0
        self.waits = []

    @property
    def finish_event(self):
        return self.load_events[-1]

    def begin_generation(self, generation):
        self.generation = generation

    def wait(self, layer_index, *, generation=None, timeout=None, pump=None):
        self.waits.append((layer_index, generation))


def _counter(max_concurrent_streams=1):
    from sglang.srt.managers import cache_controller

    with mock.patch.object(
        cache_controller, "LayerLoadingEvent", _FakeLayerLoadingEvent
    ):
        return cache_controller.LayerDoneCounter(
            NUM_LAYERS, max_concurrent_streams=max_concurrent_streams
        )


class TestLayerDoneCounterSlots(CustomTestCase):
    def test_slot_count_leaves_room_for_every_stream_plus_overlap(self):
        # Two spare slots are what the non-streaming path has always needed for
        # overlap mode; the streams are on top of that, not instead of it.
        self.assertEqual(_counter(1).num_counters, 3)
        self.assertEqual(_counter(4).num_counters, 6)

    def test_a_busy_slot_is_skipped_rather_than_reused(self):
        """Blind rotation onto an in-flight slot resets a generation another
        consumer is waiting on, which `python -O` turns from a crash into a
        forward reading KV that never arrived."""
        counter = _counter(4)
        first = counter.update_producer()
        counter.events[first].finish_event.ready = False

        # Walk the ring all the way round; the busy slot must never come back.
        taken = []
        for _ in range(counter.num_counters - 1):
            index = counter.update_producer()
            counter.events[index].finish_event.ready = False
            taken.append(index)

        self.assertNotIn(first, taken)
        self.assertEqual(len(set(taken)), len(taken), "no slot handed out twice")

    def test_running_out_of_slots_says_so_instead_of_corrupting_one(self):
        counter = _counter(1)
        for _ in range(counter.num_counters):
            index = counter.update_producer()
            counter.events[index].finish_event.ready = False

        with self.assertRaisesRegex(RuntimeError, "still in flight"):
            counter.update_producer()

    def test_generation_increases_every_time_a_slot_is_handed_out(self):
        counter = _counter(2)
        seen = []
        for _ in range(3):
            index = counter.update_producer()
            seen.append(counter.events[index].generation)
            counter.events[index].finish_event.ready = True
        self.assertEqual(seen, sorted(set(seen)), "generations must strictly rise")


class TestLayerDoneCounterConsumers(CustomTestCase):
    def test_wait_gates_on_every_open_stream_not_just_the_batch_index(self):
        """The batch's rows come from all the open streams. Waiting only on the
        one the batch happens to name would read a row whose stream is behind.
        """
        counter = _counter(4)
        counter.set_consumer(0, generation=1)
        counter.add_stream_consumer(1, 5)
        counter.add_stream_consumer(2, 6)

        counter.wait_until(3)

        self.assertEqual(counter.events[0].waits, [(3, 1)])
        self.assertEqual(counter.events[1].waits, [(3, 5)])
        self.assertEqual(counter.events[2].waits, [(3, 6)])

    def test_the_batch_index_is_not_waited_on_twice(self):
        # ready_to_load_host_cache hands the batch one of the open streams, so
        # that slot is in both places; it is waited at the batch's generation.
        counter = _counter(4)
        counter.set_consumer(1, generation=5)
        counter.add_stream_consumer(1, 5)

        counter.wait_until(2)

        self.assertEqual(counter.events[1].waits, [(2, 5)])

    def test_a_stream_still_gates_the_wait_when_the_batch_has_no_index(self):
        counter = _counter(4)
        counter.set_consumer(-1)
        counter.add_stream_consumer(2, 9)

        counter.wait_until(1)

        self.assertEqual(counter.events[2].waits, [(1, 9)])

    def test_dropping_a_stream_stops_gating_on_it(self):
        counter = _counter(4)
        counter.add_stream_consumer(2, 9)
        counter.drop_stream_consumer(2)

        counter.wait_until(1)

        self.assertEqual(counter.events[2].waits, [])

    def test_reset_forgets_the_streams_too(self):
        counter = _counter(4)
        counter.add_stream_consumer(2, 9)
        counter.reset()

        counter.wait_until(1)

        self.assertEqual(counter.events[2].waits, [])
        self.assertEqual(counter.stream_consumers, {})

    def test_a_slot_outside_the_ring_is_refused(self):
        counter = _counter(1)
        with self.assertRaisesRegex(ValueError, "stream consumer index"):
            counter.add_stream_consumer(counter.num_counters, 1)


if __name__ == "__main__":
    unittest.main()
