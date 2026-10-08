from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from streamnav.envs.habitat_client import VectorHabitatEnvs


def test_completed_environments_reset_concurrently_and_preserve_slot_identity():
    barrier = Barrier(2, timeout=5)

    class Client:
        def __init__(self, slot):
            self.slot = slot

        def reset(self, episode):
            barrier.wait()
            return (self.slot, episode)

    vector = object.__new__(VectorHabitatEnvs)
    vector.clients = [Client(i) for i in range(3)]
    with ThreadPoolExecutor(max_workers=3) as executor:
        vector.executor = executor
        assert vector.reset_at({2: "second", 0: "first"}) == {2: (2, "second"), 0: (0, "first")}
