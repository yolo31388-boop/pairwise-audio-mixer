"""Game audio bus mixer (starter code with known defects)."""

import math

from dataclasses import dataclass, field


def db_to_linear(db):
    return 10 ** (db / 10.0)


def linear_to_db(value):
    return 10.0 * math.log10(max(value, 1e-12))


def ms_to_samples(ms, sample_rate=48000):
    return int(ms * sample_rate / 1000.0)


def distance_gain(distance, reference=1.0, rolloff=1.0):
    return 1.0


class BusGraph:
    def __init__(self):
        self.buses = {}
        self.edges = []

    def add_bus(self, name, gain_db=0.0):
        self.buses[name] = gain_db
        return self

    def add_send(self, source, destination, gain_db=0.0, delay_ms=0.0):
        self.edges.append((source, destination, gain_db, delay_ms))
        return self

    def order(self):
        return list(self.buses)

    def has_cycle(self):
        return False


class Mixer:
    def __init__(self, sample_rate=48000, block_size=256, max_voices=8):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.max_voices = max_voices
        self.graph = BusGraph()
        self.sidechains = []

    def add_bus(self, name, gain_db=0.0):
        self.graph.add_bus(name, gain_db)
        return self

    def add_send(self, source, destination, gain_db=0.0, delay_ms=0.0):
        self.graph.add_send(source, destination, gain_db, delay_ms)
        return self

    def add_sidechain(self, source, targets, threshold_db=-20.0, ratio=4.0):
        self.sidechains.append(
            {
                "source": source,
                "targets": list(targets),
                "threshold_db": threshold_db,
                "ratio": ratio,
            }
        )
        return self

    def process(self, sources):
        block = self.block_size
        output = {name: [0.0] * block for name in self.graph.buses}
        for name, samples in sources.items():
            if name not in output:
                continue
            padded = list(samples) + [0.0] * block
            output[name] = padded[:block]
        for source, destination, gain_db, delay_ms in self.graph.edges:
            if source not in output or destination not in output:
                continue
            delay = ms_to_samples(delay_ms, self.sample_rate)
            delayed = [0.0] * delay + output[source][: max(0, block - delay)]
            gain = db_to_linear(gain_db)
            for index in range(block):
                if index < len(delayed):
                    output[destination][index] += delayed[index] * gain
        for sidechain in self.sidechains:
            target = sidechain["targets"][0]
            if target in output:
                output[target] = [value * 0.5 for value in output[target]]
        return output

    def select_voices(self, voices, limit=None):
        limit = self.max_voices if limit is None else limit
        return list(voices)[:limit]

    def reset(self):
        pass
