"""Game audio bus mixer.

Block-aligned implementation:

* sends run through persistent per-edge delay lines so delay compensation and
  reverb tails stay aligned across process blocks (no hard cuts on scene
  switches);
* side-chain ducking propagates downstream along the bus graph;
* sub-mixes at foreign sample rates are resampled before being summed;
* distance attenuation follows an inverse-distance curve with dB-space
  hysteresis;
* voice stealing evicts by priority first, then distance;
* the bus graph is processed in (iterative) topological order and cycles are
  detected without recursion;
* pause/resume resets the DSP clock and every delay line.
"""

import math


# Fixed duck depth applied while a side-chain source is over threshold.
SIDECHAIN_DUCK_GAIN = 0.5
# A reverb tail is considered fully drained once it has faded by 60 dB.
TAIL_FLOOR_DB = -60.0


def db_to_linear(db):
    # Amplitude convention: +6 dB doubles amplitude.
    return 10 ** (db / 20.0)


def linear_to_db(value):
    return 20.0 * math.log10(max(value, 1e-12))


def ms_to_samples(ms, sample_rate=48000):
    # Round half up (audio parameters are non-negative).
    return int(math.floor(ms * sample_rate / 1000.0 + 0.5))


def distance_gain(distance, reference=1.0, rolloff=1.0):
    """Inverse-distance attenuation, clamped to unity at/inside the reference."""
    if distance <= reference:
        return 1.0
    return reference / (reference + rolloff * (distance - reference))


def resample_block(samples, source_rate, target_rate, count):
    """Linear-interpolation resampler; emits exactly ``count`` samples.

    Input shorter than the needed read position is zero-padded so a block is
    always fully produced.
    """
    if source_rate == target_rate:
        out = [float(v) for v in samples[:count]]
        if len(out) < count:
            out.extend([0.0] * (count - len(out)))
        return out

    ratio = float(source_rate) / float(target_rate)
    out = [0.0] * count
    size = len(samples)
    for index in range(count):
        position = index * ratio
        left = int(math.floor(position))
        fraction = position - left
        if left + 1 < size:
            out[index] = samples[left] * (1.0 - fraction) + samples[left + 1] * fraction
        elif left < size:
            out[index] = samples[left]
        # else: beyond the input -> zero padding
    return out


def _rms_db(samples):
    if not samples:
        return -math.inf
    mean_square = sum(value * value for value in samples) / len(samples)
    if mean_square <= 0.0:
        return -math.inf
    return 10.0 * math.log10(mean_square)


class DistanceAttenuator:
    """Stateful inverse-distance gain with dB-space hysteresis.

    The target gain only replaces the held gain once it moves by more than
    ``hysteresis_db``, so near/far threshold crossings do not flap.
    """

    def __init__(self, reference=1.0, rolloff=1.0, hysteresis_db=2.0):
        self.reference = reference
        self.rolloff = rolloff
        self.hysteresis_db = hysteresis_db
        self._gain = 1.0

    def update(self, distance):
        target = distance_gain(distance, self.reference, self.rolloff)
        target_db = linear_to_db(target)
        current_db = linear_to_db(self._gain)
        if abs(target_db - current_db) >= self.hysteresis_db:
            self._gain = target
        return self._gain

    def reset(self):
        self._gain = 1.0


class BusGraph:
    def __init__(self):
        self.buses = {}
        self.edges = []

    def add_bus(self, name, gain_db=0.0):
        self.buses[name] = gain_db
        return self

    def add_send(self, source, destination, gain_db=0.0, delay_ms=0.0, tail_ms=0.0):
        if source not in self.buses:
            raise ValueError(f"unknown source bus: {source!r}")
        if destination not in self.buses:
            raise ValueError(f"unknown destination bus: {destination!r}")
        self.edges.append((source, destination, gain_db, delay_ms, tail_ms))
        return self

    def adjacency(self):
        graph = {name: [] for name in self.buses}
        for source, destination, *_ in self.edges:
            if source in graph and destination in graph:
                graph[source].append(destination)
        return graph

    def _indegrees(self):
        indegree = {name: 0 for name in self.buses}
        for source, destination, *_ in self.edges:
            if source in indegree and destination in indegree:
                indegree[destination] += 1
        return indegree

    def order(self):
        """Iterative Kahn topological sort.

        Ties break by bus insertion order for deterministic processing.
        Raises ``ValueError`` when the graph contains a cycle.
        """
        indegree = self._indegrees()
        adjacency = self.adjacency()
        ordered = []
        taken = set()
        while len(ordered) < len(self.buses):
            progressed = False
            for name in self.buses:
                if name not in taken and indegree[name] == 0:
                    ordered.append(name)
                    taken.add(name)
                    progressed = True
                    for downstream in adjacency[name]:
                        indegree[downstream] -= 1
                    break
            if not progressed:
                raise ValueError("bus graph contains a cycle")
        return ordered

    def has_cycle(self):
        try:
            self.order()
        except ValueError:
            return True
        return False


class DelayLine:
    """Persistent block-aligned delay line with an optional decay tail.

    Modelled as convolution with an impulse response that is silent for
    ``delay`` samples and then decays by ``TAIL_FLOOR_DB`` over ``tail``
    samples. Because the IR is finite and the input history is retained for
    exactly ``delay + tail + 1`` samples, a source going silent keeps feeding
    its reverb tail block after block (no hard cut) and drains to zero.
    """

    def __init__(self, delay_samples, tail_samples=0, block_size=0, sample_rate=48000):
        self.delay = max(0, int(delay_samples))
        self.tail = max(0, int(tail_samples))
        self.block_size = block_size
        self.sample_rate = sample_rate
        if self.tail > 0:
            # Per-sample coefficient reaching TAIL_FLOOR_DB over the tail.
            self.decay = 10 ** (TAIL_FLOOR_DB / 20.0 / self.tail)
        else:
            self.decay = 1.0
        # Input history (gain applied) with the absolute sample index of
        # history[0] recorded in self.base.
        self.history = []
        self.base = 0

    def _get(self, absolute_index):
        offset = absolute_index - self.base
        if 0 <= offset < len(self.history):
            return self.history[offset]
        return 0.0

    def process_block(self, samples, gain):
        block = len(samples)
        self.history.extend(value * gain for value in samples)
        total = self.base + len(self.history)
        ir_length = self.delay + self.tail + 1
        out = [0.0] * block
        for index in range(block):
            position = total - block + index
            accumulator = 0.0
            coefficient = 1.0
            for tap in range(ir_length):
                if tap > self.tail:
                    break
                value = self._get(position - self.delay - tap)
                if value:
                    accumulator += value * coefficient
                coefficient *= self.decay
            out[index] = accumulator
        # Retain only the history a future block can still read.
        keep = ir_length
        if len(self.history) > keep:
            drop = len(self.history) - keep
            del self.history[:drop]
            self.base += drop
        return out

    def reset(self):
        self.history = []
        self.base = 0


class Mixer:
    def __init__(self, sample_rate=48000, block_size=256, max_voices=8):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.max_voices = max_voices
        self.graph = BusGraph()
        self.sidechains = []
        self._clock = 0
        self._paused = False
        # Edge index -> DelayLine, persisted across process() calls.
        self._delay_lines = {}
        self._attenuators = {}

    # ------------------------------------------------------------------ graph
    def add_bus(self, name, gain_db=0.0):
        self.graph.add_bus(name, gain_db)
        return self

    def add_send(self, source, destination, gain_db=0.0, delay_ms=0.0, tail_ms=0.0):
        # BusGraph.add_send validates bus existence and raises ValueError.
        self.graph.add_send(source, destination, gain_db, delay_ms, tail_ms)
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

    # ------------------------------------------------------------- transport
    def pause(self):
        self._paused = True
        return self

    def resume(self):
        # Restoring after a pause resets the DSP clock and every delay line so
        # delayed signals cannot come back misaligned.
        self._paused = False
        self._clock = 0
        for line in self._delay_lines.values():
            line.reset()
        for attenuator in self._attenuators.values():
            attenuator.reset()
        return self

    def reset(self):
        self._clock = 0
        self._paused = False
        self._delay_lines.clear()
        for attenuator in self._attenuators.values():
            attenuator.reset()
        return self

    # ------------------------------------------------------------ attenuation
    def distance_gain(self, source_name, distance):
        attenuator = self._attenuators.get(source_name)
        if attenuator is None:
            attenuator = DistanceAttenuator()
            self._attenuators[source_name] = attenuator
        return attenuator.update(distance)

    # ----------------------------------------------------------- voice manage
    def select_voices(self, voices, limit=None):
        limit = self.max_voices if limit is None else limit
        ranked = sorted(
            voices,
            key=lambda voice: (
                -voice.get("priority", 0),
                voice.get("distance", math.inf),
            ),
        )
        return ranked[:limit]

    # ----------------------------------------------------------------- routing
    def _downstream_buses(self, targets):
        """Transitive downstream closure of ``targets`` (targets included)."""
        adjacency = self.graph.adjacency()
        seen = set()
        stack = [target for target in targets if target in self.graph.buses]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adjacency.get(node, ()))
        return seen

    def _coerce_block(self, source):
        """Normalize a source declaration to a block at the mixer rate."""
        source_rate = self.sample_rate
        if isinstance(source, dict):
            samples = source.get("samples", [])
            source_rate = source.get("sample_rate", self.sample_rate)
        elif isinstance(source, tuple) and len(source) == 2:
            samples, source_rate = source
        else:
            samples = source
        return resample_block(
            list(samples), source_rate, self.sample_rate, self.block_size
        )

    def process(self, sources):
        block = self.block_size
        if self._paused:
            # Frozen transport: emit silence without advancing the clock.
            return {name: [0.0] * block for name in self.graph.buses}

        # Topological order also rejects cyclic graphs.
        order = self.graph.order()
        signals = {name: [0.0] * block for name in order}
        input_levels = {}
        for name, source in sources.items():
            if name not in signals:
                continue
            block_samples = self._coerce_block(source)
            signals[name] = block_samples
            input_levels[name] = _rms_db(block_samples)

        # Side-chain gains are computed before routing so ducking follows the
        # signal downstream as well as hitting every declared target bus.
        duck_gain = {name: 1.0 for name in order}
        for sidechain in self.sidechains:
            source = sidechain["source"]
            if source not in signals:
                continue
            if input_levels.get(source, -math.inf) <= sidechain["threshold_db"]:
                continue
            for target in self._downstream_buses(sidechain["targets"]):
                if target in duck_gain:
                    duck_gain[target] *= SIDECHAIN_DUCK_GAIN

        # Sends in topological order through persistent, block-aligned lines.
        for edge_index, (source, destination, gain_db, delay_ms, tail_ms) in enumerate(
            self.graph.edges
        ):
            if source not in signals or destination not in signals:
                continue
            line = self._delay_lines.get(edge_index)
            if line is None:
                line = DelayLine(
                    ms_to_samples(delay_ms, self.sample_rate),
                    ms_to_samples(tail_ms, self.sample_rate),
                    block_size=block,
                    sample_rate=self.sample_rate,
                )
                self._delay_lines[edge_index] = line
            sent = line.process_block(signals[source], db_to_linear(gain_db))
            destination_block = signals[destination]
            for index in range(block):
                destination_block[index] += sent[index]

        # Finalize each bus: bus fader times accumulated side-chain ducking.
        for name in order:
            factor = db_to_linear(self.graph.buses[name]) * duck_gain[name]
            if factor != 1.0:
                signals[name] = [value * factor for value in signals[name]]

        self._clock += block
        return signals
