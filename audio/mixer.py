"""
pairwise-audio-mixer - AudioMixer

A small but correct audio mixing engine:

- mix_audio(): volume balancing, sidechain ducking, and a true limiter.
- apply_dsp(): multiple effects (lowpass/reverb/echo/equalizer/gain) with
  parameter smoothing and bypass support.
- calculate_3d_audio(): direction (panning + ITD), HRTF-style head shadow,
  and distance attenuation.
- compress_dynamic_range(): threshold/ratio/attack/release + makeup gain.
- route_audio(): multiple buses with send effects, mute and solo.
"""

import numpy as np


class AudioMixer:
    DEFAULT_BUSES = ("music", "sfx", "voice", "ambient")

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.state = {"dsp": {}}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_float_array(samples):
        return np.asarray(samples, dtype=np.float64)

    @staticmethod
    def _envelope(x, attack_ms, release_ms, sample_rate):
        """Peak envelope follower with separate attack/release times."""
        x = np.abs(x)
        attack_n = attack_ms * 0.001 * sample_rate
        attack_coef = 0.0 if attack_n <= 0.0 else np.exp(-1.0 / attack_n)
        release_coef = np.exp(-1.0 / max(1.0, release_ms * 0.001 * sample_rate))
        env = np.empty_like(x)
        prev = 0.0
        for i in range(x.shape[0]):
            v = x[i]
            coef = attack_coef if v > prev else release_coef
            prev = coef * prev + (1.0 - coef) * v
            env[i] = prev
        return env

    @staticmethod
    def _lowpass(x, cutoff_hz, sample_rate):
        """One-pole low-pass filter."""
        rc = 1.0 / (2.0 * np.pi * max(1.0, cutoff_hz))
        dt = 1.0 / sample_rate
        alpha = dt / (rc + dt)
        y = np.empty_like(x)
        prev = 0.0
        for i in range(x.shape[0]):
            prev = prev + alpha * (x[i] - prev)
            y[i] = prev
        return y

    @staticmethod
    def _smooth_param(target, previous, num_samples, smoothing_ms, sample_rate):
        """Ramp a parameter from its previous value to the target value."""
        smooth_n = int(smoothing_ms * 0.001 * sample_rate)
        if smooth_n <= 0:
            return np.full(num_samples, target)
        ramp_len = min(num_samples, smooth_n)
        ramp = np.linspace(previous, target, ramp_len, endpoint=True)
        if ramp_len < num_samples:
            ramp = np.concatenate([ramp, np.full(num_samples - ramp_len, target)])
        return ramp

    # ------------------------------------------------------------------
    # mixing
    # ------------------------------------------------------------------
    def mix_audio(self, tracks, sidechain=True, duck_amount=0.6,
                  duck_threshold=0.02, ceiling=1.0, sample_rate=None):
        """Mix tracks with volume balancing, sidechain ducking and limiting.

        tracks: list of dicts with keys
            samples (array-like), volume (float, default 1.0),
            kind ("music" | "sfx" | "voice" | "ambient", default "sfx").
        """
        sr = sample_rate or self.sample_rate
        if not tracks:
            return np.zeros(0)

        length = max(len(t["samples"]) for t in tracks)

        # Sidechain: voice activity ducks the music bus.
        duck_gain = np.ones(length)
        if sidechain:
            voice = np.zeros(length)
            for t in tracks:
                if t.get("kind", "sfx") == "voice":
                    s = self._as_float_array(t["samples"])
                    voice[:len(s)] += s
            voice_env = self._envelope(voice, 5.0, 150.0, sr)
            depth = np.clip(voice_env / max(duck_threshold, 1e-9), 0.0, 1.0)
            duck_gain = 1.0 - duck_amount * depth

        # Volume balancing: every track is scaled by its own volume,
        # music additionally follows the sidechain duck gain.
        mix = np.zeros(length)
        for t in tracks:
            s = self._as_float_array(t["samples"])
            volume = float(t.get("volume", 1.0))
            gain = volume * (duck_gain[:len(s)] if t.get("kind", "sfx") == "music"
                             else 1.0)
            mix[:len(s)] += s * gain

        # True limiter: instant-attack envelope, gain never lets the
        # signal exceed the ceiling (no hard clipping / no distortion).
        env = self._envelope(mix, 0.0, 50.0, sr)
        gain = np.minimum(1.0, ceiling / np.maximum(env, 1e-12))
        return mix * gain

    # ------------------------------------------------------------------
    # DSP effects
    # ------------------------------------------------------------------
    def apply_dsp(self, samples, effect="lowpass", bypass=False,
                  sample_rate=None, smoothing_ms=20.0, **params):
        """Apply a DSP effect.

        Supported effects: lowpass, reverb, echo, equalizer, gain.
        Parameter changes are smoothed to avoid zipper noise / clicks.
        bypass=True returns the input untouched (for A/B comparison).
        """
        x = self._as_float_array(samples)
        if bypass:
            return x.copy()
        sr = sample_rate or self.sample_rate

        if effect == "lowpass":
            return self._lowpass(x, params.get("cutoff_hz", 1000.0), sr)
        if effect == "reverb":
            return self._reverb(x, sr, **params)
        if effect == "echo":
            return self._echo(x, sr, **params)
        if effect == "equalizer":
            return self._equalizer(x, sr, smoothing_ms, **params)
        if effect == "gain":
            return self._gain(x, sr, smoothing_ms, **params)
        raise ValueError(f"unknown effect: {effect!r}")

    def _gain(self, x, sr, smoothing_ms, gain_db=0.0, **_):
        key = ("gain", "gain_db")
        previous = self.state["dsp"].get(key, 0.0)
        self.state["dsp"][key] = gain_db
        ramp_db = self._smooth_param(gain_db, previous, len(x), smoothing_ms, sr)
        return x * np.power(10.0, ramp_db / 20.0)

    def _equalizer(self, x, sr, smoothing_ms, low_gain_db=0.0,
                   mid_gain_db=0.0, high_gain_db=0.0, **_):
        n = len(x)
        gains = {}
        for name, target in (("low_gain_db", low_gain_db),
                             ("mid_gain_db", mid_gain_db),
                             ("high_gain_db", high_gain_db)):
            key = ("equalizer", name)
            previous = self.state["dsp"].get(key, 0.0)
            self.state["dsp"][key] = target
            ramp_db = self._smooth_param(target, previous, n, smoothing_ms, sr)
            gains[name] = np.power(10.0, ramp_db / 20.0)
        low = self._lowpass(x, 250.0, sr)
        high = x - self._lowpass(x, 5000.0, sr)
        mid = x - low - high
        return (low * gains["low_gain_db"] + mid * gains["mid_gain_db"]
                + high * gains["high_gain_db"])

    def _echo(self, x, sr, delay_ms=300.0, feedback=0.4, mix=0.5, **_):
        delay = max(1, int(delay_ms * 0.001 * sr))
        wet = np.zeros(len(x) + delay)
        wet[:len(x)] = x
        for i in range(delay, len(wet)):
            wet[i] += wet[i - delay] * feedback
        out = (1.0 - mix) * x + mix * wet[:len(x)]
        return np.concatenate([out, mix * wet[len(x):]])

    def _reverb(self, x, sr, decay=0.5, mix=0.3, **_):
        # Schroeder reverb: parallel feedback combs + series allpasses.
        combs = [(29.7, 0.773), (37.1, 0.802), (41.1, 0.753), (43.7, 0.733)]
        wet = np.zeros(len(x) + int(0.1 * sr))
        wet[:len(x)] = x
        out = np.zeros_like(wet)
        for delay_ms, base_fb in combs:
            d = max(1, int(delay_ms * 0.001 * sr))
            fb = base_fb * decay
            buf = np.zeros_like(wet)
            buf[:len(x)] = x
            for i in range(d, len(wet)):
                buf[i] += buf[i - d] * fb
            out += buf
        out *= 0.25
        for delay_ms in (5.0, 1.7):
            d = max(1, int(delay_ms * 0.001 * sr))
            g = 0.5
            y = np.zeros_like(out)
            for i in range(len(out)):
                delayed = out[i - d] if i >= d else 0.0
                y_prev = y[i - d] if i >= d else 0.0
                y[i] = -g * out[i] + delayed + g * y_prev
            out = y
        result = (1.0 - mix) * wet + mix * out
        return result[:len(x) + int(0.05 * sr)]

    # ------------------------------------------------------------------
    # 3D audio
    # ------------------------------------------------------------------
    def calculate_3d_audio(self, samples, source_pos, listener_pos=(0, 0, 0),
                           listener_yaw=0.0, reference_distance=1.0,
                           sample_rate=None):
        """Spatialize a mono source to stereo.

        Uses distance attenuation, constant-power panning, interaural time
        delay (ITD) and an HRTF-style head-shadow / front-back filter.
        Coordinates: +x right, +y forward, +z up. listener_yaw in degrees.
        """
        sr = sample_rate or self.sample_rate
        x = self._as_float_array(samples)
        src = np.asarray(source_pos, dtype=np.float64)
        lis = np.asarray(listener_pos, dtype=np.float64)

        rel = src - lis
        distance = float(np.linalg.norm(rel))

        # Distance attenuation: closer than reference stays at unity,
        # farther sources decay with 1/d.
        attenuation = min(1.0, reference_distance / max(distance, 1e-9))

        # Azimuth relative to the listener's facing direction.
        yaw = np.deg2rad(listener_yaw)
        forward = np.array([np.sin(yaw), np.cos(yaw)])
        right = np.array([np.cos(yaw), -np.sin(yaw)])
        horiz = rel[:2]
        norm = np.linalg.norm(horiz)
        if norm < 1e-9:
            azimuth = 0.0
        else:
            horiz = horiz / norm
            azimuth = np.arctan2(np.dot(horiz, right), np.dot(horiz, forward))

        # Constant-power panning (direction -> left/right level difference).
        pan_angle = (azimuth + np.pi / 2.0) / 2.0  # 0=left ... pi/2=right
        left_gain = np.cos(pan_angle) * np.sqrt(2.0)
        right_gain = np.sin(pan_angle) * np.sqrt(2.0)

        left = x * left_gain
        right = x * right_gain

        # ITD: the far ear receives the sound up to ~0.63 ms later.
        itd = 0.00063 * np.sin(azimuth)
        delay_samples = int(round(abs(itd) * sr))
        if delay_samples > 0:
            if itd > 0:  # source on the right -> left ear delayed
                left = np.concatenate([np.zeros(delay_samples), left])
                right = np.concatenate([right, np.zeros(delay_samples)])
            else:
                right = np.concatenate([np.zeros(delay_samples), right])
                left = np.concatenate([left, np.zeros(delay_samples)])

        # HRTF-style filtering: the far ear is low-passed (head shadow),
        # and sources behind the listener lose high frequencies (pinna).
        shadow = abs(np.sin(azimuth))
        far_cutoff = 20000.0 - 16000.0 * shadow
        behind = np.cos(azimuth) < 0
        if itd > 0:
            left = self._lowpass(left, far_cutoff, sr)
        elif itd < 0:
            right = self._lowpass(right, far_cutoff, sr)
        if behind:
            left = self._lowpass(left, 5000.0, sr)
            right = self._lowpass(right, 5000.0, sr)

        stereo = np.stack([left, right], axis=-1) * attenuation
        return stereo

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def compress_dynamic_range(self, samples, threshold_db=-20.0, ratio=4.0,
                               attack_ms=5.0, release_ms=100.0,
                               makeup_gain_db=None, sample_rate=None,
                               knee_db=0.0):
        """Compressor with threshold, ratio, attack/release and makeup gain.

        Signals below the threshold pass through untouched. makeup_gain_db
        defaults to auto-makeup (the gain reduction of a full-scale signal).
        """
        sr = sample_rate or self.sample_rate
        x = self._as_float_array(samples)

        env = self._envelope(x, attack_ms, release_ms, sr)
        level_db = 20.0 * np.log10(np.maximum(env, 1e-12))

        over = level_db - threshold_db
        reduction_db = np.where(over > 0.0, over * (1.0 - 1.0 / ratio), 0.0)
        gain = np.power(10.0, -reduction_db / 20.0)
        compressed = x * gain

        if makeup_gain_db is None:
            # Auto makeup: compensate the reduction of a 0 dBFS signal.
            full_over = max(0.0, 0.0 - threshold_db)
            makeup_gain_db = full_over * (1.0 - 1.0 / ratio)
        return compressed * np.power(10.0, makeup_gain_db / 20.0)

    # ------------------------------------------------------------------
    # bus routing
    # ------------------------------------------------------------------
    def route_audio(self, sources, buses=None, sample_rate=None):
        """Route named sources through buses to a master mix.

        sources: {name: {"samples": array, "bus": "music", "volume": 1.0}}
        buses:   {bus_name: {"volume": 1.0, "mute": False, "solo": False,
                             "sends": [{"effect": "reverb", "amount": 0.3,
                                        "params": {...}}]}}
        Returns {"master": array, "buses": {bus_name: array}}.
        """
        sr = sample_rate or self.sample_rate
        if buses is None:
            buses = {name: {} for name in self.DEFAULT_BUSES}

        length = 0
        for s in sources.values():
            length = max(length, len(s["samples"]))

        bus_mix = {}
        for bus_name, cfg in buses.items():
            acc = np.zeros(length)
            for s in sources.values():
                if s.get("bus", "sfx") != bus_name:
                    continue
                data = self._as_float_array(s["samples"])
                acc[:len(data)] += data * float(s.get("volume", 1.0))

            # Send effects: blend the processed (wet) signal into the bus.
            for send in cfg.get("sends", []):
                amount = float(send.get("amount", 0.5))
                wet = self.apply_dsp(acc, effect=send["effect"],
                                     sample_rate=sr,
                                     **send.get("params", {}))
                wet = wet[:length]
                if len(wet) < length:
                    wet = np.concatenate([wet, np.zeros(length - len(wet))])
                acc = (1.0 - amount) * acc + amount * wet

            bus_mix[bus_name] = acc * float(cfg.get("volume", 1.0))

        any_solo = any(cfg.get("solo", False) for cfg in buses.values())
        master = np.zeros(length)
        for bus_name, cfg in buses.items():
            if cfg.get("mute", False):
                continue
            if any_solo and not cfg.get("solo", False):
                continue
            master += bus_mix[bus_name]

        return {"master": master, "buses": bus_mix}
