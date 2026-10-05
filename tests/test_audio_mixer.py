import numpy as np
import pytest

from audio.mixer import AudioMixer


SR = 8000  # keep sample-by-sample filters fast in tests


def sine(freq, seconds, amp=1.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


@pytest.fixture
def mixer():
    return AudioMixer(sample_rate=SR)


# ---------------------------------------------------------------- mix_audio

class TestMixAudio:
    def test_volume_balance(self, mixer):
        """每条轨道按各自音量加权，背景音乐不再和音效一样大"""
        music = sine(220, 0.5, amp=0.5)
        sfx = sine(880, 0.5, amp=0.5)
        out = mixer.mix_audio([
            {"samples": music, "volume": 0.2, "kind": "music"},
            {"samples": sfx, "volume": 1.0, "kind": "sfx"},
        ], sidechain=False)
        assert out is not None
        # 峰值不触发限幅时，输出恰好是各轨按音量加权之和
        assert np.allclose(out, 0.2 * music + sfx, atol=1e-6)
        # 音乐只占 0.2 权重，不再和音效一样大
        assert rms(out - sfx) < 0.25 * rms(out)

    def test_sidechain_ducking(self, mixer):
        """语音出现时背景音乐自动降低"""
        music = sine(220, 1.0, amp=0.5)
        voice = np.concatenate([np.zeros(4000), sine(500, 0.5, amp=0.8)])
        out = mixer.mix_audio([
            {"samples": music, "volume": 1.0, "kind": "music"},
            {"samples": voice, "volume": 1.0, "kind": "voice"},
        ], duck_amount=0.8)
        # 语音出现前 vs 出现后，去掉语音分量后的音乐残量
        music_level_before = rms(out[:3000])
        residual = out[5000:] - voice[5000:]
        music_level_after = rms(residual)
        assert music_level_after < 0.6 * music_level_before

    def test_limiter_no_clipping(self, mixer):
        """多轨同时播放超过 1.0 时被限幅而不是截断爆音"""
        tracks = [{"samples": sine(100 * (i + 1), 0.5, amp=0.9), "volume": 1.0}
                  for i in range(4)]
        out = mixer.mix_audio(tracks, sidechain=False)
        assert np.max(np.abs(out)) <= 1.0 + 1e-9
        # 限幅应是平滑的，不是硬截断的平顶
        assert np.sum(np.abs(out) >= 0.999) < len(out) * 0.5


# ---------------------------------------------------------------- apply_dsp

class TestApplyDsp:
    def test_multiple_effects(self, mixer):
        """支持混响、回声、均衡器等多种效果"""
        x = sine(440, 0.3)
        rev = mixer.apply_dsp(x, effect="reverb", mix=0.5)
        echo = mixer.apply_dsp(x, effect="echo", delay_ms=50, mix=0.5)
        eq = mixer.apply_dsp(x, effect="equalizer", low_gain_db=12)
        lp = mixer.apply_dsp(x, effect="lowpass", cutoff_hz=500)
        for out in (rev, echo, eq, lp):
            assert out is not None and len(out) >= len(x)
        # 混响产生余音：输入结束后仍有能量
        assert rms(rev[len(x):]) > 1e-4
        # 回声产生延迟副本
        assert rms(echo[len(x):]) > 1e-4
        # 均衡器提升低频后输出与输入不同
        assert not np.allclose(eq[:len(x)], x)

    def test_parameter_smoothing(self, mixer):
        """参数变化经过平滑，调节音量不出现爆音"""
        x = sine(440, 0.2)
        mixer.apply_dsp(x, effect="gain", gain_db=0.0)  # 建立初始状态
        out = mixer.apply_dsp(x, effect="gain", gain_db=24,
                              smoothing_ms=200)  # 平滑窗口覆盖整个缓冲
        # 增益从 0 dB 渐变到 24 dB：开头接近原信号，结尾被放大
        assert np.allclose(out[:10], x[:10], atol=0.02)
        assert rms(out[:200]) < 1.5 * rms(x[:200])
        assert rms(out[-500:]) > 4 * rms(x[-500:])
        # 不平滑时增益瞬间跳变（爆音），平滑时从旧值连续过渡
        mixer2 = AudioMixer(sample_rate=SR)
        mixer2.apply_dsp(x, effect="gain", gain_db=0.0)
        hard = mixer2.apply_dsp(x, effect="gain", gain_db=24, smoothing_ms=0)
        assert abs(hard[5] - x[5]) > 10 * abs(out[5] - x[5])

    def test_bypass(self, mixer):
        """旁路时原样输出，可对比效果前后"""
        x = sine(440, 0.2)
        out = mixer.apply_dsp(x, effect="reverb", bypass=True)
        assert np.array_equal(out, x)
        processed = mixer.apply_dsp(x, effect="reverb")
        assert not np.array_equal(processed[:len(x)], x)


# ---------------------------------------------------------- calculate_3d_audio

class TestCalculate3dAudio:
    def test_direction_panning(self, mixer):
        """方向影响左右声道：右侧声源右耳更响"""
        x = sine(440, 0.2)
        stereo = mixer.calculate_3d_audio(x, source_pos=(2, 0, 0))
        assert stereo.shape[1] == 2
        assert rms(stereo[:, 1]) > 1.5 * rms(stereo[:, 0])
        # 左侧声源镜像
        stereo_l = mixer.calculate_3d_audio(x, source_pos=(-2, 0, 0))
        assert rms(stereo_l[:, 0]) > 1.5 * rms(stereo_l[:, 1])

    def test_hrtf_spatial_cues(self, mixer):
        """HRTF 让正前方和正后方（同距离）的声音不同，且侧方两耳有时差"""
        x = sine(440, 0.2)
        front = mixer.calculate_3d_audio(x, source_pos=(0, 2, 0))
        behind = mixer.calculate_3d_audio(x, source_pos=(0, -2, 0))
        assert not np.allclose(front, behind)
        # 侧方声源产生耳间时间差（ITD）
        side = mixer.calculate_3d_audio(x, source_pos=(2, 0, 0))
        left_peak = int(np.argmax(np.abs(side[:, 0])))
        right_peak = int(np.argmax(np.abs(side[:, 1])))
        assert left_peak != right_peak

    def test_distance_attenuation(self, mixer):
        """距离衰减：远处声音比近处小"""
        x = sine(440, 0.2)
        near = mixer.calculate_3d_audio(x, source_pos=(0, 1, 0))
        far = mixer.calculate_3d_audio(x, source_pos=(0, 10, 0))
        assert rms(far) < 0.5 * rms(near)


# ---------------------------------------------------- compress_dynamic_range

class TestCompressDynamicRange:
    def test_ratio_and_attack_release(self, mixer):
        """超过阈值的部分按压缩比压缩，攻击时间让瞬态先通过"""
        x = sine(200, 1.0, amp=0.8)  # 约 -4 dB
        out = mixer.compress_dynamic_range(
            x, threshold_db=-20.0, ratio=4.0,
            attack_ms=20.0, release_ms=100.0, makeup_gain_db=0.0)
        # 稳态电平 ≈ threshold + (in - threshold) / ratio
        in_db = 20 * np.log10(rms(x[2000:]))
        out_db = 20 * np.log10(rms(out[2000:]))
        expected_db = -20.0 + (in_db + 20.0) / 4.0
        assert abs(out_db - expected_db) < 2.0
        # 攻击时间内增益尚未完全压下：开头比稳态响
        assert rms(out[200:400]) > rms(out[4000:4200])

    def test_threshold_leaves_quiet_signals(self, mixer):
        """低于阈值的声音不被压缩"""
        quiet = sine(200, 0.5, amp=0.01)  # 约 -43 dB
        out = mixer.compress_dynamic_range(
            quiet, threshold_db=-20.0, ratio=4.0, makeup_gain_db=0.0)
        assert np.allclose(out, quiet, atol=1e-9)

    def test_makeup_gain(self, mixer):
        """makeup gain 补偿压缩损失的音量"""
        x = sine(200, 0.5, amp=0.8)
        dry = mixer.compress_dynamic_range(
            x, threshold_db=-20.0, ratio=4.0, makeup_gain_db=0.0)
        auto = mixer.compress_dynamic_range(
            x, threshold_db=-20.0, ratio=4.0)  # 自动 makeup
        assert rms(auto[2000:]) > 2 * rms(dry[2000:])
        # 自动 makeup 后整体响度接近原始信号
        assert rms(auto[2000:]) > 0.7 * rms(x[2000:])


# -------------------------------------------------------------- route_audio

class TestRouteAudio:
    def test_multiple_buses_with_send_effects(self, mixer):
        """多总线分离背景音乐/音效/语音，且支持发送效果"""
        music = np.concatenate([sine(220, 0.2), np.zeros(2000)])
        voice = sine(500, 0.2)
        result = mixer.route_audio(
            {
                "bgm": {"samples": music, "bus": "music"},
                "line": {"samples": voice, "bus": "voice"},
            },
            buses={
                "music": {"sends": [{"effect": "reverb", "amount": 0.8,
                                     "params": {"mix": 0.8}}]},
                "voice": {},
                "sfx": {},
                "ambient": {},
            })
        assert set(result["buses"]) == {"music", "voice", "sfx", "ambient"}
        # 音乐总线有混响余音（源结束后仍有能量）
        assert rms(result["buses"]["music"][2100:]) > 1e-4
        # 语音总线无发送效果，源结束后安静
        assert rms(result["buses"]["voice"][2100:]) < 1e-9
        # master 包含两路
        assert rms(result["master"][:1000]) > 0

    def test_mute_and_solo(self, mixer):
        """静音和独奏：调试时可单独听某一路"""
        sources = {
            "bgm": {"samples": sine(220, 0.2), "bus": "music"},
            "line": {"samples": sine(500, 0.2), "bus": "voice"},
        }
        solo = mixer.route_audio(sources, buses={
            "music": {}, "voice": {"solo": True}})
        assert np.allclose(solo["master"], solo["buses"]["voice"])
        # 独奏时听不到音乐
        assert abs(np.corrcoef(solo["master"], sources["bgm"]["samples"])[0, 1]) < 0.1

        muted = mixer.route_audio(sources, buses={
            "music": {"mute": True}, "voice": {}})
        assert np.allclose(muted["master"], muted["buses"]["voice"])
