"""音效混音表驱动测试：优先级抢占、3D衰减、淡入淡出、静音、循环。"""

from __future__ import annotations

import pytest

from src import AudioMixer, Sound


def _eng():
    return AudioMixer(channels=2)


def test_basic_play():
    eng = _eng()
    assert eng.play_sound(Sound("s1", priority=5)) is True
    assert eng.active_count() == 1


def test_high_priority_preempts():
    """高优先级应抢占低优先级声道。"""
    eng = _eng()
    eng.play_sound(Sound("low1", priority=1))
    eng.play_sound(Sound("low2", priority=1))
    # 声道满了，高优先级应该抢占
    result = eng.play_sound(Sound("high", priority=10))
    assert result is True, "高优先级应抢占低优先级"
    assert eng.active_count() == 2
    # 高优先级应该在播放
    playing = [s.sid for s in eng.channels.values() if s]
    assert "high" in playing


def test_low_priority_rejected():
    eng = _eng()
    eng.play_sound(Sound("high1", priority=10))
    eng.play_sound(Sound("high2", priority=10))
    # 低优先级不能抢占高优先级
    result = eng.play_sound(Sound("low", priority=1))
    assert result is False


def test_same_priority_fifo():
    eng = AudioMixer(channels=1)
    eng.play_sound(Sound("a", priority=5))
    # 相同优先级不抢占
    result = eng.play_sound(Sound("b", priority=5))
    assert result is False


def test_3d_volume_logarithmic():
    """3D音量应该是对数/距离反比衰减，不是线性。"""
    eng = _eng()
    v_near = eng.calculate_3d_volume((0,0,0), (1,0,0))
    v_far = eng.calculate_3d_volume((0,0,0), (50,0,0))
    # 对数衰减：远处不应该是0.5（线性），应该更小
    assert v_far < v_near * 0.5, \
        "3D音量应对数衰减: near=%.3f far=%.3f" % (v_near, v_far)


def test_muted_no_play():
    eng = _eng()
    eng.set_muted(True)
    assert eng.play_sound(Sound("s1")) is False


def test_pause_resume():
    eng = _eng()
    eng.play_sound(Sound("s1"))
    eng.pause()
    assert eng.paused is True
    eng.resume()
    assert eng.paused is False


def test_stop_sound():
    eng = _eng()
    eng.play_sound(Sound("s1"))
    eng.stop_sound("s1")
    assert eng.active_count() == 0


def test_independent_volumes():
    eng = _eng()
    eng.master_volume = 0.5
    eng.sfx_volume = 0.8
    assert eng.master_volume == 0.5
    assert eng.sfx_volume == 0.8


def test_deterministic():
    def run():
        eng = _eng()
        eng.play_sound(Sound("a", priority=3))
        eng.play_sound(Sound("b", priority=7))
        eng.play_sound(Sound("c", priority=5))
        return eng.snapshot()
    assert run() == run()


def test_fifo_tiebreak_on_preempt():
    """多个同优先级低优先级声道时，FIFO淘汰最早开始的音效。"""
    eng = AudioMixer(channels=2)
    eng.play_sound(Sound("old", priority=1))
    eng.play_sound(Sound("new_low", priority=1))
    assert eng.play_sound(Sound("high", priority=5)) is True
    playing = [s.sid for s in eng.channels.values() if s]
    assert "old" not in playing, "同优先级最低音效应按FIFO淘汰最早者"
    assert "new_low" in playing
    assert "high" in playing


def test_low_priority_cannot_preempt_high():
    """低优先级不能抢占高优先级声道。"""
    eng = AudioMixer(channels=3)
    eng.play_sound(Sound("h1", priority=9))
    eng.play_sound(Sound("h2", priority=9))
    eng.play_sound(Sound("h3", priority=9))
    assert eng.play_sound(Sound("footstep", priority=0)) is False
    assert eng.active_count() == 3
    assert eng.is_playing("footstep") is False


def test_same_priority_fifo_reject_full():
    """声道全被同优先级占用时，新音效被拒绝（不抢占）。"""
    eng = AudioMixer(channels=2)
    assert eng.play_sound(Sound("a", priority=4)) is True
    assert eng.play_sound(Sound("b", priority=4)) is True
    assert eng.play_sound(Sound("c", priority=4)) is False
    assert eng.active_count() == 2


def test_fade_in_envelope():
    """新播放音效必须淡入：增益从0平滑上升到1，不能突然满音量。"""
    eng = AudioMixer(channels=1, fade_in=0.1)
    eng.play_sound(Sound("s", priority=5), duration=10.0)
    assert eng.gains()["s"] == 0.0
    eng.update(0.05)
    mid = eng.gains()["s"]
    assert 0.0 < mid < 1.0, "淡入中途增益应在(0,1)之间: %r" % mid
    eng.update(0.05)
    assert eng.gains()["s"] == 1.0


def test_fade_out_on_preempt_no_cut():
    """被抢占音效立即释放声道但保留淡出尾音，不突然切断。"""
    eng = AudioMixer(channels=1, fade_in=0.0, fade_out=0.1)
    eng.play_sound(Sound("low", priority=1), duration=10.0)
    eng.update(0.01)
    assert eng.play_sound(Sound("high", priority=8), duration=10.0) is True
    assert eng.is_playing("low") is False
    assert eng.active_count() == 1
    assert "low" in eng.tail_gains()
    assert eng.tail_gains()["low"] == 1.0, "抢占瞬间不应突然跳变音量"
    eng.update(0.05)
    g = eng.tail_gains()["low"]
    assert 0.0 < g < 1.0, "被抢占旧音效应平滑淡出而非立刻归零: %r" % g
    eng.update(0.05)
    assert "low" not in eng.tail_gains(), "淡出结束后尾音应被清理"


def test_fade_out_on_stop():
    """主动停止也必须淡出，声道立即释放。"""
    eng = AudioMixer(channels=1, fade_in=0.0, fade_out=0.1)
    eng.play_sound(Sound("s", priority=5), duration=10.0)
    eng.update(0.01)
    eng.stop_sound("s")
    assert eng.active_count() == 0
    assert "s" in eng.tail_gains()
    assert eng.tail_gains()["s"] > 0.0
    eng.update(0.1)
    assert eng.tail_gains() == {}


def test_3d_volume_monotonic_log_curve():
    """3D音量按距离反比衰减：单调递减，且比线性衰减更快、更安静。"""
    eng = _eng()
    v0 = eng.calculate_3d_volume((0, 0, 0), (0, 0, 0))
    v10 = eng.calculate_3d_volume((0, 0, 0), (10, 0, 0))
    v20 = eng.calculate_3d_volume((0, 0, 0), (20, 0, 0))
    v50 = eng.calculate_3d_volume((0, 0, 0), (50, 0, 0))
    assert v0 == 1.0
    assert v10 > v20 > v50, "距离越远音量必须越小"
    assert v10 == 0.5
    # 线性衰减在20单位处为0.8，对数/反比衰减必须明显更低
    assert v20 < 0.8
    assert v50 < 0.5
    v_max = eng.calculate_3d_volume((0, 0, 0), (100, 0, 0), max_distance=100.0)
    assert v_max == 0.0


def test_muted_frees_channels():
    """静音时不占用声道，且静音期间新音效不播放。"""
    eng = AudioMixer(channels=2, fade_in=0.0, fade_out=0.1)
    eng.play_sound(Sound("a", priority=5), duration=10.0)
    eng.play_sound(Sound("b", priority=5), duration=10.0)
    eng.update(0.01)
    eng.set_muted(True)
    assert eng.active_count() == 0
    assert eng.tail_gains() == {}, "静音不应保留可发声的尾音"
    assert eng.play_sound(Sound("c", priority=5)) is False
    assert eng.active_count() == 0
    eng.set_muted(False)
    assert eng.active_count() == 0


def test_looping_sound_repeats():
    """循环音效不会因时长结束而被释放。"""
    eng = AudioMixer(channels=1, fade_in=0.0, fade_out=0.01)
    assert eng.play_sound(Sound("loop", priority=3, looping=True), duration=0.1) is True
    for _ in range(10):
        eng.update(0.05)
    assert eng.is_playing("loop") is True
    assert eng.gains()["loop"] == 1.0


def test_oneshot_sound_finishes():
    """一次性音效播放完毕后淡出并释放声道。"""
    eng = AudioMixer(channels=1, fade_in=0.0, fade_out=0.02)
    assert eng.play_sound(Sound("shot", priority=3, looping=False), duration=0.1) is True
    eng.update(0.08)
    assert eng.is_playing("shot") is True
    eng.update(0.05)
    assert eng.is_playing("shot") is False
    assert eng.tail_gains() == {}
    # 释放出的声道可以再次使用
    assert eng.play_sound(Sound("shot2", priority=1), duration=0.1) is True


def test_pause_freezes_all_voices():
    """暂停冻结所有音效（含淡出尾音）；恢复后时间继续推进。"""
    eng = AudioMixer(channels=1, fade_in=0.1, fade_out=0.1)
    eng.play_sound(Sound("s", priority=5), duration=10.0)
    eng.update(0.05)
    frozen = eng.gains()["s"]
    eng.pause()
    for _ in range(5):
        eng.update(0.1)
    assert eng.gains()["s"] == frozen, "暂停期间包络必须冻结"
    eng.resume()
    eng.update(0.05)
    assert eng.gains()["s"] == 1.0, "恢复后淡入应继续到满音量"


def test_independent_volumes_realtime():
    """主音量/音效音量/音乐音量相互独立，修改实时生效。"""
    eng = AudioMixer(channels=2, fade_in=0.0, fade_out=0.0)
    eng.play_sound(Sound("sfx", priority=5), duration=10.0, category="sfx")
    eng.play_sound(Sound("mus", priority=5), duration=10.0, category="music")
    eng.update(0.01)
    eng.master_volume = 0.5
    assert eng.gains()["sfx"] == 0.5
    assert eng.gains()["mus"] == 0.5
    eng.sfx_volume = 0.4
    assert eng.gains()["sfx"] == 0.2
    assert eng.gains()["mus"] == 0.5, "音乐音量不应受音效音量影响"
    eng.music_volume = 0.6
    assert eng.gains()["mus"] == 0.3
    assert eng.gains()["sfx"] == 0.2, "音效音量不应受音乐音量影响"


def test_deterministic_preemption_decisions():
    """相同输入序列必须产生完全相同的播放决策。"""

    def run():
        eng = AudioMixer(channels=2, fade_in=0.0, fade_out=0.0)
        eng.play_sound(Sound("a", priority=3), duration=1.0)
        eng.play_sound(Sound("b", priority=7), duration=1.0)
        eng.play_sound(Sound("c", priority=5), duration=1.0)
        eng.update(0.05)
        eng.play_sound(Sound("d", priority=9), duration=1.0)
        eng.update(0.05)
        return (eng.play_log, eng.snapshot(), eng.events)

    first = run()
    for _ in range(5):
        assert run() == first


def test_channel_count_configurable():
    """声道数可配置：1声道严格串行，非法配置报错。"""
    eng = AudioMixer(channels=1)
    assert eng.play_sound(Sound("a", priority=5)) is True
    assert eng.play_sound(Sound("b", priority=5)) is False
    eng4 = AudioMixer(channels=4)
    for i in range(4):
        assert eng4.play_sound(Sound("x%d" % i, priority=5)) is True
    assert eng4.active_count() == 4
    assert eng4.play_sound(Sound("overflow", priority=5)) is False
    with pytest.raises(ValueError):
        AudioMixer(channels=0)
    with pytest.raises(ValueError):
        AudioMixer(channels=-1)
