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
