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


# ---- 补充断言：覆盖全部恢复的不变量 ----

def test_preempt_picks_lowest_priority_first():
    """声道满时抢占最低优先级声道，而不是FIFO挤掉先进来的。"""
    eng = _eng()
    eng.play_sound(Sound("first_mid", priority=5))
    eng.play_sound(Sound("second_low", priority=1))
    eng.play_sound(Sound("boss", priority=10))
    playing = [s.sid for s in eng.channels.values() if s]
    # 被挤掉的必须是优先级最低的 second_low，而不是先进的 first_mid
    assert "first_mid" in playing
    assert "boss" in playing
    assert "second_low" not in playing


def test_same_priority_fifo_victim_selection():
    """多个同优先级候选时，淘汰最早分配（FIFO）的那个。"""
    eng = _eng()
    a = Sound("old", priority=1)
    b = Sound("new", priority=1)
    eng.play_sound(a)
    eng.play_sound(b)
    eng.play_sound(Sound("boss", priority=9))
    playing = [s.sid for s in eng.channels.values() if s]
    assert "old" not in playing  # 同优先级中最早分配的先被淘汰
    assert "new" in playing
    assert "boss" in playing


def test_preempted_sound_fades_out_not_cut():
    """被抢占的旧音效必须淡出，不能突然切断。"""
    eng = _eng()
    low = Sound("low", priority=1, fade_in=0.0, fade_out=0.1)
    eng.play_sound(low)
    eng.play_sound(Sound("low2", priority=1))
    eng.play_sound(Sound("boss", priority=10))
    assert low.state == "fading_out"
    assert low.channel is None  # 声道已让给新音效
    assert low.fade > 0.0  # 仍有残余音量，未突断
    eng.update(0.2)  # 超过淡出时长
    assert low.state == "stopped"
    assert low.fade == 0.0


def test_fade_in_ramps_up():
    """正在播放的音效必须有淡入，音量从0爬升到1。"""
    eng = _eng()
    s = Sound("s", fade_in=0.1)
    eng.play_sound(s)
    assert s.fade == 0.0
    eng.update(0.05)
    assert 0.0 < s.fade < 1.0
    eng.update(0.05)
    assert s.fade == 1.0


def test_3d_volume_monotonic_and_logarithmic():
    """距离越远音量越小，且为对数/反比曲线而非线性。"""
    eng = _eng()
    d = [1, 2, 10, 50, 100]
    v = [eng.calculate_3d_volume((0, 0, 0), (x, 0, 0)) for x in d]
    assert all(v[i] > v[i + 1] for i in range(len(v) - 1))  # 严格单调递减
    assert v[0] == 1.0  # 参考距离内满音量
    assert v[3] < 0.1  # 线性衰减会给0.5，对数/反比应远小于它
    assert eng.calculate_3d_volume((0, 0, 0), (0, 0, 0)) == 1.0  # 零距离安全


def test_mute_releases_channels():
    """静音时不占用声道。"""
    eng = _eng()
    eng.play_sound(Sound("s1"))
    eng.play_sound(Sound("s2"))
    eng.set_muted(True)
    assert eng.active_count() == 0
    assert eng.play_sound(Sound("s3")) is False
    eng.set_muted(False)
    assert eng.play_sound(Sound("s4")) is True


def test_looping_sound_survives_oneshot_ends():
    """循环音效一直播放，一次性音效到时结束并释放声道。"""
    eng = _eng()
    loop = Sound("bgm", looping=True)
    one = Sound("shot", duration=0.5)
    eng.play_sound(loop)
    eng.play_sound(one)
    eng.update(1.0)
    playing = [s.sid for s in eng.channels.values() if s]
    assert "bgm" in playing
    assert "shot" not in playing
    assert eng.active_count() == 1


def test_pause_freezes_all_resume_continues():
    """暂停冻结所有正在播放的音效，恢复后继续。"""
    eng = _eng()
    s = Sound("s", duration=1.0, fade_in=0.2, fade_out=5.0)
    eng.play_sound(s)
    eng.update(0.1)
    eng.pause()
    eng.update(10.0)  # 暂停期间不推进
    assert s.elapsed == 0.1
    assert s.state == "playing"
    eng.resume()
    eng.update(1.0)  # 恢复后一次性音效结束
    assert s.state == "fading_out"


def test_independent_volumes_realtime():
    """主/音效/音乐音量独立且实时生效。"""
    eng = _eng()
    sfx = Sound("fx", bus="sfx", fade_in=0.0)
    mus = Sound("bgm", bus="music", fade_in=0.0)
    eng.play_sound(sfx)
    eng.play_sound(mus)
    eng.set_master_volume(0.5)
    eng.set_sfx_volume(0.2)
    eng.set_music_volume(1.0)
    assert eng.effective_volume(sfx) == pytest.approx(0.5 * 0.2)
    assert eng.effective_volume(mus) == pytest.approx(0.5 * 1.0)
    eng.set_sfx_volume(1.0)  # 实时生效，无需重启播放
    assert eng.effective_volume(sfx) == pytest.approx(0.5)
    eng.set_muted(True)
    assert eng.effective_volume(mus) == 0.0


def test_deterministic_playback_decisions():
    """相同输入产生完全相同的播放决策与声道布局。"""
    def run():
        eng = AudioMixer(channels=3)
        for sid, p in [("a", 3), ("b", 7), ("c", 1), ("d", 9), ("e", 5), ("f", 2)]:
            eng.play_sound(Sound(sid, priority=p))
        return eng.play_log, eng.snapshot()
    assert run() == run()
    log, snap = run()
    assert log == ["a", "b", "c", "d", "e"]  # f(2) 无法抢占最低优先级5
    assert snap["channels"] == {0: "e", 1: "b", 2: "d"}


def test_configurable_channel_count():
    """声道数可配置，且配置立即生效。"""
    eng = AudioMixer(channels=1)
    eng.play_sound(Sound("a"))
    assert eng.play_sound(Sound("b")) is False  # 单声道满
    eng.set_channel_count(3)
    assert eng.play_sound(Sound("b")) is True
    assert eng.play_sound(Sound("c")) is True
    assert eng.play_sound(Sound("d")) is False
    eng.set_channel_count(1)  # 收缩声道，多余的淡出
    assert eng.active_count() == 1


def test_no_channel_leak_after_preemption():
    """抢占后声道总数不变，旧音效不再占用声道。"""
    eng = _eng()
    low = Sound("low", priority=1)
    eng.play_sound(low)
    eng.play_sound(Sound("low2", priority=1))
    eng.play_sound(Sound("boss", priority=10))
    assert eng.active_count() == 2
    assert low.channel is None
    assert all(s is not low for s in eng.channels.values())
