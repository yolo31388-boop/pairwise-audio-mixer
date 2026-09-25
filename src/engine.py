"""音效混音引擎：优先级抢占、3D对数衰减、淡入淡出、循环、暂停恢复。

核心不变量：
- 声道满时由优先级决定新音效是否播放
- 高优先级可抢占低优先级声道；低优先级不能抢占高优先级
- 相同优先级不互相抢占；多个可抢占目标时按FIFO（最早开始者淘汰）
- 抢占/停止的旧音效走淡出尾音（tail），不突然切断，且立即释放声道
- 3D音量按距离反比（对数感知）衰减，距离越远音量越小
- 静音时不播放、不占用声道
- 支持循环与一次性播放；暂停/恢复作用于全部音效（含淡出尾音）
- 主音量/音效音量/音乐音量相互独立，实时生效
- 相同输入产生相同播放决策（确定性，无随机/系统时钟依赖）
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

FADE_IN = "fade_in"
PLAYING = "playing"
FADE_OUT = "fade_out"


@dataclass
class Sound:
    sid: str
    priority: int = 0  # 越高越优先
    volume: float = 1.0
    looping: bool = False
    channel: Optional[int] = None
    position: Optional[tuple] = None  # (x,y,z) for 3D


class _Voice:
    """一个正在播放（或淡出中）的音效实例。"""

    __slots__ = (
        "sound", "seq", "channel", "state", "envelope", "elapsed",
        "duration", "fade_in", "fade_out", "category", "fade_from", "fade_t",
    )

    def __init__(
        self,
        sound: Sound,
        seq: int,
        channel: int,
        duration: float,
        fade_in: float,
        fade_out: float,
        category: str,
    ) -> None:
        self.sound = sound
        self.seq = seq
        self.channel: Optional[int] = channel
        self.state: str = FADE_IN
        self.envelope: float = 0.0
        self.elapsed: float = 0.0
        self.duration = duration
        self.fade_in = fade_in
        self.fade_out = fade_out
        self.category = category
        self.fade_from: float = 1.0
        self.fade_t: float = 0.0

    @property
    def sid(self) -> str:
        return self.sound.sid


class AudioMixer:
    def __init__(
        self,
        channels: int = 8,
        fade_in: float = 0.05,
        fade_out: float = 0.05,
        ref_distance: float = 10.0,
        max_distance: float = 100.0,
    ) -> None:
        if not isinstance(channels, int) or channels < 1:
            raise ValueError("channels 必须是正整数")
        self.max_channels = channels
        self.channels: Dict[int, Optional[_Voice]] = {i: None for i in range(channels)}
        # 被抢占/停止但仍在淡出的尾音，不再占用声道
        self.tails: List[_Voice] = []
        self.master_volume: float = 1.0
        self.sfx_volume: float = 1.0
        self.music_volume: float = 1.0
        self.muted: bool = False
        self.paused: bool = False
        self.listener_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.ref_distance = float(ref_distance)
        self.max_distance = float(max_distance)
        self.default_fade_in = float(fade_in)
        self.default_fade_out = float(fade_out)
        self.play_log: List[str] = []
        self.events: List[Tuple[str, str]] = []
        self._seq: int = 0

    # ------------------------------------------------------------------
    # 声道分配
    # ------------------------------------------------------------------
    def _find_free_channel(self) -> Optional[int]:
        for i, voice in self.channels.items():
            if voice is None:
                return i
        return None

    def _find_preempt_victim(self, priority: int) -> Optional[int]:
        """返回可抢占声道：优先级最低者；同优先级取最早开始（FIFO）。

        只有严格更低的优先级才会被抢占；相同优先级不抢占。
        """
        victim: Optional[int] = None
        victim_key: Optional[Tuple[int, int]] = None
        for i, voice in self.channels.items():
            if voice is None:
                continue
            if voice.sound.priority >= priority:
                continue
            key = (voice.sound.priority, voice.seq)
            if victim_key is None or key < victim_key:
                victim_key = key
                victim = i
        return victim

    def allocate_channel(self, sound: Sound) -> Optional[int]:
        """为音效决定声道：优先空闲声道；满了按优先级/FIFO抢占；否则拒绝。"""
        free = self._find_free_channel()
        if free is not None:
            return free
        return self._find_preempt_victim(sound.priority)

    # ------------------------------------------------------------------
    # 播放控制
    # ------------------------------------------------------------------
    def play_sound(
        self,
        sound: Sound,
        duration: float = 1.0,
        category: str = "sfx",
        fade_in: Optional[float] = None,
        fade_out: Optional[float] = None,
    ) -> bool:
        """播放音效。返回是否获得声道。

        静音时直接拒绝且不占用声道。声道满时，仅当新音效优先级严格高于
        某个在播音效时才能抢占；同优先级按FIFO拒绝。被抢占者进入淡出尾音。
        """
        if self.muted:
            self.events.append(("reject_muted", sound.sid))
            return False

        ch = self._find_free_channel()
        preempted: Optional[_Voice] = None
        if ch is None:
            victim = self._find_preempt_victim(sound.priority)
            if victim is None:
                self.events.append(("reject", sound.sid))
                return False
            preempted = self.channels[victim]
            ch = victim

        if preempted is not None:
            self._begin_fade_out(preempted)
            self.events.append(("preempt", preempted.sound.sid))

        self._seq += 1
        voice = _Voice(
            sound=sound,
            seq=self._seq,
            channel=ch,
            duration=max(0.0, float(duration)),
            fade_in=self.default_fade_in if fade_in is None else max(0.0, float(fade_in)),
            fade_out=self.default_fade_out if fade_out is None else max(0.0, float(fade_out)),
            category=category,
        )
        sound.channel = ch
        self.channels[ch] = voice
        self.play_log.append(sound.sid)
        self.events.append(("play", sound.sid))
        return True

    def stop_sound(self, sid: str) -> None:
        """停止音效：声道立即释放，声音走淡出尾音，不突然切断。"""
        for i, voice in list(self.channels.items()):
            if voice is not None and voice.sound.sid == sid:
                self.channels[i] = None
                self._begin_fade_out(voice)

    def _begin_fade_out(self, voice: _Voice) -> None:
        if voice.state == FADE_OUT:
            return
        voice.state = FADE_OUT
        voice.channel = None
        voice.sound.channel = None
        voice.fade_from = voice.envelope
        voice.fade_t = 0.0
        if self.muted:
            # 静音状态下不保留尾音，取消静音后也不会继续发声
            self.events.append(("stop", voice.sound.sid))
            return
        self.tails.append(voice)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def set_muted(self, m: bool) -> None:
        """切换静音。静音时所有声道立即释放（尾音淡出，实际输出为静音）。"""
        self.muted = bool(m)
        if self.muted:
            for i, voice in list(self.channels.items()):
                if voice is not None:
                    self.channels[i] = None
                    voice.sound.channel = None
                    self.events.append(("stop", voice.sound.sid))
            for voice in self.tails:
                voice.sound.channel = None
                self.events.append(("stop", voice.sound.sid))
            self.tails.clear()

    def set_listener_position(self, pos: tuple) -> None:
        self.listener_position = (float(pos[0]), float(pos[1]), float(pos[2]))

    # ------------------------------------------------------------------
    # 音量
    # ------------------------------------------------------------------
    def calculate_3d_volume(
        self,
        listener_pos: tuple,
        sound_pos: tuple,
        max_distance: Optional[float] = None,
    ) -> float:
        """3D音量：距离反比衰减（参考距离模型），单调递减，范围[0,1]。

        gain = ref_distance / (ref_distance + distance)，比线性衰减更陡，
        符合对数感知的听感；超过最大距离后衰减为0。
        """
        dx = listener_pos[0] - sound_pos[0]
        dy = listener_pos[1] - sound_pos[1]
        dz = listener_pos[2] - sound_pos[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        gain = self.ref_distance / (self.ref_distance + dist)
        if max_distance is None:
            max_distance = self.max_distance
        if max_distance is not None and dist >= float(max_distance):
            gain = 0.0
        return max(0.0, min(1.0, gain))

    def _category_volume(self, category: str) -> float:
        if category == "music":
            return self.music_volume
        return self.sfx_volume

    def _base_volume(self, voice: _Voice) -> float:
        gain = voice.sound.volume
        if voice.sound.position is not None:
            gain *= self.calculate_3d_volume(
                self.listener_position, voice.sound.position
            )
        return gain

    def effective_volume(self, voice: _Voice) -> float:
        if self.muted:
            return 0.0
        return (
            voice.envelope
            * self._base_volume(voice)
            * self.master_volume
            * self._category_volume(voice.category)
        )

    # ------------------------------------------------------------------
    # 时间推进（确定性：由调用方显式传入dt，不读取系统时钟）
    # ------------------------------------------------------------------
    def update(self, dt: float) -> None:
        dt = max(0.0, float(dt))

        # 静音状态下声道必须为空：任何残留全部转入淡出尾音
        if self.muted:
            for i, voice in list(self.channels.items()):
                if voice is not None:
                    self.channels[i] = None
                    self._begin_fade_out(voice)
        elif self.paused or dt == 0.0:
            return

        finished: List[int] = []
        for i, voice in self.channels.items():
            if voice is None:
                continue
            voice.elapsed += dt
            if not voice.sound.looping and voice.duration > 0.0:
                # 一次性音效：按时间点计算淡出，duration处增益归零
                fade_start = max(0.0, voice.duration - voice.fade_out)
                if voice.elapsed >= voice.duration:
                    voice.envelope = 0.0
                    finished.append(i)
                    continue
                if voice.elapsed >= fade_start:
                    voice.state = FADE_OUT
                    if voice.fade_out > 0.0:
                        voice.envelope = max(
                            0.0, (voice.duration - voice.elapsed) / voice.fade_out
                        )
                    else:
                        voice.envelope = 1.0
                    continue
            # 淡入或稳定播放
            if voice.state == FADE_IN:
                if voice.fade_in > 0.0:
                    voice.envelope = min(1.0, voice.elapsed / voice.fade_in)
                else:
                    voice.envelope = 1.0
                if voice.envelope >= 1.0:
                    voice.state = PLAYING
            else:
                voice.envelope = 1.0
                if voice.sound.looping and voice.duration > 0.0:
                    voice.elapsed %= voice.duration

        for i in finished:
            voice = self.channels[i]
            self.channels[i] = None
            if voice is not None:
                voice.sound.channel = None
                self.events.append(("finish", voice.sound.sid))

        dead: List[_Voice] = []
        for voice in self.tails:
            voice.fade_t += dt
            if voice.fade_out > 0.0:
                voice.envelope = max(
                    0.0, voice.fade_from - voice.fade_t / voice.fade_out
                )
            else:
                voice.envelope = 0.0
            if voice.envelope <= 0.0:
                voice.envelope = 0.0
                dead.append(voice)
        for voice in dead:
            voice.sound.channel = None
            self.tails.remove(voice)
            self.events.append(("stop", voice.sound.sid))

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    def active_count(self) -> int:
        return sum(1 for voice in self.channels.values() if voice is not None)

    def gains(self) -> Dict[str, float]:
        """当前占用声道的音效 sid -> 实际输出增益（含包络与各类音量）。"""
        return {
            voice.sound.sid: self.effective_volume(voice)
            for voice in self.channels.values()
            if voice is not None
        }

    def tail_gains(self) -> Dict[str, float]:
        """淡出尾音 sid -> 实际输出增益。"""
        return {voice.sound.sid: self.effective_volume(voice) for voice in self.tails}

    def is_playing(self, sid: str) -> bool:
        return any(
            voice is not None and voice.sound.sid == sid
            for voice in self.channels.values()
        )

    def snapshot(self) -> dict:
        return {
            "channels": {
                i: (voice.sound.sid if voice else None)
                for i, voice in self.channels.items()
            },
            "active": self.active_count(),
            "muted": self.muted,
            "paused": self.paused,
            "tails": [voice.sound.sid for voice in self.tails],
            "gains": self.gains(),
        }
