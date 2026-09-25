"""音效混音引擎：优先级抢占、3D对数衰减、淡入淡出、循环、暂停恢复。

核心不变量：
- 声道满时高优先级抢占低优先级，低优先级不能抢占高优先级
- 相同优先级按FIFO（最早分配的先被淘汰）处理
- 3D音量按距离反比（对数级）衰减
- 播放/停止/被抢占均有淡入淡出，不突然切断
- 静音时不占用声道
- 主音量/音效音量/音乐音量独立且实时生效
- 播放决策确定性（无随机、无墙钟依赖）
- 声道数可配置
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Sound:
    sid: str
    priority: int = 0  # 越高越优先
    volume: float = 1.0
    looping: bool = False
    channel: Optional[int] = None
    position: Optional[tuple] = None  # (x,y,z) for 3D
    duration: float = 1.0  # 一次性播放时长（秒），循环音效忽略
    bus: str = "sfx"  # "sfx" 或 "music"，决定受哪个独立音量控制
    fade_in: float = 0.01  # 淡入时长（秒）
    fade_out: float = 0.05  # 淡出时长（秒）
    # 运行时状态（由引擎维护）
    state: str = "stopped"  # "playing" | "fading_out" | "stopped"
    elapsed: float = 0.0
    fade: float = 0.0  # 当前淡入淡出增益 0..1
    seq: int = -1  # 分配序号，用于同优先级FIFO


class AudioMixer:
    def __init__(self, channels: int = 8):
        if channels < 1:
            raise ValueError("channels must be >= 1")
        self.max_channels = channels
        self.channels: Dict[int, Optional[Sound]] = {i: None for i in range(channels)}
        self.master_volume: float = 1.0
        self.sfx_volume: float = 1.0
        self.music_volume: float = 1.0
        self.muted: bool = False
        self.paused: bool = False
        self.play_log: List[str] = []
        self._seq_counter: int = 0
        self._fading: List[Sound] = []  # 已让出声道、正在淡出的音效

    # ---- 声道分配 ----

    def _find_free_channel(self) -> Optional[int]:
        for i in sorted(self.channels):
            if self.channels[i] is None:
                return i
        return None

    def _find_preempt_victim(self, priority: int) -> Optional[int]:
        """在优先级低于 priority 的声道中，返回最低优先级且最早分配（FIFO）的声道。"""
        best_ch = None
        best_prio = None
        best_seq = None
        for i in sorted(self.channels):
            s = self.channels[i]
            if s is None or s.priority >= priority:
                continue
            if (best_prio is None or s.priority < best_prio
                    or (s.priority == best_prio and s.seq < best_seq)):
                best_ch, best_prio, best_seq = i, s.priority, s.seq
        return best_ch

    def _allocate(self, sound: Sound, ch: int) -> None:
        sound.channel = ch
        sound.state = "playing"
        sound.elapsed = 0.0
        sound.fade = 1.0 if sound.fade_in <= 0 else 0.0
        sound.seq = self._seq_counter
        self._seq_counter += 1
        self.channels[ch] = sound
        self.play_log.append(sound.sid)

    def _begin_fade_out(self, sound: Sound) -> None:
        """让音效淡出而不是突然切断；立即释放其声道。"""
        if sound.channel is not None and self.channels.get(sound.channel) is sound:
            self.channels[sound.channel] = None
        sound.channel = None
        sound.state = "fading_out"
        self._fading.append(sound)

    # ---- 播放控制 ----

    def play_sound(self, sound: Sound) -> bool:
        """播放音效。声道满时按优先级决定：仅当新音效优先级严格更高时抢占。"""
        if self.muted:
            return False
        ch = self._find_free_channel()
        if ch is None:
            ch = self._find_preempt_victim(sound.priority)
            if ch is None:
                return False  # 低优先级或同优先级不能抢占
            self._begin_fade_out(self.channels[ch])
        self._allocate(sound, ch)
        return True

    def stop_sound(self, sid: str) -> None:
        for s in list(self.channels.values()):
            if s and s.sid == sid:
                self._begin_fade_out(s)

    def update(self, dt: float) -> None:
        """推进播放状态（淡入淡出、一次性结束）。暂停/静音时不推进。"""
        if dt < 0:
            raise ValueError("dt must be >= 0")
        if self.paused or self.muted:
            return
        for s in list(self.channels.values()):
            if s is None:
                continue
            if s.fade < 1.0 and s.fade_in > 0:
                s.fade = min(1.0, s.fade + dt / s.fade_in)
            s.elapsed += dt
            if not s.looping and s.elapsed >= s.duration:
                self._begin_fade_out(s)
        for s in list(self._fading):
            if s.fade_out > 0:
                s.fade = max(0.0, s.fade - dt / s.fade_out)
            else:
                s.fade = 0.0
            if s.fade <= 0.0:
                s.state = "stopped"
                self._fading.remove(s)

    # ---- 3D 音量 ----

    def calculate_3d_volume(self, listener_pos: tuple, sound_pos: tuple) -> float:
        """3D音量：按距离反比衰减（对数级曲线），距离越远音量越小。"""
        dx = listener_pos[0] - sound_pos[0]
        dy = listener_pos[1] - sound_pos[1]
        dz = listener_pos[2] - sound_pos[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        ref = 1.0  # 参考距离：此距离内音量不衰减
        if dist <= ref:
            return 1.0
        return min(1.0, ref / dist)

    def effective_volume(self, sound: Sound, listener_pos: Optional[tuple] = None) -> float:
        """实时计算音效实际音量：自身音量 × 总线音量 × 主音量 × 淡入淡出 × 3D衰减。"""
        if self.muted:
            return 0.0
        bus = self.music_volume if sound.bus == "music" else self.sfx_volume
        v = sound.volume * bus * self.master_volume * sound.fade
        if sound.position is not None and listener_pos is not None:
            v *= self.calculate_3d_volume(listener_pos, sound.position)
        return v

    # ---- 全局状态 ----

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def set_muted(self, m: bool) -> None:
        self.muted = m
        if m:
            # 静音时释放所有声道（淡出，不突然切断）
            for s in list(self.channels.values()):
                if s is not None:
                    self._begin_fade_out(s)

    def set_master_volume(self, v: float) -> None:
        self.master_volume = max(0.0, min(1.0, v))

    def set_sfx_volume(self, v: float) -> None:
        self.sfx_volume = max(0.0, min(1.0, v))

    def set_music_volume(self, v: float) -> None:
        self.music_volume = max(0.0, min(1.0, v))

    def set_channel_count(self, n: int) -> None:
        """重新配置声道数；被裁掉的声道上的音效淡出。"""
        if n < 1:
            raise ValueError("channels must be >= 1")
        new_channels: Dict[int, Optional[Sound]] = {}
        for i in range(n):
            s = self.channels.get(i)
            if s is not None:
                s.channel = i
            new_channels[i] = s
        for i, s in self.channels.items():
            if i >= n and s is not None:
                self._begin_fade_out(s)
        self.max_channels = n
        self.channels = new_channels

    # ---- 查询 ----

    def active_count(self) -> int:
        return sum(1 for s in self.channels.values() if s is not None)

    def snapshot(self) -> dict:
        return {
            "channels": {i: (s.sid if s else None) for i, s in self.channels.items()},
            "active": self.active_count(),
            "muted": self.muted,
            "paused": self.paused,
        }
