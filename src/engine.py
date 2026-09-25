"""音效混音引擎：优先级抢占、3D对数衰减、淡入淡出、循环、暂停恢复。

核心不变量：
- 声道满时高优先级抢占低优先级
- 相同优先级FIFO
- 3D音量对数衰减
- 淡入淡出
- 静音不占声道
- 独立音量
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Sound:
    sid: str
    priority: int = 0  # 越高越优先
    volume: float = 1.0
    looping: bool = False
    channel: Optional[int] = None
    position: Optional[tuple] = None  # (x,y,z) for 3D


class AudioMixer:
    def __init__(self, channels: int = 8):
        self.max_channels = channels
        self.channels: Dict[int, Optional[Sound]] = {i: None for i in range(channels)}
        self.master_volume: float = 1.0
        self.sfx_volume: float = 1.0
        self.music_volume: float = 1.0
        self.muted: bool = False
        self.paused: bool = False
        self.play_log: List[str] = []

    def _find_free_channel(self) -> Optional[int]:
        for i, s in self.channels.items():
            if s is None:
                return i
        return None

    def _find_lowest_priority_channel(self) -> Optional[int]:
        lowest = None
        lowest_prio = float('inf')
        for i, s in self.channels.items():
            if s and s.priority < lowest_prio:
                lowest_prio = s.priority
                lowest = i
        return lowest

    def play_sound(self, sound: Sound) -> bool:
        """播放音效。BUG：声道满时用FIFO而不是优先级抢占。"""
        if self.muted:
            return False
        ch = self._find_free_channel()
        if ch is None:
            # BUG：直接返回False，不检查是否可以抢占低优先级
            # 正确：找到最低优先级声道，如果新音效优先级更高则抢占
            return False
        sound.channel = ch
        self.channels[ch] = sound
        self.play_log.append(sound.sid)
        return True

    def stop_sound(self, sid: str) -> None:
        for i, s in self.channels.items():
            if s and s.sid == sid:
                self.channels[i] = None

    def calculate_3d_volume(self, listener_pos: tuple, sound_pos: tuple) -> float:
        """3D音量。BUG：线性衰减而不是对数。"""
        dx = listener_pos[0] - sound_pos[0]
        dy = listener_pos[1] - sound_pos[1]
        dz = listener_pos[2] - sound_pos[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        # BUG：线性衰减，应该是对数/距离反比
        return max(0.0, 1.0 - dist / 100.0)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def set_muted(self, m: bool) -> None:
        self.muted = m

    def active_count(self) -> int:
        return sum(1 for s in self.channels.values() if s is not None)

    def snapshot(self) -> dict:
        return {
            "channels": {i: (s.sid if s else None) for i, s in self.channels.items()},
            "active": self.active_count(),
            "muted": self.muted,
            "paused": self.paused,
        }
