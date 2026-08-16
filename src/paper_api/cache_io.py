"""健壮的 JSON 缓存读写：规避 Windows 下 OneDrive / IDE 文件监视器造成的瞬态
PermissionError。

设计要点
--------
* ``atomic_write_json``：先写同目录临时文件，再 ``os.replace`` 原子替换目标；若目标被
  外部进程锁住，重试若干次（锁通常是瞬态的），仍失败则降级写到 ``<path>.bakretry``
  兄弟文件并继续，**绝不中断主流程**。
* ``load_json_cache``：优先读主文件，主文件缺失/损坏时回退读 ``.bakretry``，保证缓存
  可续跑、被杀重跑只补未完成的片段。
"""

from __future__ import annotations

import json
import os
import tempfile
import time


def atomic_write_json(path: str, data, retries: int = 12, pause: float = 1.0) -> bool:
    """原子写 JSON。成功返回 True；目标被持续锁住、降级到 ``.bakretry`` 也返回 True；
    连降级都失败才返回 False（主流程不应因此中断）。"""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)
    except Exception:  # noqa: BLE001 - 序列化失败直接放弃本轮落盘
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False

    for _ in range(retries):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(pause)
    # 目标仍被锁：降级到兄弟文件（不同文件名通常不会被同一把锁占用）
    try:
        os.replace(tmp, path + ".bakretry")
        return True
    except Exception:  # noqa: BLE001
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def load_json_cache(path: str) -> dict:
    """读取缓存；主文件不可用时回退 ``.bakretry``。返回 dict（无缓存则为空 dict）。"""
    for candidate in (path, path + ".bakretry"):
        if os.path.exists(candidate):
            try:
                with open(candidate, encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception:  # noqa: BLE001 - 损坏就忽略，当作无缓存
                continue
    return {}
