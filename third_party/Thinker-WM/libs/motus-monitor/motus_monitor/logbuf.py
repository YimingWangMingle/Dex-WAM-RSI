"""内存 ring buffer 日志 handler（server 侧；供管理后台 /admin/api/logs 读取）。

serve 启动时挂到 root logger（见 cli.py），与 stderr / RotatingFileHandler 并存；
管理后台（admin.py）与主站同进程，共享同一份 buffer，API 直接从内存过滤返回。
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, List


class RingBufferHandler(logging.Handler):
    """把日志记录存进定长 deque（默认 2000 条），超出自动丢弃最旧记录。"""

    def __init__(self, capacity: int = 2000):
        super().__init__()
        self._records: Deque[Dict[str, Any]] = deque(maxlen=int(capacity))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        self._records.append(
            {
                "ts": record.created,
                "level": record.levelname,
                "levelno": record.levelno,
                "logger": record.name,
                "message": message,
            }
        )

    def snapshot(self) -> List[Dict[str, Any]]:
        """旧→新的记录副本（deque.append 本身线程安全，这里再借 handler 锁取一致快照）。"""
        self.acquire()
        try:
            return list(self._records)
        finally:
            self.release()
