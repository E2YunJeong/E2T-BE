import asyncio
import base64
import traceback

import cv2
import numpy as np
from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer

from .engine import BlinkToMorseEngine

# 세션ID → 엔진 (메모리). 배포 시 Redis dump/load 추가 권장.
ENGINES = {}


def _decode_image_from_b64(data_url: str):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(data_url), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@sync_to_async
def _get_engine_by_sid(sid: str):
    eng = ENGINES.get(sid)
    if eng is None:
        eng = BlinkToMorseEngine()
        ENGINES[sid] = eng
    return eng


@sync_to_async
def _reset_engine(eng: BlinkToMorseEngine):
    eng.reset()


@sync_to_async
def _process_frame(eng: BlinkToMorseEngine, frame, ts=None):
    return eng.process_frame(frame, now_ts=ts)


class BlinkConsumer(AsyncJsonWebsocketConsumer):
    """
    - 클라이언트가 보내는 frame 메시지는 코얼레싱(가장 최신 1장 유지)
    - 백그라운드 루프가 주기적으로 최신 프레임만 처리 → 지연/백로그 방지
    """
    PROCESS_HZ = 25  # 처리 루프 빈도 (초당 25회 ≈ 40ms)
    PROCESS_INTERVAL = 1.0 / PROCESS_HZ

    async def connect(self):
        self.sid = self.scope["url_route"]["kwargs"]["session_id"]
        await self.accept()
        await _get_engine_by_sid(self.sid)

        # 최신 프레임 버퍼(코얼레싱)
        self._latest_b64 = None
        self._latest_ts = None

        # 백그라운드 처리 루프 시작
        self._runner = asyncio.create_task(self._process_loop())

        await self.send_json({"type": "ready"})

    async def disconnect(self, code):
        if hasattr(self, "_runner") and self._runner:
            self._runner.cancel()
            try:
                await self._runner
            except asyncio.CancelledError:
                pass
        # 엔진을 계속 유지하고 싶으면 pass
        # 메모리 회수하려면 ↓
        # ENGINES.pop(self.sid, None)

    async def receive_json(self, content, **kwargs):
        try:
            t = content.get("type")

            if t == "reset":
                eng = await _get_engine_by_sid(self.sid)
                await _reset_engine(eng)
                await self.send_json({"type": "reset_ok"})
                return

            if t == "frame":
                # 최신 프레임만 유지(코얼레싱)
                self._latest_b64 = content.get("data")
                self._latest_ts = content.get("ts")
                return

            await self.send_json({"type": "error", "message": f"unknown type: {t}"})
        except Exception as e:
            await self.send_json({"type": "error", "message": str(e)})
            traceback.print_exc()

    async def _process_loop(self):
        """가장 최신 프레임만 주기적으로 처리."""
        eng = await _get_engine_by_sid(self.sid)
        try:
            while True:
                await asyncio.sleep(self.PROCESS_INTERVAL)
                if not self._latest_b64:
                    continue

                # 최신 프레임 스냅샷 후 버퍼 비우기
                b64 = self._latest_b64
                ts = self._latest_ts
                self._latest_b64 = None
                self._latest_ts = None

                frame = _decode_image_from_b64(b64)
                if frame is None:
                    continue

                out = await _process_frame(eng, frame, ts)
                for ev in out.get("events", []):
                    await self.send_json(ev)
        except asyncio.CancelledError:
            # 정상 종료
            return
        except Exception as e:
            await self.send_json({"type": "error", "message": str(e)})
            traceback.print_exc()
