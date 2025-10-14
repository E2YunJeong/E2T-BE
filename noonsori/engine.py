import os
import statistics
import time
from collections import deque
from typing import Optional

import cv2
import dlib
import numpy as np
from django.conf import settings
from tensorflow.keras.models import load_model

from .morse_decoder import decode_morse_to_korean
from .unicode import join_jamos

# -------------------- 전역(프로세스) 캐시 --------------------
_MODEL = None
_PREDICTOR = None
_DETECTOR = dlib.get_frontal_face_detector()


def get_model():
    global _MODEL
    if _MODEL is None:
        path = settings.BLINK_MODEL_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[BLINK_MODEL_PATH] not found: {path}")
        _MODEL = load_model(path)
    return _MODEL


def get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        path = settings.DLIB_PREDICTOR_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[DLIB_PREDICTOR_PATH] not found: {path}")
        _PREDICTOR = dlib.shape_predictor(path)
    return _PREDICTOR


class BlinkToMorseEngine:
    """
    프레임 1장을 입력받아 상태를 갱신하고, '이벤트'만 돌려줍니다.
    반환 형식: {"events": [ ... ]}
      - {"type":"calib", "base":0.36, "close":0.288, "open":0.331}
      - {"type":"start"}
      - {"type":"morse", "symbol":"."|"-", "morse":"-.-"}
      - {"type":"space", "morse":"-.- "}
      - {"type":"end", "morse":"-.- .", "text":"안녕"}
    """

    def __init__(
            self,
            *,
            # 추론/상태
            smooth_n: int = 3,
            close_thr: float = 0.65,
            open_thr: float = 0.35,
            ear_close: float = 0.21,
            ear_open: float = 0.24,
            consec_k: int = 2,
            k_open: int = 2,
            closed_hold: float = 0.10,  # 최소 닫힘 유지(초) — 너무 짧은 노이즈 무시
            eye_size: int = 80,
            start_end_sec: float = 3.0,  # 3초 감기 → 세션 토글
            dash_sec: float = 0.50,  # '-' 기준
            letter_gap: float = 1.5,
            # 안정화 가드
            min_dot: float = 0.10,  # 닫힘이 이보다 짧으면 입력 무시
            min_open_gap: float = 0.12,  # 입력 후 다음 입력까지 최소 열린 시간
            min_active_guard: float = 0.70,  # START 직후 즉시 END 방지
            # 성능 최적화
            redetect_every: int = 5,  # N프레임마다 전체 재검출, 그 사이엔 추적
            closed_skip_mod: int = 2,  # CLOSED 상태일 때 CNN/EAR N프레임에 1회만 갱신
            DEBUG: bool = False,
            DEBUG_EVERY: int = 10,
    ):
        # 모델/검출기
        self.model = get_model()
        self.predictor = get_predictor()
        self.detector = _DETECTOR

        # 하이퍼파라미터
        self.SM_N = smooth_n
        self.CLOSE_THR = close_thr
        self.OPEN_THR = open_thr
        self.EAR_CLOSE = ear_close
        self.EAR_OPEN = ear_open
        self.K_CLOSE = consec_k
        self.K_OPEN = k_open
        self.CLOSED_HOLD = closed_hold
        self.EYE_SIZE = eye_size
        self.START_END_SEC = start_end_sec
        self.DASH_SEC = dash_sec
        self.LETTER_GAP = letter_gap

        self.MIN_DOT = min_dot
        self.MIN_OPEN_GAP = min_open_gap
        self.MIN_ACTIVE_GUARD = min_active_guard

        # 성능
        self.redetect_every = redetect_every
        self.closed_skip_mod = closed_skip_mod

        self.DEBUG = DEBUG
        self.DEBUG_EVERY = DEBUG_EVERY
        self._frame_count = 0

        # 동적 캘리브
        self.auto_calib = True
        self.calib_needed = True
        self.calib_buffer = []
        self.calib_frames = 30
        self.ear_base = None
        self.dyn_ear_close = ear_close
        self.dyn_ear_open = ear_open

        # 추적기 관련
        self.tracker: Optional[dlib.correlation_tracker] = None
        self._track_countdown = 0
        self._last_rect: Optional[dlib.rectangle] = None

        # 최근 값(스킵 시 재사용)
        self._last_p_sm = 0.0
        self._last_ear_sm = 0.0
        self._closed_skip_i = 0

        self.reset()

    # -------------------- 상태 초기화 --------------------
    def reset(self):
        self.prob_buf = deque(maxlen=self.SM_N)
        self.ear_buf = deque(maxlen=self.SM_N)

        self.state = "OPEN"  # "OPEN" or "CLOSED"
        self.consec_close = 0
        self.consec_open = 0

        self.conversion_active = False
        self.conversion_ended = False
        self.closed_started_at = None
        self.open_started_at = None
        self._early_open_ts = None
        self._session_started_at = None
        self._symbols_since_start = 0

        self.output_str = ""

        # 캘리브 상태도 리셋
        self.calib_needed = True
        self.calib_buffer.clear()
        self.ear_base = None
        self.dyn_ear_close = self.EAR_CLOSE
        self.dyn_ear_open = self.EAR_OPEN

        # 추적 리셋
        self.tracker = None
        self._track_countdown = 0
        self._last_rect = None

        # 캐시 리셋
        self._last_p_sm = 0.0
        self._last_ear_sm = 0.0
        self._closed_skip_i = 0

        self._frame_count = 0

    # -------------------- 유틸 --------------------
    @staticmethod
    def _eye_pts(shape, idxs):
        return np.array([(shape.part(i).x, shape.part(i).y) for i in idxs])

    @staticmethod
    def _calc_ear(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3]) + 1e-6
        return (A + B) / (2.0 * C)

    def _crop_eye_box(self, rgb, shape, size):
        left = self._eye_pts(shape, range(36, 42))
        right = self._eye_pts(shape, range(42, 48))
        both = np.concatenate([left, right], axis=0)
        x, y, w, h = cv2.boundingRect(both)
        cx, cy = x + w // 2, y + h // 2
        half = size // 2
        crop = rgb[max(cy - half, 0):cy + half, max(cx - half, 0):cx + half]
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        ch, cw = crop.shape[:2]
        canvas[:ch, :cw] = crop
        return canvas

    # -------------------- 검출/추적 --------------------
    def _detect_or_track(self, rgb) -> Optional[dlib.rectangle]:
        """N프레임마다 전체 검출, 그 사이엔 correlation tracker로 박스 추적."""
        h, w = rgb.shape[:2]

        # 재검출 타이밍 또는 추적기 없음 → 전체 검출
        need_redetect = (self.tracker is None) or (self._track_countdown <= 0)
        if need_redetect:
            dets = self.detector(rgb, 1)
            if len(dets) == 0:
                self.tracker = None
                self._last_rect = None
                return None
            rect = dets[0]
            self.tracker = dlib.correlation_tracker()
            self.tracker.start_track(rgb, rect)
            self._track_countdown = self.redetect_every
            self._last_rect = rect
            return rect

        # 추적만 수행
        self.tracker.update(rgb)
        pos = self.tracker.get_position()
        l, t, r, b = int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())

        # 경계 보정 + 최소 크기 보장
        l = max(0, l);
        t = max(0, t);
        r = min(w - 1, r);
        b = min(h - 1, b)
        if (r - l) < 20 or (b - t) < 20:
            # 너무 작으면 다음 루프에서 재검출
            self._track_countdown = 0
            return None

        rect = dlib.rectangle(l, t, r, b)
        self._last_rect = rect
        self._track_countdown -= 1
        return rect

    # -------------------- 캘리브레이션 --------------------
    def _update_dynamic_ear_thresholds(self):
        """OPEN 상태 EAR 샘플로 동적 임계 계산."""
        if len(self.calib_buffer) < self.calib_frames:
            return None

        base = statistics.median(self.calib_buffer)
        est_close = min(base * 0.86, base - 0.04)
        est_open = min(base * 0.93, base - 0.02)

        close = max(self.EAR_CLOSE, est_close)
        open_ = max(self.EAR_OPEN, est_open)

        if open_ <= close + 0.01:
            open_ = close + 0.015

        self.ear_base = base
        self.dyn_ear_close = close
        self.dyn_ear_open = open_
        self.calib_needed = False
        self.calib_buffer.clear()

        return {"base": base, "close": close, "open": open_}

    # -------------------- 메인 처리 --------------------
    def process_frame(self, frame_bgr, now_ts=None):
        """
        한 프레임 처리 → {"events":[...]} 반환.
        이벤트가 없으면 빈 배열이 옵니다.
        """
        events = []
        now = now_ts if now_ts is not None else time.time()
        self._frame_count += 1

        # --- 얼굴 검출/추적 ---
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rect = self._detect_or_track(rgb)
        if rect is None:
            return {"events": events}

        # --- 랜드마크 ---
        shape = self.predictor(rgb, rect)

        # --- 성능 최적화: CLOSED 상태에서 N-1 프레임은 CNN/EAR 갱신 스킵 ---
        do_heavy = True
        if self.state == "CLOSED":
            self._closed_skip_i = (self._closed_skip_i + 1) % self.closed_skip_mod
            do_heavy = (self._closed_skip_i == 0)

        if do_heavy:
            eye_img = self._crop_eye_box(rgb, shape, self.EYE_SIZE).astype(np.float32) / 255.0
            X = np.expand_dims(eye_img, axis=0)
            # predict() 대신 직접 호출이 보통 더 가벼움
            p = float(self.model(X, training=False)[0, 0])

            left = self._eye_pts(shape, range(36, 42))
            right = self._eye_pts(shape, range(42, 48))
            ear = (self._calc_ear(left) + self._calc_ear(right)) / 2.0

            self.prob_buf.append(p)
            p_sm = sum(self.prob_buf) / len(self.prob_buf)

            self.ear_buf.append(ear)
            ear_sm = sum(self.ear_buf) / len(self.ear_buf)

            self._last_p_sm = p_sm
            self._last_ear_sm = ear_sm
        else:
            # 스킵: 이전 스무딩값 재사용
            p_sm = self._last_p_sm
            ear_sm = self._last_ear_sm

        # --- 자동 캘리브레이션(OPEN에서만 샘플링) ---
        if self.auto_calib and self.state == "OPEN":
            self.calib_buffer.append(ear_sm)
            if self.calib_needed:
                calib_info = self._update_dynamic_ear_thresholds()
                if calib_info:
                    events.append({
                        "type": "calib",
                        "base": round(calib_info["base"], 3),
                        "close": round(calib_info["close"], 3),
                        "open": round(calib_info["open"], 3),
                    })

        # --- 동적 임계 (히스테리시스) ---
        closing = (p_sm > self.CLOSE_THR) or (ear_sm < self.dyn_ear_close)
        opening = (p_sm < self.OPEN_THR) and (ear_sm > self.dyn_ear_open)

        # --- 상태머신 ---
        if self.state == "OPEN":
            if self.open_started_at is None:
                self.open_started_at = now

            if closing:
                self.consec_close += 1
                if self.consec_close >= self.K_CLOSE:
                    self.state = "CLOSED"
                    self.closed_started_at = now
                    self.consec_close = 0

                    # 글자 간 공백: 충분히 길게 열려 있었으면 space 이벤트
                    if self.conversion_active and self.open_started_at is not None:
                        if (now - self.open_started_at) >= max(self.LETTER_GAP, self.MIN_OPEN_GAP + 0.3):
                            self.output_str += " "
                            events.append({"type": "space", "morse": self.output_str})
                    self.open_started_at = None
        else:
            # 너무 짧은 닫힘은 무시
            if (now - (self.closed_started_at or now)) < self.CLOSED_HOLD:
                self.consec_open = 0
                self._early_open_ts = None
            else:
                if opening:
                    if self._early_open_ts is None:
                        self._early_open_ts = now  # K_OPEN 지연 보정
                    self.consec_open += 1
                    if self.consec_open >= self.K_OPEN:
                        closed_dur = max(0.0, (self._early_open_ts or now) - (self.closed_started_at or now))
                        self.state = "OPEN"
                        self.consec_open = 0
                        self.open_started_at = now
                        self._early_open_ts = None

                        # 세션 토글 or 입력
                        if closed_dur >= self.START_END_SEC:
                            if not self.conversion_active:
                                # START: 캘리브 완료 후 시작
                                if not self.calib_needed:
                                    self.conversion_active = True
                                    self.conversion_ended = False
                                    self.output_str = ""
                                    self._session_started_at = now
                                    self._symbols_since_start = 0
                                    events.append({"type": "start"})
                            else:
                                # END: 시작 직후 즉시 종료 방지
                                if (self._symbols_since_start > 0) or (
                                        (self._session_started_at is not None)
                                        and (now - self._session_started_at >= self.MIN_ACTIVE_GUARD)
                                ):
                                    self.conversion_active = False
                                    self.conversion_ended = True
                                    morse_str = self.output_str
                                    text = decode_morse_to_korean(morse_str)
                                    final_text = join_jamos(text)
                                    events.append({"type": "end", "morse": morse_str, "text": final_text or ""})
                        else:
                            if self.conversion_active and closed_dur >= self.MIN_DOT:
                                symbol = "-" if closed_dur >= self.DASH_SEC else "."
                                self.output_str += symbol
                                self._symbols_since_start += 1
                                events.append({"type": "morse", "symbol": symbol, "morse": self.output_str})
                else:
                    self.consec_open = 0
                    self._early_open_ts = None

        return {"events": events}
