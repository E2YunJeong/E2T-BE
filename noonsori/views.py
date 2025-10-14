# noonsori/views.py
import base64
import uuid

import cv2
import numpy as np
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import MultiPartParser, JSONParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .engine import BlinkToMorseEngine

SESSIONS = {}


def get_engine(session_id: str) -> BlinkToMorseEngine:
    eng = SESSIONS.get(session_id)
    if eng is None:
        eng = BlinkToMorseEngine()
        SESSIONS[session_id] = eng
    return eng


def _decode_image(file_or_b64) -> np.ndarray:
    if isinstance(file_or_b64, (bytes, bytearray)):
        data = np.frombuffer(file_or_b64, np.uint8)
    else:
        s = file_or_b64
        if "," in s:
            s = s.split(",", 1)[1]
        data = np.frombuffer(base64.b64decode(s), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


@method_decorator(csrf_exempt, name="dispatch")
class NewSession(APIView):
    def post(self, request):
        sid = uuid.uuid4().hex
        get_engine(sid)
        return Response({"session_id": sid})


@method_decorator(csrf_exempt, name="dispatch")
class ResetSession(APIView):
    def post(self, request):
        sid = request.data.get("session_id")
        if not sid:
            return Response({"error": "session_id required"}, status=400)
        eng = get_engine(sid)
        eng.reset()
        return Response({"ok": True})


@method_decorator(csrf_exempt, name="dispatch")
class FrameAPI(APIView):
    parser_classes = (MultiPartParser, JSONParser, FormParser)

    def post(self, request):
        sid = request.data.get("session_id")
        if not sid:
            return Response({"error": "session_id required"}, status=400)
        eng = get_engine(sid)

        if "frame" in request.FILES:
            frame_bgr = _decode_image(request.FILES["frame"].read())
        elif "frame_b64" in request.data:
            frame_bgr = _decode_image(request.data["frame_b64"])
        else:
            return Response({"error": "frame or frame_b64 required"}, status=400)

        if frame_bgr is None:
            return Response({"error": "failed to decode image"}, status=400)

        out = eng.process_frame(frame_bgr)
        return Response(out)
