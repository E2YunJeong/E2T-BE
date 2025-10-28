# syntax=docker/dockerfile:1

############################
# (A) 모델 레이어
############################
FROM 2myungpil/model-base:latest AS models
# /opt/models 내에 best_conv2d.h5, shape_predictor_68_face_landmarks.dat 존재

############################
# (B) Build stage (Python 3.9)
############################
FROM python:3.9-slim AS build
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# dlib/opencv 컴파일 도구(빌드 스테이지만)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake python3-dev pkg-config \
    libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

# 파이썬 의존성
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 앱 소스
COPY . .

# 모델을 코드 경로(/app/models)로 복사 → settings.py(MODELS_DIR)와 일치
RUN mkdir -p /app/models && cp -a /opt/models/. /app/models/

############################
# (C) Runtime stage (가볍게)
############################
FROM python:3.9-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# 런타임 의존성만 남김
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

# 빌드 산출물 복사
COPY --from=build /usr/local/lib/python3.9 /usr/local/lib/python3.9
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /app /app

EXPOSE 8000

# ASGI 서버 (Daphne)
CMD ["daphne","-b","0.0.0.0","-p","8000","config.asgi:application"]
