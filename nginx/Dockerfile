# nginx 베이스 이미지 지정
FROM nginx:1.27.0-alpine

# 기존 설정 파일 지우고
RUN rm -f /etc/nginx/conf.d/default.conf
# 직접 설정한 파일로 기존 설정 파일 대체
COPY nginx.conf /etc/nginx/nginx.conf
COPY conf.d/*.conf /etc/nginx/conf.d/