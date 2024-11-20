# Python 3.11.3 버전의 slim 이미지 사용
FROM python:3.11.3-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 ffmpeg 설치
RUN apt-get update && \
    apt-get install -y ffmpeg git gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

# pip 최신 버전으로 업데이트
RUN python -m pip install --upgrade pip

# 필요한 Python 패키지를 설치하기 위해 requirements.txt 파일을 컨테이너에 복사
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# Flask 애플리케이션 파일을 컨테이너에 복사
COPY . .

# 환경 변수 설정
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5001

# 컨테이너 실행 시 Flask 애플리케이션 시작
CMD ["flask", "run"]
