# Python 3.11.3 버전의 slim 이미지 사용
FROM python:3.11.3-slim

# 작업 디렉토리 설정
WORKDIR /app

# MariaDB Connector/C 최신 버전 설치
RUN apt-get update && \
    apt-get install -y wget gnupg curl && \
    wget https://downloads.mariadb.com/MariaDB/mariadb_repo_setup && \
    chmod +x mariadb_repo_setup && \
    ./mariadb_repo_setup --mariadb-server-version=10.6 && \
    apt-get install -y mariadb-client libmariadb-dev-compat libmariadb-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 시스템 필수 패키지 설치
RUN apt-get update && \
    apt-get install -y ffmpeg git gcc build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# wait-for-it.sh 다운로드
RUN curl -o /usr/local/bin/wait-for-it.sh https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh && \
    chmod +x /usr/local/bin/wait-for-it.sh

# pip 최신 버전으로 업데이트
RUN python -m pip install --upgrade pip

# 필요한 Python 패키지를 설치하기 위해 requirements.txt 파일을 컨테이너에 복사
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# Flask 애플리케이션 파일을 컨테이너에 복사
COPY . .

# wait-for-it.sh 스크립트 복사
COPY wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

# Flask 애플리케이션 실행
CMD ["/wait-for-it.sh", "mariadb:3306", "--", "flask", "run"]
