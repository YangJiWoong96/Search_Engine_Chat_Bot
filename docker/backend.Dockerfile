# docker/backend.Dockerfile

# 1. Base Image: Python 3.11 슬림 
FROM python:3.11-slim

# 2. 환경변수 
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /app

# 3. System Dependencies 설치 (Chrome & ChromeDriver)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    unzip \ 
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    fonts-liberation \
    && : # End of package installation

# Google Chrome 설치 
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list' && \
    apt-get update && \
    apt-get install -y google-chrome-stable --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# --- ChromeDriver 수동 설치 ---
# 설치된 Chrome 버전에 맞는 ChromeDriver 버전 명시 

ARG CHROME_VERSION=135.0.7049.114
ARG CHROMEDRIVER_URL=https://storage.googleapis.com/chrome-for-testing-public/${CHROME_VERSION}/linux64/chromedriver-linux64.zip
ARG CHROMEDRIVER_INSTALL_PATH=/usr/local/bin/chromedriver

RUN wget -q ${CHROMEDRIVER_URL} -O chromedriver_linux64.zip && \
    unzip chromedriver_linux64.zip -d /tmp/chromedriver_temp && \
    mv /tmp/chromedriver_temp/chromedriver-linux64/chromedriver ${CHROMEDRIVER_INSTALL_PATH} && \
    chmod +x ${CHROMEDRIVER_INSTALL_PATH} && \
    rm chromedriver_linux64.zip && \
    rm -rf /tmp/chromedriver_temp && \
    ${CHROMEDRIVER_INSTALL_PATH} --version

# 4. Working Directory 
WORKDIR /app

# 5. Requirements 설치 
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. docker 빌드시 Application Code 복사 
COPY ./api ./api
COPY ./config ./config
COPY ./core ./core
COPY ./search ./search
COPY ./utils ./utils
COPY .env .
COPY google-search-api.json .

# 7. Expose Port 
EXPOSE 8000

# 8. Run Application (FastAPI)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]