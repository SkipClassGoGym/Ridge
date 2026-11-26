# 1. Image gốc: Python bản nhẹ
FROM python:3.14-slim

# 2. Thư mục làm việc trong container
WORKDIR /src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY src/ .


ENV DATA_PATH=/data/train-house-prices-advanced-regression-techniques.csv
ENV MODEL_PATH=/models/model.joblib

EXPOSE 8000

# 6. Lệnh chạy khi start container: mặc định chạy API + giao diện
#    (muốn train: docker run ... house-price python -m house_price.train)
CMD ["uvicorn", "house_price.api:app", "--host", "0.0.0.0", "--port", "8000"]
