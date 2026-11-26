# Ridge — House Price Prediction

Dự đoán giá nhà (dataset Kaggle *House Prices: Advanced Regression Techniques*) bằng **Ridge Regression**, **FastAPI** + **Gradio**, đóng gói bằng **Docker**.

## Cấu trúc

```
data/      dữ liệu train (.csv)
eda/       notebook phân tích dữ liệu
models/    model đã train (model.joblib)
src/house_price/
  preprocessing.py   tiền xử lý (impute, scale, one-hot)
  model.py           pipeline Ridge
  train.py           train + đánh giá (R2, RMSE) + lưu model
  api.py             FastAPI + Gradio UI
```

## Chạy local

```bash
pip install -r requirements.txt
cd src
python -m house_price.train                     # train, lưu models/model.joblib
uvicorn house_price.api:app --port 8000         # chạy API
```

## Chạy bằng Docker

```bash
docker build -t house-price .

# Train (ghi model ra thư mục models/ trên máy)
docker run --rm -v "$(pwd)/data:/data" -v "$(pwd)/models:/models" house-price python -m house_price.train

# Chạy API
docker run --rm -p 8000:8000 -v "$(pwd)/models:/models" house-price
```

## Sử dụng

| URL | Mô tả |
|---|---|
| http://localhost:8000/docs | Swagger UI |
| http://localhost:8000/ui | Giao diện Gradio |
| `POST /predict` | Dự đoán giá |


Cột thiếu sẽ được tự điền (impute). Tên cột / giá trị chữ không hợp lệ trả về lỗi 422 kèm danh sách giá trị đúng.
