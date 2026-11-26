import joblib
from fastapi import FastAPI, HTTPException
from .train import MODEL_PATH
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title = "House Price API")
model = joblib.load(MODEL_PATH)

# Lấy từ chính model đã train: tên các cột, và các giá trị chữ mà OneHotEncoder đã học
ALL_COLUMNS = set(model.feature_names_in_)
cat_pipe = model.named_steps["preprocess"].named_transformers_["cat"]
CATEGORIES = {col: list(values) for col, values in zip(cat_pipe.feature_names_in_, cat_pipe["onehot"].categories_)}


def validate(features):
    errors = []
    for col, value in features.items():
        if col not in ALL_COLUMNS:
            errors.append(f"Cột không tồn tại: {col}")
        elif value is None:
            continue
        elif col in CATEGORIES:
            if value not in CATEGORIES[col]:
                errors.append(f"{col}={value!r} không hợp lệ. Chọn một trong: {CATEGORIES[col]}")
        elif isinstance(value, str):
            errors.append(f"{col} phải là số, nhận được {value!r}")
    if errors:
        raise HTTPException(status_code=422, detail=errors)


@app.get('/')
def hello():
    return {"message": "House Price API đang chạy"}

class HouseInput(BaseModel):
    features: dict[str, float | str | None]

@app.post("/predict")
def predict(house: HouseInput):
    validate(house.features)
    X =pd.DataFrame([house.features]).reindex(columns = model.feature_names_in_)
    price = model.predict(X)[0]

    return {"predicted_price": round(float(price),2)}


# ===== Giao diện Gradio =====
import gradio as gr

# (chữ hiển thị, mã trong dữ liệu)
QUALITY = [("Xuất sắc", "Ex"), ("Tốt", "Gd"), ("Trung bình", "TA"), ("Kém", "Fa")]


def predict_ui(overall_qual, gr_liv_area, year_built, garage_cars, neighborhood,
               total_bsmt_sf, full_bath, exter_qual, kitchen_qual):
    house = HouseInput(features={
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "YearBuilt": year_built,
        "GarageCars": garage_cars,
        "Neighborhood": neighborhood,
        "TotalBsmtSF": total_bsmt_sf,
        "FullBath": full_bath,
        "ExterQual": exter_qual,
        "KitchenQual": kitchen_qual,
    })
    return predict(house)["predicted_price"]


demo = gr.Interface(
    fn=predict_ui,
    inputs=[
        gr.Slider(1, 10, value=6, step=1, label="Chất lượng tổng thể (1-10)"),
        gr.Number(value=1500, label="Diện tích sống (sqft)"),
        gr.Number(value=1975, label="Năm xây dựng"),
        gr.Slider(0, 4, value=2, step=1, label="Số chỗ đậu xe trong garage"),
        gr.Dropdown(CATEGORIES["Neighborhood"], value="NAmes", label="Khu vực"),
        gr.Number(value=990, minimum=0, label="Diện tích tầng hầm (sqft, 0 = không có hầm)"),
        gr.Slider(0, 3, value=2, step=1, label="Số phòng tắm đầy đủ"),
        gr.Dropdown(QUALITY, value="TA", label="Chất lượng ngoại thất"),
        gr.Dropdown(QUALITY, value="TA", label="Chất lượng bếp"),
    ],
    outputs=gr.Number(label="Giá dự đoán ($)"),
    title="Dự đoán giá nhà",
)

app = gr.mount_gradio_app(app, demo, path="/ui")