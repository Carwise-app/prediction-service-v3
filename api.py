from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="CARWISE API", version="1.0")

model = joblib.load("carwise_hgb_model_20250518_2026.pkl")
r2 = joblib.load("hgb_model_r2_20250518_2026.pkl")
mae = joblib.load("hgb_model_mae_20250518_2026.pkl")

class CarInput(BaseModel):
    Marka: str
    Seri: str
    Model: str
    Yıl: int
    Kilometre: float
    Motor_Hacmi: float
    Motor_Gücü: float
    Tramer: float
    Boyalı_sayısı: int
    Değişen_sayısı: int
    Orjinal_sayısı: int
    Vites_Tipi: str
    Yakıt_Tipi: str
    Kasa_Tipi: str
    Renk: str

@app.post("/predict")
def predict(car: CarInput):
    try:
        Vites_Yakıt = f"{car.Vites_Tipi}_{car.Yakıt_Tipi}"
        Araç_Yaşı = 2025 - car.Yıl

        input_df = pd.DataFrame([{
            "Marka": car.Marka,
            "Seri": car.Seri,
            "Model": car.Model,
            "Yıl": car.Yıl,
            "Kilometre": car.Kilometre,
            "Motor Hacmi": car.Motor_Hacmi,
            "Motor Gücü": car.Motor_Gücü,
            "Tramer": car.Tramer,
            "Boyalı_sayısı": car.Boyalı_sayısı,
            "Değişen_sayısı": car.Değişen_sayısı,
            "Orjinal_sayısı": car.Orjinal_sayısı,
            "Araç_Yaşı": Araç_Yaşı,
            "Vites Tipi": car.Vites_Tipi,
            "Yakıt Tipi": car.Yakıt_Tipi,
            "Kasa Tipi": car.Kasa_Tipi,
            "Renk": car.Renk,
            "Vites_Yakıt": Vites_Yakıt
        }])

        log_pred = model.predict(input_df)[0]
        tahmini_fiyat = np.expm1(log_pred)

        return {
            "tahmini_fiyat": round(tahmini_fiyat, 2),
            "r2_skoru": round(r2, 4),
            "mae": round(mae, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
