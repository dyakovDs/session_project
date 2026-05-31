"""
predict.py — утилита загрузки модели и инференса.

Использование:
    from api.predict import load_model, predict_session

Или из командной строки:
    python predict.py --input input.json --output output.json
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Union

# Путь до папки models/ относительно корня проекта
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_model      = None
_meta       = None

logger = logging.getLogger(__name__)


def load_model():
    """Загрузить модель и мета-конфигурацию (кешируется в памяти)."""
    global _model, _meta

    if _model is None:
        model_path = os.path.join(MODELS_DIR, "final_model.pkl")
        meta_path  = os.path.join(MODELS_DIR, "feature_config.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}\n"
                "Сначала запустите notebooks/analysis.ipynb до конца, "
                "чтобы обучить и сохранить модель."
            )

        logger.info("Загрузка модели из %s", model_path)
        _model = joblib.load(model_path)
        _meta  = joblib.load(meta_path) if os.path.exists(meta_path) else {}
        logger.info("Модель загружена. ROC-AUC на тесте: %.4f",
                    _meta.get("test_roc_auc", 0))

    return _model, _meta


def _build_input_df(data: dict) -> pd.DataFrame:
    """Создать DataFrame из словаря входных данных для одной сессии."""
    _, meta = load_model()
    feature_cols = meta.get("feature_cols", [])

    row = {}
    for col in feature_cols:
        row[col] = data.get(col, np.nan)

    df = pd.DataFrame([row])

    # Подстановка 'unknown' для категориальных пропусков
    cat_cols = meta.get("ohe_features", []) + meta.get("high_card_features", [])
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)
            df[col] = df[col].replace({"nan": "unknown", "None": "unknown"})

    return df


def predict_session(data: dict) -> dict:
    """
    Получить вероятность конверсии для одной сессии.

    Parameters
    ----------
    data : dict
        Словарь с признаками сессии (см. SessionInput в main.py).

    Returns
    -------
    dict
        {"conversion_probability": float, "prediction": int}
    """
    model, _ = load_model()
    df = _build_input_df(data)

    proba      = float(model.predict_proba(df)[0, 1])
    prediction = int(proba >= 0.5)

    return {"conversion_probability": round(proba, 6), "prediction": prediction}


# ──────────────────────────────────────────────
# CLI-интерфейс
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Предсказание вероятности конверсии для сессий."
    )
    parser.add_argument("--input",  required=True, help="Путь к JSON-файлу с входными данными")
    parser.add_argument("--output", required=True, help="Путь для записи результатов")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    with open(args.input, "r", encoding="utf-8") as f:
        sessions = json.load(f)

    if isinstance(sessions, dict):
        sessions = [sessions]

    results = []
    for i, sess in enumerate(sessions):
        try:
            result = predict_session(sess)
            result["input_index"] = i
            results.append(result)
        except Exception as e:
            results.append({"input_index": i, "error": str(e)})

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Результаты сохранены в {args.output}")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
