"""
FastAPI-приложение для предсказания вероятности конверсии.

Запуск:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Пример запроса:
    curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{
        "device_category": "mobile",
        "utm_source": "google",
        "utm_medium": "cpc",
        "visit_number": 1,
        "total_hits": 12,
        "hour": 15,
        "day_of_week": 3,
        "is_weekend": 0
      }'
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.predict import predict_session, load_model

# ──────────────────────────────────────────────
# Логирование
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Приложение FastAPI
# ──────────────────────────────────────────────
app = FastAPI(
    title="СберАвтоподписка — API предсказания конверсии",
    description=(
        "Принимает признаки пользовательской сессии и возвращает "
        "вероятность совершения целевого действия."
    ),
    version="1.0.0",
)


# ──────────────────────────────────────────────
# Схема входных данных
# ──────────────────────────────────────────────
class SessionInput(BaseModel):
    """Входные признаки одной сессии."""

    # Основные признаки устройства
    device_category: str = Field(
        default="desktop",
        description="Тип устройства: mobile / desktop / tablet",
        example="mobile",
    )
    device_os: Optional[str] = Field(
        default="unknown",
        description="ОС устройства (Android, iOS, Windows и т.д.)",
        example="Android",
    )
    device_browser: Optional[str] = Field(
        default="unknown",
        description="Браузер",
        example="Chrome",
    )
    device_model_grp: Optional[str] = Field(
        default="other",
        description="Модель устройства (топ-20 или 'other')",
        example="Samsung Galaxy",
    )

    # UTM-метки
    utm_source: Optional[str] = Field(
        default="unknown",
        description="Источник трафика (google, yandex, vk и т.д.)",
        example="google",
    )
    utm_medium: Optional[str] = Field(
        default="unknown",
        description="Канал трафика (cpc, organic, referral и т.д.)",
        example="cpc",
    )
    utm_medium_group: Optional[str] = Field(
        default=None,
        description="Группа канала: organic / social / paid (вычисляется автоматически, если не задана)",
        example="paid",
    )

    # Параметры визита
    visit_number: int = Field(
        default=1,
        ge=1,
        description="Порядковый номер визита пользователя",
        example=1,
    )
    total_hits: Optional[int] = Field(
        default=0,
        ge=0,
        description="Количество событий (хитов) в сессии",
        example=12,
    )

    # Временные признаки
    hour: Optional[int] = Field(
        default=12,
        ge=0,
        le=23,
        description="Час начала визита (0–23)",
        example=15,
    )
    day_of_week: Optional[int] = Field(
        default=0,
        ge=0,
        le=6,
        description="День недели (0=Понедельник, 6=Воскресенье)",
        example=3,
    )
    month: Optional[int] = Field(
        default=6,
        ge=1,
        le=12,
        description="Месяц визита (1–12)",
        example=5,
    )
    is_weekend: Optional[int] = Field(
        default=0,
        ge=0,
        le=1,
        description="Выходной день (1=да, 0=нет)",
        example=0,
    )

    # Производные признаки
    is_mobile: Optional[int] = Field(
        default=None,
        description="Мобильное устройство (вычисляется из device_category, если не задано)",
        example=1,
    )
    visit_number_gt1: Optional[int] = Field(
        default=None,
        description="Повторный визит (вычисляется из visit_number, если не задано)",
        example=0,
    )
    is_car_page: Optional[int] = Field(
        default=0,
        ge=0,
        le=1,
        description="Просматривал страницы с автомобилями",
        example=1,
    )
    rolling_cr_7d: Optional[float] = Field(
        default=None,
        description="Скользящая средняя конверсии по источнику за 7 дней (необязательно)",
        example=0.05,
    )

    # Гео
    geo_city_grp: Optional[str] = Field(
        default="other",
        description="Город (топ-20 или 'other')",
        example="Moscow",
    )
    utm_source_grp: Optional[str] = Field(
        default=None,
        description="Группированный utm_source (вычисляется из utm_source, если не задано)",
        example="google",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "device_category": "mobile",
                "utm_source": "google",
                "utm_medium": "cpc",
                "visit_number": 1,
                "total_hits": 12,
                "hour": 15,
                "day_of_week": 3,
                "is_weekend": 0,
            }
        }


class PredictionResponse(BaseModel):
    conversion_probability: float
    prediction: int


# ──────────────────────────────────────────────
# Загрузка модели при старте приложения
# ──────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Предзагрузка модели при старте сервера."""
    try:
        load_model()
        logger.info("Модель успешно загружена при старте.")
    except FileNotFoundError as e:
        logger.warning("Модель не найдена при старте: %s", e)


# ──────────────────────────────────────────────
# Эндпоинты
# ──────────────────────────────────────────────
@app.get("/", summary="Статус сервиса")
async def root():
    return {
        "service": "СберАвтоподписка — Conversion Prediction API",
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health", summary="Healthcheck")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Предсказание вероятности конверсии",
    description=(
        "Принимает признаки одной пользовательской сессии "
        "и возвращает вероятность совершения целевого действия."
    ),
)
async def predict(session: SessionInput):
    """
    Возвращает:
    - **conversion_probability**: вероятность конверсии [0, 1]
    - **prediction**: бинарное предсказание (1 = конверсия, порог 0.5)
    """
    request_time = datetime.now().isoformat()
    logger.info("[%s] POST /predict  device=%s utm=%s/%s hits=%s",
                request_time,
                session.device_category,
                session.utm_source,
                session.utm_medium,
                session.total_hits)

    # Вычисляем производные признаки, если не переданы
    data = session.model_dump()

    # is_mobile
    if data.get("is_mobile") is None:
        data["is_mobile"] = int(str(data.get("device_category", "")).lower() == "mobile")

    # visit_number_gt1
    if data.get("visit_number_gt1") is None:
        data["visit_number_gt1"] = int((data.get("visit_number") or 1) > 1)

    # utm_medium_group
    if data.get("utm_medium_group") is None:
        ORGANIC_MEDIUMS = {"organic", "referral", "(none)", "none", ""}
        SOCIAL_SOURCES  = {"vk", "vkontakte", "facebook", "fb", "instagram",
                           "ok", "odnoklassniki", "tiktok", "telegram", "youtube"}
        medium = str(data.get("utm_medium") or "").lower().strip()
        source = str(data.get("utm_source") or "").lower().strip()
        if medium in ORGANIC_MEDIUMS:
            data["utm_medium_group"] = "organic"
        elif any(s in source for s in SOCIAL_SOURCES):
            data["utm_medium_group"] = "social"
        else:
            data["utm_medium_group"] = "paid"

    # utm_source_grp
    if data.get("utm_source_grp") is None:
        data["utm_source_grp"] = data.get("utm_source") or "unknown"

    # rolling_cr_7d: если не передан, используем 0 (будет заменён медианой в препроцессоре)
    if data.get("rolling_cr_7d") is None:
        data["rolling_cr_7d"] = 0.0

    # Безопасная замена неизвестных категорий на 'unknown'
    cat_fields = ["device_category", "device_os", "device_browser", "device_model_grp",
                  "utm_source", "utm_medium", "utm_medium_group", "utm_source_grp",
                  "geo_city_grp"]
    for field in cat_fields:
        if data.get(field) is None or str(data.get(field, "")).lower() in ("none", "nan", ""):
            data[field] = "unknown"

    try:
        result = predict_session(data)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Ошибка предсказания: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {e}")

    logger.info("[%s] Результат: prob=%.4f pred=%d",
                request_time, result["conversion_probability"], result["prediction"])

    return PredictionResponse(**result)
