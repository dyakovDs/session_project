# СберАвтоподписка - предсказание конверсии

Проект для курса ML. Задача: по данным веб-сессии предсказать, оставит ли пользователь заявку (позвонит, напишет в чат и т.д.).

Данные: Google Analytics - сессии (`ga_sessions.csv`) и хиты (`ga_hits.csv`).

## Что внутри

```
sber_autosub/
├── data/                  # папка для данных (csv не коммитятся)
├── notebooks/
│   └── analysis.ipynb     # EDA + обучение модели
├── models/                # сюда сохраняется модель после обучения
├── api/
│   ├── main.py            # FastAPI
│   └── predict.py         # загрузка модели, CLI для батч-предсказаний
└── requirements.txt
```

## Запуск

```bash
pip install -r requirements.txt
```

Положить данные в папку `data/`, потом открыть ноутбук и запустить все ячейки (`Kernel -> Restart & Run All`). Модель сохранится в `models/`.

После этого можно поднять API:

```bash
uvicorn api.main:app --reload --port 8000
```

Документация: http://localhost:8000/docs

## API

`POST /predict` - предсказание для одной сессии.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"device_category":"desktop","utm_medium":"cpc","visit_number":3,"total_hits":20,"hour":14,"day_of_week":2}'
```

Ответ:
```json
{"conversion_probability": 0.88, "prediction": 1}
```

Все поля необязательные - пропущенные заполняются медианами из обучающей выборки.

`GET /health` - проверка что сервис живой.

## Модель

LightGBM с подбором гиперпараметров через RandomizedSearchCV (темпоральный сплит 80/20 по дате).

ROC-AUC на тесте: **0.90**

Основные признаки: `total_hits`, `rolling_cr_7d` (скользящая конверсия за 7 дней по каналу), `is_car_page`, `visit_number`, временные фичи.
