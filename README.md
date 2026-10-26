# Customer Churn MLOps Project

## Мета проєкту
Побудова базового MLOps-пайплайну для задачі прогнозування відтоку клієнтів (Customer Churn): валідація вхідних даних, версіонування датасету, відстеження експериментів у MLflow, реєстрація моделей та фіксація метрик.

## Джерело даних
Telco Customer Churn dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Структура директорій
- `configs` — конфігурації, зокрема схема даних (`data_schema.yaml`)
- `data/raw` — початкові дані (`churn.csv`)
- `data/processed` — підготовлені версії даних (`churn_v1.csv`) та їх метадані
- `src` — код валідації, версіонування даних, підготовки даних та навчання моделі
- `models` — збережені моделі
- `reports` — метрики та звіти (звіт валідації, метрики моделей, вибір моделі)
- `notebooks` — експерименти
- `tests` — тести

## Встановлення залежностей
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск пайплайну
Усі команди виконуються з кореня проєкту в такій послідовності:

```powershell
python src\validate_data.py
python src\version_data.py
python src\train.py --model-type logistic_regression
python src\train.py --model-type random_forest
```

1. `validate_data.py` — перевіряє датасет `data/raw/churn.csv` за схемою `configs/data_schema.yaml` і зберігає звіт у `reports/data_validation_report.json`.
2. `version_data.py` — очищає дані (порожні значення `TotalCharges` → пропуски → 0, видалення дублікатів), зберігає версію `data/processed/churn_v1.csv` та метадані `data/processed/churn_v1_metadata.json` (хеші SHA-256, кількість рядків, дата створення).
3. `train.py` — навчає модель на версії даних `v1`, логує запуск у MLflow та реєструє модель. Тип моделі задається параметром `--model-type` (`logistic_regression` або `random_forest`).

## Валідація даних
Схема даних описана в `configs/data_schema.yaml`: обов'язкові колонки, цільова змінна `Churn` та її допустимі значення (`Yes`, `No`), числові колонки, максимальна частка пропусків (20%).

Перевіряються: наявність обов'язкових колонок, дублікати, частка пропусків, допустимі значення цільової змінної та коректність числових типів.

Статус у звіті `reports/data_validation_report.json`:
- `passed` — критичних помилок (`errors`) немає, дані можна використовувати для навчання. Попередження (`warnings`), наприклад дублікати, не блокують пайплайн.
- `failed` — є хоча б одна помилка в `errors`; скрипт завершується з кодом 1, навчання запускати не можна, поки дані не буде виправлено.

## Версіонування даних
Версія датасету зберігається в назві файлу та в метаданих:
- дані: `data/processed/churn_v1.csv`
- метадані: `data/processed/churn_v1_metadata.json`

**Моделі використовують версію даних `v1`.** Ця версія логується в MLflow як параметр `data_version` і записується у звіти метрик.

## Експерименти та MLflow
Для відстеження експериментів і реєстру моделей використовується MLflow (експеримент `customer-churn-baseline`).

Запуск MLflow UI (в окремому терміналі, з кореня проєкту):
```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```
Адреса в браузері: http://127.0.0.1:5000

Навчання з логуванням у MLflow:
```powershell
python src\train.py --model-type logistic_regression
python src\train.py --model-type random_forest
```

Де що зберігається:
- записи про запуски (параметри, метрики) — у локальній базі `mlflow.db` у корені проєкту;
- артефакти запусків і моделі — у папці `mlruns/`;
- звіти метрик — `reports/metrics_<model_type>.json`;
- обидва шляхи (`mlflow.db`, `mlruns/`) додані до `.gitignore` і не комітяться.

Як переглянути результати:
- **Training runs** — список запусків, параметри та метрики, порівняння (кнопка Compare);
- **Model registry** (меню ліворуч) — зареєстрована модель `CustomerChurnClassifier` з версіями.

Вибір найкращої моделі описано в `reports/model_selection.md`.

## Отримані метрики
Версія даних `v1`, однакове розбиття train/test.

| Метрика | logistic_regression | random_forest |
|---|---|---|
| accuracy | 0.7842 | 0.8006 |
| precision | 0.7215 | 0.6768 |
| recall | 0.3048 | 0.4759 |
| f1 | 0.4286 | 0.5589 |
| roc_auc | 0.8307 | 0.8335 |

Обрана модель: **random_forest** (`CustomerChurnClassifier`, версія 2), основна метрика — recall.

## Висновок
Валідація та очищення даних (приведення `TotalCharges` до числового типу) покращили якість моделі відносно сирих даних. Відстеження запусків у MLflow дозволяє порівнювати моделі та відтворювати результати, а реєстр моделей фіксує версії `CustomerChurnClassifier`.