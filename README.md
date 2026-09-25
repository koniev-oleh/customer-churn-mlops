# Customer Churn MLOps Project

## Мета проєкту
Побудова базового MLOps-пайплайну для задачі прогнозування відтоку клієнтів (Customer Churn).

## Джерело даних
Telco Customer Churn dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Структура директорій
- `data/raw` — початкові дані
- `data/processed` — підготовлені дані
- `src` — код підготовки даних та навчання моделі
- `models` — збережені моделі
- `reports` — метрики та звіти
- `notebooks` — експерименти
- `tests` — тести

## Встановлення залежностей
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Запуск навчання
\`\`\`bash
cd src
python train.py
\`\`\`

## Отримані метрики
{
  "accuracy": 0.7636621717530163,
  "precision": 0.7303370786516854,
  "recall": 0.17379679144385027,
  "f1": 0.28077753779697623,
  "roc_auc": 0.8194864243457594
}

## Висновок
Базова модель логістичної регресії дає стартову точку для подальших покращень пайплайну.