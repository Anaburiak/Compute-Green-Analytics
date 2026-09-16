# 🌱 Green-Compute: AI Data Center Analytics Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![R2 Score](https://img.shields.io/badge/Model%20R2-0.92-emerald)

Платформа предиктивной аналитики для прогнозирования энергопотребления, углеродного следа ($CO_2$) и финансово-экологических рисков (Carbon Tax) вычислительных узлов дата-центров.

---

## 🚀 Основные возможности
* **Предиктивный ML-анализ:** Предсказание энергопотребления ($kW$) и выбросов $CO_2$ до запуска вычислений.
* **Финансовая оценка:** Расчет потенциального углеродного налога (Carbon Tax) по ставке EU ETS ($90/т $CO_2$).
* **Экологические эквиваленты:** Перевод выбросов в километры автопробега и количество деревьев для компенсации.
* **Интерактивный дашборд:** Визуализация суточных профилей нагрузки и факторов влияния (Feature Importances).

---

## 🛠️ Стек технологий
* **Core & ML (Python):** `Pandas`, `NumPy`, `Scikit-Learn` (Random Forest Regressor).
* **Исследование:** `Jupyter Notebook`, `Matplotlib`, `Seaborn`.
* **Frontend / Demo:** HTML5, Tailwind CSS, Chart.js.

---

## 📊 Результаты моделирования (Python)

| Модель | Целевая метрика | $R^2$ Score | MAE |
| :--- | :--- | :--- | :--- |
| **Power Predictor** | Энергопотребление ($kW$) | **0.81** | 0.12 kW |
| **Carbon Predictor** | Выбросы $CO_2$ ($kg$) | **0.92** | 0.04 kg |

---

## 💻 Быстрый запуск (Python)

1. Клонируйте репозиторий:
   ```bash
   git clone [https://github.com/your-username/green-compute-analytics.git](https://github.com/your-username/green-compute-analytics.git)
   cd green-compute-analytics
