import os
import glob
import kagglehub
import pandas as pd
import numpy as np
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ==========================================
# 1. СКАЧИВАНИЕ ДАТАСЕТА И ОБУЧЕНИЕ МОДЕЛЕЙ
# ==========================================
print("Скачивание датасета с Kaggle...")
path = kagglehub.dataset_download("deeplumiere/hyperscale-data-center-dataset")
csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)

if not csv_files:
    raise FileNotFoundError("CSV файлы не найдены в датасете!")

df = pd.read_csv(csv_files[0])

# Определение целевых и входных признаков
TARGET_POWER = 'power_consumption_kw'
TARGET_CARBON = 'carbon_emission_kg'
exclude_cols = [TARGET_POWER, TARGET_CARBON, 'timestamp', 'date', 'id', 'Id']
feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

# Заполнение пропусков
X = df[feature_cols].fillna(df[feature_cols].median())
y_power = df[TARGET_POWER].fillna(df[TARGET_POWER].median())
y_carbon = df[TARGET_CARBON].fillna(df[TARGET_CARBON].median())

# Разделение выборки
X_train, X_test, y_power_train, y_power_test, y_carbon_train, y_carbon_test = train_test_split(
    X, y_power, y_carbon, test_size=0.2, random_state=42
)

# Обучение RandomForest
print("Обучение ML-моделей...")
model_power = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_power.fit(X_train, y_power_train)

model_carbon = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_carbon.fit(X_train, y_carbon_train)

r2_p = r2_score(y_power_test, model_power.predict(X_test))
r2_c = r2_score(y_carbon_test, model_carbon.predict(X_test))
print(f"Модели успешно обучены! R2 Power: {r2_p:.3f}, R2 Carbon: {r2_c:.3f}")

# ==========================================
# 2. ФУНКЦИЯ ИНФЕРЕНСА И ПОСТРОЕНИЯ ГРАФИКОВ
# ==========================================
def predict_and_visualize(*args):
    # Формируем входной DataFrame из полученных значений слайдеров
    input_dict = {col: val for col, val in zip(feature_cols, args)}
    input_df = pd.DataFrame([input_dict])

    # Точечный прогноз на 1 час
    pred_kw_hourly = float(model_power.predict(input_df)[0])
    pred_co2_hourly = float(model_carbon.predict(input_df)[0])

    # Экстраполяция на 1 год (24 часа * 365 дней = 8760 часов)
    pred_co2_yearly_tons = (pred_co2_hourly * 8760) / 1000.0
    km_car_yearly = (pred_co2_hourly * 8760) / 0.12
    trees_yearly = (pred_co2_hourly * 8760) / 21.0

    # Текст для KPI карточек
    kpi_power_str = f"⚡ {pred_kw_hourly:.2f} kW / ч"
    kpi_carbon_str = f"💨 {pred_co2_hourly:.2f} kg / ч"
    kpi_yearly_str = f"🌍 {pred_co2_yearly_tons:.2f} тонн / год"

    # Экологическая сводка
    eco_summary = (
        f"### 🚗 Экологический эквивалент (в расчете на 1 год работы):\n"
        f"* **Годовой выброс CO₂:** `{pred_co2_yearly_tons:.2f} тонн`\n"
        f"* **Эквивалент пробега авто:** `{km_car_yearly:,.0f} км`\n"
        f"* **Необходимо деревьев для нейтрализации:** `{int(trees_yearly)} шт.`"
    )

    # --- График 1: Суточная экстраполяция нагрузки ---
    hours = list(range(24))
    hourly_power = [pred_kw_hourly * (1 + 0.15 * np.sin(h / 3.5)) for h in hours]
    hourly_carbon = [pred_co2_hourly * (1 + 0.15 * np.sin(h / 3.5)) for h in hours]

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=hours, y=hourly_power, mode='lines+markers', name='Мощность (kW)', line=dict(color='#00CC96', width=3)))
    fig_line.add_trace(go.Scatter(x=hours, y=hourly_carbon, mode='lines+markers', name='CO₂ (kg)', line=dict(color='#EF553B', width=3, dash='dash')))
    fig_line.update_layout(
        title="📈 Суточный профиль потребления (24 часа)",
        xaxis_title="Час суток",
        yaxis_title="Значение",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # --- График 2: Важность признаков (Feature Importance) ---
    feature_importances = model_power.feature_importances_
    df_importance = pd.DataFrame({
        'Признак': feature_cols,
        'Важность': feature_importances
    }).sort_values('Важность', ascending=True).tail(8)

    fig_bar = px.bar(
        df_importance,
        x='Важность',
        y='Признак',
        orientation='h',
        color='Важность',
        color_continuous_scale='Viridis',
        title="📊 Топ факторов, влияющих на мощность"
    )
    fig_bar.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))

    return kpi_power_str, kpi_carbon_str, kpi_yearly_str, eco_summary, fig_line, fig_bar

# ==========================================
# 3. ПОСТРОЕНИЕ ИНТЕРФЕЙСА GRADIO
# ==========================================
theme = gr.themes.Soft(primary_hue="emerald", neutral_hue="slate")

with gr.Blocks(theme=theme, title="Green-Compute Monitor") as demo:
    gr.Markdown(
        f"""
        # 🌱 Green-Compute: AI Data Center Analytics Platform
        **Система оперативного прогнозирования энергопотребления и углеродного следа узлов дата-центра**  
        *(Точность ML-моделей $R^2$: Power = `{r2_p:.2f}`, Carbon = `{r2_c:.2f}`)*
        """
    )
    
    with gr.Row():
        # ЛЕВАЯ КОЛОНКА: Слайдеры управления
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Параметры нагрузки")
            inputs = []
            for col in feature_cols:
                min_v = float(df[col].min())
                max_v = float(df[col].max())
                mean_v = float(df[col].mean())
                slider = gr.Slider(minimum=min_v, maximum=max_v, value=mean_v, label=col)
                inputs.append(slider)
            
            btn = gr.Button("🚀 Рассчитать прогноз", variant="primary")

        # ПРАВАЯ КОЛОНКА: Дашборд с результатами
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Метрики и Экстраполяция")
            
            # Карточки KPI в один ряд
            with gr.Row():
                out_kpi1 = gr.Textbox(label="Мощность (Час)", interactive=False)
                out_kpi2 = gr.Textbox(label="Выбросы CO₂ (Час)", interactive=False)
                out_kpi3 = gr.Textbox(label="Годовой CO₂ след", interactive=False)

            out_eco = gr.Markdown()

            # Интерактивные Plotly графики
            with gr.Row():
                out_plot1 = gr.Plot(label="Суточный профиль")
                out_plot2 = gr.Plot(label="Важность признаков")

    # Связываем изменение каждого слайдера и нажатие кнопки с обновлением дашборда
    for inp in inputs:
        inp.change(fn=predict_and_visualize, inputs=inputs, outputs=[out_kpi1, out_kpi2, out_kpi3, out_eco, out_plot1, out_plot2])
    
    btn.click(fn=predict_and_visualize, inputs=inputs, outputs=[out_kpi1, out_kpi2, out_kpi3, out_eco, out_plot1, out_plot2])

    # Автоматический запуск при первой загрузке страницы
    demo.load(fn=predict_and_visualize, inputs=inputs, outputs=[out_kpi1, out_kpi2, out_kpi3, out_eco, out_plot1, out_plot2])

if __name__ == "__main__":
    demo.launch()
