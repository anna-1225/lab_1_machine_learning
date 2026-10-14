
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.stats.outliers_influence import variance_inflation_factor


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "Life Expectancy Data.csv"
FIGURES_DIR = BASE_DIR / "results" / "figures"
TABLES_DIR = BASE_DIR / "results" / "tables"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("Загрузка данных")
print("Размер датасета:", df.shape)
print(df.head())
df.info()


print("\nОписательные статистики")

statistics = df.describe().T
print(statistics.round(2))

statistics.to_csv(
    TABLES_DIR / "descriptive_statistics.csv",
    encoding="utf-8-sig"
)

print("\nПропущенные значения")

missing = df.isna().sum()

missing_table = pd.DataFrame({
    "Количество": missing,
    "Доля (%)": missing / len(df) * 100
})

print(missing_table.sort_values("Количество", ascending=False))

missing_table.to_csv(
    TABLES_DIR / "missing_values.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(10, 6))
missing[missing > 0].sort_values().plot(kind="barh")
plt.title("Количество пропущенных значений")
plt.xlabel("Количество пропусков")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "missing_values.png", dpi=200)
plt.close()

print("\nДубликаты")
print("Полных дубликатов:", df.duplicated().sum())
print(
    "Дубликатов Country-Year:",
    df.duplicated(subset=["Country", "Year"]).sum()
)


plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="Life expectancy",
    bins=30,
    kde=True
)

plt.title("Распределение продолжительности жизни")
plt.xlabel("Продолжительность жизни, лет")
plt.ylabel("Количество наблюдений")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "life_expectancy_distribution.png", dpi=200)
plt.close()


numeric_columns = df.select_dtypes(include=np.number).columns

df[numeric_columns].hist(figsize=(18, 18), bins=25)

plt.suptitle("Распределения числовых признаков")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "numeric_distributions.png", dpi=200)
plt.close()


features = ["Schooling", "Adult Mortality", "GDP", "BMI"]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, feature in zip(axes.flat, features):
    sns.scatterplot(
        data=df,
        x=feature,
        y="Life expectancy",
        alpha=0.5,
        ax=ax
    )
    ax.set_title(feature)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "features_vs_target.png", dpi=200)
plt.close()


corr_initial = df.select_dtypes(
    include=np.number
).corr(method="pearson")

plt.figure(figsize=(16, 12))

sns.heatmap(
    corr_initial,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Корреляционная матрица исходного датасета")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=200)
plt.close()



outlier_features = [
    "Life expectancy",
    "Adult Mortality",
    "GDP",
    "Schooling"
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, feature in zip(axes.flat, outlier_features):
    sns.boxplot(data=df, x=feature, ax=ax)
    ax.set_title(feature)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "outliers_boxplot.png", dpi=200)
plt.close()



print("\nОчистка данных")

df_clean = df.dropna().copy()

print("До очистки:", df.shape)
print("После очистки:", df_clean.shape)
print("Удалено строк:", len(df) - len(df_clean))

rows = pd.Series({
    "До очистки": len(df),
    "После очистки": len(df_clean)
})

plt.figure(figsize=(7, 4))
rows.plot(kind="bar")
plt.title("Количество наблюдений до и после очистки")
plt.ylabel("Количество наблюдений")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "data_cleaning.png", dpi=200)
plt.close()



df_clean["Status"] = df_clean["Status"].map({
    "Developing": 0,
    "Developed": 1
})

plt.figure(figsize=(7, 4))
sns.countplot(data=df_clean, x="Status")
plt.title("Распределение по статусу страны")
plt.xlabel("0 — Developing, 1 — Developed")
plt.ylabel("Количество наблюдений")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "status_distribution.png", dpi=200)
plt.close()



X = df_clean.drop(columns=["Country", "Life expectancy"])
y = df_clean["Life expectancy"]
groups = df_clean["Country"]

print("Размер X:", X.shape)
print("Размер y:", y.shape)


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

groups_train = groups.iloc[train_idx]

print("\nРазделение данных")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

common_countries = (
    set(groups.iloc[train_idx]) &
    set(groups.iloc[test_idx])
)

print("Общие страны:", common_countries)

sizes = pd.Series({
    "Обучающая": len(X_train),
    "Тестовая": len(X_test)
})

plt.figure(figsize=(7, 4))
sizes.plot(kind="bar")
plt.title("Разделение на train и test")
plt.ylabel("Количество наблюдений")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "train_test_split.png", dpi=200)
plt.close()


train_data = X_train.copy()
train_data["Life expectancy"] = y_train

corr_train = train_data.corr(method="pearson")

corr_train.to_csv(
    TABLES_DIR / "train_correlations.csv",
    encoding="utf-8-sig"
)

print("\nКорреляции с целевой переменной")

target_corr = corr_train["Life expectancy"].drop(
    "Life expectancy"
)

target_corr = target_corr.reindex(
    target_corr.abs().sort_values(ascending=False).index
)

print(target_corr.round(3))


print("\nVIF до PCA")

scaler_vif = StandardScaler()
X_vif = scaler_vif.fit_transform(X_train)

vif_table = pd.DataFrame({
    "Признак": X_train.columns,
    "VIF": [
        variance_inflation_factor(X_vif, i)
        for i in range(X_vif.shape[1])
    ]
})

vif_table = vif_table.sort_values("VIF", ascending=False)

print(vif_table.round(2))

vif_table.to_csv(
    TABLES_DIR / "vif_before_pca.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=vif_table,
    x="VIF",
    y="Признак"
)

plt.axvline(10, color="red", linestyle="--")
plt.title("VIF до PCA")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "vif_before_pca.png", dpi=200)
plt.close()



models = {
    "Linear": LinearRegression(),
    "Lasso": Lasso(max_iter=30000),
    "Ridge": Ridge()
}

alphas = [0.001, 0.01, 0.1, 1, 10, 100]

cv = GroupKFold(n_splits=5)

trained_models = {}
results = []
cv_results = []


print("\nОбучение моделей")

for name, model in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    if name == "Linear":
        params = {}
    else:
        params = {"model__alpha": alphas}

    search = GridSearchCV(
        pipeline,
        param_grid=params,
        cv=cv,
        scoring="neg_root_mean_squared_error"
    )

    search.fit(X_train, y_train, groups=groups_train)

    trained_models[name] = search.best_estimator_

    cv_results.append({
        "Модель": name,
        "Alpha": search.best_params_.get("model__alpha", np.nan),
        "CV RMSE": -search.best_score_
    })

    print(name)
    print("Лучшие параметры:", search.best_params_)
    print("CV RMSE:", round(-search.best_score_, 3))

    for sample, X_part, y_part in [
        ("Train", X_train, y_train),
        ("Test", X_test, y_test)
    ]:

        y_pred = search.predict(X_part)

        rmse = np.sqrt(mean_squared_error(y_part, y_pred))
        r2 = r2_score(y_part, y_pred)
        mape = mean_absolute_percentage_error(y_part, y_pred) * 100

        results.append({
            "Модель": name,
            "Выборка": sample,
            "RMSE": rmse,
            "R²": r2,
            "MAPE (%)": mape
        })


results_df = pd.DataFrame(results)
cv_df = pd.DataFrame(cv_results)

print("\nМетрики до PCA")
print(results_df.round(3).to_string(index=False))

print("\nРезультаты кросс-валидации")
print(cv_df.round(3).to_string(index=False))

results_df.to_csv(
    TABLES_DIR / "metrics_before_pca.csv",
    index=False,
    encoding="utf-8-sig"
)

cv_df.to_csv(
    TABLES_DIR / "cv_results.csv",
    index=False,
    encoding="utf-8-sig"
)

test_results = results_df[
    results_df["Выборка"] == "Test"
]

plt.figure(figsize=(8, 5))

sns.barplot(
    data=test_results,
    x="Модель",
    y="RMSE"
)

plt.title("Сравнение RMSE до PCA")
plt.ylabel("RMSE, лет")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "rmse_before_pca.png", dpi=200)
plt.close()


print("\nКоэффициенты моделей")

coef_table = pd.DataFrame({
    name: trained_models[name].named_steps["model"].coef_
    for name in models
}, index=X_train.columns)

print(coef_table.round(3))

coef_table.to_csv(
    TABLES_DIR / "model_coefficients.csv",
    encoding="utf-8-sig"
)


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (name, model) in zip(axes, trained_models.items()):

    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, color="red", linestyle="--")

    ax.set_title(name)
    ax.set_xlabel("Предсказанное значение")
    ax.set_ylabel("Остатки, лет")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "residuals_before_pca.png", dpi=200)
plt.close()



print("\nPCA")

scaler_pca = StandardScaler()
X_train_scaled = scaler_pca.fit_transform(X_train)

pca_full = PCA()
pca_full.fit(X_train_scaled)

variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(variance)

n_components = np.argmax(cumulative_variance >= 0.95) + 1

print("Исходное число признаков:", X_train.shape[1])
print("Выбрано компонент:", n_components)
print(
    "Сохранённая дисперсия:",
    round(cumulative_variance[n_components - 1] * 100, 2),
    "%"
)

pca_variance = pd.DataFrame({
    "Компонента": range(1, len(variance) + 1),
    "Доля дисперсии": variance,
    "Накопленная доля": cumulative_variance
})

pca_variance.to_csv(
    TABLES_DIR / "pca_variance.csv",
    index=False,
    encoding="utf-8-sig"
)


plt.figure(figsize=(10, 5))

plt.plot(
    range(1, len(variance) + 1),
    variance,
    marker="o"
)

plt.title("График каменистой осыпи")
plt.xlabel("Номер главной компоненты")
plt.ylabel("Доля объяснённой дисперсии")
plt.grid(True)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "pca_scree.png", dpi=200)
plt.close()


plt.figure(figsize=(10, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)

plt.axhline(
    0.95,
    color="red",
    linestyle="--",
    label="95% дисперсии"
)

plt.title("Накопленная объяснённая дисперсия")
plt.xlabel("Количество компонент")
plt.ylabel("Накопленная доля дисперсии")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "pca_cumulative.png", dpi=200)
plt.close()


pca_selected = PCA(n_components=0.95)
X_pca = pca_selected.fit_transform(X_train_scaled)

print("\nРазмер матрицы после PCA:", X_pca.shape)

vif_after = pd.DataFrame({
    "Компонента": [
        f"PC{i+1}" for i in range(X_pca.shape[1])
    ],
    "VIF": [
        variance_inflation_factor(X_pca, i)
        for i in range(X_pca.shape[1])
    ]
})

print("\nVIF после PCA")
print(vif_after.round(3))

vif_after.to_csv(
    TABLES_DIR / "vif_after_pca.csv",
    index=False,
    encoding="utf-8-sig"
)

loadings = pd.DataFrame(
    pca_selected.components_.T,
    index=X_train.columns,
    columns=[
        f"PC{i+1}" for i in range(X_pca.shape[1])
    ]
)

print("\nНагрузки первых трех компонент")
print(loadings.iloc[:, :3].round(3))

loadings.to_csv(
    TABLES_DIR / "pca_loadings.csv",
    encoding="utf-8-sig"
)


print("\nМодели после PCA")

trained_pca_models = {}
pca_results = []

for name, model in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("model", model)
    ])

    if name == "Linear":
        params = {}
    else:
        params = {"model__alpha": alphas}

    search = GridSearchCV(
        pipeline,
        param_grid=params,
        cv=cv,
        scoring="neg_root_mean_squared_error"
    )

    search.fit(X_train, y_train, groups=groups_train)

    model_name = name + " + PCA"
    trained_pca_models[model_name] = search.best_estimator_

    print(model_name)
    print("Лучшие параметры:", search.best_params_)
    print("CV RMSE:", round(-search.best_score_, 3))

    for sample, X_part, y_part in [
        ("Train", X_train, y_train),
        ("Test", X_test, y_test)
    ]:

        y_pred = search.predict(X_part)

        rmse = np.sqrt(mean_squared_error(y_part, y_pred))
        r2 = r2_score(y_part, y_pred)
        mape = mean_absolute_percentage_error(y_part, y_pred) * 100

        pca_results.append({
            "Модель": model_name,
            "Выборка": sample,
            "RMSE": rmse,
            "R²": r2,
            "MAPE (%)": mape
        })


pca_results_df = pd.DataFrame(pca_results)

print("\nМетрики после PCA")
print(pca_results_df.round(3).to_string(index=False))

pca_results_df.to_csv(
    TABLES_DIR / "metrics_after_pca.csv",
    index=False,
    encoding="utf-8-sig"
)


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (name, model) in zip(axes, trained_pca_models.items()):

    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, color="red", linestyle="--")

    ax.set_title(name)
    ax.set_xlabel("Предсказанное значение")
    ax.set_ylabel("Остатки, лет")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "residuals_after_pca.png", dpi=200)
plt.close()


comparison_df = pd.concat(
    [results_df, pca_results_df],
    ignore_index=True
)

comparison_df = comparison_df[
    comparison_df["Выборка"] == "Test"
].copy()

print("\nИтоговое сравнение")
print(comparison_df.round(3).to_string(index=False))

comparison_df.to_csv(
    TABLES_DIR / "final_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)


rmse_data = pd.DataFrame({
    "Модель": ["Linear", "Lasso", "Ridge"],
    "До PCA": [
        comparison_df.loc[
            comparison_df["Модель"] == name, "RMSE"
        ].iloc[0]
        for name in ["Linear", "Lasso", "Ridge"]
    ],
    "После PCA": [
        comparison_df.loc[
            comparison_df["Модель"] == name + " + PCA", "RMSE"
        ].iloc[0]
        for name in ["Linear", "Lasso", "Ridge"]
    ]
})

ax = rmse_data.plot(
    x="Модель",
    y=["До PCA", "После PCA"],
    kind="bar",
    figsize=(10, 6),
    width=0.7
)

plt.title("Сравнение RMSE до и после PCA")
plt.xlabel("Модель")
plt.ylabel("RMSE, лет")
plt.xticks(rotation=0)

plt.ylim(4.15, 4.20)

for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=3)

plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "rmse_comparison.png",
    dpi=300
)

plt.show()


best_model = comparison_df.loc[
    comparison_df["RMSE"].idxmin()
]

print("\nЛучшая модель по RMSE")
print("Модель:", best_model["Модель"])
print("RMSE:", round(best_model["RMSE"], 3))
print("R²:", round(best_model["R²"], 3))
print("MAPE:", round(best_model["MAPE (%)"], 3), "%")

