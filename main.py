
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit

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

print(missing_table.sort_values("Количество", ascending=False).round(2))

missing_table.to_csv(
    TABLES_DIR / "missing_values.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(10, 6))
missing[missing > 0].sort_values().plot(kind="barh")

plt.title("Количество пропущенных значений")
plt.xlabel("Количество пропусков")
plt.ylabel("Признаки")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "missing_values.png", dpi=300)
plt.show()
plt.close()


print("\nДубликаты")

print("Полных дубликатов:", df.duplicated().sum())

print(
    "Дубликатов по Country и Year:",
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
plt.savefig(FIGURES_DIR / "life_expectancy_distribution.png", dpi=300)
plt.show()
plt.close()

numeric_columns = df.select_dtypes(include=np.number).columns

df[numeric_columns].hist(
    figsize=(18, 18),
    bins=25
)

plt.suptitle("Распределения числовых признаков")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "numeric_distributions.png", dpi=200)
plt.show()
plt.close()


features = [
    "Schooling",
    "Adult Mortality",
    "GDP",
    "BMI"
]

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
plt.savefig(FIGURES_DIR / "features_vs_target.png", dpi=300)
plt.show()
plt.close()


print("\nКорреляционная матрица")

corr_matrix = df.select_dtypes(
    include=np.number
).corr(method="pearson")

plt.figure(figsize=(16, 12))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Корреляционная матрица признаков")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=300)
plt.show()
plt.close()


outlier_features = [
    "Life expectancy",
    "Adult Mortality",
    "GDP",
    "Schooling"
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, feature in zip(axes.flat, outlier_features):
    sns.boxplot(
        data=df,
        x=feature,
        ax=ax
    )
    ax.set_title(feature)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "outliers_boxplot.png", dpi=300)
plt.show()
plt.close()


print("\nОчистка данных")

df_clean = df.dropna().copy()

print("До очистки:", df.shape)
print("После очистки:", df_clean.shape)
print("Удалено строк:", len(df) - len(df_clean))
print("Осталось пропусков:", df_clean.isna().sum().sum())

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
plt.savefig(FIGURES_DIR / "data_cleaning.png", dpi=300)
plt.show()
plt.close()


df_clean["Status"] = df_clean["Status"].map({
    "Developing": 0,
    "Developed": 1
})

print(df_clean["Status"].value_counts())

plt.figure(figsize=(7, 4))

sns.countplot(data=df_clean, x="Status")

plt.title("Количество наблюдений по статусу страны")
plt.xlabel("Статус (0 — Developing, 1 — Developed)")
plt.ylabel("Количество наблюдений")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "status_distribution.png", dpi=300)
plt.show()
plt.close()

print("\nФормирование X и y")

X = df_clean.drop(columns=["Country", "Life expectancy"])
y = df_clean["Life expectancy"]
groups = df_clean["Country"]

print("Размер X:", X.shape)
print("Размер y:", y.shape)


print("\nРазделение данных")

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

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print(
    "Общие страны:",
    set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
)

sizes = pd.Series({
    "Обучающая": len(X_train),
    "Тестовая": len(X_test)
})

plt.figure(figsize=(7, 4))
sizes.plot(kind="bar")

plt.title("Разделение на обучающую и тестовую выборки")
plt.ylabel("Количество наблюдений")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "train_test_split.png", dpi=300)
plt.show()
plt.close()

print("\nПодготовка данных завершена.")
