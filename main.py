from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "Life Expectancy Data.csv"

FIGURES_DIR = BASE_DIR / "results" / "figures"
TABLES_DIR = BASE_DIR / "results" / "tables"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("Размер датасета:", df.shape)
print(df.head())
df.info()

print("\nОписательные статистики")

statistics = df.describe().T

print(statistics)

statistics.to_csv(
    TABLES_DIR / "descriptive_statistics.csv",
    encoding="utf-8-sig"
)

print("\nПропущенные значения")

missing_count = df.isna().sum()

missing_percent = (
    missing_count / len(df) * 100
)

missing_table = pd.DataFrame({
    "Пропуски": missing_count,
    "Доля (%)": missing_percent
})

missing_table = missing_table.sort_values(
    by="Пропуски",
    ascending=False
)

print(missing_table)

missing_table.to_csv(
    TABLES_DIR / "missing_values.csv",
    encoding="utf-8-sig"
)

print("\nПроверка дубликатов")

duplicates = df.duplicated().sum()

print("Количество полных дубликатов:", duplicates)

country_year_duplicates = df.duplicated(
    subset=["Country", "Year"]
).sum()

print(
    "Повторения сочетания Country-Year:",
    country_year_duplicates
)

print("\nРаспределение целевой переменной")

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Life expectancy",
    bins=30,
    kde=True,
    color="steelblue"
)

plt.title("Распределение продолжительности жизни")
plt.xlabel("Продолжительность жизни, лет")
plt.ylabel("Количество наблюдений")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "life_expectancy_distribution.png",
    dpi=300
)

plt.show()
plt.close()

numeric_columns = df.select_dtypes(
    include=np.number
).columns

df[numeric_columns].hist(
    figsize=(18, 18),
    bins=25
)

plt.suptitle(
    "Распределения числовых признаков",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "numeric_distributions.png",
    dpi=200
)

plt.show()
plt.close()

features = [
    "Schooling",
    "Adult Mortality",
    "GDP",
    "BMI"
]

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 10)
)

for ax, feature in zip(axes.flat, features):
    sns.scatterplot(
        data=df,
        x=feature,
        y="Life expectancy",
        alpha=0.5,
        ax=ax
    )

    ax.set_title(f"{feature} и продолжительность жизни")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "features_vs_target.png",
    dpi=300
)

plt.show()
plt.close()

outlier_features = [
    "Life expectancy",
    "Adult Mortality",
    "GDP",
    "Schooling"
]

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 8)
)

for ax, feature in zip(axes.flat, outlier_features):
    sns.boxplot(
        data=df,
        x=feature,
        ax=ax
    )

    ax.set_title(f"Выбросы: {feature}")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "outliers_boxplot.png",
    dpi=300
)

plt.show()
plt.close()

print("\nОчистка данных")

df_clean = df.dropna().copy()

print("Строк до очистки:", len(df))
print("Строк после очистки:", len(df_clean))

print(
    "Удалено строк:",
    len(df) - len(df_clean)
)

print(
    "Осталось пропусков:",
    df_clean.isna().sum().sum()
)

df_clean.to_csv(
    TABLES_DIR / "cleaned_data.csv",
    index=False,
    encoding="utf-8-sig"
)