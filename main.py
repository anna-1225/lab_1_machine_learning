from pathlib import Path
import pandas as pd

# Путь к папке, в которой находится main.py
BASE_DIR = Path(__file__).resolve().parent

# Путь к исходному датасету
DATA_PATH = BASE_DIR / "data" / "Life Expectancy Data.csv"

# Загружаем данные
df = pd.read_csv(DATA_PATH)

# Удаляем лишние пробелы в названиях столбцов
df.columns = df.columns.str.strip()

# Выводим общую информацию
print("Размер датасета:", df.shape)

print("\nПервые пять строк:")
print(df.head())

print("\nИнформация о столбцах:")
df.info()