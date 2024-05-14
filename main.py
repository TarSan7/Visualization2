import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

data = pd.read_csv('insurance.csv')

# Підготовка даних
X = data.drop('charges', axis=1)
y = data['charges']

# Кодування категоріальних змінних
categorical_features = ['sex', 'smoker', 'region']
transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_features)], remainder='passthrough')
X_encoded = transformer.fit_transform(X)

# Розділення на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Побудова моделі
model = LinearRegression()
model.fit(X_train, y_train)

# Оцінка моделі
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Аналіз адекватності та стійкості (може бути додано)
# Наприклад, можна провести аналіз залишкових діаграм, тест на гетероскедастичність тощо.

# Прогнозування наступного періоду
new_data = pd.DataFrame({'age': [30], 'sex': ['male'], 'bmi': [25], 'children': [2], 'smoker': ['no'], 'region': ['southwest']})
new_data_encoded = transformer.transform(new_data)
predicted_charge = model.predict(new_data_encoded)
print("Predicted charge for new data:", predicted_charge)

sample_data = data.sample(100)

# Побудова scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='charges', data=sample_data)
plt.title('Залежність між віком та вартістю медичних витрат')
plt.xlabel('Вік')
plt.ylabel('Вартість медичних витрат')
plt.grid(True)
plt.show()


# Визначення залежних та незалежних змінних
X = sample_data['age']
y = sample_data['charges']

# Використання методу найменших квадратів для побудови парної лінійної регресії
slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

# Виведення статистичних оцінок параметрів
print("Коефіцієнт нахилу (slope):", slope)
print("Зсув (intercept):", intercept)
print("Коефіцієнт кореляції (r_value):", r_value)
print("p-значення (p_value):", p_value)
print("Стандартна помилка (std_err):", std_err)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Дані')
plt.plot(X, slope*X + intercept, color='red', label='Лінійна регресія')
plt.title('Парна лінійна регресія')
plt.xlabel('Вік')
plt.ylabel('Вартість медичних витрат')
plt.legend()
plt.grid(True)
plt.show()

# Розрахунок коефіцієнта кореляції
r_value = stats.pearsonr(X, y)[0]

# Розрахунок коефіцієнта детермінації
r_squared = r_value**2

print("Коефіцієнт кореляції (r-value):", r_value)
print("Коефіцієнт детермінації (R-squared):", r_squared)

from scipy.stats import pearsonr

# Обчислення коефіцієнта кореляції та p-значення
corr_coef, p_value = pearsonr(X, y)

# Виведення результатів
print("Коефіцієнт кореляції:", corr_coef)
print("P-значення:", p_value)

# Перевірка статистичної значущості кореляції
alpha = 0.05  # Рівень значущості
if p_value < alpha:
    print("Кореляція статистично значуща")
else:
    print("Кореляція нестатистично значуща")

import statsmodels.api as sm

# Додавання константного стовпця до незалежних змінних
X_with_const = sm.add_constant(X)

# Побудова моделі лінійної регресії
model = sm.OLS(y, X_with_const).fit()

# Обчислення прогнозу середнього значення залежної змінної
mean_prediction = model.predict(X_with_const)

# Обчислення довірчого інтервалу
confidence_interval = model.conf_int(alpha=0.05)  # alpha = 0.05 для надійності 0.95

# Виведення результатів
print("Прогноз середнього значення залежної змінної:")
print(mean_prediction)

print("\nДовірчий інтервал для теоретичної лінійної парної регресії:")
print(confidence_interval)
