from LinearRegression import LinearRegression
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def r_2(y,y_pred):
  ss_res = np.sum((y_pred-y)**2)
  ss_tot = np.sum((y-np.mean(y))**2)
  return 1 - ss_res/ss_tot

def mae(y, y_pred):
  return np.sum(np.abs(y-y_pred))

def mse(y, y_pred):
  return np.sum((y-y_pred)**2)


def main():
  regression = LinearRegression()

  file_path = "multiple_linear_regression_dataset.csv"

  df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "hussainnasirkhan/multiple-linear-regression-dataset",
    file_path,
  )

  X = df[['age', 'experience']]
  y = df['income']

  scaler = StandardScaler()

  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)

  X_train = scaler.fit_transform(X_train)
  regression.fit(X_train, y_train)

  X_test = scaler.transform(X_test)
  y_pred = regression.predict(X_test)

  print(f'R2 score: {r_2(y_test, y_pred)}')
  print(f'MAE: {mae(y_test, y_pred)}')
  print(f'MSE: {mse(y_test, y_pred)}')

if __name__ == "__main__":
  main()