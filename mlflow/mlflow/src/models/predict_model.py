import mlflow
logged_model = 'runs:/65eb9e73914449c7a6abb1fd6fe6c17a/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

data = pd.read_csv('Data/casas_X.csv')
predicted = loaded_model.predict(pd.data)
data['predicted'] = predicted
data.to_csv('precos.csv')
