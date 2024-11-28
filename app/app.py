import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the model if it exists, else None
# try:
model = joblib.load('model.joblib')
# except FileNotFoundError:
#     model = None

# def train_model(df):
#     """ Train the model if it doesn't exist. """
#     X = df.drop(columns='Target')
#     y = df['Target']
    
#     # Train the model
#     model = LinearRegression()
#     model.fit(X, y)
    
#     # Save the trained model
#     joblib.dump(model, 'model.joblib')
#     return model

def process_csv(file_path):
    """ Process the uploaded CSV file, make predictions, and return MSE. """
    df = pd.read_csv(file_path)
    
    # Train the model if not already trained
    # if model is None:
    #     model = train_model(df)
    
    X_test = df.drop(columns='Target')
    y_test = df['Target']
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Compute Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'Predicted values: {y_pred}')

    return mse

if __name__ == "__main__":
    # File path to your CSV data
    file_path = 'data.csv'  # Replace with your actual file path
    process_csv(file_path)
