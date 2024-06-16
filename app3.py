from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import pickle
import tensorflow as tf
from keras.models import load_model
from langchain_community.llms import Ollama
from flask_cors import CORS
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from keras.models import load_model
from keras.initializers import Orthogonal
from keras.utils import CustomObjectScope
from keras.layers import LSTM as BaseLSTM
from keras.models import load_model

app = Flask(__name__)
CORS(app)

model = load_model('cpu_usage_forecast_model.keras')
    
def process_input(anomaly_output, forecast_output):
    model_local = Ollama(model="mistral")
    prompt = f"Summarize the following for a Kubernetes Cluster CPU usage data: {anomaly_output} and {forecast_output} and also provide analysis and insights"
    prompts = [prompt]  # convert prompt to a list
    llm_result = model_local.generate(prompts)  # pass list of prompts to generate method
    summary = llm_result.generations # get the summary from the first element of the list of results
    return summary

def perform_analysis(data):
    # Anomaly detection
    clf = IsolationForest(random_state=42)
    clf.fit(data[['cpu_usage_percentage']])
    scores_pred = clf.decision_function(data[['cpu_usage_percentage']])
    anomaly_indices = np.where(scores_pred < 0)[0]
    anomaly_output = "No anomalies detected."
    if anomaly_indices.size > 0:
        anomaly_list = []
        for idx in anomaly_indices:
            if data['cpu_usage_percentage'][idx] > 100:
                anomaly_list.append(f"The CPU usage was unusually high on {data['timestamp'][idx].date()} at {data['timestamp'][idx].time()} with a value of {data['cpu_usage_percentage'][idx]}%.")
        anomaly_output = "\n".join(anomaly_list)
        
        
    # Prepare data for LSTM
    cpu_usage = data['cpu_usage_percentage'].values
    cpu_usage = cpu_usage.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler()
    cpu_usage_scaled = scaler.fit_transform(cpu_usage)

    # Create input and output sequences
    def create_sequences(data, seq_length=24):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    seq_length = 24  # Number of time steps for input sequence
    X_test = cpu_usage_scaled[-seq_length:]
    X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])

    # Forecast future values
    future_steps = 24  # Number of future steps to forecast
    future_predictions = []
    for _ in range(future_steps):
        prediction = model.predict(X_test)
        future_predictions.append(prediction[0, 0])
        X_test = np.roll(X_test, -1, axis=1)
        X_test[0, -1, :] = prediction

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    print(future_predictions)
    forecast_output = "The forecast for the next 24 hours is stable with no expected spikes."
    if np.any(future_predictions > 0.8):  # Assuming 80% is a high CPU usage
        high_index = np.where(future_predictions > 18)[0][0]
        forecast_output = f"High CPU usage is expected in the next 24 hours at step {high_index + 1}."

    return anomaly_output, forecast_output

# forecast_output = "High CPU usage is expected in the next 24 hours at steps: "
#     high_indices = np.where(future_predictions > 18)[0] 
#     if high_indices.size > 0:  # Check if any values exceed the threshold
#         for i, index in enumerate(high_indices):
#             forecast_output += f"{index + 1}"
#             if i < len(high_indices) - 1:  # Add comma if not the last step 
#                 forecast_output += ", "
#     else:
#         forecast_output = "The forecast for the next 24 hours is stable with no expected spikes."


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = 'uploaded_file.csv'
        file.save(os.path.join('/Users/tusharbhatia/Desktop/SAGE/flask-app', filename))

    data = pd.read_csv('/Users/tusharbhatia/Desktop/SAGE/flask-app/' + filename)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['hour_of_day'] = data['timestamp'].dt.hour

    anomaly_output, forecast_output = perform_analysis(data)

    # # Generate summary
    # summary = process_input(anomaly_output, forecast_output)
    # return jsonify({'summary': summary[0][0].text})

@app.route('/summary', methods=['GET'])
def get_summary():
    # Load the latest uploaded CSV file
    data = pd.read_csv('/Users/tusharbhatia/Desktop/SAGE/flask-app/uploaded_file.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['hour_of_day'] = data['timestamp'].dt.hour

    anomaly_output, forecast_output = perform_analysis(data)

    summary = process_input(anomaly_output, forecast_output)
    return jsonify({'summary': summary[0][0].text,
                    'anomaly': anomaly_output,
                    'forecast': forecast_output}) 

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port = 8000)
