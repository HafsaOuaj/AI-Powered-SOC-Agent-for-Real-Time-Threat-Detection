from kafka import KafkaConsumer
import json
import time
import random
import torch
import pandas as pd
import sys
import pickle
sys.path.append('src/')
from model.TabTransformer import TabTransformer
from datetime import datetime
import numpy as np

def handle_alert(log):
    # Convert any NumPy types to native Python types
    cleaned_log = {
        k: (v.item() if isinstance(v, (np.integer, np.floating)) else v)
        for k, v in log.items()
    }
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'log': cleaned_log
    }
    
    with open("alerts.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# Load the model from the pickle file
model = torch.load('resources/model/tab_transformer_model.pkl')
    # Initialize model
x_train=pd.read_csv("data/starter/train_val_data/x_train.csv").drop(columns=['Unnamed: 0'])

with open("resources/data_preprocessing/label_encoders.pkl","rb") as f:
    labels_encoder=pickle.load(f)
num_col = [col for col in x_train.columns if col not in labels_encoder.keys()]

categorical_cardinates = [len(label_encoder.classes_) for label_encoder in labels_encoder.values()]

model = TabTransformer(
        categorical_cardinates,
        num_numerical=x_train[num_col].shape[1]
    )

# Load the state dict
state_dict = torch.load('resources/model/tab_transformer_model.pkl', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Set to evaluation mode
model.eval()


if __name__ == "__main__":
    consumer = KafkaConsumer(
    'logs',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    for message in consumer:
        log = message.value
        # Preprocess the log data
        # Convert categorical features to numerical using label encoders
        print(log)
        for col, label_encoder in labels_encoder.items():
            log[col] = label_encoder.transform([log[col]])[0]

        # Convert the log to a DataFrame
        log_df = pd.DataFrame([log])

        # Separate numerical and categorical features
        x_categorical = log_df.drop(columns=num_col).values
        x_numerical = log_df[num_col].values

        # Convert to PyTorch tensors
        x_categorical_tensor = torch.tensor(x_categorical, dtype=torch.long)
        x_numerical_tensor = torch.tensor(x_numerical, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            prediction = model(x_categorical_tensor, x_numerical_tensor)

        print(f"Prediction: {prediction.numpy()}")
        if prediction[0]==1.:
            handle_alert(log)
