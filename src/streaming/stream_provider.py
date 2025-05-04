from kafka import KafkaConsumer,KafkaProducer
import json
import time
import random
import torch
import pandas as pd
import sys
import pickle
import os
sys.path.append('src/')
from model.TabTransformer import TabTransformer

print(os.getcwd())
# Example function to generate synthetic logs
def generate_log():
    log = {
        'dur': random.randint(1, 100),
        'proto': random.choice(labels_encoder["proto"].classes_),
        'service': random.choice(labels_encoder["service"].classes_),
        'state': random.choice(labels_encoder["state"].classes_),
        'spkts': random.randint(1, 100),
        'dpkts': random.randint(1, 100),
        'sbytes': random.randint(100, 1000),
        'dbytes': random.randint(100, 1000),
        'rate': random.random(),
        'sload': random.random(),
        'dload': random.random(),
        'sloss': random.random(),
        'dloss': random.random(),
        'sinpkt': random.random(),
        'dinpkt': random.random(),
        'sjit': random.random(),
        'djit': random.random(),
        'swin': random.randint(1, 1000),
        'stcpb': random.randint(1, 1000),
        'dtcpb': random.randint(1, 1000),
        'dwin': random.randint(1, 1000),
        'tcprtt': random.random(),
        'synack': random.random(),
        'ackdat': random.random(),
        'smean': random.random(),
        'dmean': random.random(),
        'trans_depth': random.randint(1, 10),
        'response_body_len': random.randint(0, 1000),
        'ct_src_dport_ltm': random.randint(1, 100),
        'ct_dst_sport_ltm': random.randint(1, 100),
        'is_ftp_login': random.choice([0, 1]),
        'ct_ftp_cmd': random.randint(0, 10),
        'ct_flw_http_mthd': random.choice([ 1,  0, 25,  9,  2,  4, 16,  6, 30, 12,  3]),
        'is_sm_ips_ports': random.choice([0, 1]),
        'attack_cat': random.choice(labels_encoder["attack_cat"].classes_)
    }
    return log
if __name__ == "__main__":
    
    producer= KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )

    x_train=pd.read_csv("data/starter/train_val_data/x_train.csv").drop(columns=['Unnamed: 0'])

    with open("resources/data_preprocessing/label_encoders.pkl","rb") as f:
        labels_encoder=pickle.load(f)
    num_col = [col for col in x_train.columns if col not in labels_encoder.keys()]

    categorical_cardinates = [len(label_encoder.classes_) for label_encoder in labels_encoder.values()]

    while True:
        log = generate_log()
        producer.send('logs', log)
        print(f"Sent: {log}")
        time.sleep(1)