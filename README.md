# AI-Powered-SOC-Agent-for-Real-Time-Threat-Detection
Objective:
Develop an AI-powered Security Operations Center (SOC) agent capable of real-time network traffic log analysis for threat detection. The agent will leverage advanced machine learning techniques to classify security events and autonomously make decisions (e.g., generating alerts, blocking malicious traffic) based on the network logs.


## 📊 Dataset Features

The AI-Powered SOC Agent uses a subset of the UNSW-NB15 dataset, designed for network intrusion detection tasks.  
Each data point describes a network flow with detailed information:

- `dur`: Duration of the flow (seconds)
- `proto`: Protocol used (e.g., TCP, UDP)
- `service`: Network service (e.g., HTTP, SSH)
- `state`: Status of the session (e.g., connection established)
- `spkts` / `dpkts`: Number of packets sent by source/destination
- `sbytes` / `dbytes`: Number of bytes sent by source/destination
- `rate`: Packet transmission rate
- `sload` / `dload`: Source and destination load (bits/sec)
- `sloss` / `dloss`: Packets lost by source/destination
- `sinpkt` / `dinpkt`: Inter-arrival packet time at source/destination
- `sjit` / `djit`: Jitter at source/destination
- `swin` / `dwin`: TCP window advertisement by source/destination
- `stcpb` / `dtcpb`: TCP base sequence numbers
- `tcprtt`: TCP Round-Trip Time
- `synack`: SYN to SYN-ACK time
- `ackdat`: SYN-ACK to ACK time
- `smean` / `dmean`: Mean packet size for source/destination
- `trans_depth`: HTTP transaction depth
- `response_body_len`: HTTP response body size
- `ct_src_dport_ltm` / `ct_dst_sport_ltm`: Counts of unique ports used by source/destination
- `is_ftp_login`: FTP login success indicator
- `ct_ftp_cmd`: Number of FTP commands issued
- `ct_flw_http_mthd`: Number of HTTP methods used
- `is_sm_ips_ports`: Whether source and destination IPs and ports are the same
- `attack_cat`: Category of attack (e.g., Reconnaissance, DoS, Shellcode)
- `label`: Traffic label (0 = normal, 1 = attack)

These features allow the AI agent to model normal behavior and detect anomalies or intrusions efficiently.


## 🧠 TabTransformer Model Training

This project uses a **TabTransformer** architecture to perform binary classification on tabular data with both categorical and numerical features.

### 🏋️‍♀️ Training Details

- **Model**: `TabTransformer` with embedding layers for categorical variables and fully connected layers for numerical inputs.
- **Loss Function**: Binary Cross Entropy Loss (`nn.BCELoss`)
- **Optimizer**: Adam (`lr=0.001`)
- **Device**: GPU (if available)

### 📊 Dataset

The data is split into **training** and **validation** sets. Categorical features are encoded with `LabelEncoder`, and numerical features are normalized as needed.

### 🔄 Training Loop

- Each epoch:
  - The model is trained using the training set (`train_loader`).
  - Loss and accuracy are evaluated on the validation set (`val_loader`).
  - Validation accuracy and losses are printed after every epoch using `tqdm` for progress tracking.

### 💾 Model Saving

After training, the model's weights are saved as a pickle file:

```bash
tab_transformer_model.pkl
```

You can reload the model for inference using:

```python
model.load_state_dict(torch.load("tab_transformer_model.pkl"))
model.eval()
```



## 🧪 Synthetic Log Generator

This module simulates realistic cybersecurity logs for stream-based threat detection systems. It mimics the structure and distribution of the [UNSW-NB15](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) dataset, generating both normal and attack traffic to test the classification model in a live environment.

### ✅ Features

* Generates logs matching real-world network events.
* Supports streaming to Apache Kafka.
* Covers multiple attack categories: DoS, Reconnaissance, Exploits, etc.
* Tunable distribution for numeric and categorical fields.

### 📘 Log Schema (subset)

| Field        | Description                         |
| ------------ | ----------------------------------- |
| `dur`        | Duration of connection              |
| `proto`      | Protocol used (e.g., tcp, udp)      |
| `service`    | Application-level service           |
| `state`      | Connection state (e.g., CON, REQ)   |
| `spkts`      | Source packets                      |
| `dpkts`      | Destination packets                 |
| `sbytes`     | Source bytes                        |
| `dbytes`     | Destination bytes                   |
| `rate`       | Packet rate                         |
| `sload`      | Source load                         |
| `dload`      | Destination load                    |
| `sloss`      | Source packet loss                  |
| `dloss`      | Destination packet loss             |
| `attack_cat` | Attack category (e.g., Normal, DoS) |

### 🛠️ Tech Stack

* Python 3
* NumPy
* Kafka (via `kafka-python`)
* JSON serialization

### 🚀 Usage Example

```bash
# Install dependencies
pip install kafka-python numpy
```


### 🔧 Customization

* Adjust weights in `random.choices()` to control attack frequency.
* Extend the schema for IPs, timestamps, ports.
* Chain logs into sequences for multi-step attacks.

---

