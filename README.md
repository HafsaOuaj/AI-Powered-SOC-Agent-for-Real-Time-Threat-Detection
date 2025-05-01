# -AI-Powered-SOC-Agent-for-Real-Time-Threat-Detection
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
