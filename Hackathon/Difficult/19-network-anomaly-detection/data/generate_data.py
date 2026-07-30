#!/usr/bin/env python3
"""Generate synthetic network traffic with anomalies."""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

n_records = 10000
anomaly_rate = 0.02
protocols = ["TCP", "UDP", "HTTP", "DNS"]
anomaly_types = ["DDoS", "Exfiltration", "PortScan"]

# Normal traffic parameters
base_packet_size = np.random.lognormal(5, 0.5, n_records)  # ~150 bytes avg
base_duration = np.random.exponential(2, n_records)
base_bytes_sent = np.random.lognormal(8, 1, n_records)
base_bytes_received = np.random.lognormal(9, 1, n_records)
base_num_packets = np.random.poisson(10, n_records)
base_error_rate = np.random.exponential(0.01, n_records)
base_time_of_day = np.random.uniform(0, 24, n_records)

# Generate anomalies in bursts
anomaly_mask = np.zeros(n_records, dtype=bool)
anomaly_type_labels = np.full(n_records, "None", dtype=object)

# DDoS: sudden spike in packet count and bytes
n_ddos = int(n_records * 0.007)
ddos_starts = np.random.choice(n_records - 50, n_ddos // 10, replace=False)
for start in ddos_starts:
    burst_len = np.random.randint(5, 15)
    for i in range(start, min(start + burst_len, n_records)):
        if np.random.random() < 0.8:
            anomaly_mask[i] = True
            anomaly_type_labels[i] = "DDoS"
            base_num_packets[i] = np.random.poisson(500)
            base_bytes_sent[i] = np.random.lognormal(12, 1)
            base_packet_size[i] = np.random.lognormal(4, 0.3)

# Data exfiltration: slow steady transfer
n_exfil = int(n_records * 0.006)
exfil_starts = np.random.choice(n_records - 100, n_exfil // 15, replace=False)
for start in exfil_starts:
    exfil_len = np.random.randint(10, 30)
    for i in range(start, min(start + exfil_len, n_records)):
        if np.random.random() < 0.6:
            anomaly_mask[i] = True
            anomaly_type_labels[i] = "Exfiltration"
            base_bytes_sent[i] = np.random.lognormal(11, 0.5)
            base_duration[i] = np.random.exponential(10)
            base_num_packets[i] = 1
            base_packet_size[i] = np.random.lognormal(7, 0.3)

# Port scanning: many small packets to different IPs
n_scan = int(n_records * 0.007)
scan_starts = np.random.choice(n_records - 30, n_scan // 8, replace=False)
for start in scan_starts:
    scan_len = np.random.randint(5, 15)
    for i in range(start, min(start + scan_len, n_records)):
        if np.random.random() < 0.7:
            anomaly_mask[i] = True
            anomaly_type_labels[i] = "PortScan"
            base_num_packets[i] = np.random.poisson(20)
            base_bytes_sent[i] = np.random.lognormal(6, 0.5)
            base_packet_size[i] = np.random.lognormal(3, 0.3)
            base_error_rate[i] = np.random.exponential(0.2)

# Build DataFrame
df = pd.DataFrame({
    "Timestamp": pd.date_range("2024-01-01", periods=n_records, freq="5s"),
    "SourceIP": np.random.randint(1000000, 9999999, n_records),
    "DestIP": np.random.randint(1000000, 9999999, n_records),
    "Protocol": np.random.choice(protocols, n_records),
    "PacketSize": np.round(base_packet_size, 1),
    "Duration": np.round(base_duration, 3),
    "BytesSent": np.round(base_bytes_sent, 1),
    "BytesReceived": np.round(base_bytes_received, 1),
    "NumPackets": base_num_packets.astype(int),
    "ErrorRate": np.round(base_error_rate, 4),
    "TimeOfDay": np.round(base_time_of_day, 2),
    "AnomalyType": anomaly_type_labels
})

# Shuffle timestamps out-of-order to simulate streaming
df = df.sample(frac=1, random_state=42).sort_values("Timestamp").reset_index(drop=True)

out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "network_traffic.csv")
df.to_csv(out_path, index=False)
print(f"Generated {len(df)} records (anomalies: {(df['AnomalyType'] != 'None').sum()}) -> {out_path}")
print(f"Anomaly breakdown:")
print(df[df["AnomalyType"] != "None"]["AnomalyType"].value_counts())
