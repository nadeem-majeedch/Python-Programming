# Problem 19: Network Anomaly Detection

## Domain
Time Series + Anomaly Detection

## Problem Statement
Your company's SOC (Security Operations Center) needs an automated system to detect network anomalies in real-time. The network generates ~10,000 traffic records containing different types of anomalous behavior: DDoS attacks (sudden traffic spikes), data exfiltration (slow, stealthy transfers), and port scanning (many small packets to different IPs). Only ~2% of records are anomalous, making this a highly imbalanced detection problem.

The detector must operate in near-real-time: it cannot look at future data to flag current events. It must not only detect anomalies but also distinguish between anomaly types so the response team can take appropriate action.

## Objectives
1. Detect network anomalies with F1 > 0.85
2. Keep false positive rate below 5%
3. Detect anomalies within 3 time steps of onset (low latency)
4. Distinguish between DDoS, data exfiltration, and port scanning patterns

## Dataset
- 10000 network traffic records
- Columns: Timestamp, SourceIP (hashed), DestIP (hashed), Protocol, PacketSize, Duration, BytesSent, BytesReceived, NumPackets, ErrorRate, TimeOfDay
- Ground truth: AnomalyType (DDoS/Exfiltration/PortScan/None)
- CSV at `data/network_traffic.csv`

## Success Criteria
- **F1-score**: >0.85 on anomaly detection (binary: anomaly vs normal)
- **False Positive Rate**: <5%
- **Detection Latency**: Average <3 time steps from anomaly onset
- **Type Classification**: Per-type F1 > 0.70 for each anomaly category

## Starter Code
`starter_code.py` loads data and implements basic threshold-based detection. You must extend with proper anomaly detection approaches.

## Constraints
- Cannot use future data for detection (streaming/online scenario)
- Must evaluate both detection and classification performance
- Must report detection latency
