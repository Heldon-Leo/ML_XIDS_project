"""
src/mitre_mapping.py
=====================
Maps CSE-CIC-IDS2018 attack labels to MITRE ATT&CK (Enterprise) technique
IDs, so ML-XIDS predictions carry the same vocabulary the rest of a larger
threat-intel platform (threat graphs, dashboards, ontologies) expects.

NOTE: these are reasonable best-effort mappings based on each attack's
mechanism, not an official, audited mapping. CICIDS2018 predates widespread
dataset-to-ATT&CK crosswalks, so there is no single authoritative source.
Before using this in anything beyond a prototype/demo, have someone with
security domain expertise review and adjust these.
"""

from typing import Optional, Tuple

# label (as it appears in label_encoder.classes_) -> (technique_id, technique_name)
CICIDS2018_TO_MITRE = {
    "Benign": (None, None),
    "Bot": ("T1071", "Application Layer Protocol (C2 / Botnet)"),
    "Brute Force -Web": ("T1110", "Brute Force"),
    "Brute Force -XSS": ("T1190", "Exploit Public-Facing Application"),
    "DDOS attack-HOIC": ("T1498", "Network Denial of Service"),
    "DDOS attack-LOIC-UDP": ("T1498", "Network Denial of Service"),
    "DDoS attacks-LOIC-HTTP": ("T1498.001", "Direct Network Flood"),
    "DoS attacks-GoldenEye": ("T1499", "Endpoint Denial of Service"),
    "DoS attacks-Hulk": ("T1499", "Endpoint Denial of Service"),
    "DoS attacks-SlowHTTPTest": ("T1499.002", "Service Exhaustion Flood"),
    "DoS attacks-Slowloris": ("T1499.002", "Service Exhaustion Flood"),
    "FTP-BruteForce": ("T1110.001", "Password Guessing"),
    "Infilteration": ("T1190", "Exploit Public-Facing Application"),
    "SQL Injection": ("T1190", "Exploit Public-Facing Application"),
    "SSH-Bruteforce": ("T1110.001", "Password Guessing"),
}


def _normalize(label: str) -> str:
    return " ".join(str(label).strip().lower().split())


_NORMALIZED_LOOKUP = {_normalize(k): v for k, v in CICIDS2018_TO_MITRE.items()}


def get_mitre_info(label: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (technique_id, technique_name) for a CICIDS2018 label.
    Returns (None, None) for Benign or any unrecognized label.
    """
    return _NORMALIZED_LOOKUP.get(_normalize(label), (None, None))
