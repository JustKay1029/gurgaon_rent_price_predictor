from http.server import BaseHTTPRequestHandler
from pathlib import Path
import json
import pickle

import pandas as pd


SECTOR_MEANS = {
    "Sector 53": 6.275000e07,
    "DLF City Phase 5": 5.433333e07,
    "Sector 54": 5.250000e07,
    "Sector 26": 4.666667e07,
    "Sector 50": 4.535000e07,
    "Sector 28": 4.485000e07,
    "DLF City Phase 1": 4.077143e07,
    "NH 8": 4.000000e07,
    "DLF Golf Course Road": 3.916667e07,
    "Golf course Extension Road": 3.830000e07,
    "Other": 2.500000e07,
}

PIPELINE_PATH = Path(__file__).resolve().parents[1] / "property_pipeline.pkl"
with PIPELINE_PATH.open("rb") as model_file:
    TRANSFORMER, MODEL = pickle.load(model_file)


def _to_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def predict(payload):
    sector = payload.get("sector") or "Other"
    input_data = pd.DataFrame(
        [
            {
                "status": payload.get("status") or "Ready to Move",
                "transaction": payload.get("transaction") or "Resale",
                "bathroom": _to_int(payload.get("bathroom"), 3),
                "balcony": _to_int(payload.get("balcony"), 2),
                "bedroom": _to_int(payload.get("bedroom"), 3),
                "total_area": _to_int(payload.get("total_area"), 1500),
                "clean_sector": SECTOR_MEANS.get(sector, SECTOR_MEANS["Other"]),
            }
        ]
    )
    transformed = TRANSFORMER.transform(input_data)
    return float(MODEL.predict(transformed)[0])


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode("utf-8"))

    def do_OPTIONS(self):
        self._send_json(200, {"ok": True})

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(content_length) or b"{}")
            price = predict(payload)
            self._send_json(
                200,
                {
                    "predicted_price": price,
                    "formatted_price": f"Rs {price / 10000000:.2f} Cr"
                    if price >= 10000000
                    else f"Rs {price / 100000:.2f} L",
                },
            )
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
