import os
import requests
import numpy as np
from typing import Optional

class TariffModel:
    def __init__(self, lat: float = 39.7391, lon: float = -104.9847, api_key: Optional[str] = None):
        self.lat = lat
        self.lon = lon
        self.api_key = api_key or os.getenv("NREL_API_KEY", "DEMO_KEY")
        self.base_url = "https://api.openei.org/utility_rates"
        self.rate_info = self._fetch_rate()
        
    def _fetch_rate(self):
        try:
            params = {
                "version": "8",
                "format": "json",
                "lat": self.lat,
                "lon": self.lon,
                "api_key": self.api_key,
                "detail": "true"
            }
            resp = requests.get(self.base_url, params=params, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", [])
                if items:
                    return items[0]
        except Exception:
            return None
        return None
    
    def _extract_flat_rate(self) -> Optional[float]:
        item = self.rate_info or {}
        # Try direct field
        for key in ["energy_rate", "energyrate", "fixedcharge"]:
            val = item.get(key)
            if isinstance(val, (int, float)):
                return float(val)
        # Try energyratestructure tiers
        ers = item.get("energyratestructure")
        if isinstance(ers, list) and ers:
            for block in ers:
                if isinstance(block, list):
                    for tier in block:
                        rate = tier.get("rate") if isinstance(tier, dict) else None
                        if isinstance(rate, (int, float)):
                            return float(rate)
                        if isinstance(rate, str):
                            try:
                                return float(rate)
                            except Exception:
                                continue
        return None

    def get_hourly_prices(self, hours: int = 24) -> np.ndarray:
        # If URDB provided a flat rate, use it
        flat = self._extract_flat_rate()
        if flat is not None and flat > 0:
            return np.full(hours, flat, dtype=float)
        # Fallback TOU schedule (USD/kWh)
        prices = np.zeros(hours, dtype=float)
        for h in range(hours):
            if 0 <= h < 7:
                prices[h] = 0.12
            elif 7 <= h < 17:
                prices[h] = 0.15
            elif 17 <= h < 22:
                prices[h] = 0.20
            else:
                prices[h] = 0.15
        return prices
