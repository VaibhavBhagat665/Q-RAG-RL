import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class NRELDataLoader:
    def __init__(self):
        self.base_url = "https://developer.nrel.gov/api"
        self.api_key = os.getenv("NREL_API_KEY", "DEMO_KEY")
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'nrel_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _read_or_cache(self, cache_name: str, url: str, params: Dict) -> pd.DataFrame | None:
        cache_path = os.path.join(self.cache_dir, cache_name)
        if os.path.exists(cache_path):
            try:
                return pd.read_csv(cache_path)
            except Exception:
                pass
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 0:
                lines = resp.text.strip().split('\n')
                data_lines = lines[2:]
                df = pd.read_csv(pd.StringIO('\n'.join(data_lines)))
                try:
                    df.to_csv(cache_path, index=False)
                except Exception:
                    pass
                return df
        except Exception:
            return None
        return None

    def get_solar_data(self, lat: float = 39.7391, lon: float = -104.9847, 
                      year: int = 2022) -> pd.DataFrame:
        try:
            # Prefer PSM 3.2.2 if available
            url = f"{self.base_url}/nsrdb/v2/solar/psm3-2-2-download.csv"
            params = {
                'api_key': self.api_key,
                'wkt': f'POINT({lon} {lat})',
                'names': year,
                'leap_day': 'false',
                'interval': '60',
                'utc': 'false',
                'full_name': 'Solar+Data',
                'email': 'test@nrel.gov',
                'attributes': 'ghi,dni,dhi,wind_speed,air_temperature'
            }
            df = self._read_or_cache(
                cache_name=f"solar_psm3_{lat}_{lon}_{year}.csv".replace(' ', ''),
                url=url,
                params=params,
            )
            if df is not None:
                return df
            # fallback to older endpoint
            url_fallback = f"{self.base_url}/nsrdb/v2/solar/psm3-download.csv"
            df = self._read_or_cache(
                cache_name=f"solar_psm3_legacy_{lat}_{lon}_{year}.csv".replace(' ', ''),
                url=url_fallback,
                params=params,
            )
            if df is not None:
                return df
            print("NREL Solar API unavailable, using synthetic data")
            return self._generate_synthetic_solar_data()
        except Exception as e:
            print(f"NREL API failed: {e}, using synthetic data")
            return self._generate_synthetic_solar_data()
    
    def _generate_synthetic_solar_data(self) -> pd.DataFrame:
        dates = pd.date_range('2022-01-01', periods=8760, freq='H')
        
        data = []
        for i, date in enumerate(dates):
            hour = date.hour
            day_of_year = date.timetuple().tm_yday
            
            solar_elevation = np.sin(2 * np.pi * (day_of_year - 80) / 365)
            hourly_pattern = np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 0
            
            ghi = max(0, 800 * solar_elevation * hourly_pattern * (0.8 + 0.2 * np.random.random()))
            
            wind_speed = 5 + 3 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 1)
            wind_speed = max(0, wind_speed)
            
            temperature = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + \
                         5 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 2)
            
            data.append({
                'Year': date.year,
                'Month': date.month,
                'Day': date.day,
                'Hour': date.hour,
                'GHI': ghi,
                'DNI': ghi * 0.8,
                'DHI': ghi * 0.2,
                'Wind Speed': wind_speed,
                'Temperature': temperature
            })
        
        return pd.DataFrame(data)
    
    def get_wind_data(self, lat: float = 39.7391, lon: float = -104.9847, 
                     year: int = 2022) -> pd.DataFrame:
        try:
            url = f"{self.base_url}/wind/wtk/v2/download.csv"
            params = {
                'api_key': self.api_key,
                'wkt': f'POINT({lon} {lat})',
                'names': year,
                'leap_day': 'false',
                'interval': '60',
                'utc': 'false',
                'full_name': 'Wind+Data',
                'email': 'test@nrel.gov',
                'attributes': 'windspeed_100m,winddirection_100m,temperature_100m,pressure_100m'
            }
            df = self._read_or_cache(
                cache_name=f"wind_wtk_{lat}_{lon}_{year}.csv".replace(' ', ''),
                url=url,
                params=params,
            )
            if df is not None:
                return df
            print("NREL Wind API unavailable, using synthetic data")
            return self._generate_synthetic_wind_data()
        except Exception as e:
            print(f"NREL Wind API failed: {e}, using synthetic data")
            return self._generate_synthetic_wind_data()
    
    def _generate_synthetic_wind_data(self) -> pd.DataFrame:
        dates = pd.date_range('2022-01-01', periods=8760, freq='H')
        
        data = []
        for i, date in enumerate(dates):
            hour = date.hour
            day_of_year = date.timetuple().tm_yday
            
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            daily_factor = 1 + 0.2 * np.sin(2 * np.pi * hour / 24)
            
            wind_speed = 8 * seasonal_factor * daily_factor + np.random.weibull(2) * 3
            wind_speed = np.clip(wind_speed, 0, 25)
            
            wind_direction = np.random.uniform(0, 360)
            
            temperature = 10 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + \
                         3 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 2)
            
            pressure = 1013 + np.random.normal(0, 10)
            
            data.append({
                'Year': date.year,
                'Month': date.month,
                'Day': date.day,
                'Hour': date.hour,
                'Wind Speed': wind_speed,
                'Wind Direction': wind_direction,
                'Temperature': temperature,
                'Pressure': pressure
            })
        
        return pd.DataFrame(data)
    
    def convert_to_power(self, solar_df: pd.DataFrame, wind_df: pd.DataFrame, 
                        solar_capacity: float = 3.0, wind_capacity: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        
        solar_power = []
        for _, row in solar_df.iterrows():
            ghi = row['GHI']
            temp = row.get('Temperature', 25)
            
            efficiency = 0.2 * (1 - 0.004 * (temp - 25))
            panel_area = solar_capacity * 1000 / 200
            
            power = (ghi / 1000) * panel_area * efficiency / 1000
            solar_power.append(min(power, solar_capacity))
        
        wind_power = []
        for _, row in wind_df.iterrows():
            wind_speed = row['Wind Speed']
            
            if wind_speed < 3:
                power = 0
            elif wind_speed < 12:
                power = wind_capacity * ((wind_speed - 3) / 9) ** 3
            elif wind_speed < 25:
                power = wind_capacity
            else:
                power = 0
            
            wind_power.append(power)
        
        return np.array(solar_power), np.array(wind_power)
    
    def get_load_profile(self, base_load: float = 2.5) -> np.ndarray:
        dates = pd.date_range('2022-01-01', periods=8760, freq='H')
        
        load_profile = []
        for date in dates:
            hour = date.hour
            day_of_week = date.weekday()
            day_of_year = date.timetuple().tm_yday
            
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            
            if day_of_week < 5:
                if 6 <= hour <= 9 or 17 <= hour <= 21:
                    daily_factor = 1.3
                elif 10 <= hour <= 16:
                    daily_factor = 1.1
                else:
                    daily_factor = 0.7
            else:
                daily_factor = 1.0 + 0.2 * np.sin(2 * np.pi * hour / 24)
            
            noise = 1 + 0.1 * np.random.normal()
            
            load = base_load * seasonal_factor * daily_factor * noise
            load_profile.append(max(0.5, load))
        
        return np.array(load_profile)
