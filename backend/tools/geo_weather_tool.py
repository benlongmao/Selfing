#!/usr/bin/env python3
"""
Geo + weather (GeoWeatherTool)

- ``get_location`` — coarse public-IP location via ip-api.com (no API key)
- ``get_weather`` — current conditions + ~12h hourly via Open-Meteo (no API key)

City table covers common CN + ASCII aliases for fast lookup; still accepts mixed-language user strings.
[2026-03-13] initial version
"""
import logging
import requests
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# CN + pinyin slugs (used for string matching, not as external API keys)
CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "北京": (39.9042, 116.4074),
    "上海": (31.2304, 121.4737),
    "广州": (23.1291, 113.2644),
    "深圳": (22.5431, 114.0579),
    "杭州": (30.2741, 120.1551),
    "成都": (30.5728, 104.0668),
    "武汉": (30.5928, 114.3052),
    "重庆": (29.5630, 106.5516),
    "西安": (34.3416, 108.9398),
    "南京": (32.0603, 118.7969),
    "苏州": (31.2990, 120.5853),
    "天津": (39.3434, 117.3616),
    "长沙": (28.2282, 112.9388),
    "郑州": (34.7472, 113.6249),
    "青岛": (36.0671, 120.3826),
    "厦门": (24.4798, 118.0894),
    "宁波": (29.8683, 121.5440),
    "合肥": (31.8612, 117.2834),
    "沈阳": (41.8057, 123.4315),
    "哈尔滨": (45.8038, 126.5349),
    "大连": (38.9140, 121.6147),
    "济南": (36.6512, 117.1201),
    "昆明": (25.0389, 102.7183),
    "贵阳": (26.6470, 106.6302),
    "兰州": (36.0611, 103.8343),
    "南昌": (28.6820, 115.8579),
    "福州": (26.0745, 119.2965),
    "海口": (20.0440, 110.1999),
    "南宁": (22.8170, 108.3665),
    "石家庄": (38.0428, 114.5149),
    "太原": (37.8706, 112.5489),
    "呼和浩特": (40.8423, 111.7496),
    "银川": (38.4872, 106.2309),
    "乌鲁木齐": (43.8256, 87.6168),
    "拉萨": (29.6520, 91.1721),
    "香港": (22.3193, 114.1694),
    "台北": (25.0330, 121.5654),
    "beijing": (39.9042, 116.4074),
    "shanghai": (31.2304, 121.4737),
    "guangzhou": (23.1291, 113.2644),
    "shenzhen": (22.5431, 114.0579),
}

WMO_CODES: Dict[int, str] = {
    0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm, slight hail", 99: "Thunderstorm, heavy hail",
}


def _get(url: str, params: Optional[Dict] = None, timeout: int = 8) -> Optional[Dict]:
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching {url}")
        return None
    except Exception as e:
        logger.warning(f"Error fetching {url}: {e}")
        return None


def _resolve_city(city: str) -> Optional[Tuple[float, float, str]]:
    """Map city text to (lat, lon, resolved label). Tries static table, then geocoding."""
    key = city.strip()
    if key in CITY_COORDS:
        lat, lon = CITY_COORDS[key]
        return lat, lon, key
    lk = key.lower()
    if lk in CITY_COORDS:
        lat, lon = CITY_COORDS[lk]
        return lat, lon, key

    data = _get("https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en", "format": "json"})
    if data and data.get("results"):
        r = data["results"][0]
        return float(r["latitude"]), float(r["longitude"]), r.get("name", city)
    return None


class GeoWeatherTool:
    """IP geolocation and Open-Meteo weather reads."""

    def get_location(self) -> Dict[str, Any]:
        """Rough public-IP city-level location (ip-api.com, no key)."""
        data = _get("http://ip-api.com/json/?lang=en&fields=status,message,country,regionName,city,lat,lon,isp,query")
        if not data:
            return {"success": False, "error": "Network error or timeout while resolving IP location"}
        if data.get("status") != "success":
            return {"success": False, "error": data.get("message", "Location lookup failed")}
        return {
            "success": True,
            "ip": data.get("query", ""),
            "country": data.get("country", ""),
            "region": data.get("regionName", ""),
            "city": data.get("city", ""),
            "latitude": data.get("lat"),
            "longitude": data.get("lon"),
            "isp": data.get("isp", ""),
            "note": "City-level accuracy (IP based), not a street address",
        }

    def get_weather(
        self,
        city: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Current weather + 12h hourly. Provide lat/lon, or a ``city`` string, or nothing (uses IP location).
        """
        city_name = city or ""

        if latitude is None or longitude is None:
            if city:
                result = _resolve_city(city)
                if not result:
                    return {
                        "success": False,
                        "error": (
                            f"Could not geocode city: {city!r}. "
                            "Use latitude/longitude, or a known place name (Chinese or English table keys)."
                        ),
                    }
                latitude, longitude, city_name = result
            else:
                loc = self.get_location()
                if not loc.get("success"):
                    return {
                        "success": False,
                        "error": f"Auto location failed: {loc.get('error')}. Pass city=... or explicit lat/lon.",
                    }
                latitude = loc["latitude"]
                longitude = loc["longitude"]
                city_name = f"{loc.get('city', '')} {loc.get('region', '')}".strip() or "current location (IP-based)"

        params = {
            "latitude": round(latitude, 4),
            "longitude": round(longitude, 4),
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,wind_direction_10m,weathercode,precipitation",
            "hourly": "temperature_2m,weathercode,precipitation_probability",
            "forecast_days": 1,
            "timezone": "Asia/Shanghai",
            "wind_speed_unit": "ms",
        }
        data = _get("https://api.open-meteo.com/v1/forecast", params=params)
        if not data:
            return {"success": False, "error": "Open-Meteo request failed — try again"}
        if data.get("error"):
            return {"success": False, "error": data.get("reason", "Open-Meteo error")}

        cur = data.get("current", {})
        hourly = data.get("hourly", {})

        wcode = cur.get("weathercode", -1)
        condition = WMO_CODES.get(wcode, f"Code {wcode} (see WMO table)")

        forecast = []
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        wcodes = hourly.get("weathercode", [])
        prec_probs = hourly.get("precipitation_probability", [])
        for i, t in enumerate(times[:12]):
            forecast.append({
                "time": t,
                "temp_c": temps[i] if i < len(temps) else None,
                "condition": WMO_CODES.get(wcodes[i] if i < len(wcodes) else -1, "Unknown"),
                "rain_prob_pct": prec_probs[i] if i < len(prec_probs) else None,
            })

        return {
            "success": True,
            "location": city_name,
            "latitude": latitude,
            "longitude": longitude,
            "current": {
                "temperature_c": cur.get("temperature_2m"),
                "feels_like_c": cur.get("apparent_temperature"),
                "humidity_pct": cur.get("relative_humidity_2m"),
                "wind_speed_ms": cur.get("wind_speed_10m"),
                "wind_direction_deg": cur.get("wind_direction_10m"),
                "precipitation_mm": cur.get("precipitation"),
                "condition": condition,
                "weathercode": wcode,
            },
            "hourly_forecast_12h": forecast,
            "data_source": "Open-Meteo (open-meteo.com)",
        }

    def get_tool_definitions(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_location",
                    "description": "Approximate city-level location for the current public IP (ip-api.com, no API key).",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Live weather plus ~12h hourly forecast (Open-Meteo, no key). Pass city name or lat/lon; if empty, uses get_location first.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "Place name (built-in table includes CN + ASCII aliases; optional—falls back to IP location)",
                            },
                            "latitude": {
                                "type": "number",
                                "description": "Latitude with longitude",
                            },
                            "longitude": {
                                "type": "number",
                                "description": "Longitude with latitude",
                            },
                        },
                        "required": [],
                    },
                },
            },
        ]


_geo_weather_instance: Optional[GeoWeatherTool] = None


def get_geo_weather_tool() -> GeoWeatherTool:
    global _geo_weather_instance
    if _geo_weather_instance is None:
        _geo_weather_instance = GeoWeatherTool()
    return _geo_weather_instance
