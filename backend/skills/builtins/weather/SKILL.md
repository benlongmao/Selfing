---
name: weather
description: Weather and coarse location via get_weather / get_location (Open-Meteo)
version: "2.0"
requires_bins: []
always_load: false
os: ["linux", "darwin", "windows"]
---

# Weather & location skill

## Important: use the dedicated tools, not curl

From v2.0 onward the stack ships ``get_location`` and ``get_weather`` backed by **Open-Meteo** (no API key, stable). **Do not** use ``shell_exec`` + ``curl wttr.in``; that path is brittle and hard to parse.

## Usage

### Weather
```
get_weather(city="Beijing")       # city name in any common language
get_weather(city="Tokyo")
get_weather(latitude=39.9, longitude=116.4)   # precise coords
get_weather()                    # omit args → IP-based coarse location
```

Response includes live temperature, apparent temperature, humidity, wind, human-readable condition (API locale), and ~12h hourly outlook.

### Current location (coarse)
```
get_location()    # city / region / country + lat-lon from public IP (city-level)
```

## Typical flows

- User asks “what’s the weather?” → ``get_location()`` then ``get_weather(city=...)``, or ``get_weather()`` alone.
- User asks “will it rain in Beijing?” → ``get_weather(city="Beijing")`` and read ``current.condition`` plus ``hourly_forecast_12h`` / ``rain_prob_pct``.
- User asks “where am I?” → ``get_location()``.

## Notes

- Location is IP-based and **city-level**, not a street address.
- Open-Meteo data is cached on the provider side (~15-minute freshness).
