"""Weather skill using OpenWeatherMap API"""

import os
import requests
from typing import Optional


def get_weather(city: str, units: str = "metric") -> dict:
    """
    Get current weather for a city.

    Args:
        city: City name (e.g., "London", "New York")
        units: "metric" (Celsius) or "imperial" (Fahrenheit)

    Returns:
        Weather data or error
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        # Fallback: use wttr.in (no API key needed)
        return _get_weather_wttr(city)

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": units
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        return {
            "success": True,
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
            "units": units
        }

    except requests.RequestException as e:
        return {"success": False, "error": str(e)}


def _get_weather_wttr(city: str) -> dict:
    """Fallback weather using wttr.in (no API key needed)."""
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        current = data["current_condition"][0]

        return {
            "success": True,
            "city": city,
            "temperature": int(current["temp_C"]),
            "feels_like": int(current["FeelsLikeC"]),
            "humidity": int(current["humidity"]),
            "description": current["weatherDesc"][0]["value"],
            "wind_speed": int(current["windspeedKmph"]),
            "units": "metric",
            "source": "wttr.in"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def get_forecast(city: str, days: int = 3) -> dict:
    """
    Get weather forecast for upcoming days.

    Args:
        city: City name
        days: Number of days (1-5)

    Returns:
        Forecast data
    """
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        forecast = []
        for day in data["weather"][:days]:
            forecast.append({
                "date": day["date"],
                "max_temp": int(day["maxtempC"]),
                "min_temp": int(day["mintempC"]),
                "description": day["hourly"][4]["weatherDesc"][0]["value"],
                "chance_of_rain": day["hourly"][4]["chanceofrain"]
            })

        return {
            "success": True,
            "city": city,
            "forecast": forecast
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
