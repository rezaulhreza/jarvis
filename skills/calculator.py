"""Calculator and math operations"""

import math
import re
from typing import Union


def calculate(expression: str) -> dict:
    """
    Safely evaluate a mathematical expression.

    Args:
        expression: Math expression (e.g., "2 + 2", "sqrt(16)", "sin(pi/2)")

    Returns:
        Calculation result
    """
    # Allowed names for safe evaluation
    safe_dict = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        # Math functions
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "factorial": math.factorial,
        # Constants
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
    }

    try:
        # Basic sanitization - only allow safe characters
        if not re.match(r'^[\d\s\+\-\*\/\(\)\.\,\^a-zA-Z_]+$', expression):
            return {"success": False, "error": "Invalid characters in expression"}

        # Replace ^ with ** for exponentiation
        expression = expression.replace("^", "**")

        # Evaluate safely
        result = eval(expression, {"__builtins__": {}}, safe_dict)

        return {
            "success": True,
            "expression": expression,
            "result": result,
            "type": type(result).__name__
        }

    except ZeroDivisionError:
        return {"success": False, "error": "Division by zero"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def convert_units(value: float, from_unit: str, to_unit: str) -> dict:
    """
    Convert between common units.

    Args:
        value: The value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted value
    """
    # Conversion factors to base units
    conversions = {
        # Length (base: meters)
        "m": 1, "km": 1000, "cm": 0.01, "mm": 0.001,
        "mi": 1609.344, "ft": 0.3048, "in": 0.0254, "yd": 0.9144,

        # Weight (base: grams)
        "g": 1, "kg": 1000, "mg": 0.001,
        "lb": 453.592, "oz": 28.3495,

        # Temperature handled separately

        # Volume (base: liters)
        "l": 1, "ml": 0.001, "gal": 3.78541, "qt": 0.946353,

        # Time (base: seconds)
        "s": 1, "min": 60, "h": 3600, "d": 86400,

        # Data (base: bytes)
        "b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4,
    }

    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    # Handle temperature separately
    if from_unit in ["c", "f", "k"] and to_unit in ["c", "f", "k"]:
        return _convert_temperature(value, from_unit, to_unit)

    try:
        if from_unit not in conversions or to_unit not in conversions:
            return {"success": False, "error": f"Unknown unit. Supported: {list(conversions.keys())}"}

        # Convert to base unit, then to target
        base_value = value * conversions[from_unit]
        result = base_value / conversions[to_unit]

        return {
            "success": True,
            "original": f"{value} {from_unit}",
            "result": result,
            "formatted": f"{result:.6g} {to_unit}"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> dict:
    """Convert temperature between Celsius, Fahrenheit, and Kelvin."""
    # Convert to Celsius first
    if from_unit == "f":
        celsius = (value - 32) * 5 / 9
    elif from_unit == "k":
        celsius = value - 273.15
    else:
        celsius = value

    # Convert from Celsius to target
    if to_unit == "f":
        result = celsius * 9 / 5 + 32
    elif to_unit == "k":
        result = celsius + 273.15
    else:
        result = celsius

    return {
        "success": True,
        "original": f"{value}°{from_unit.upper()}",
        "result": result,
        "formatted": f"{result:.2f}°{to_unit.upper()}"
    }


def percentage(value: float, total: float) -> dict:
    """Calculate percentage."""
    try:
        pct = (value / total) * 100
        return {
            "success": True,
            "value": value,
            "total": total,
            "percentage": pct,
            "formatted": f"{pct:.2f}%"
        }
    except ZeroDivisionError:
        return {"success": False, "error": "Total cannot be zero"}
