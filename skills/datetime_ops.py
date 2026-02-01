"""Date and time operations"""

from datetime import datetime, timedelta
import zoneinfo


def get_current_time(timezone: str = "Europe/London") -> dict:
    """
    Get current date and time.

    Args:
        timezone: Timezone name (e.g., "Europe/London", "America/New_York")

    Returns:
        Current time info
    """
    try:
        tz = zoneinfo.ZoneInfo(timezone)
        now = datetime.now(tz)

        return {
            "success": True,
            "datetime": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day": now.strftime("%A"),
            "timezone": timezone,
            "utc_offset": now.strftime("%z")
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def convert_timezone(time_str: str, from_tz: str, to_tz: str) -> dict:
    """
    Convert time between timezones.

    Args:
        time_str: Time string (ISO format or HH:MM)
        from_tz: Source timezone
        to_tz: Target timezone

    Returns:
        Converted time
    """
    try:
        from_zone = zoneinfo.ZoneInfo(from_tz)
        to_zone = zoneinfo.ZoneInfo(to_tz)

        # Parse the time
        if "T" in time_str:
            dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        else:
            # Assume today's date
            today = datetime.now().date()
            time_parts = time_str.split(":")
            dt = datetime(today.year, today.month, today.day,
                          int(time_parts[0]), int(time_parts[1]))

        # If naive, localize to from_tz
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=from_zone)

        # Convert to target timezone
        converted = dt.astimezone(to_zone)

        return {
            "success": True,
            "original": dt.isoformat(),
            "converted": converted.isoformat(),
            "from_timezone": from_tz,
            "to_timezone": to_tz
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def add_time(days: int = 0, hours: int = 0, minutes: int = 0,
             from_time: str = None, timezone: str = "Europe/London") -> dict:
    """
    Add time to current or specified datetime.

    Args:
        days: Days to add
        hours: Hours to add
        minutes: Minutes to add
        from_time: Starting time (ISO format), defaults to now
        timezone: Timezone

    Returns:
        Resulting datetime
    """
    try:
        tz = zoneinfo.ZoneInfo(timezone)

        if from_time:
            start = datetime.fromisoformat(from_time.replace("Z", "+00:00"))
            if start.tzinfo is None:
                start = start.replace(tzinfo=tz)
        else:
            start = datetime.now(tz)

        delta = timedelta(days=days, hours=hours, minutes=minutes)
        result = start + delta

        return {
            "success": True,
            "start": start.isoformat(),
            "result": result.isoformat(),
            "added": f"{days}d {hours}h {minutes}m"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def time_until(target: str, timezone: str = "Europe/London") -> dict:
    """
    Calculate time until a target datetime.

    Args:
        target: Target datetime (ISO format or YYYY-MM-DD)
        timezone: Timezone

    Returns:
        Time remaining
    """
    try:
        tz = zoneinfo.ZoneInfo(timezone)
        now = datetime.now(tz)

        if "T" in target:
            target_dt = datetime.fromisoformat(target.replace("Z", "+00:00"))
        else:
            target_dt = datetime.strptime(target, "%Y-%m-%d")

        if target_dt.tzinfo is None:
            target_dt = target_dt.replace(tzinfo=tz)

        diff = target_dt - now

        days = diff.days
        hours, remainder = divmod(diff.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return {
            "success": True,
            "target": target_dt.isoformat(),
            "now": now.isoformat(),
            "days": days,
            "hours": hours,
            "minutes": minutes,
            "total_seconds": int(diff.total_seconds()),
            "human": f"{days} days, {hours} hours, {minutes} minutes"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
