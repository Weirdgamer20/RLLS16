import math
from typing import Callable


class MultiRateScheduler:
    """
    RLLS 16 Multi-Rate Simulation Scheduler.
    Decouples rendering loop (60 FPS) from simulation physics.
    Fundamental temporal tick = 1 simulation hour (3600 seconds).

    Schedules:
      - Hourly (every tick): Solar radiation, surface energy balance, wind, humidity, precipitation.
      - 6-Hourly (every 6 ticks): Hydrological & atmospheric pressure re-balancing.
      - Daily (every 24 ticks): Soil moisture, emergent vegetation biomass, human physiological/demographic updates.
      - Monthly (every 720 ticks / 30 days): Long-term climate statistics.
    """

    def __init__(self, initial_time_sec: float = 0.0):
        self.sim_time_sec = initial_time_sec
        self.time_speed = 1.0  # Multiplier: 1x, 10x, 100x, 1000x
        self.is_paused = False

        # Accumulator for sub-tick real-time progression
        self._accumulator_sec = 0.0

        # Subscribed callbacks
        self.hourly_callbacks: list[Callable[[float], None]] = []
        self.six_hourly_callbacks: list[Callable[[float], None]] = []
        self.daily_callbacks: list[Callable[[], None]] = []
        self.monthly_callbacks: list[Callable[[], None]] = []

        # Internal tick counters
        self.total_hours_elapsed = int(initial_time_sec // 3600.0)

    def subscribe_hourly(self, callback: Callable[[float], None]):
        self.hourly_callbacks.append(callback)

    def subscribe_six_hourly(self, callback: Callable[[float], None]):
        self.six_hourly_callbacks.append(callback)

    def subscribe_daily(self, callback: Callable[[], None]):
        self.daily_callbacks.append(callback)

    def subscribe_monthly(self, callback: Callable[[], None]):
        self.monthly_callbacks.append(callback)

    def advance(self, real_dt_sec: float) -> int:
        """
        Advance simulation time based on real frame dt and current playback speed.
        Returns the number of hourly ticks executed in this frame.
        """
        if self.is_paused:
            return 0

        # Progress simulation time
        sim_delta = real_dt_sec * self.time_speed
        self.sim_time_sec += sim_delta
        self._accumulator_sec += sim_delta

        ticks_executed = 0
        tick_duration = 3600.0 # 1 hour

        # Prevent runaway spiral of death if paused or large dt: cap to at most 48 ticks per frame
        max_ticks = 48
        while self._accumulator_sec >= tick_duration and ticks_executed < max_ticks:
            self._accumulator_sec -= tick_duration
            self.total_hours_elapsed += 1
            current_time = self.total_hours_elapsed * tick_duration
            ticks_executed += 1

            # 1. Hourly Callbacks (Solar, Temperature, Wind, Rain)
            for cb in self.hourly_callbacks:
                cb(current_time)

            # 2. Six-Hourly Callbacks (every 6 hours)
            if self.total_hours_elapsed % 6 == 0:
                for cb in self.six_hourly_callbacks:
                    cb(current_time)

            # 3. Daily Callbacks (every 24 hours)
            if self.total_hours_elapsed % 24 == 0:
                for cb in self.daily_callbacks:
                    cb()

            # 4. Monthly Callbacks (every 30 days = 720 hours)
            if self.total_hours_elapsed % 720 == 0:
                for cb in self.monthly_callbacks:
                    cb()

        if ticks_executed >= max_ticks:
            self._accumulator_sec = 0.0 # Discard overflow under extreme speed

        return ticks_executed

    def get_formatted_time(self) -> tuple[str, str, str]:
        """Convert current simulation seconds into Year, Day, HH:MM:SS format."""
        total_seconds = int(self.sim_time_sec)
        seconds_in_day = 86400
        days_in_year = 365

        total_days = total_seconds // seconds_in_day
        year = (total_days // days_in_year) + 1
        day = (total_days % days_in_year) + 1

        sec_of_day = total_seconds % seconds_in_day
        h = sec_of_day // 3600
        m = (sec_of_day % 3600) // 60
        s = sec_of_day % 60

        year_str = f"Year {year:07,d}"
        day_str = f"Day {day:03d}"
        time_str = f"{h:02d}:{m:02d}:{s:02d}"
        return year_str, day_str, time_str

    def reset(self, initial_time_sec: float = 0.0):
        self.sim_time_sec = initial_time_sec
        self._accumulator_sec = 0.0
        self.total_hours_elapsed = int(initial_time_sec // 3600.0)
        self.is_paused = False
