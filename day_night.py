"""Day/night cycle and perception radius."""

import config_rl


class DayNight:
    def __init__(self, length_ticks=None):
        self.length = length_ticks or config_rl.DAY_LENGTH_TICKS
        self.tick = 0

    def advance(self):
        self.tick = (self.tick + 1) % self.length

    @property
    def time_of_day(self):
        return self.tick / float(self.length)

    @property
    def is_night(self):
        t = self.time_of_day
        return t < 0.25 or t > 0.75

    def perception_radius(self, torch_active=False):
        if self.is_night and not torch_active:
            return config_rl.NIGHT_PERCEPTION_RADIUS
        if torch_active:
            return config_rl.TORCH_PERCEPTION_RADIUS
        return config_rl.PERCEPTION_RADIUS
