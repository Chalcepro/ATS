"""Day/night cycle and perception radius."""

import config_rl


class DayNight:
    def __init__(self, length_ticks=None, start_fraction=None):
        self.length = length_ticks or config_rl.DAY_LENGTH_TICKS

        # Where in the cycle an episode opens.
        #
        # This used to be tick 0, which is time_of_day 0.0, which is_night
        # calls night — so every episode began in darkness at perception 4
        # instead of 7, and the agent's first impression of the world was
        # always a blind one. Nothing chose that; it fell out of starting a
        # counter at zero.
        #
        # Dawn is 0.25. Starting a little after it gives a full working day
        # before the first night, which is the order the agent needs to learn
        # them in: find out what the world looks like, then find out what
        # happens when the lights go out.
        f = config_rl.DAY_START_FRACTION if start_fraction is None else start_fraction
        self.tick = int(self.length * float(f)) % self.length

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
