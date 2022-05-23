class Muscle:
    def __init__(self, line_segment):
        self.p1 = line_segment.p1
        self.p2 = line_segment.p2
        self.activated = False
        self.activate_time = None

    def update(self, t):
        if self.activated and t - self.activate_time > 2:
            self.activated = False
