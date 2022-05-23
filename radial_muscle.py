class RadialMuscle:
    def __init__(self, line_segment):
        self.p1 = line_segment.p1
        self.p2 = line_segment.p2
        self.activated = False
        self.activate_time = None
        self.refraction = False
        self.refraction_time = None

    def update(self, t):
        if self.activated and t - self.activate_time > .8:
            self.activated = False
            self.refraction = True
            self.refraction_time = t
        if self.refraction and t - self.refraction_time > 1:
            self.refraction = False
