import time


class PIDController:
    def __init__(self, kp=0.2, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._prev_e   = 0
        self._integral = 0
        self._prev_t   = None

    def reset(self):
        self._prev_e   = 0
        self._integral = 0
        self._prev_t   = None

    def compute(self, e):
        now = time.time()
        if self._prev_t is None:
            dt         = 0.033
            derivative = 0.0
        else:
            dt = now - self._prev_t
            dt = max(dt, 0.005)
            derivative = (e - self._prev_e) / dt

        self._integral += e * dt
        self._integral  = max(-200, min(200, self._integral))

        output = self.kp * e + self.ki * self._integral + self.kd * derivative

        self._prev_e = e
        self._prev_t = now
        return float(output)


def pid_to_speeds(pid_output, base_speed=35):
    correction = int(pid_output)
    left  = max(0, min(100, base_speed + correction))
    right = max(0, min(100, base_speed - correction))
    return left, right


if __name__ == "__main__":
    pid = PIDController(kp=0.2, ki=0.0, kd=0.0)
    for e in [0, 20, -20, 50, -50, 5]:
        out = pid.compute(e)
        l, r = pid_to_speeds(out)
        print(f"e={e:4d}px → pid={out:6.1f} → L={l} R={r}")
