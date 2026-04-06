import time
import math
import smbus2 as smbus
from gpiozero import LED

# --- PCA9685 驅動類別 ---
class PCA9685:
    __MODE1 = 0x00
    __PRESCALE = 0xFE
    __LED0_ON_L = 0x06
    __LED0_ON_H = 0x07
    __LED0_OFF_L = 0x08
    __LED0_OFF_H = 0x09

    def __init__(self, address=0x40):
        self.bus = smbus.SMBus(1)
        self.address = address
        self.write(self.__MODE1, 0x00)

    def write(self, reg, value):
        self.bus.write_byte_data(self.address, reg, value)

    def read(self, reg):
        return self.bus.read_byte_data(self.address, reg)

    def setPWMFreq(self, freq):
        prescaleval = 25000000.0 / 4096.0 / float(freq) - 1.0
        prescale = math.floor(prescaleval + 0.5)
        oldmode = self.read(self.__MODE1)
        newmode = (oldmode & 0x7F) | 0x10
        self.write(self.__MODE1, newmode)
        self.write(self.__PRESCALE, int(prescale))
        self.write(self.__MODE1, oldmode)
        time.sleep(0.005)
        self.write(self.__MODE1, oldmode | 0x80)

    def setPWM(self, channel, on, off):
        self.write(self.__LED0_ON_L + 4 * channel, on & 0xFF)
        self.write(self.__LED0_ON_H + 4 * channel, on >> 8)
        self.write(self.__LED0_OFF_L + 4 * channel, off & 0xFF)
        self.write(self.__LED0_OFF_H + 4 * channel, off >> 8)

    def setDutycycle(self, channel, pulse):
        self.setPWM(channel, 0, int(pulse * (4095 / 100)))

    def setLevel(self, channel, value):
        self.setPWM(channel, 0, 4095 if value == 1 else 0)


# --- 馬達控制類別（供 app.py 呼叫）---
class MotorController:
    def __init__(self, speed=60):
        self.pwm = PCA9685(0x40)
        self.pwm.setPWMFreq(50)
        self.speed = speed

        self.CHANNELS = {
            'A_PWM': 0, 'A_IN1': 2, 'A_IN2': 1,
            'B_PWM': 5, 'B_IN1': 3, 'B_IN2': 4,
            'C_PWM': 6, 'C_IN1': 8, 'C_IN2': 7,
            'D_PWM': 11
        }
        self.motorD1 = LED(25)
        self.motorD2 = LED(24)
        print("✅ MotorController 初始化成功")

    def _set_motors(self, a_fwd, b_fwd, c_fwd, d_fwd, speed=None):
        spd = speed if speed is not None else self.speed

        # 馬達 A
        self.pwm.setDutycycle(self.CHANNELS['A_PWM'], spd)
        self.pwm.setLevel(self.CHANNELS['A_IN1'], 0 if a_fwd else 1)
        self.pwm.setLevel(self.CHANNELS['A_IN2'], 1 if a_fwd else 0)
        # 馬達 B
        self.pwm.setDutycycle(self.CHANNELS['B_PWM'], spd)
        self.pwm.setLevel(self.CHANNELS['B_IN1'], 1 if b_fwd else 0)
        self.pwm.setLevel(self.CHANNELS['B_IN2'], 0 if b_fwd else 1)
        # 馬達 C
        self.pwm.setDutycycle(self.CHANNELS['C_PWM'], spd)
        self.pwm.setLevel(self.CHANNELS['C_IN1'], 1 if c_fwd else 0)
        self.pwm.setLevel(self.CHANNELS['C_IN2'], 0 if c_fwd else 1)
        # 馬達 D（GPIO）
        self.pwm.setDutycycle(self.CHANNELS['D_PWM'], spd)
        if d_fwd:
            self.motorD1.off()
            self.motorD2.on()
        else:
            self.motorD1.on()
            self.motorD2.off()

    def forward(self):
        print("🚗 前進")
        self._set_motors(True, True, True, True)

    def backward(self):
        print("🚗 後退")
        self._set_motors(False, False, False, False)

    def left(self):
        print("🚗 左轉")
        # 左側馬達(A,C)倒退，右側馬達(B,D)前進
        self._set_motors(False, True, False, True)

    def right(self):
        print("🚗 右轉")
        # 左側馬達(A,C)前進，右側馬達(B,D)倒退
        self._set_motors(True, False, True, False)

    def stop(self):
        print("🛑 停止")
        for ch in [0, 5, 6, 11]:
            self.pwm.setDutycycle(ch, 0)
        self.motorD1.off()
        self.motorD2.off()


# --- 單獨測試 ---
if __name__ == '__main__':
    mc = MotorController()
    try:
        mc.forward(); time.sleep(2)
        mc.stop();   time.sleep(0.5)
        mc.backward(); time.sleep(2)
        mc.stop();   time.sleep(0.5)
        mc.left();   time.sleep(1)
        mc.stop();   time.sleep(0.5)
        mc.right();  time.sleep(1)
        mc.stop()
    except KeyboardInterrupt:
        mc.stop()
        print("\n中斷")
