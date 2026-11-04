import spidev
from gpiozero import OutputDevice
# LED bar: 850 nm IR LEDs switched by the LR7843 MOSFET module, and six WS2812B RGB LEDs.
# IR: GPIO18 (pin 12) to the module's PWM input. RGB: GPIO10 (pin 19, SPI0 MOSI) to DIN, needs dtparam=spi=on.
# The usual WS2812 library (rpi_ws281x) does not run on the Pi 5, so the bit timing comes from the SPI clock: each LED
# bit is one SPI byte, high for 2 of 8 clocks for a 0 and 6 of 8 for a 1.
IR_GPIO = 18
RGB_N   = 6
SPI_HZ  = 6400000       # the Pi 5 divides this to 6.25 MHz, 1.28 us per LED bit
RESET   = bytes(200)    # 320 us low latches the colours
ENC     = [bytes(0xFC if v >> b & 1 else 0xC0 for b in range(7, -1, -1)) for v in range(256)]

class Lights:
    """Either half can be missing (no SPI yet, say): its calls then do nothing, so web.py still runs."""
    def __init__(self):
        self.pin = self.spi = None
        try: self.pin = OutputDevice(IR_GPIO, initial_value=False)
        except Exception as e: print('leds: no IR pin:', e, flush=True)
        try:
            self.spi = spidev.SpiDev(); self.spi.open(0, 0); self.spi.max_speed_hz = SPI_HZ
        except Exception as e: print('leds: no SPI:', e, flush=True); self.spi = None

    def ir(self, on):
        if self.pin: self.pin.value = bool(on)

    def rgb(self, r, g, b):
        """All six the same colour, 0-255. WS2812B takes green first."""
        if self.spi: self.spi.writebytes2(bytes(4) + (ENC[g] + ENC[r] + ENC[b]) * RGB_N + RESET)

    def close(self):
        """Blank them: the WS2812Bs hold their colour on the 5 V rail after the Pi has shut down."""
        self.ir(False); self.rgb(0, 0, 0)
