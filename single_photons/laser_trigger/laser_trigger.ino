/* laser_trigger.ino -- Arduino Uno WiFi
 *
 * Waits for the character 's' on the USB serial port, then emits a 5 V,
 * 1 ms pulse on TRIG_PIN to fire the laser.
 *
 * You can drive it two ways:
 *   - open the Arduino IDE Serial Monitor (115200 baud) and type s + ENTER
 *   - let measure_delay.py hold the port and send the 's' for you
 * Only one of the two can have the port open at a time.
 *
 * The Arduino's own timing jitter (microseconds) does not matter: the
 * measurement clock starts on the laser's sync output, not on this pulse.
 */

const int TRIG_PIN = 8;               // digital pin wired to the laser trigger input
const unsigned int PULSE_US = 1000;   // pulse width, 1000 us = 1 ms
const int N_PULSES = 1;               // pulses per keypress (raise to average faster)
const unsigned long PERIOD_MS = 10;   // spacing between pulses when N_PULSES > 1

void setup() {
  pinMode(TRIG_PIN, OUTPUT);
  digitalWrite(TRIG_PIN, LOW);
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
}

void loop() {
  if (Serial.available() == 0) return;

  char c = Serial.read();
  if (c != 's' && c != 'S') return;   // ignore newlines and anything else

  for (int i = 0; i < N_PULSES; i++) {
    if (i > 0) delay(PERIOD_MS);
    digitalWrite(LED_BUILTIN, HIGH);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(PULSE_US);
    digitalWrite(TRIG_PIN, LOW);
    digitalWrite(LED_BUILTIN, LOW);
  }

  Serial.println("fired");
}
