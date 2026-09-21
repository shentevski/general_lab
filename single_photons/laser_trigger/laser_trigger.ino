/* laser_trigger.ino -- Arduino Uno WiFi
 *
 *

const int TRIG_PIN = 8;              
const unsigned int PULSE_US = 1000;   // pulse width, 1000 us = 1 ms
const int N_PULSES = 1;               
const unsigned long PERIOD_MS = 10;   // spacing between pulses

void setup() {
  pinMode(TRIG_PIN, OUTPUT);
  digitalWrite(TRIG_PIN, LOW);
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
}

void loop() {
  if (Serial.available() == 0) return;

  char c = Serial.read();
  if (c != 's' && c != 'S') return;   

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
