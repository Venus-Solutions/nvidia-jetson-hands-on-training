import Jetson.GPIO as GPIO
import time

led_pin = 12

def main(): 
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(led_pin, GPIO.OUT, intial=GPIO.LOW)
    
    try:
        while True:
            GPIO.output(led_pin, GPIO.HIGH)
            time.sleep(1)
            GPIO.output(led_pin, GPIO.LOW)
            time.sleep(1)
    finally:
            GPIO.cleanup()

if __name__ == ‘__main__’:
    main()
