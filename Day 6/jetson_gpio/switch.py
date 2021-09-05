import Jetson.GPIO as GPIO
import time

switch_pin = 18

def main(): 
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(switch_pin, GPIO.IN)

    try:
        while True:
            if GPIO.input(switch_pin) == GPIO.HIGH:
                print(“Button was pushed.”)
            
            time.sleep(1)
    finally:
            GPIO.cleanup()

if __name__ == ‘__main__’:
    main()
