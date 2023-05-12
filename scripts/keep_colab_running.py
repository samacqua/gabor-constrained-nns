"""Script to trick Colab into not timing out."""

import time
import random
import string

import pyautogui

def main():
    
    # Wait a few seconds to navigate to colab + make new cell + select it.
    time.sleep(5)

    # Randomly type things every 30 seconds.
    MAX_TIME = 12   # 12 hours.
    TIME_PER_LOOP = 30  # 30 seconds.
    n_loops = int(MAX_TIME * 60 * 60 / TIME_PER_LOOP)
    for _ in range(n_loops):

        # Type a random string (takes 5 seconds).
        time_to_type = 5
        rand_string = ''.join(random.choice(string.ascii_letters) for _ in range(10))
        pyautogui.write(rand_string, interval=0.25)
        for _ in range(10):
            pyautogui.press("backspace", interval=0.25)

        time.sleep(TIME_PER_LOOP - time_to_type)



if __name__ == '__main__':
    main()
