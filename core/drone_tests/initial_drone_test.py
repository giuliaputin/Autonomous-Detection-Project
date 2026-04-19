from drone_controls.take_off import CrazyflieTakeoff
from drone_controls.landing import land
import time

drone = CrazyflieTakeoff()

try:
    drone.connect()
    drone.takeoff(height=0.5, duration=2.0)

    # later you will append more motions here
    time.sleep(5)

    # Call landing BEFORE disconnect
    land(drone.cf, duration=4.0)

finally:
    drone.disconnect()