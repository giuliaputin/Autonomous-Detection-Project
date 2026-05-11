import time
import cflib

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie


class CrazyflieTakeoff:
    def __init__(self, uri="radio://0/100/2M/247E000016", cache="./cache"):
        self.uri = uri
        self.cache = cache
        self.scf = None
        self.cf = None

    def connect(self):
        """Connect to the Crazyflie and enable high-level commander."""
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self.scf = SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache=self.cache))
        self.scf.open_link()
        self.cf = self.scf.cf
        self.cf.param.set_value("commander.enHighLevel", "1")
        print("Connected to Crazyflie.")

    def disconnect(self):
        """Close the link to the Crazyflie."""
        if self.scf is not None:
            self.scf.close_link()
            print("Disconnected from Crazyflie.")

    def takeoff(self, height=0.5, duration=3.0):
        """
        Take off to a given height and stay there.
        
        Parameters
        ----------
        height : float
            Target takeoff height in meters.
        duration : float
            Time in seconds to reach target height.
        """
        if self.cf is None:
            raise RuntimeError("Drone is not connected. Call connect() first.")

        print(f"Taking off to {height:.2f} m...")
        self.cf.high_level_commander.takeoff(height, duration)
        time.sleep(duration)
