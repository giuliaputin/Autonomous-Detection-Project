import time


def land(cf, duration=3.0):
    """
    Land the Crazyflie slowly and safely.

    Parameters
    ----------
    cf : Crazyflie
        Connected Crazyflie object, for example `scf.cf`
    duration : float
        Landing time in seconds. Larger = slower descent.
    """
    if cf is None:
        raise RuntimeError("Crazyflie object is None. Make sure the drone is connected.")

    print(f"Starting landing over {duration:.1f} seconds...")
    cf.high_level_commander.land(0.0, duration)
    time.sleep(duration)
    cf.high_level_commander.stop()
    print("Landing complete.")