"""Main entrypoint to run webcam display."""

from vision.camera import Camera
import cv2


def run_camera_feed(camera_index=0, width=640, height=480):
    camera = Camera(camera_index=camera_index, width=width, height=height)

    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("Warning: frame is None, retrying...")
                continue

            # Show the frame in a window
            cv2.imshow("Webcam Feed", frame)

            # Check for ESC key to exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    except KeyboardInterrupt:
        print("Exiting camera feed.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_feed()
