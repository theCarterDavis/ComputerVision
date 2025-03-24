import cv2
from ultralytics import SAM
import numpy as np
import time

# Load the SAM2 tiny model (smallest available)
model = SAM("sam2.1_t.pt")  # 't' for tiny version

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, change if needed

# Check if webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set display window
cv2.namedWindow("SAM2 Segmentation", cv2.WINDOW_NORMAL)

# Variables to store click position
click_position = None
result_overlay = None
original_frame = None


# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global click_position
    if event == cv2.EVENT_LBUTTONDOWN:
        click_position = (x, y)
        print(f"Click detected at coordinates: ({x}, {y})")


# Set mouse callback
cv2.setMouseCallback("SAM2 Segmentation", mouse_callback)

try:
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Store the original frame
        original_frame = frame.copy()

        # Display frame with segmentation overlay if available
        display_frame = original_frame.copy()
        if result_overlay is not None:
            # Blend the result overlay with the original frame
            display_frame = cv2.addWeighted(display_frame, 0.7, result_overlay, 0.3, 0)

        # Process new click
        if click_position is not None:
            x, y = click_position

            # Prepare point prompt
            point_prompt = np.array([[x, y]])

            # Process frame with SAM2 using point prompt
            start_time = time.time()
            results = model(frame,
                            points=point_prompt,
                            labels=np.array([1]),  # 1 = foreground point
                            verbose=False)
            inference_time = time.time() - start_time

            # Visualize results
            result_overlay = results[0].plot()

            # Add inference time info
            fps_text = f"Inference: {inference_time:.2f}s"
            cv2.putText(display_frame, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Reset click position
            click_position = None

        # Display the current point (if any)
        if click_position is not None:
            cv2.circle(display_frame, click_position, 5, (0, 0, 255), -1)

        # Display the processed frame
        cv2.imshow("SAM2 Segmentation", display_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Clear segmentation on 'c' key press
        if cv2.waitKey(1) & 0xFF == ord('c'):
            result_overlay = None

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed")