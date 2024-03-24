import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture(r"data/sample_video.mp4")

# Function to generate random colors for each label
def generate_random_colors(num_colors):
    return np.random.randint(0, 255, (num_colors, 3), dtype=np.uint8)

# Function to perform segmentation on a single frame
def segment_frame(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform denoising on the grayscale frame
    denoised = cv2.medianBlur(gray_frame, 5)

    # Compute the gradient of the denoised frame
    gradient = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT, np.ones((2, 2), np.uint8))

    # Threshold the gradient image to obtain markers
    markers = gradient < 6 # 3  # Adjust this threshold for desired segmentation

    # Label the markers using connected components
    _, markers = cv2.connectedComponents(markers.astype(np.uint8))

    # Convert markers to the correct data type expected by the watershed function
    markers = markers.astype(np.int32)

    # Apply watershed algorithm with the gradient image and markers
    labels = cv2.watershed(cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR), markers)

    return labels

# Function to visualize the segmented frame
def visualize_segmentation(labels):
    # Generate random colors for each label
    color_map = generate_random_colors(np.max(labels) + 1)
    color_map[-1] = [0, 0, 0]  # Assign background color to black

    # Create RGB image using the label map
    segmented_frame_rgb = color_map[labels]

    return segmented_frame_rgb

# Read the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform segmentation
    segmented_frame = segment_frame(frame)

    # Display original frame
    cv2.imshow("Original Frame", frame)

    # Display segmented frame
    cv2.imshow("Segmented Frame", (segmented_frame * 255).astype(np.uint8))

    # Press 'q' to exit
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
