import cv2
import numpy as np
from PIL import Image, ImageSequence
import os

# Function to perform segmentation on a single frame
def segment_frame(frame, mask):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.bitwise_and(gray_frame, mask)

    # Perform denoising on the grayscale frame
    denoised = cv2.medianBlur(gray_frame, 1)

    # Compute the gradient of the denoised frame
    gradient = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT, np.ones((2, 2), np.uint8))

    # Threshold the gradient image to obtain markers
    markers = gradient < 3  # Adjust this threshold for desired segmentation

    # Label the markers using connected components
    _, markers = cv2.connectedComponents(markers.astype(np.uint8))

    # Apply watershed algorithm with the gradient image and markers
    labels = cv2.watershed(cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR), markers)
    
    labels += 1
    
    # Assign labels in the mask to 0
    labels[mask == 0] = 0

    return labels

def colorize_segments_cv(labels):
    # Find unique labels excluding the background label (0)
    unique_labels = np.unique(labels)[1:]

    # Apply a predefined color map (e.g., COLORMAP_JET)
    colored_segments = cv2.applyColorMap((labels * 10).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Create a mask for label 0 (background)
    background_mask = labels == 0

    # Set the pixels corresponding to label 0 to black
    colored_segments[background_mask] = [0, 0, 0]

    return colored_segments

# Function to draw rectangles over connected components (blobs)
def draw_rectangles(frame, blobs):
    for blob in blobs:
        x, y, w, h = blob
        # Draw rectangle over the blob
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_rectangles_gray(frame, blobs):
    for blob in blobs:
        x, y, w, h = blob
        # Draw rectangle over the blob
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255), 2)
    
    return frame

# Function to merge touching segments into one blob (connected component)
def merge_touching_segments(labels):
    # Create a binary mask for segments
    segments_mask = labels > 0
    # Find connected components of segments
    _, segmented_labels = cv2.connectedComponents(segments_mask.astype(np.uint8))
    return segmented_labels

# Function to remove internal blobs
def remove_internal_blobs(blobs):
    # Initialize list to store valid blobs (external blobs)
    valid_blobs = []
    # Iterate through blobs
    for i in range(len(blobs)):
        # Flag to check if blob lies completely within another blob
        inside = False
        x1, y1, w1, h1 = blobs[i]
        # Iterate through other blobs
        for j in range(len(blobs)):
            if i != j:
                x2, y2, w2, h2 = blobs[j]
                # Check if bounding rectangle of blob i lies completely within bounding rectangle of blob j
                if x2 <= x1 and y2 <= y1 and (x2 + w2) >= (x1 + w1) and (y2 + h2) >= (y1 + h1):
                    inside = True
                    break
        # If blob i does not lie completely within any other blob, add it to valid blobs
        if not inside:
            valid_blobs.append(blobs[i])
    return valid_blobs
    
# Function to remove small components from the mask
def remove_small_components(mask, min_size, kernel = np.ones((3, 3), np.uint8)):
    # Perform morphological opening to remove small objects
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find connected components after opening
    _, labels = cv2.connectedComponents(opened_mask)

    # Iterate through unique labels
    for label in np.unique(labels):
        # Remove small components by setting their pixels to 0
        if np.sum(labels == label) < min_size:
            opened_mask[labels == label] = 0
    
    return opened_mask


# Open the video file
cap = cv2.VideoCapture(r"highway.mp4")
bg_frame = cv2.imread("cleanBG.jpg")

# Create a temporary directory to store individual frames
temp_dir = 'temp_frames'
os.makedirs(temp_dir, exist_ok=True)

# Read the video frame by frame
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Compute the absolute difference between the background and the current frame
    difference = cv2.absdiff(bg_frame, frame)

    # Apply median blur to the difference image
    blur = cv2.medianBlur(difference, 1)
    gray_blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a mask
    _, mask = cv2.threshold(gray_blur, 20, 255, cv2.THRESH_BINARY)
    
    # Remove small components from the mask
    mask = remove_small_components(mask, min_size=50)

    # Perform segmentation
    segmented_frame = segment_frame(frame, mask)
    
    # Colorize the segmented frame
    colored_segments = colorize_segments_cv(segmented_frame)

    # Merge touching segments into one blob (connected component)
    merged_segments = merge_touching_segments(segmented_frame)

    # Find bounding rectangles of blobs
    blobs = []
    # Find unique labels excluding the background label (0)
    unique_labels = np.unique(merged_segments)[1:]
    for label in unique_labels:
        blob_mask = merged_segments == label
        x, y, w, h = cv2.boundingRect(np.uint8(blob_mask))
        blobs.append((x, y, w, h))

    # Remove internal blobs
    valid_blobs = remove_internal_blobs(blobs)

    # Draw rectangles over blobs
    draw_rectangles(colored_segments, valid_blobs)
    
    segmented_frame =  (segmented_frame * 255).astype(np.uint8)
    draw_rectangles_gray(segmented_frame, valid_blobs)
    
    # Overlay the original frame with the colored segmented frame
    overlay = cv2.addWeighted(frame, 0.5, colored_segments, 0.5, 1)

    # Save frame to temporary directory
    cv2.imwrite(os.path.join(temp_dir, f'{frame_count:05d}.jpg'), segmented_frame)
    frame_count += 1

# Release the video capture object
cap.release()

# Create GIF from saved frames
frames = []
total_size = 0
for i in range(frame_count):
    frame_path = os.path.join(temp_dir, f'{i:05d}.jpg')
    frame = Image.open(frame_path)
    total_size += os.path.getsize(frame_path)
    frames.append(frame)
    os.remove(frame_path)  # Delete temporary frame file
    
    # if total_size > 15 * 1024 * 1024:  # Check if size exceeds 25 MB
        # frames.pop()  # Remove the last frame to keep the size within limit
        # break

# Save GIF using Pillow with palette-based compression
frames[0].save('output.gif', format='GIF', append_images=frames[1:], save_all=True, duration=50, loop=0, optimize=True)

# Remove temporary directory
os.rmdir(temp_dir)
