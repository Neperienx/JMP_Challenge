import cv2
import numpy as np

def detect_lanes(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Define a region of interest (ROI)
    height, width = edges.shape
    roi_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height)
    ]
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.int32([roi_vertices]), 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Apply Hough line transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
    
    # Draw detected lines on the original image
    line_image = np.zeros_like(image)
    draw_lines(line_image, lines)
    
    # Combine the original image with the detected lines
    result = cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)
    
    return result

def draw_lines(image, lines, color=(0, 0, 255), thickness=2):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)

""" # Load the image
image = cv2.imread('/home/rachid/JmpChallenge/JMP_Challenge_Repo/data/raw-data/nuscenes/samples/CAM_FRONT/n015-2018-11-21-19-38-26+0800__CAM_FRONT__1542800865412815.jpg')

# Detect lanes
result_image = detect_lanes(image)

# Display the result
cv2.imshow('Lanes Detection', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
 """