import cv2

def detect_brightness(image):
    # Load the image
    #image = cv2.imread('path/to/image.jpg')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the average pixel intensity
    average_intensity = gray.mean()

    # Define a threshold to classify the brightness level
    brightness_threshold = 100  # Adjust as needed

    # Determine the environmental condition based on the average intensity
    if average_intensity < brightness_threshold:
        condition = "Low brightness"
    else:
        condition = "Normal brightness"

    # Print the detected environmental condition
    #print("Environmental Condition:", condition)
    return condition
