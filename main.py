from data.raw_data import load_data
from perception.camera.obj_det import detect_objects, visualize_objects
from perception.camera.lane_det import detect_lanes
from perception.camera.env_det import detect_brightness
import cv2

if __name__ == '__main__':
    # Replace <data_path> with the path to your raw sensor data
    data_path = '/home/rachid/JmpChallenge/JMP_Challenge_Repo/data/raw-data/nuscenes'

    # Load the raw sensor data
    lidar_data, radar_data, camera_data = load_data(data_path)

    # Print the shape of the raw sensor data
    print('Raw sensor data shape loaded')

    # Assuming camera_data is a list of numpy arrays containing pre-processed images
    for image in camera_data:

        # Perform object detection on the image:
        # person, bicycle, car, motorcycle, bus, truck
        # traffic light
        # stop sign

        boxes, labels, scores = detect_objects(image)
        # Visualize the detections
        visualize_objects(image, boxes, labels, scores)

        # LANE DETECTION:
        # Detect lanes
        result_image = detect_lanes(image)

        # Display the result
        cv2.imshow('Lanes Detection', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # BRIGHTNESS DETECTION:
        condition = detect_brightness(image)

# Lane markings --> Lane classical algorithm
# Traffic signs and signals --> Training???             X Can you write me a code that trains a model --> define training set, validation set, architecture...
        # Chatgpt allows training (R&D) and not just quick/mediocre code
# Pedestrians and cyclists --> pre-trained mscoco
# vehicles --> pre-trained mscoco
# Osbtacles and objects --> Research --> pre-trained mscoco
# Road markings and boundaries --> Research --> Lane classical algorithm 
# Environmental conditions: --> Research: brightness code, still missing: weather, road surface, traffic density, 

# Scope: talk to OFE with proposal: perception --> Nicolas(?)

""" 
    Lane markings: The camera can detect and track lane markings on the road, helping the system understand the vehicle's position within the lanes.

    Traffic signs and signals: The camera can detect and recognize various traffic signs, such as speed limits, stop signs, yield signs, and traffic lights. This information is important for understanding and obeying traffic rules.

    Pedestrians and cyclists: Cameras can identify and track pedestrians and cyclists in the vicinity of the vehicle, allowing the system to anticipate their movements and take appropriate actions to avoid collisions.

    Vehicles: The camera can detect other vehicles on the road, including cars, trucks, motorcycles, and bicycles. By analyzing their positions, speeds, and trajectories, the system can make decisions based on the surrounding traffic.

    Obstacles and objects: Cameras can detect and classify objects such as obstacles, debris, and road furniture (e.g., traffic cones). This information helps the system navigate safely and avoid collisions.

    Road markings and boundaries: Cameras can identify road boundaries, such as curbs, dividers, and guardrails, which aid in determining the drivable area and path planning.

    Environmental conditions: Cameras can assess environmental factors, such as lighting conditions, weather conditions, and visibility, which can impact driving decisions and system performance. """