from data.raw_data import load_data
from perception.camera.obj_det import detect_objects, visualize_objects

if __name__ == '__main__':
    # Replace <data_path> with the path to your raw sensor data
    data_path = '/home/rachid/JmpChallenge/JMP_Challenge_Repo/data/raw-data/nuscenes'

    # Load the raw sensor data
    lidar_data, radar_data, camera_data = load_data(data_path)

    # Print the shape of the raw sensor data
    print('Raw sensor data shape loaded')

    # Assuming camera_data is a list of numpy arrays containing pre-processed images
    for image in camera_data:
        # Perform object detection on the image
        boxes, labels, scores = detect_objects(image)

        # Visualize the detections
        visualize_objects(image, boxes, labels, scores)