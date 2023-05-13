from data.raw_data import load_data

if __name__ == '__main__':
    # Replace <data_path> with the path to your raw sensor data
    data_path = '/home/rachid/JmpChallenge/JMP_Challenge_Repo/data/raw-data/nuscenes'

    # Load the raw sensor data
    rawlidar_data, radar_data, camera_data_sensor_data = load_data(data_path)

    # Print the shape of the raw sensor data
    print('Raw sensor data shape loaded')