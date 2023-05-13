from data.raw_data import load_data

if __name__ == '__main__':
    # Replace <data_path> with the path to your raw sensor data
    data_path = '<data_path>'

    # Load the raw sensor data
    raw_sensor_data = load_data(data_path)

    # Print the shape of the raw sensor data
    print('Raw sensor data shape:', raw_sensor_data.shape)