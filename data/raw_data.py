import numpy as np
import sys
sys.path.append('../nuscenes-devkit-master')
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import sys
sys.path.append('../nuscenes-devkit-master')

def load_data(data_path):
    """
    Loads the nuscenes dataset from the given path and returns a numpy array.
    
    Args:
        data_path (str): Path to the nuscenes dataset.
        
    Returns:
        numpy array: Numpy array containing the dataset.
    """
    dataset = []
    nusc = NuScenes(version='v1.0-mini', dataroot=data_path, verbose=True)
    
    # Loop through each sample in the dataset
    for sample in nusc.sample_iter():
        lidar_data = []
        radar_data = []
        camera_data = []
        
        # Get the lidar data for the current sample
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_path, boxes, camera_intrinsic = nusc.get_sample_data(lidar_token)
        lidar_data = LidarPointCloud.from_file(lidar_path).points.T
        
        # Get the radar data for the current sample
        for sensor_name in ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']:
            radar_token = sample['data'][sensor_name]
            radar_path, boxes, camera_intrinsic = nusc.get_sample_data(radar_token)
            radar_data.append(np.load(radar_path)['arr_0'])
        
        # Get the camera data for the current sample
        for sensor_name in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
            camera_token = sample['data'][sensor_name]
            camera_path, boxes, camera_intrinsic = nusc.get_sample_data(camera_token)
            camera_data.append(np.load(camera_path)['arr_0'])
        
        # Combine the lidar, radar, and camera data into one sample
        sample_data = np.concatenate((lidar_data, np.concatenate(radar_data, axis=1), np.concatenate(camera_data, axis=1)), axis=1)
        dataset.append(sample_data)
    
    return np.array(dataset)
