import yaml
import colorsys
import numpy as np
import open3d as o3d
from snow_generator import SnowGenerator

def kitti_to_open3d(kitti_pc, intensity_as_rgb=True):   
    xyz = np.zeros((kitti_pc.shape[0], 3))
    xyz[:, 0] = kitti_pc[:, 0]  # x position of point
    xyz[:, 1] = kitti_pc[:, 1]  # y position of point
    xyz[:, 2] = kitti_pc[:, 2]  # z position of point
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    if intensity_as_rgb:
        r = kitti_pc[:, 3]  # reflectance value of point

        r_color = np.zeros_like(r)
        g_color = np.zeros_like(r)
        b_color = np.zeros_like(r)

        min_i = 0.0
        max_i = 1.0

        min_c = 0.
        max_c = 0.99
        for i in range(len(r)):
            norm_i = (r[i] - min_i) / (max_i - min_i)
            final_c = (norm_i * max_c) + ((1 - norm_i) * min_c)
            r_c, g_c, b_c = colorsys.hsv_to_rgb(final_c, 0.9, 0.9)

            r_color[i] = r_c
            g_color[i] = g_c
            b_color[i] = b_c
        
        colors = np.zeros((len(r), 3))
        colors[:, 0] = r_color
        colors[:, 1] = g_color
        colors[:, 2] = b_color
    
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def main():
    # Load sensor correction params
    sensor_dict = {}
    with open('calib/20171102_64E_S3.yaml', 'r') as stream:
        sensor_dict = yaml.safe_load(stream)
    
    # Load KITTI cloud
    path_to_kitti_cloud = "KITTI/3D/training/velodyne/000000.bin"
    pointcloud = np.fromfile(str(path_to_kitti_cloud), dtype=np.float32, count=-1).reshape([-1,4])
    # pcd = kitti_to_open3d(pointcloud, intensity_as_rgb=True)
    # o3d.visualization.draw_geometries([pcd])
    
    # Calculate elevation angle for every channel (for Velodyne HDL-64E)
    channel_angles = np.linspace(-24.8, 2.0, 64)

    # Elevation angles for every point
    elevations = np.rad2deg(np.arctan2(
        pointcloud[:, 2], 
        np.sqrt(pointcloud[:, 0] ** 2 + pointcloud[:, 1] ** 2)))
    
    # Calculate channels for every point
    channels = []
    for e in elevations:
        c = np.argmin(np.abs(channel_angles - e))
        channels.append(c)
    channels = np.array(channels)

    # Create point cloud with size N x 5: (x,y,z,intensity,channel)
    pc = np.zeros((pointcloud.shape[0], 5))
    pc[:, :4] = pointcloud
    pc[:, 4] = channels

    # Create snow generator and set it parameters
    sg = SnowGenerator(42)
    sg.set_snowfall_rate(0.5)           # 0.5-2.5 mm/h
    sg.set_terminal_velocity(0.2)       # 0.2-2 m/s
    sg.set_snow_density(0.2)            # 0.01-0.2 g/cm^3
    sg.set_snowflake_diameter(0.003)    # m
    sg.set_max_sampling_radius(50.)     # m
    sg.set_distribution("sekhon")       #'sekhon', 'gunn'

    # Generate snowflake per channel
    sg.generate_snowflakes(np.unique(channels))
    
    # Augment pointcloud with snowflakes
    stats, aug_pc = sg.augment_cloud(pc, sensor_dict)
    num_attenuated, num_removed, avg_intensity_diff = stats

    # Normalize intensity
    aug_pc[:, 3] = aug_pc[:, 3] / 255.

    # Visualize snowy cloud
    pcd = kitti_to_open3d(aug_pc, intensity_as_rgb=True)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()