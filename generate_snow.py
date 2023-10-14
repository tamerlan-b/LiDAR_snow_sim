
import numpy as np
import open3d as o3d
import colorsys


def kitti_to_open3d(kitti_pc, intensity_as_rgb=True):
    x = kitti_pc[:, 0]  # x position of point
    y = kitti_pc[:, 1]  # y position of point
    z = kitti_pc[:, 2]  # z position of point
    r = kitti_pc[:, 3]  # reflectance value of point
    
    xyz = np.zeros((len(x), 3))
    xyz[:, 0] = x
    xyz[:, 1] = y
    xyz[:, 2] = z
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    if intensity_as_rgb:
        pass
        r_color = np.zeros_like(r)
        g_color = np.zeros_like(r)
        b_color = np.zeros_like(r)
        
        for i in range(len(r)):
            r_c, g_c, b_c = colorsys.hsv_to_rgb(r[i], 0.9, 0.9)
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
    
    # Load KITTI cloud
    path_to_kitti_cloud = "KITTI/3D/training/velodyne/000001.bin"
    pointcloud = np.fromfile(str(path_to_kitti_cloud), dtype=np.float32, count=-1).reshape([-1,4])
    
    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point
    r = pointcloud[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    
    pcd = kitti_to_open3d(pointcloud, intensity_as_rgb=True)
    o3d.visualization.draw_geometries([pcd])
           
    print(pointcloud.shape)
    # print("Hello, world")

if __name__ == "__main__":
    main()