import cv2
import numpy as np
import open3d as o3d

# def compute_depth_map(img1, img2):
#     gray1 = img1 if len(img1.shape) == 2 else cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     gray2 = img2 if len(img2.shape) == 2 else cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#     window_size = 5
#     min_disp = 0
#     num_disp = 16 * 5

#     stereo = cv2.StereoSGBM_create(
#         minDisparity=min_disp,
#         numDisparities=num_disp,
#         blockSize=window_size,
#         P1=8 * 3 * window_size ** 2,
#         P2=32 * 3 * window_size ** 2,
#         disp12MaxDiff=1,
#         uniquenessRatio=10,
#         speckleWindowSize=100,
#         speckleRange=32
#     )

#     disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0

#     disparity[disparity <= 0] = 0.1

#     depth_map = 1000.0 / disparity

#     white_mask = gray1 > 240

#     depth_map[white_mask] = 0.0

#     return depth_map

# def mvs_pipeline(images, images_color, global_poses, K, all_3d_points):
#     print(f"[MVS] Starting")

#     volume = o3d.pipelines.integration.ScalableTSDFVolume(
#         voxel_length=0.35, 
#         sdf_trunc=1.15,
#         color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
#     )

#     h, w = images[0].shape[:2]

#     intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, K[0,0], K[1,1], K[0,2], K[1,2])

#     for i in range(len(global_poses) - 1):
#         img1_gray = images[i]
#         img2_gray = images[i + 1]

#         img1_color = images_color[i]

#         depth_map = compute_depth_map(img1_gray, img2_gray)

#         color_o3d = o3d.geometry.Image(cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB))
#         depth_o3d = o3d.geometry.Image(depth_map.astype(np.float32))

#         rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#             color_o3d, depth_o3d, 
#             depth_scale=1.0, 
#             depth_trunc=3000.0,
#             convert_rgb_to_intensity=False
#         )

#         R = global_poses[i]['R']
#         t = global_poses[i]['t']

#         extrinsic = np.eye(4)
#         extrinsic[:3, :3] = R
#         extrinsic[:3, 3] = t.flatten()

#         volume.integrate(rgbd, intrinsic, extrinsic)

#     print(f"[MVS] Extracting final 3d mesh")

#     mesh = volume.extract_triangle_mesh()
#     mesh.compute_vertex_normals()

#     if len(all_3d_points) > 10:
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(all_3d_points)

#         bbox = pcd.get_axis_aligned_bounding_box()

#         bbox.scale(1.2, bbox.get_center())

#         mesh = mesh.crop(bbox)

#     return mesh

def mvs_pipeline(loaded_images, loaded_images_colored, global_poses, K, global_points_3d):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(global_points_3d)
    

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)


    print("[MVS] Alpha Shape Meshing")
    alpha = 0.05
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

    # print("[MVS] Taubin Smoothing")
    # mesh = mesh.filter_smooth_taubin(number_of_iterations=5)
    
    mesh.compute_vertex_normals()
    
    return mesh