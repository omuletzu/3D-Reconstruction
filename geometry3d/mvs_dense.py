import cv2
import numpy as np
import open3d as o3d

def mesh_reconstruction(loaded_images, loaded_images_colored, global_poses, K, global_points_3d):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(global_points_3d)
    

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)


    print("[MVS] Alpha Shape Meshing")
    alpha = 0.1
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

    # print("[MVS] Taubin Smoothing")
    # mesh = mesh.filter_smooth_taubin(number_of_iterations=5)
    
    mesh.compute_vertex_normals()
    
    return mesh

# def mvs_pipeline(loaded_images, loaded_images_colored, global_poses, K, global_points_3d):
#     volume = o3d.pipelines.integration.ScalableTSDFVolume(
#         voxel_length=50.0,
#         sdf_trunc=150.0,
#         color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
#     )

#     h, w = loaded_images[0].shape[:2]

#     intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
#         w, h, K[0,0], K[1,1], K[0,2], K[1,2]
#     )

#     for i in range(len(loaded_images) - 1):

#         if i not in global_poses or (i+1) not in global_poses:
#             continue

#         stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=5)

#         disparity = stereo.compute(loaded_images[i], loaded_images[i+1])
        
#         t1 = global_poses[i]['t'].flatten()
#         t2 = global_poses[i+1]['t'].flatten()

#         baseline = np.linalg.norm(t1 - t2)

#         disp_float = disparity / 16.0
#         disp_float[disp_float <= 0] = 0.1 

#         depth_map = (K[0,0] * baseline) / disp_float

#         depth_map_o3d = o3d.geometry.Image(depth_map.astype(np.float32))

#         color_bgr = loaded_images_colored[i]
#         color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
#         color_img_o3d = o3d.geometry.Image(color_rgb)

#         extrinsic = np.eye(4)
#         extrinsic[:3, :3] = global_poses[i]['R']
#         extrinsic[:3, 3] = global_poses[i]['t'].flatten()

#         extrinsic_inv = np.linalg.inv(extrinsic)
        
#         rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#             color_img_o3d, 
#             depth_map_o3d,
#             depth_scale=1.0, 
#             depth_trunc=15000.0,
#             convert_rgb_to_intensity=False
#         )
        
#         print("-" * 30)
#         print(f"DEBUG FRAME {i}")
#         print(f"Extrinsic Matrix:\n{extrinsic}")
#         print(f"Baseline calculat: {baseline:.4f}")
#         print(f"Harta adancime - Min: {np.min(depth_map):.2f}, Max: {np.max(depth_map):.2f}, Mean: {np.mean(depth_map):.2f}")

#         volume.integrate(rgbd, intrinsic_o3d, extrinsic_inv)


#     mesh = volume.extract_triangle_mesh()

#     mesh.compute_vertex_normals()

#     return mesh