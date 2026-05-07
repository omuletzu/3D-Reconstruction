import os
import config
import cv2
import numpy as np
import open3d as o3d
import time
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix
from geometry3d.utils import triangulate_points, save_to_ply, get_best_solution, extract_extrinsics_E
from geometry3d.mvs_dense import alpha_shapes_reconstruction, poisson_mesh_reconstruction

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    
    projected_2d = np.zeros_like(points_2d, dtype=np.float64)
    
    for c in range(n_cameras):
        mask = (camera_indices == c)
        if not np.any(mask): continue
        
        proj, _ = cv2.projectPoints(points_3d[point_indices[mask]], 
                                    camera_params[c, :3], 
                                    camera_params[c, 3:], K, None)
        
        projected_2d[mask] = proj.reshape(-1, 2)
    
    res = (projected_2d - points_2d).ravel()
    return res

def run_scipy_bundle_adjustment(global_poses, ba_points_3d, ba_camera_indices, ba_point_indices, ba_points_2d, K):
    print("\n[BUNDLE ADJUSTMENT] Starting")

    n_cameras = max(global_poses.keys()) + 1
    n_points = len(ba_points_3d)
    
    camera_params = np.zeros((n_cameras, 6))
    
    for cam_id, pose in global_poses.items():
        R = pose['R']
        t = pose['t']
        rvec, _ = cv2.Rodrigues(R)
        camera_params[cam_id, :3] = rvec.flatten()
        camera_params[cam_id, 3:] = t.flatten()
        
    x0 = np.hstack((camera_params.flatten(), np.array(ba_points_3d).flatten()))
    
    camera_indices = np.array(ba_camera_indices)
    point_indices = np.array(ba_point_indices)
    points_2d = np.array(ba_points_2d)
    
    print(f"Cameras: {n_cameras}, 3D points: {n_points}, 2D observations: {len(points_2d)}")
    
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    
    t0 = time.time()
    
    if np.any(np.isnan(x0)) or np.any(np.isinf(x0)):
        x0 = np.nan_to_num(x0, nan=0.0, posinf=10000.0, neginf=-10000.0)

    res = least_squares(
        fun, 
        x0, 
        jac_sparsity=A, 
        x_scale='jac', 
        ftol=1e-2,
        method='trf', 
        loss='huber', 
        f_scale=20.0, 
        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K),
        verbose=2
    )
    
    t1 = time.time()
    
    print(f"[BUNDLE ADJUSTMENT] Finished in {t1 - t0:.2f} seconds")
    
    optimized_points_3d = res.x[n_cameras * 6:].reshape((n_points, 3))
    optimized_camera_params = res.x[:n_cameras * 6].reshape((n_cameras, 6))
    
    for cam_id in list(global_poses.keys()):
        if cam_id < n_cameras:
            rvec = optimized_camera_params[cam_id, :3]
            tvec = optimized_camera_params[cam_id, 3:].reshape(3, 1)
            R_opt, _ = cv2.Rodrigues(rvec)
            global_poses[cam_id] = {'R': R_opt, 't': tvec}
    
    return optimized_points_3d.tolist(), global_poses

def filter_outlier_observations(observations, global_poses, global_points_3d, K, threshold=5.0):
    clean_observations = []
    for obs in observations:
        cam_id, pt_idx, u, v = obs
        if cam_id not in global_poses: continue
        
        R = global_poses[cam_id]['R']
        t = global_poses[cam_id]['t']
        p3d = np.array(global_points_3d[pt_idx])
        
        p_cam = R @ p3d.reshape(3,1) + t
        if p_cam[2] <= 0: continue
        
        p_2d = K @ p_cam
        u_proj, v_proj = p_2d[0]/p_2d[2], p_2d[1]/p_2d[2]
        
        err = np.sqrt((u_proj - u)**2 + (v_proj - v)**2)
        if err < threshold:
            clean_observations.append(obs)
            
    return clean_observations

def initialize_reconstruction(all_matches, K, cam1, cam2):
    pair_name = f"{cam1}_{cam2}"
    pair = all_matches.get(pair_name)
    
    if not pair:
        pair = all_matches.get(f"{cam2}_{cam1}")
        
    pts0_final, pts1_final = pair['ptsA'], pair['ptsB']
    idx0_final, idx1_final = pair['indicesA'], pair['indicesB']

    R = pair['R_local']
    t = pair['t_local']

    global_poses = {
        cam1: {'R': np.eye(3), 't': np.zeros((3, 1))},
        cam2: {'R': R, 't': t}
    }

    P0 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = np.hstack((R, t))

    K_inv = np.linalg.inv(K)
    N_pts = len(pts0_final)

    pts0_homo = np.hstack((pts0_final, np.ones((N_pts, 1))))
    pts1_homo = np.hstack((pts1_final, np.ones((N_pts, 1))))

    pts0_norm = (K_inv @ pts0_homo.T).T[:, :2]
    pts1_norm = (K_inv @ pts1_homo.T).T[:, :2]

    # pts4D = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T)
    # pts3D = (pts4D[:3, :] / pts4D[3, :]).T
    pts3D = triangulate_points(pts0_norm, pts1_norm, P0, P1)

    global_points_3d = []
    track_map = {} 
    # mask = mask.flatten()
    observations = []

    # for i in range(len(pts3D)):
    #     if mask_pose[i]:
    #         pt_idx = len(global_points_3d)
    #         global_points_3d.append(pts3D[i])
    #         track_map[(cam1, int(idx0[i]))] = pt_idx
    #         track_map[(cam2, int(idx1[i]))] = pt_idx

    #         observations.append((cam1, pt_idx, pts0[i][0], pts0[i][1]))
    #         observations.append((cam2, pt_idx, pts1[i][0], pts1[i][1]))

    for i in range(len(pts3D)):
        pt_idx = len(global_points_3d)
        global_points_3d.append(pts3D[i])
        
        track_map[(cam1, int(idx0_final[i]))] = pt_idx
        track_map[(cam2, int(idx1_final[i]))] = pt_idx

        observations.append((cam1, pt_idx, pts0_final[i][0], pts0_final[i][1]))
        observations.append((cam2, pt_idx, pts1_final[i][0], pts1_final[i][1]))

    return global_poses, global_points_3d, track_map, observations

def find_3d_2d_correspondences(curr_cam, all_matches, track_map, global_points_3d, global_poses):
    object_points, image_points, tracking_info = [], [], []
    
    for look_back in sorted(global_poses.keys(), reverse=True):
        pair_name_A = f"{look_back}_{curr_cam}"
        pair_name_B = f"{curr_cam}_{look_back}"
        
        m = all_matches.get(pair_name_A)
        is_reverse = False
        
        if not m:
            m = all_matches.get(pair_name_B)
            is_reverse = True
            pair_name = pair_name_B
        else:
            pair_name = pair_name_A
        
        if not m or m == "PENDING" or m == "FAILED" or len(m['ptsA']) == 0:
            continue
            
        total_matches = len(m['indicesA'])
        found_in_track_map = 0
        
        for j in range(total_matches):
            if is_reverse:
                idx_back = int(m['indicesB'][j])
                idx_curr = int(m['indicesA'][j])
                pt_2d_curr = m['ptsA'][j]
            else:
                idx_back = int(m['indicesA'][j])
                idx_curr = int(m['indicesB'][j])
                pt_2d_curr = m['ptsB'][j]
            
            if (look_back, idx_back) in track_map:
                found_in_track_map += 1
                pt_idx = track_map[(look_back, idx_back)]
                
                if pt_idx not in [t['pt_idx'] for t in tracking_info]:
                    object_points.append(global_points_3d[pt_idx])
                    image_points.append(pt_2d_curr)
                    tracking_info.append({
                        's_curr': idx_curr, 
                        'pt_idx': pt_idx, 
                        'pt_2d': pt_2d_curr
                    })
        
        print(f"[PAIR] {pair_name}: {total_matches} matches found. Only {found_in_track_map} are 3D anchors.")

    if len(object_points) > 0:
        print(f"[RESULT] Sending to PnP: {len(object_points)} unique points.")

    return np.array(object_points), np.array(image_points), tracking_info

def estimate_camera_pose(obj_pts, img_pts, K, dist_coeffs=None):
    if len(obj_pts) < 10: 
        return False, None, None, None

    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    obj_pts_np = obj_pts.astype(np.float32)
    img_pts_np = img_pts.astype(np.float32)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts_np, 
        img_pts_np, 
        K, 
        dist_coeffs, 
        reprojectionError=5.0, 
        iterationsCount=100000,
        flags=cv2.SOLVEPNP_EPNP 
    )
    
    if not success or inliers is None or len(inliers) < 15:
        return False, None, None, None

    inliers_obj = obj_pts_np[inliers.flatten()]
    inliers_img = img_pts_np[inliers.flatten()]

    cv2.solvePnPRefineLM(
        inliers_obj, 
        inliers_img, 
        K, 
        dist_coeffs, 
        rvec, 
        tvec
    )

    return success, rvec, tvec, inliers

def triangulate_new_points(curr_cam, global_poses, all_matches, track_map, global_points_3d, K, observations):
    count = 0

    for prev_cam in sorted(global_poses.keys()):
        if prev_cam == curr_cam: continue
        
        pair = all_matches.get(f"{prev_cam}_{curr_cam}")
        
        if not pair or pair == "FAILED" or 'ptsA' not in pair or pair['ptsA'] is None or len(pair['ptsA']) == 0:
            continue

        R_p, t_p = global_poses[prev_cam]['R'], global_poses[prev_cam]['t']
        R_c, t_c = global_poses[curr_cam]['R'], global_poses[curr_cam]['t']

        P_prev = np.hstack((R_p, t_p))
        P_curr = np.hstack((R_c, t_c))

        ptsA = pair['ptsA']
        ptsB = pair['ptsB']
        N_pts = len(ptsA)

        K_inv = np.linalg.inv(K)

        ptsA_hom = np.hstack((ptsA, np.ones((N_pts, 1))))
        ptsB_hom = np.hstack((ptsB, np.ones((N_pts, 1))))

        ptsA_norm = (K_inv @ ptsA_hom.T).T[:, :2]
        ptsB_norm = (K_inv @ ptsB_hom.T).T[:, :2]

        pts4D = cv2.triangulatePoints(P_prev, P_curr, ptsA_norm.T, ptsB_norm.T)
        pts3D_new = (pts4D[:3, :] / pts4D[3, :]).T

        # pts3D_new = triangulate_points(ptsA_norm, ptsB_norm, P_prev, P_curr)

        for k in range(len(pts3D_new)):
            id_p, id_c = int(pair['indicesA'][k]), int(pair['indicesB'][k])
            
            if (prev_cam, id_p) not in track_map and (curr_cam, id_c) not in track_map:

                p_local_c = R_c @ pts3D_new[k].reshape(3,1) + t_c
                p_local_p = R_p @ pts3D_new[k].reshape(3,1) + t_p
                
                if p_local_c[2] > 0.0 and p_local_p[2] > 0.0:
                    
                    pt3d_reshaped = pts3D_new[k].reshape(1, 1, 3)
                    
                    proj_p, _ = cv2.projectPoints(pt3d_reshaped, R_p, t_p, K, None)
                    proj_c, _ = cv2.projectPoints(pt3d_reshaped, R_c, t_c, K, None)

                    err_p = np.linalg.norm(proj_p.flatten() - pair['ptsA'][k])
                    err_c = np.linalg.norm(proj_c.flatten() - pair['ptsB'][k])
                    
                    if err_p < 8.0 and err_c < 8.0:
                        pt_idx = len(global_points_3d)
                        global_points_3d.append(pts3D_new[k])
                        track_map[(prev_cam, id_p)] = pt_idx
                        track_map[(curr_cam, id_c)] = pt_idx

                        observations.append((prev_cam, pt_idx, pair['ptsA'][k][0], pair['ptsA'][k][1]))
                        observations.append((curr_cam, pt_idx, pair['ptsB'][k][0], pair['ptsB'][k][1]))

                        count += 1
    return count

def attempt_incremental_sfm(camA, camB, all_matches, processed_data, K, dist_coeffs):
    print(f"[SEED] Start reconstruction chaining for {camA} - {camB} cameras")
    
    t_init_start = time.perf_counter()

    global_poses, global_points_3d, track_map, observations = initialize_reconstruction(all_matches, K, camA, camB)
    
    t_init_end = time.perf_counter()

    metrics = {
        'init_time': t_init_end - t_init_start,
        'pnp_triang_time_total': 0.0,
        'ba_times': [],
        'ba_time_total': 0.0
    }

    unregistered_cameras = set(range(config.TOTAL_PHOTOS))
    unregistered_cameras.remove(camA)
    unregistered_cameras.remove(camB)

    while unregistered_cameras:
        camera_ranking = []

        for candidate_cam in unregistered_cameras:
            obj_pts, img_pts, tracking_info = find_3d_2d_correspondences(
                candidate_cam, all_matches, track_map, global_points_3d, global_poses
            )
            
            num_anchors = len(obj_pts)
            if num_anchors >= 4:
                camera_ranking.append({
                    'cam_id': candidate_cam,
                    'anchors': num_anchors,
                    'obj_pts': obj_pts,
                    'img_pts': img_pts,
                    'tracking_info': tracking_info
                })

        if not camera_ranking:
            break

        camera_ranking.sort(key=lambda x: x['anchors'], reverse=True)
        pnp_success_for_this_round = False
        
        for candidate in camera_ranking:
            best_cam = candidate['cam_id']

            t_geom_start = time.perf_counter()

            success, rvec, tvec, inliers = estimate_camera_pose(candidate['obj_pts'], candidate['img_pts'], K, dist_coeffs)

            if success and inliers is not None and len(inliers) >= 8:
                R_curr, _ = cv2.Rodrigues(rvec)
                global_poses[best_cam] = {'R': R_curr, 't': tvec}
                pnp_success_for_this_round = True
                
                unregistered_cameras.remove(best_cam)

                for prev_cam in sorted(global_poses.keys()):
                    if prev_cam == best_cam: continue
                    
                    is_reverse = False
                    pair = all_matches.get(f"{prev_cam}_{best_cam}")
                    if not pair: 
                        pair = all_matches.get(f"{best_cam}_{prev_cam}")
                        is_reverse = True
                    
                    if not pair or pair == "FAILED" or 'ptsA' not in pair or pair['ptsA'] is None or len(pair['ptsA']) == 0:
                        continue
                    
                    for k in range(len(pair['indicesA'])):
                        if is_reverse:
                            id_p, id_c = int(pair['indicesB'][k]), int(pair['indicesA'][k])
                            pt_2d_c = pair['ptsA'][k]
                        else:
                            id_p, id_c = int(pair['indicesA'][k]), int(pair['indicesB'][k])
                            pt_2d_c = pair['ptsB'][k]
                        
                        if (prev_cam, id_p) in track_map and (best_cam, id_c) not in track_map:
                            pt_idx = track_map[(prev_cam, id_p)]
                            track_map[(best_cam, id_c)] = pt_idx
                            observations.append((best_cam, pt_idx, pt_2d_c[0], pt_2d_c[1]))

                triangulate_new_points(best_cam, global_poses, all_matches, track_map, global_points_3d, K, observations)

                t_geom_end = time.perf_counter()
                metrics['pnp_triang_time_total'] += (t_geom_end - t_geom_start)

                if len(global_poses) >= 3 and len(global_poses) % 2 == 0:
                    t_ba_start = time.perf_counter()

                    ba_observations = filter_outlier_observations(observations, global_poses, global_points_3d, K)
                    ba_cam_idx = np.array([o[0] for o in ba_observations], dtype=int)
                    ba_pt_idx = np.array([o[1] for o in ba_observations], dtype=int)
                    ba_pts2d = np.array([[o[2], o[3]] for o in ba_observations], dtype=np.float64)
                    
                    global_points_3d_np, global_poses = run_scipy_bundle_adjustment(
                        global_poses, global_points_3d, ba_cam_idx, ba_pt_idx, ba_pts2d, K
                    )

                    global_points_3d = list(global_points_3d_np)

                    t_ba_end = time.perf_counter()
                    ba_duration = t_ba_end - t_ba_start
                    metrics['ba_times'].append(ba_duration)
                    metrics['ba_time_total'] += ba_duration
                
                break
            
        if not pnp_success_for_this_round:
            break

    return global_poses, global_points_3d, metrics


def global_reconstruction(all_matches, processed_data, K, dist_coeffs=None):

    t_sfm_global_start = time.perf_counter()

    seed_ranking = []

    for pair_name, m in all_matches.items():

        if m == "PENDING" or m == "FAILED" or not m or len(m['ptsA']) < 15:
            continue

        seed_ranking.append({
            'pair_name': pair_name,
            'matches': len(m['ptsA'])
        })
        
    seed_ranking.sort(key=lambda x: x['matches'], reverse=True)
    
    best_reconstruction_poses = {}
    best_reconstruction_points = []

    best_metrics = {}

    max_cameras_registered = 0
    
    MAX_SEEDS_TO_TRY = 10
    seeds_tried = 0
    
    for seed in seed_ranking:
        if seeds_tried >= MAX_SEEDS_TO_TRY:
            break
            
        camA, camB = map(int, seed['pair_name'].split('_'))
        
        poses, points, current_metrics = attempt_incremental_sfm(camA, camB, all_matches, processed_data, K, dist_coeffs)
        
        num_registered = len(poses)

        print(f"Seed result for {camA}-{camB}: {num_registered}/{config.TOTAL_PHOTOS}")
        
        if num_registered > max_cameras_registered:
            max_cameras_registered = num_registered
            best_reconstruction_poses = poses
            best_reconstruction_points = points
            best_metrics = current_metrics
            
        if max_cameras_registered >= config.TOTAL_PHOTOS - 2:
            print("Found final cameras for reconstruction")
            break
            
        seeds_tried += 1

    t_sfm_global_end = time.perf_counter()
    timp_total_sfm = t_sfm_global_end - t_sfm_global_start

    print(f"Integrating {max_cameras_registered} cameras")
    
    if max_cameras_registered < 3:
        print("Final fail")
        return
        
    print("\n==================================================")
    print("TIMER - SfM")
    print("==================================================")

    timp_evaluare_seeds = timp_total_sfm - best_metrics.get('pnp_triang_time_total', 0) - best_metrics.get('ba_time_total', 0)

    print(f"1 - Seed evaluation: {timp_evaluare_seeds:.4f} secunde")
    print(f"2 - Total PnP - Triangulation:   {best_metrics.get('pnp_triang_time_total', 0):.4f} secunde")
    print(f"3 - Total BA:   {best_metrics.get('ba_time_total', 0):.4f} secunde")
    print(f"--------------------------------------------------")
    print(f"Total SfM:              {timp_total_sfm:.4f} secunde")
    print("==================================================\n")

    ba_times = best_metrics.get('ba_times', [])
    
    if len(ba_times) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(ba_times) + 1), ba_times, marker='o', linestyle='-', color='r')
        plt.title("Evoluția timpului Bundle Adjustment per iterație")
        plt.xlabel("Iterație BA (la fiecare 2 cadre adăugate)")
        plt.ylabel("Timp de execuție")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(config.DATA_FOLDER, "ba_performance_graph.png"))
        print(f"[METRICA] Graficul a fost salvat cu succes în folderul de date: ba_performance_graph.png")

    save_to_ply(config.SAVE_POINT_CLOUD_PATH, best_reconstruction_points, best_reconstruction_poses)

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(np.array(best_reconstruction_points))

    pcd.paint_uniform_color([0.7, 0.7, 0.7])

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([pcd, axes], window_name="Sparse Point Cloud")

    if config.COMPUTE_MVS:
        print(f"[MVS] Starting dense reconstruction")

        loaded_images = []
        loaded_images_colored = []

        for i in range(config.TOTAL_PHOTOS):
            if i in processed_data and 'image_gray' in processed_data[i] and 'image_color' in processed_data[i]:
                loaded_images.append(processed_data[i]['image_gray'])
                loaded_images_colored.append(processed_data[i]['image_color'])
            else:
                print(f"[MVS] Image data {i} missing")

        if len(loaded_images) > 1:

            if config.USE_ALPHA_SHAPES:
                mesh = alpha_shapes_reconstruction(best_reconstruction_points)
            else:
                mesh = poisson_mesh_reconstruction(best_reconstruction_points)

            o3d.io.write_triangle_mesh(config.SAVE_MESH_PATH, mesh)

            print("[MVS] Mesh saved")

            o3d.visualization.draw_geometries([mesh], window_name="Mesh", mesh_show_wireframe=True, mesh_show_back_face=True)
        else:
            print("[MVS] Not enough images found")