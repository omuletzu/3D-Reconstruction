import config
import cv2

import numpy as np

def save_to_ply(filename, points_3d, global_poses=None):

    print(f"[PLY] Saving point cloud to {filename}...")
    
    points = np.array(points_3d).reshape(-1, 3)

    cam_centers = []
    if global_poses is not None:
        for cam_id, pose in global_poses.items():
            R = pose['R']
            t = pose['t']
            
            C = -np.matrix(R).T @ np.matrix(t)

            cam_centers.append(np.array(C).flatten())
            
    num_points = len(points) + len(cam_centers)
    
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for p in points:
            if np.any(np.isnan(p)) or np.any(np.isinf(p)):
                continue
            f.write(f"{p[0]} {p[1]} {p[2]} 200 200 200\n")
            
        for c in cam_centers:
            f.write(f"{c[0]} {c[1]} {c[2]} 255 0 0\n")
            
    print(f"[PLY] Done")

def find_essential_mat(pts1, pts2, K):
    # if len(pts1) < 8:
    #     return None, None

    # ransac_thresholds = [1.0, 2.0, 3.0]
    ransac_thresholds = [1.5]
    
    for threshold in ransac_thresholds:

        E, mask = cv2.findEssentialMat(
            pts1, pts2, 
            cameraMatrix=K, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=threshold
        )
        
        if mask is not None:
            return E, mask                
            
    return None, None

def extract_extrinsics_E(E):
    U, _, Vt = np.linalg.svd(E)

    if np.linalg.det(U) < 0:
        U = -U

    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    t = U[:, 2].reshape(3, 1)

    solutions = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]

    return solutions

def triangulate_points(pts1, pts2, P1, P2):
    N = pts1.shape[0]

    A = np.zeros((N, 4, 4))

    x1 = pts1[:, 0].reshape(-1, 1)
    y1 = pts1[:, 1].reshape(-1, 1)

    x2 = pts2[:, 0].reshape(-1, 1)
    y2 = pts2[:, 1].reshape(-1, 1)

    A[:, 0, :] = x1 * P1[2, :] - P1[0, :]
    A[:, 1, :] = y1 * P1[2, :] - P1[1, :]
    A[:, 2, :] = x2 * P2[2, :] - P2[0, :]
    A[:, 3, :] = y2 * P2[2, :] - P2[1, :]

    _, _, Vt = np.linalg.svd(A)

    X = Vt[:, -1, :]

    X_3d = X[:, :3] / X[:, 3].reshape(-1, 1)

    return X_3d

def get_best_solution(solutions, pts1, pts2, K):
    K_inv = np.linalg.inv(K)

    N = len(pts1)

    pts1_homogeneus = np.hstack((pts1, np.ones((N, 1))))
    pts2_homogeneus = np.hstack((pts2, np.ones((N, 1))))

    pts1_norm = (K_inv @ pts1_homogeneus.T).T[:, :2]
    pts2_norm = (K_inv @ pts2_homogeneus.T).T[:, :2]

    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))

    best_R, best_t = None, None

    max_count = 0

    best_ind_filt = []

    for R, t in solutions:
        P2 = np.hstack((R, t))

        pts3d = triangulate_points(pts1_norm, pts2_norm, P1, P2)

        z1_mask = pts3d[:, 2] > 0

        pts3d_cam2 = (R @ pts3d.T).T + t.T

        z2_mask = pts3d_cam2[:, 2] > 0

        valid_mask = z1_mask & z2_mask

        count = np.sum(valid_mask)

        if count > max_count:
            max_count = count
            best_R = R
            best_t = t

            best_ind_filt = np.where(valid_mask)[0].tolist()

    return best_R, best_t, best_ind_filt