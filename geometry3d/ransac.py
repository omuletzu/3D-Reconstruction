import numpy as np
import random

def normalize_points(pts, K):
    pts_homogeneous = np.hstack((pts, np.ones((pts.shape[0], 1))))

    K_inv = np.linalg.inv(K)

    pts_normalized = (K_inv @ pts_homogeneous.T).T

    return pts_normalized

def eight_points_algorithm(pts1, pts2):
    N = pts1.shape[0]

    A = np.zeros((N, 9))

    for i in range(N):
        x1, y1 = pts1[i, 0], pts1[i, 1]
        x2, y2 = pts2[i, 0], pts2[i, 1]

        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]

    _, _, Vt = np.linalg.svd(A)

    E = Vt[-1].reshape(3, 3)

    U_e, S_e, Vt_e = np.linalg.svd(E)

    S_e[2] = 0.0

    E_corrected = U_e @ np.diag(S_e) @ Vt_e

    return E_corrected

def ransac(pts1, pts2, K, max_iters=1000, threshold=0.01):
    pts1_normalized = normalize_points(pts1, K)
    pts2_normalized = normalize_points(pts2, K)

    N = pts1_normalized.shape[0]

    E = None
    best_inliers_idx = []
    max_inliers_count = 0

    for _ in range(max_iters):
        random_idx = random.sample(range(N), 8)

        pts1_sample = pts1_normalized[random_idx]
        pts2_sample = pts2_normalized[random_idx]

        E_current = eight_points_algorithm(pts1_sample, pts2_sample)

        inliers_current = []

        for i in range(N):
            pt1 = pts1_normalized[i]
            pt2 = pts2_normalized[i]

            error = np.abs(pt2 @ E_current @ pt1.T)

            if error < threshold:
                inliers_current.append(i)

        if len(inliers_current) > max_inliers_count:
            max_inliers_count = len(inliers_current)
            best_inliers_idx = inliers_current
            E = E_current

    if max_inliers_count >= 8:
        pts1_best = pts1_normalized[best_inliers_idx]
        pts2_best = pts2_normalized[best_inliers_idx]
        best_E = eight_points_algorithm(pts1_best, pts2_best)

    return best_E, pts1[best_inliers_idx], pts2[best_inliers_idx]