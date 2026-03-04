import numpy as np

def extract_extrinsics_E(E):
    U, S, Vt = np.linalg.svd(E)

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

def triangulate_points(pt1_normalized, pt2_normalized, P1, P2):
    A = np.zeros((4, 4))

    A[0] = pt1_normalized[0] * P1[2, :] - P1[0, :]
    A[1] = pt1_normalized[1] * P1[2, :] - P1[1, :]
    
    A[2] = pt2_normalized[0] * P2[2, :] - P2[0, :]
    A[3] = pt2_normalized[1] * P2[2, :] - P2[1, :]

    _, _, Vt = np.linalg.svd(A)

    X = Vt[-1]

    X = X[:3] / X[3]

    return X

def get_best_solution(solutions, pts1, pts2, K):
    K_inv = np.linalg.inv(K)

    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))

    best_R, best_t = None, None

    max_count = 0
    best_3d_points = []

    for R, t in solutions:
        P2 = np.hstack((R, t))

        count = 0
        current_3d_points = []

        for i in range(len(pts1)):
            pt1_homogeneus = np.array([pts1[i, 0], pts1[i, 1], 1.0])
            pt2_homogeneus = np.array([pts2[i, 0], pts2[i, 1], 1.0])

            pt1_normalized = K_inv @ pt1_homogeneus
            pt2_normalized = K_inv @ pt2_homogeneus

            pt_3d = triangulate_points(pt1_normalized, pt2_normalized, P1, P2)

            z1 = pt_3d[2]

            pt_3d_cam2 = R @ pt_3d.reshape(3, 1) + t

            z2 = pt_3d_cam2[2, 0]

            if z1 > 0 and z2 > 0:
                count += 1
                current_3d_points.append(pt_3d)

        if count > max_count:
            max_count = count
            best_R = R
            best_t = t
            best_3d_points = current_3d_points

    return best_R, best_t, np.array(best_3d_points)