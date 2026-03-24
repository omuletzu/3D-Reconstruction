import cv2
import numpy as np
from geometry3d.ransac import ransac
from geometry3d.utils import extract_extrinsics_E, get_best_solution

def geometry_worker(geometry_queue, lock, all_matches):
    while True:
        item = geometry_queue.get()

        if item is None:
            break

        pair_name = item['pair_name']
        
        ptsA = np.float32(item['ptsA'])
        ptsB = np.float32(item['ptsB'])

        idA = item['indicesA']
        idB = item['indicesB']

        K = item['K']

        print(f"[GEOMETRY] Ransac for {pair_name}")

        # E, ptsA_filt, ptsB_filt = ransac(ptsA, ptsB, K)
        # if E is None or ptsA_filt is None or ptsB_filt is None:
        #     geometry_queue.task_done()
        #     continue
        # solutions = extract_extrinsics_E(E)
        # R_local, t_local = get_best_solution(solutions, ptsA_filt, ptsB_filt, K)
        
        if ptsA is None or len(ptsA) < 8 or ptsB is None or len(ptsB) < 8:
            with lock:
                all_matches[pair_name] = "FAILED"

            geometry_queue.task_done()

            continue

        E, mask = cv2.findEssentialMat(
            ptsA, 
            ptsB, 
            K,
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=2.5
        )

        if E is None or mask is None:
            print(f"[GEOMETRY] FAILED for {pair_name} - E not found")

            with lock:
                all_matches[pair_name] = "FAILED"
            
            geometry_queue.task_done()
            continue

        # _, R_local, t_local, mask_pose = cv2.recoverPose(E, ptsA, ptsB, K, mask=mask)

        # final_mask = mask_pose.ravel() > 0

        # ptsA_filt = ptsA[final_mask]
        # ptsB_filt = ptsB[final_mask]

        # idA_filtered = np.array(idA)[final_mask]
        # idB_filtered = np.array(idB)[final_mask]

        solutions = extract_extrinsics_E(E)

        mask = mask.ravel() > 0

        ptsA_filt1 = ptsA[mask]
        ptsB_filt1 = ptsB[mask]

        idA_filt1 = np.array(idA)[mask]
        idB_filt1 = np.array(idB)[mask]

        R_local, t_local, ind_filt = get_best_solution(solutions, ptsA_filt1, ptsB_filt1, K)

        ptsA_final = ptsA_filt1[ind_filt]
        ptsB_final = ptsB_filt1[ind_filt]

        idA_final = idA_filt1[ind_filt]
        idB_final = idB_filt1[ind_filt]

        # E, ind_filt = ransac(ptsA, ptsB, K)

        # if E is None or ind_filt is None:
        #     print(f"[GEOMETRY] FAILED for {pair_name} - E not found")

        #     with lock:
        #         all_matches[pair_name] = "FAILED"
            
        #     geometry_queue.task_done()
        #     continue

        # ptsA_filt = ptsA[ind_filt]
        # ptsB_filt = ptsB[ind_filt]

        # idA_filtered = np.array(idA)[ind_filt]
        # idB_filtered = np.array(idA)[ind_filt]

        # solutions = extract_extrinsics_E(E)

        # R_local, t_local = get_best_solution(solutions, ptsA_filt, ptsB_filt, K)

        with lock:
            all_matches[pair_name] = {
                'ptsA': ptsA_final,
                'ptsB': ptsB_final,
                'indicesA': idA_final,
                'indicesB': idB_final,
                'R_local': R_local,
                't_local': t_local
            }

        print(f"[GEOMETRY] Done for {pair_name}. Inliers: {len(ptsA_final)}/{len(ptsA)}")

        geometry_queue.task_done()