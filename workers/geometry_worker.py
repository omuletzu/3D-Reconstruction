import cv2
import config
import os
import time
import numpy as np
from geometry3d.ransac import ransac
from geometry3d.utils import extract_extrinsics_E, get_best_solution

def geometry_worker(geometry_queue, lock, all_matches, thread_metrics):

    while True:
        item = geometry_queue.get()

        if item is None:
            break

        pair_name = item['pair_name']
        
        start_geo_img = time.perf_counter()

        ptsA = np.float32(item['ptsA'])
        ptsB = np.float32(item['ptsB'])

        idA = item['indicesA']
        idB = item['indicesB']

        K = item['K']

        print(f"[GEOMETRY] Processing for {pair_name}")
        
        if ptsA is None or len(ptsA) < 8 or ptsB is None or len(ptsB) < 8:
            with lock:
                all_matches[pair_name] = "FAILED"

            geometry_queue.task_done()

            continue

        if config.USE_MANUAL_RANSAC:
            E, mask = ransac(ptsA, ptsB, K)
        else:
            E, mask = cv2.findEssentialMat(
                ptsA, 
                ptsB, 
                K,
                method=cv2.RANSAC, 
                prob=0.999, 
                threshold=config.ESSENTIAL_MATRIX_THRESHOLD
            )

            if mask is not None:
                mask = mask.ravel() > 0

        if E is None or mask is None:
            print(f"[GEOMETRY] FAILED for {pair_name} - E not found")

            with lock:
                all_matches[pair_name] = "FAILED"
            
            geometry_queue.task_done()
            continue

        ptsA_filt1 = ptsA[mask]
        ptsB_filt1 = ptsB[mask]

        idA_filt1 = np.array(idA)[mask]
        idB_filt1 = np.array(idB)[mask]

        solutions = extract_extrinsics_E(E)

        R_local, t_local, ind_filt = get_best_solution(solutions, ptsA_filt1, ptsB_filt1, K)

        ptsA_final = ptsA_filt1[ind_filt]
        ptsB_final = ptsB_filt1[ind_filt]

        idA_final = idA_filt1[ind_filt]
        idB_final = idB_filt1[ind_filt]

        end_geo_img = time.perf_counter()

        with lock:
            thread_metrics['geo_time_sum'] += end_geo_img - start_geo_img
            thread_metrics['geo_count'] += 1

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