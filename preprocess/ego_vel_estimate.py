import numpy as np

# Define indices for the fields in the vectorized target array.
idx_ = {
    'azimuth': 0,
    'elevation': 1,
    'x_r': 2,
    'y_r': 3,
    'z_r': 4,
    'peak_db': 5,
    'r_x': 6,
    'r_y': 7,
    'r_z': 8,
    'v_d': 9,
    'noise_db': 10
}

def estimate_ego_vel(radar_scan, C_stab_r=None, retain_vel=False, maximum_range=120):
    """
    Estimate ego velocity from the radar scan and return the inlier (static) point cloud 
    along with the estimated ego velocity.
    
    - Computes ranges, azimuths, and elevations in a vectorized manner.
    - Filters valid points based on range and a transformed height condition.
    - Constructs a consolidated array (valid_targets) containing all required fields.
    
    """
    if radar_scan.shape[0] == 0:
        return False, None, radar_scan

    pts = radar_scan[:, :3]
    r = np.linalg.norm(pts, axis=1)
    azimuth = np.arctan2(pts[:, 1], pts[:, 0])
    elevation = np.arctan2(np.linalg.norm(pts[:, :2], axis=1), pts[:, 2]) - np.pi/2

    # First filtering by range.
    valid_mask = (r > 0.25) & (r < maximum_range)
    if not np.any(valid_mask):
        print('No valid points found')
        return False, None, radar_scan

    pts_valid = pts[valid_mask]

    # Apply stabilization transformation if provided.
    if C_stab_r is not None:
        p_stab = pts_valid @ C_stab_r.T
    else:
        p_stab = pts_valid

    valid_mask2 = (p_stab[:, 2] > -100) & (p_stab[:, 2] < 1000)
    if not np.any(valid_mask2):
        print('No points passed the stabilization filter')
        return False, None, radar_scan

    # Determine final valid indices (indices in the original radar_scan that passed both filters).
    valid_indices = np.where(valid_mask)[0][valid_mask2]
    M = valid_indices.shape[0]

    # Build the valid_targets array.
    # Each row: [azimuth, elevation, x, y, z, intensity, x/r, y/r, z/r, -doppler, 0]
    valid_targets = np.empty((M, 11), dtype=np.float32)
    valid_targets[:, 0] = azimuth[valid_indices]
    valid_targets[:, 1] = elevation[valid_indices]
    valid_targets[:, 2:5] = radar_scan[valid_indices, :3]
    valid_targets[:, 5] = radar_scan[valid_indices, 4]  # intensity (Power)
    valid_targets[:, 6] = radar_scan[valid_indices, 0] / r[valid_indices]
    valid_targets[:, 7] = radar_scan[valid_indices, 1] / r[valid_indices]
    valid_targets[:, 8] = radar_scan[valid_indices, 2] / r[valid_indices]
    valid_targets[:, 9] = -radar_scan[valid_indices, 3]  # negative Doppler
    valid_targets[:, 10] = 0

    if M > 2:
        v_dopplers = np.abs(valid_targets[:, idx_['v_d']])
        n = int(M * (1.0 - 0.25))  # allowed outlier percentage = 25%
        median = np.partition(v_dopplers, n)[n]
        if median < 0.05:
            # Static frame: select targets with near-zero Doppler.
            inlier_mask = v_dopplers < 0.05
            if retain_vel:
                # Preserve the original relative radial velocity.
                radar_scan_inlier = valid_targets[inlier_mask][:, [idx_['x_r'], idx_['y_r'], idx_['z_r'], idx_['v_d'], idx_['peak_db']]]
            else:
                # Transform Doppler into relative azimuth angle using the precomputed r_x value.
                radar_scan_inlier = valid_targets[inlier_mask][:, [idx_['x_r'], idx_['y_r'], idx_['z_r'], idx_['r_x'], idx_['peak_db']]]
            v_e = np.array([0, 0, 0], dtype=np.float32)
            return True, v_e, radar_scan_inlier.astype(radar_scan.dtype)
        else:
            # Use RANSAC-based estimation.
            radar_data = valid_targets[:, [idx_['r_x'], idx_['r_y'], idx_['r_z'], idx_['v_d']]]
            # For output, default to columns [x_r, y_r, z_r, v_d, peak_db]
            radar_scan_inlier = valid_targets[:, [idx_['x_r'], idx_['y_r'], idx_['z_r'], idx_['v_d'], idx_['peak_db']]]
            success, v_e, inlier_idx_best = solve3DLsqRansac(radar_data)
            if success:
                v_e_norm = v_e / np.linalg.norm(v_e)
                radar_scan_inlier = radar_scan_inlier[inlier_idx_best]
                if not retain_vel:
                    H = radar_data[inlier_idx_best, :3]
                    radar_scan_inlier[:, 3] = np.dot(H, v_e_norm)
                return success, v_e, radar_scan_inlier.astype(radar_scan.dtype)
            else:
                if not retain_vel:
                    radar_scan_inlier[:, 3] = radar_data[:, 0]
                return success, None, radar_scan_inlier.astype(radar_scan.dtype)
    else:
        print('No more than 2 valid points')
        radar_scan[:, :-2] = 0  # Set all but the last two columns to zero.
        return False, None, radar_scan

def solve3DLsqRansac(radar_data, N_ransac_points=3, ransac_iter_=18, inlier_thresh=0.15):
    """
    Perform RANSAC to solve a 3D least-squares regression on the radar data.
    """
    H_all = radar_data[:, :3]
    y_all = radar_data[:, 3]
    num_points = radar_data.shape[0]
    inlier_idx_best = []
    if num_points >= N_ransac_points:
        indices = np.arange(num_points)
        for _ in range(ransac_iter_):
            np.random.shuffle(indices)
            sample_idx = indices[:N_ransac_points]
            radar_data_iter = radar_data[sample_idx]
            flag, v_r = solve3DLsq(radar_data_iter, False)
            if flag:
                err = np.abs(y_all - H_all.dot(v_r))
                inlier_idx = np.where(err < inlier_thresh)[0]
                if len(inlier_idx) > len(inlier_idx_best):
                    inlier_idx_best = inlier_idx
    if len(inlier_idx_best) > 0:
        radar_data_inlier = radar_data[inlier_idx_best]
        flag, v_r = solve3DLsq(radar_data_inlier, True)
        return flag, v_r, inlier_idx_best
    return False, None, None

def solve3DLsq(radar_data, estimate_sigma):
    """
    Solve the 3D least-squares problem.
    """
    H = radar_data[:, :3]
    HTH = H.T.dot(H)
    y = radar_data[:, 3]
    try:
        U, singular_values, Vt = np.linalg.svd(HTH)
    except np.linalg.LinAlgError:
        return False, None
    cond = singular_values[0] / singular_values[-1]
    if abs(cond) < 1.0e3:
        try:
            L = np.linalg.cholesky(HTH)
        except np.linalg.LinAlgError:
            return False, None
        v_r_ = np.linalg.solve(L, H.T.dot(y))
        v_r = np.linalg.solve(L.T, v_r_)
        if estimate_sigma:
            e = H.dot(v_r) - y
            P_v_r = np.linalg.inv(HTH) * (e.T.dot(e)) / (H.shape[0] - 3)
            sigma_v_r = np.sqrt(np.diag(P_v_r))
            offset = np.array([0.05, 0.025, 0.05])
            P_v_r += np.diag(offset**2)
            if np.all(sigma_v_r >= 0) and np.all(sigma_v_r < np.array([0.2, 0.15, 0.2])):
                return True, v_r
        else:
            return True, v_r
    return False, None

if __name__ == "__main__":
    # Example usage for testing.
    pc = np.load("example_data.npy").astype("float32")
    success, v_e, inlier_scan = estimate_ego_vel(pc)
    print("Success:", success)
    if success:
        print("Estimated ego velocity:", v_e)
