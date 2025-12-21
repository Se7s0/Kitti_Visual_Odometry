def estimate_motion(match, kp1, kp2, k, depth1=None, lidar_depth=None, lidar_only=False, max_depth=3000):
"""
Estimate motion between two frames using PnP or Essential Matrix.

    Args:
        match: List of cv2.DMatch objects
        kp1, kp2: Keypoints from frame 1 and 2
        k: Camera intrinsic matrix
        depth1: Stereo depth map
        lidar_depth: LiDAR projected depth map (sparse)
        lidar_only: If True, only use keypoints with LiDAR depth
        max_depth: Maximum valid depth

    Returns:
        rmat, tvec, image1_points, image2_points
    """
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))

    image1_points = np.float32([kp1[m.queryIdx].pt for m in match])
    image2_points = np.float32([kp2[m.trainIdx].pt for m in match])
    if depth1 is not None:
        cx = k[0, 2]
        cy = k[1, 2]
        fx = k[0, 0]
        fy = k[1, 1]
        object_points = np.zeros((0, 3))
        delete = []
        # Extract depth information of query image at match points and build 3D positions
        for i, (u, v) in enumerate(image1_points):
            u_int, v_int = int(u), int(v)

            # If lidar_only mode, ONLY use points with LiDAR depth
            if lidar_only and lidar_depth is not None:
                z = lidar_depth[v_int, u_int]
                # Skip points without LiDAR coverage
                if z <= 0:
                    delete.append(i)
                    continue
            else:
                # Use combined depth (LiDAR where available, else stereo)
                z = depth1[v_int, u_int]

            # Filter out missing/invalid depth information
            if z > max_depth or z <= 0:
                delete.append(i)
                continue

            # Use arithmetic to extract x and y (faster than using inverse of k)
            x = z * (u - cx) / fx
            y = z * (v - cy) / fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)

        # Need minimum points for PnP
        if len(object_points) < 6:
            print(f"Warning: Only {len(object_points)} points with LiDAR depth, falling back to stereo")
            # Recursive call without lidar_only restriction
            return estimate_motion(match, kp1, kp2, k, depth1, None, False, max_depth)

        # Use PnP algorithm with RANSAC for robustness to outliers
        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k, None)

        if lidar_only:
            print(f'LiDAR-only PnP: {len(inliers)}/{len(object_points)} inliers from {len(match)} matches')

        # Above function returns axis angle rotation representation rvec, use Rodrigues formula
        # to convert this to our desired format of a 3x3 rotation matrix
        rmat = cv2.Rodrigues(rvec)[0]
    else:
        # With no depth provided, use essential matrix decomposition instead
        E = cv2.findEssentialMat(image1_points, image2_points, k)[0]
        _, rmat, tvec, mask = cv2.recoverPose(E, image1_points, image2_points, k)

    return rmat, tvec, image1_points, image2_points

def visual_odometry(handler, lidar=False, lidar_only=False, detector='sift', matching='BF',
filter_match_distance=None, stereo_matcher='bm', mask=False,
depth_type='stereo', subset=None, plot=False):
"""
Visual Odometry pipeline with optional LiDAR enhancement.

    Args:
        handler: Dataset handler
        lidar: If True, enhance stereo depth with LiDAR
        lidar_only: If True, ONLY use keypoints with LiDAR depth (more accurate but fewer points)
        ... other args ...
    """
    # Report methods being used to user
    print('Generating disparities with Stereo{}'.format(str.upper(stereo_matcher)))
    print('Detecting features with {} and matching with {}'.format(str.upper(detector), matching))
    if filter_match_distance is not None:
        print('Filtering feature matches at threshold of {}*distance'.format(filter_match_distance))
    if lidar_only:
        print('Using ONLY keypoints with LiDAR depth (high accuracy mode)')
    elif lidar:
        print('Improving stereo depth estimation with lidar data')
    if subset is not None:
        num_frames = subset
    else:
        num_frames = handler.num_frames

    if plot:
        fig = plt.figure(figsize=(14, 14))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=-20, azim=270)
        xs = handler.gt[:, 0, 3]
        ys = handler.gt[:, 1, 3]
        zs = handler.gt[:, 2, 3]
        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        ax.plot(xs, ys, zs, c='k')
    imheight = handler.img_height
    imwidth = handler.img_width
    # Establish Total homogeneous transformation matrix. First pose is identity
    T_tot = np.eye(4)
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]
    # Decompose left camera projection matrix to get intrinsic k matrix
    k_left, _, _ = decompose_proj_mat(handler.P0)
    handler.reset_frames()
    image_plus1 = next(handler.img_L)
    # Iterate through all frames of the sequence
    for i in range(num_frames - 1):
        start = datetime.datetime.now()
        # Get our stereo images for depth estimation
        image_left = image_plus1
        image_right = next(handler.img_R)
        image_plus1 = next(handler.img_L)

        # Estimate depth using stereo
        if depth_type == 'stereo':
            depth, f_mask = stereo_to_depth(image_left,
                                   image_right,
                                   P0=handler.P0,
                                   P1=handler.P1,
                                   matcher=stereo_matcher)
        else:
            depth = None
        # Get LiDAR depth if using lidar
        lidar_depth_map = None
        if lidar or lidar_only:
            pointcloud = next(handler.pcs)
            lidar_depth_map = pc_to_img(pointcloud,
                                        imheight=imheight,
                                        imwidth=imwidth,
                                        Tr=handler.Tr,
                                        P0=handler.P0)

            # If not lidar_only, merge LiDAR into stereo depth
            if not lidar_only and depth is not None:
                indices = np.where(lidar_depth_map > 0)
                depth[indices] = lidar_depth_map[indices]
        kp0, kp1, matches = generate_matches(image_left, image_plus1, detector, matching,
                                              k=2, mask=f_mask, dist_threshold=filter_match_distance)
        # Estimate motion between sequential images of the left camera
        rmat, tvec, img1_points, img2_points = estimate_motion(
            matches, kp0, kp1, k_left,
            depth1=depth,
            lidar_depth=lidar_depth_map,
            lidar_only=lidar_only
        )
        # Create blank homogeneous transformation matrix
        Tmat = np.eye(4)
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        T_tot = T_tot.dot(np.linalg.inv(Tmat))

        trajectory[i+1, :, :] = T_tot[:3, :]

        end = datetime.datetime.now()
        print('Time to compute frame {}:'.format(i+1), end-start)

        if plot:
            xs = trajectory[:i+2, 0, 3]
            ys = trajectory[:i+2, 1, 3]
            zs = trajectory[:i+2, 2, 3]
            ax.plot(xs, ys, zs, c='chartreuse')

    if plot:
        plt.close()
    return trajectory
