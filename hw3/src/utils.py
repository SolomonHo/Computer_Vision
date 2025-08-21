import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
        return None
    
    # TODO: 1.forming A
    # A = np.zeros((2*N, 9))
    # for i in range(N):
    #     u_x, u_y = u[i]
    #     v_x, v_y = v[i]

    #     A[2 * i] = [u_x, u_y, 1, 0, 0, 0, -u_x * v_x, -u_y * v_x, -v_x]
    #     A[2 * i + 1] = [0, 0, 0, u_x, u_y, 1, -u_x * v_y, -u_y * v_y, -v_y]

    u_x, u_y = u[:, 0], u[:, 1]
    v_x, v_y = v[:, 0], v[:, 1]
    zeros = np.zeros(N)
    ones = np.ones(N)
    A = np.vstack([
        np.column_stack([u_x, u_y, ones, zeros, zeros, zeros, -u_x*v_x, -u_y*v_x, -v_x]),
        np.column_stack([zeros, zeros, zeros, u_x, u_y, ones, -u_x*v_y, -u_y*v_y, -v_y])
    ])
    
    # TODO: 2.solve H with A
    [U,S,V] = np.linalg.svd(A)
    h = V[-1]
    H = h.reshape(3, 3)
    H = H / H[2, 2]

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    xx, yy = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax), sparse=False)
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    grid_x = xx.ravel()
    grid_y = yy.ravel()
    ones = np.ones((grid_x.shape))
    grid = np.vstack((grid_x, grid_y, ones))

    
    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        
        coords_src = H_inv @ grid
        coords_src = coords_src[:2] / coords_src[2:3]  # Normalize 

        u = coords_src[0].reshape(ymax - ymin, xmax - xmin)
        v = coords_src[1].reshape(ymax - ymin, xmax - xmin)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = (u >= 0) & (u < w_src - 1) & (v >= 0) & (v < h_src - 1)
        mask_flat = mask.ravel()
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        
        u_floor = np.floor(u).astype(int).ravel()
        v_floor = np.floor(v).astype(int).ravel()
        u_ceil = u_floor + 1
        v_ceil = v_floor + 1
        wu = u.ravel() - u_floor
        wv = v.ravel() - v_floor

        f00 = src[v_floor[mask_flat], u_floor[mask_flat]]
        f01 = src[v_floor[mask_flat], u_ceil[mask_flat]]
        f10 = src[v_ceil[mask_flat], u_floor[mask_flat]]
        f11 = src[v_ceil[mask_flat], u_ceil[mask_flat]] 

        
        wu = wu[mask_flat][..., None]
        wv = wv[mask_flat][..., None]

        interpolated = (
            f00 * (1 - wu) * (1 - wv) +
            f01 * wu * (1 - wv) +
            f10 * (1 - wu) * wv +
            f11 * wu * wv
        )
        # interpolated = (f00 * (1 - wu[mask])[..., None] * (1 - wv[mask])[..., None] +
        #                 f01 * wu[mask][..., None] * (1 - wv[mask])[..., None] +
        #                 f10 * (1 - wu[mask])[..., None] * wv[mask][..., None] +
        #                 f11 * wu[mask][..., None] * wv[mask][..., None])
        
        # TODO: 6. assign to destination image with proper masking
        xx_flat = xx.ravel()[mask_flat]
        yy_flat = yy.ravel()[mask_flat]
        dst[yy_flat, xx_flat] = interpolated
            

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        
        coords_dst = H @ grid
        coords_dst = coords_dst[:2] / coords_dst[2:3]  # Normalize 

        u = coords_dst[0].reshape(ymax - ymin, xmax - xmin)
        v = coords_dst[1].reshape(ymax - ymin, xmax - xmin)
        
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = (u >= 0) & (u < w_dst) & (v >= 0) & (v < h_dst)
        # TODO: 5.filter the valid coordinates using previous obtained mask

        mask_flat = mask.ravel()
        u_valid = u.ravel()[mask_flat].astype(int)
        v_valid = v.ravel()[mask_flat].astype(int)
        x_src_valid = grid_x[mask_flat].astype(int)
        y_src_valid = grid_y[mask_flat].astype(int)

        # TODO: 6. assign to destination image using advanced array indicing
        dst[v_valid, u_valid] = src[y_src_valid, x_src_valid]

    return dst 
