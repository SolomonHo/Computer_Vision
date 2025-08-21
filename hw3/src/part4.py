import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def ransac(src_pts, dst_pts, threshold, num_iter, sample_num):
    assert src_pts.shape[0] >= sample_num, "Not enough points to sample"
    max_inliers = 0
    best_H = None

    for _ in range(num_iter):
        idx = np.random.choice(len(src_pts), sample_num, replace=False)
        src_sample = src_pts[idx]
        dst_sample = dst_pts[idx]

        try:
            H = solve_homography(src_sample, dst_sample)
        except np.linalg.LinAlgError:
            continue

        src_homo = np.hstack([src_pts, np.ones((len(src_pts), 1))])  # [N, 3]
        projected = (H @ src_homo.T).T  # [N, 3]
        projected /= projected[:, 2:]   # normalize

        error = np.linalg.norm(projected[:, :2] - dst_pts, axis=1)
        inliers_mask = error < threshold
        num_inliers = np.sum(inliers_mask)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = solve_homography(src_pts[inliers_mask], dst_pts[inliers_mask])

        if best_H is None:
            raise ValueError("RANSAC failed to find a valid homography")

    return best_H

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)


    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)

        matches = bf.knnMatch(des2, des1, k=2)
        src_pts, dst_pts = [], []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                src_pts.append(kp2[m.queryIdx].pt)
                dst_pts.append(kp1[m.trainIdx].pt)

        src_pts = np.float32(src_pts)
        dst_pts = np.float32(dst_pts)
        # TODO: 2. apply RANSAC to choose best H
        best_H = ransac(src_pts, dst_pts, threshold=4, num_iter=1000, sample_num=10)
        # TODO: 3. chain the homographies
        last_best_H = last_best_H @ best_H
        # TODO: 4. apply warping
        dst = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction='b')
        mask = np.any(dst != 0, axis=2).astype(np.float32)
        if idx < len(imgs) - 1:
            im2_resized = cv2.resize(im2, (w_max, h_max))
            dst = (mask[:, :, None] * dst + (1 - mask[:, :, None]) * im2_resized).astype(np.uint8)

    return dst 

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)