import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6 * sigma_s + 1
        self.pad_w = 3 * sigma_s
        # LUT of spatial and range kernal
        size = self.wndw_size // 2
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        self.LUT_spatial = np.exp(-(x ** 2 + y ** 2) / (2 * self.sigma_s ** 2)).reshape(-1)
        self.LUT_range = np.exp(-0.5 * (np.arange(256) / 255) ** 2 / self.sigma_r ** 2)

    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)


        ### TODO ###
        h, w = img.shape[:2]
        w_T = np.zeros((h, w), dtype=np.float64)
        output = np.zeros_like(img, dtype=np.float64)

        rgb_guidance = len(guidance.shape) == 3

        for i in range(self.wndw_size ** 2):
            x = i % self.wndw_size
            y = i // self.wndw_size
            s_w = self.LUT_spatial[i]

            shifted_guidance = padded_guidance[y:y + h, x:x + w]
            shifted_img = padded_img[y:y + h, x:x + w]

            if rgb_guidance:  # RGB guidance
                r_w = (np.take(self.LUT_range, np.abs(shifted_guidance[:, :, 0] - guidance[:, :, 0])) *
                       np.take(self.LUT_range, np.abs(shifted_guidance[:, :, 1] - guidance[:, :, 1])) *
                       np.take(self.LUT_range, np.abs(shifted_guidance[:, :, 2] - guidance[:, :, 2])))
            else:  # gray guidance
                r_w = np.take(self.LUT_range, np.abs(shifted_guidance - guidance))

            t_w = s_w * r_w
            w_T += t_w

            if len(img.shape) == 3:  # consider 3D (RGB) 
                for channel in range(3):
                    output[:, :, channel] += t_w * shifted_img[:, :, channel]
            else:  
                output += t_w * shifted_img

        if len(img.shape) == 3:
            for channel in range(3):
                output[:, :, channel] /= w_T
        else:
            output /= w_T

        return np.clip(output, 0, 255).astype(np.uint8)