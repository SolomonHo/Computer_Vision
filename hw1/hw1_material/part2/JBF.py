import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6 * sigma_s + 1
        self.pad_w = 3 * sigma_s
        self.LUT_s = np.exp(-0.5 * (np.arange(self.pad_w + 1) ** 2) / self.sigma_s ** 2)

    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        # # spatial kernal Gs
        # Gs = np.zeros((self.wndw_size, self.wndw_size))
        # half_w = self.pad_w 
        # for i in range(self.wndw_size):
        #     for j in range(self.wndw_size):
        #         x_diff = i - half_w
        #         y_diff = j - half_w
        #         Gs[i, j] = np.exp(-(x_diff**2 + y_diff**2) / (2 * self.sigma_s**2))

        # padded_guidance = padded_guidance.astype(np.float64) / 255.0
        # padded_img = padded_img.astype(np.float64) 

        # # range kernal Gr                
        # output = np.zeros(img.shape)
        # for i in range(self.pad_w, padded_guidance.shape[0] - self.pad_w):
        #     for j in range(self.pad_w, padded_guidance.shape[1] - self.pad_w):
        #         Tp = padded_guidance[i, j]  # center pixel
        #         Tq = padded_guidance[i - self.pad_w:i + self.pad_w + 1, j - self.pad_w:j + self.pad_w + 1] 
                        
        LUT_r = np.exp(-0.5 * (np.arange(256) / 255) ** 2 / self.sigma_r ** 2)

        wgt_sum = np.zeros(padded_img.shape)
        result = np.zeros(padded_img.shape)


        for x in range(-self.pad_w, self.pad_w + 1):
            for y in range(-self.pad_w, self.pad_w + 1):
                # Gs
                shifted_guidance = np.roll(padded_guidance, [y, x], axis=[0, 1])
                dT = np.abs(shifted_guidance - padded_guidance)
                r_w = LUT_r[dT]
                if r_w.ndim == 3:  # RGB 
                    r_w = np.prod(r_w, axis=2)

                s_w = self.LUT_s[abs(x)] * self.LUT_s[abs(y)]
                t_w = s_w * r_w

                shifted_img = np.roll(padded_img, [y, x], axis=[0, 1])
                for channel in range(padded_img.shape[2] if padded_img.ndim == 3 else 1):
                    result[:, :, channel] += shifted_img[:, :, channel] * t_w
                    wgt_sum[:, :, channel] += t_w

        output = result / wgt_sum
        output = output[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w]

        return np.clip(output, 0, 255).astype(np.uint8)