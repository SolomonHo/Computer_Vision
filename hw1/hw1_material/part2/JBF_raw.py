  
import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1 # adjust
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        # spatial kernal Gs
        Gs = np.zeros((self.wndw_size, self.wndw_size))
        half_w = self.pad_w 
        for i in range(self.wndw_size):
            for j in range(self.wndw_size):
                x_diff = i - half_w
                y_diff = j - half_w
                Gs[i, j] = np.exp(-(x_diff**2 + y_diff**2) / (2 * self.sigma_s**2))

        padded_guidance = padded_guidance.astype(np.float64) / 255.0
        padded_img = padded_img.astype(np.float64) 

        # range kernal Gr                
        output = np.zeros(img.shape)
        for i in range(self.pad_w, padded_guidance.shape[0] - self.pad_w):
            for j in range(self.pad_w, padded_guidance.shape[1] - self.pad_w):
                Tp = padded_guidance[i, j]  # center pixel
                Tq = padded_guidance[i - self.pad_w:i + self.pad_w + 1, j - self.pad_w:j + self.pad_w + 1] 
                
                channel = (np.square(Tp - Tq) / (-2 * self.sigma_r ** 2))
                if len(channel.shape) == 3:  #RGB
                    channel = channel.sum(axis=2)  
                Gr = np.exp(channel)  

                G = np.multiply(Gs, Gr)
                kernel_sum = np.sum(G)
                G /= kernel_sum
                Iq = padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1]        
                for k in range(img.shape[2]):
                    output[i-self.pad_w, j-self.pad_w, k] = np.multiply(G, Iq[:,:,k]).sum() 

        return np.clip(output, 0, 255).astype(np.uint8)
