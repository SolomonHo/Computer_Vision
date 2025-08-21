import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        first_octave = []
        second_octave = []
        for i in range(self.num_guassian_images_per_octave):
            if i ==0:
                blur = image.copy()
            else:
                blur = cv2.GaussianBlur(image, (0, 0), self.sigma**i) 
            first_octave.append(blur)  

        base_image = cv2.resize(first_octave[-1], (image.shape[1]//2, image.shape[0]//2), interpolation=cv2.INTER_NEAREST)

        for i in range(self.num_guassian_images_per_octave): 
            if i ==0:
                blur = base_image.copy()
            else: 
                blur = cv2.GaussianBlur(base_image, (0, 0), self.sigma**i) 
            second_octave.append(blur) 

        gaussian_images = [first_octave, second_octave]
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        first_results = []
        for i in range(self.num_DoG_images_per_octave):  
            DoG = cv2.subtract(gaussian_images[0][i+1], gaussian_images[0][i])
            first_results.append(DoG)
            MAX, Min = DoG.max(), DoG.min()
            norm = ((DoG - Min) * 255 / (MAX - Min + 1e-10)).astype(np.uint8)  
            cv2.imwrite(f'testdata/DoG1-{i+1}.png', norm)

        second_results = []
        for i in range(self.num_DoG_images_per_octave):  
            DoG = cv2.subtract(gaussian_images[1][i+1], gaussian_images[1][i])
            second_results.append(DoG)
            MAX, Min = DoG.max(), DoG.min()
            norm = ((DoG - Min) * 255 / (MAX - Min + 1e-10)).astype(np.uint8) 
            cv2.imwrite(f'testdata/DoG2-{i+1}.png', norm)

        dog_images = [first_results, second_results]

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = np.array([],dtype='int64').reshape((0,2))
        for i in range(self.num_octaves):
            dogs = np.array(dog_images[i])
            cube = np.array([np.roll(dogs,(x,y,z),axis=(2,1,0)) 
                            for z in range(-1,2) for y in range(-1,2) for x in range(-1,2)])
            mask = (np.absolute(dogs)>=self.threshold) & ((np.min(cube,axis=0)==dogs) | (np.max(cube,axis=0)==dogs))
            for j in range(1, self.num_DoG_images_per_octave-1):
                m = mask[j]
                x, y = np.meshgrid(np.arange(m.shape[1]),np.arange(m.shape[0]))
                scale = 2**i
                kp = np.stack([y[m],x[m]]).T*scale
                keypoints = np.concatenate([keypoints,kp])
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(np.array(keypoints), axis = 0)
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 

        return keypoints
