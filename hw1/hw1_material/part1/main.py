import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian


def plot_keypoints(img_gray, keypoints, save_path):
    img_gray = img_gray.astype(np.uint8)
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=5.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default="C:/Users/user/Documents/CV/hw1/hw1_material/part1/testdata/2.png", help='path to input image')
    args = parser.parse_args()


    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float32)

    ### TODO ###
    dog_detector = Difference_of_Gaussian(args.threshold)
    keypoints = dog_detector.get_keypoints(img)

    thresholds = [2.0, 5.0, 7.0]
    for threshold in thresholds:
        dog = Difference_of_Gaussian(threshold)
        keypoints = dog.get_keypoints(img)
        print(f"Number of keypoints: {keypoints.shape[0]}")
        save_path = f"{args.image_path[:-4]}_threshold={int(threshold)}.png"
        plot_keypoints(img, keypoints, save_path)

if __name__ == '__main__':
    main()