import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/2.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/2_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    setting_path = args.setting_path
    rgb_weights = []
    with open(setting_path, 'r') as f:
        next(f)
        for line in f:
            values = line.strip().split(',')
            if len(values) == 3:  
                r, g, b = map(float, values)
                rgb_weights.append((r, g, b))
            elif len(values) == 4 and values[0] == "sigma_s": 
                sigma_s = int(values[1])
                sigma_r = float(values[3])

    costs = {}
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    bf = JBF.joint_bilateral_filter(img_rgb, img_rgb)
    jbf = JBF.joint_bilateral_filter(img_rgb, img_gray)
    costs['cv2.COLOR_BGR2GRAY'] = np.sum(np.abs(bf.astype('int32')-jbf.astype('int32')))

    jbf_out = cv2.cvtColor(jbf, cv2.COLOR_BGR2RGB) 
    cv2.imwrite(args.image_path[:-4]+'_img_gray_COLOR_BGR2GRAY.png', img_gray)
    cv2.imwrite(args.image_path[:-4]+'_jbf_COLOR_BGR2GRAY.png', jbf_out) 

    for idx, (r, g, b) in enumerate(rgb_weights):
        gray_img = (r * img_rgb[:, :, 0] + g * img_rgb[:, :, 1] + b * img_rgb[:, :, 2]).astype(np.uint8)
        jbf_out = JBF.joint_bilateral_filter(img_rgb, gray_img).astype(np.uint8)
        l1_norm = np.sum(np.abs(jbf_out.astype('int32') - bf.astype('int32')))
        costs[f'R*{r}+G*{g}+B*{b}'] = l1_norm

        jbf_out = cv2.cvtColor(jbf_out,cv2.COLOR_BGR2RGB)
        cv2.imwrite(args.image_path[:-4]+f'_jbf_{r}_{g}_{b}.png', jbf_out)
        cv2.imwrite(args.image_path[:-4]+f'_img_gray_{r}_{g}_{b}.png', gray_img)
    for method, cost in costs.items():
        print(f'[Cost] {method}: {cost}')

if __name__ == '__main__':
    main()