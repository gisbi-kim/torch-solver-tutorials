import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia_moons.feature import *
import pathlib

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# import tkinter


def abspathdir():
    return pathlib.Path(__file__).parent.absolute()

def load_torch_image(fname):
    img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.0
    img = K.color.bgr_to_rgb(img)
    return img


def main():
    print(f"The main have been started.")

    img_extension = "png"
    fname1 = f"{abspathdir()}/data/1.{img_extension}"
    fname2 = f"{abspathdir()}/data/2.{img_extension}"
    print(fname1)
    print(fname2)

    img1 = load_torch_image(fname1).cuda()
    img2 = load_torch_image(fname2).cuda()

    matcher = KF.LoFTR(pretrained="outdoor").cuda()

    input_dict = {
        "image0": K.color.rgb_to_grayscale(
            img1
        ),  # LofTR works on grayscale images only
        "image1": K.color.rgb_to_grayscale(img2),
    }

    with torch.no_grad():
        correspondences = matcher(input_dict)

    for k,v in correspondences.items():
        print (k)

    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    H, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
    inliers = inliers > 0

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    draw_LAF_matches(
        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1,-1, 2),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1)),

        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1,-1, 2),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1)),
        torch.arange(mkpts0.shape[0]).view(-1,1).repeat(1,2),
        K.tensor_to_image(img1),
        K.tensor_to_image(img2),
        inliers,
        draw_dict={'inlier_color': (0.2, 1, 0.2),
                'tentative_color': None, 
                'feature_color': (0.2, 0.5, 1), 'vertical': True},
        ax=ax, # here 
    )

    plt.savefig(f'{abspathdir()}/result.png')
    plt.show()


if __name__ == "__main__":
    main()
