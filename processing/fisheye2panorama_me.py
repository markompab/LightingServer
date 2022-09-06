import math

import cv2
import numpy as np


def fishe2PanoCrongi(src_im, height, width):
    src_height, src_width, _ = src_im.shape
    dst_height, dst_width = src_height/2 , src_width * 2
    pano_im = np.zeros((int(dst_height),int(dst_width),3))
    for w in range(0,int(dst_width-1)):
        for h in range(0,int(dst_height-1)):
            radius = dst_height - h;
            theta = math.pi * 2 / dst_width * w * -1;

            x = int(radius * math.cos(theta) + dst_height)
            y = int(dst_height - radius * math.sin(theta))
            if (x >= 0 and x < src_width and y >= 0 and y < src_height):
                hval = int(dst_height - h - 1)
                val  = src_im[x][y]
                pano_im[hval][w]= val

        cv2.imshow("pano", pano_im)
        cv2.waitKey(200)

    cv2.waitKey(0)
    return pano_im
