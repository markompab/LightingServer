import csv
import math
import traceback

import cv2
import numpy as np

from utils.file_utils import CFileUtils

def lerp(y0, y1, x0, x1, x):
    m = (y1-y0) / (x1-x0)
    b = y0
    return m * (x-x0) + b

def rad2vec(x_dst_norm,  y_dst_norm,  f,  c,  aperture):
    longitude = x_dst_norm * math.pi + math.pi / 2;
    latitude = y_dst_norm * math.pi / 2;

    p_x = -math.cos(latitude) * math.cos(longitude)
    p_y = math.cos(latitude) * math.sin(longitude)
    p_z = math.sin(latitude)

    p   = (p_x, p_y, p_z)
    rot =  np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    pt  = rot * p

    p_xz  = math.sqrt(math.pow(pt[0, 0], 2) + math.pow(pt[0, 2], 2)) # cv::cuda::pow float(imprecision)
    theta = math.atan2(p_xz, pt[0, 1])
    phi   = math.atan2(pt[0, 2], pt[0, 0])

    x_src_norm = math.cos(phi)
    y_src_norm = math.sin(phi)

    x_src = 2 * f[0] * x_src_norm * math.sin(theta / 2)
    y_src = 2 * f[1] * y_src_norm * math.sin(theta / 2)

    return [x_src+c[0], y_src+c[1]];

def generateLUT(img, f, c, newsize, aperture):

    viewSize = {
                "height": img.shape[0],
                "width": img.shape[1]
            }

    lut = []

    for y in range(0,int((newsize["height"] / 2 )+ 10)):#10 for margin
        y_dst_norm = lerp(-1, 1, 0, newsize["height"], y)

        for x in range(0, newsize["width"]):
            x_dst_norm = lerp(-1, 1, 0, newsize["width"], x)

            uv = rad2vec(x_dst_norm, y_dst_norm, f, c, aperture)
            tx = min(viewSize["width"] - 1, int(math.floor(uv[0])))
            ty = min(viewSize["height"] - 1, int(math.floor(uv[1])))

            a = uv[0] - tx
            b = uv[1] - ty

            if (tx >= 0 and tx < viewSize["width"] - 1 and ty >= 0 and ty < viewSize["height"] - 1):

                bl = (1.-a) * (1.-b)
                tl = (1.-a) * (b)
                br = (a) * (1.-b)
                tr = (a) * (b)

                md = [x, y, tx, ty, bl, tl, br, tr]
                lut.append(md)

    try:
        lut = np.array(lut)
        savepath = "../luts/{}_{}.raw".format("lut_seqImgs", lut.shape )
        lut.astype('float').tofile(savepath, sep =" ")
        np.save("../luts/lut.csv", lut)
        #CFileUtils.saveToFile(savepath, lut)

        print("save look-up table in \"lut_seqImgs.csv\"")
        return True

    except Exception:
        print("Unable to save look-up table : ")
        print(traceback.format_exc())

        return False;

def equisolid2Equirect(img_dev, width,  xy_dev,  txty_dev, coefs_dev,  newsize):
    #x = blockDim.x * blockIdx.x + threadIdx.x
    #y = blockDim.y * blockIdx.y + threadIdx.y
    #offset = x + y * blockDim.x * gridDim.x

    xy_dev = xy_dev.astype(int)
    equirect_dev = np.zeros((newsize[0],newsize[1],3 ))
    for n in range(len(coefs_dev)):#10 for margin
        try:

            curr_pos = int(txty_dev[n][1] * width + txty_dev[n][0])

            bl_b = img_dev[xy_dev[n][0]][xy_dev[n][1]][0] * coefs_dev[n][0]
            bl_g = img_dev[xy_dev[n][0]][xy_dev[n][1]][1] * coefs_dev[n][0]
            bl_r = img_dev[xy_dev[n][0]][xy_dev[n][1]][2] * coefs_dev[n][0]

            tl_b = img_dev[xy_dev[n][0]][xy_dev[n][1]][0] * coefs_dev[n][1]
            tl_g = img_dev[xy_dev[n][0]][xy_dev[n][1]][1] * coefs_dev[n][1]
            tl_r = img_dev[xy_dev[n][0]][xy_dev[n][1]][2] * coefs_dev[n][1]

            br_b = img_dev[xy_dev[n][0]][xy_dev[n][1]][0] * coefs_dev[n][2]
            br_g = img_dev[xy_dev[n][0]][xy_dev[n][1]][1] * coefs_dev[n][2]
            br_r = img_dev[xy_dev[n][0]][xy_dev[n][1]][2] * coefs_dev[n][2]

            tr_b = img_dev[xy_dev[n][0]][xy_dev[n][1]][0] * coefs_dev[n][3]
            tr_g = img_dev[xy_dev[n][0]][xy_dev[n][1]][1] * coefs_dev[n][3]
            tr_r = img_dev[xy_dev[n][0]][xy_dev[n][1]][2] * coefs_dev[n][3]

            equirect_dev[xy_dev[n][0]][xy_dev[n][1]][0] = (bl_b + tl_b + br_b + tr_b) *255
            equirect_dev[xy_dev[n][0]][xy_dev[n][1]][1] = (bl_g + tl_g + br_g + tr_g) *255
            equirect_dev[xy_dev[n][0]][xy_dev[n][1]][2] = (bl_r + tl_r + br_r + tr_r) *255
        except Exception as e:
            print(traceback.format_exc())

    return  equirect_dev

def equisolid2Equirect2(img_dev, width,  xy_dev,  txty_dev, coefs_dev,  newsize):
    #x = blockDim.x * blockIdx.x + threadIdx.x
    #y = blockDim.y * blockIdx.y + threadIdx.y
    #offset = x + y * blockDim.x * gridDim.x
    n = 0
    equirect_dev = np.zeros((newsize[0],newsize[1],3 ))
    for y in range(0, int(newsize[0] / 2 + 10)):#10 for margin
        for x in range(0, newsize[1]):

            if (n < len(xy_dev)):
                curr_pos = int(txty_dev[n][1] * width + txty_dev[n][0])

                bl_b = img_dev[x][y][0] * coefs_dev[n][0]
                bl_g = img_dev[x][y][1] * coefs_dev[n][0]
                bl_r = img_dev[x][y][2] * coefs_dev[n][0]

                tl_b = img_dev[x][y][0] * coefs_dev[n][1]
                tl_g = img_dev[x][y][1] * coefs_dev[n][1]
                tl_r = img_dev[x][y][2] * coefs_dev[n][1]

                br_b = img_dev[x][y][0] * coefs_dev[n][2]
                br_g = img_dev[x][y][1] * coefs_dev[n][2]
                br_r = img_dev[x][y][2] * coefs_dev[n][2]

                tr_b = img_dev[x][y][0] * coefs_dev[n][3]
                tr_g = img_dev[x][y][1] * coefs_dev[n][3]
                tr_r = img_dev[x][y][2] * coefs_dev[n][3]

                equirect_dev[y][x][0] = bl_b + tl_b + br_b + tr_b
                equirect_dev[y][x][1] = bl_g + tl_g + br_g + tr_g
                equirect_dev[y][x][2] = bl_r + tl_r + br_r + tr_r
                n = n + 1

    return  equirect_dev


def loadLUT(lutpath):
    xy, txty, coefs = [],[],[]
    count = 1
    with open(lutpath, 'r') as csvfile:
        # reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
        reader = csv.reader(csvfile)  # retain
        for row2 in reader:  # each row is a list
            row = np.array(row2).astype(float)
            xy.append(row[:2])
            txty.append(row[2:4])
            coefs.append({
                "bl": row[4],
                "tl": row[5],
                "br": row[6],
                "tr": row[7]
            })
            count =count + 1
            if(count == 1000):
                break

    return xy, txty, coefs

def loadLUT2(lutpath):
    xy, txty, coefs = [],[],[]
    count = 1
    with open(lutpath, 'r') as csvfile:
        # reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
        reader = csv.reader(csvfile)  # retain
        for row2 in reader:  # each row is a list
            row = np.array(row2).astype(float)
            xy.append(row[:2])
            txty.append(row[2:4])
            coefs.append({
                "bl": row[4],
                "tl": row[5],
                "br": row[6],
                "tr": row[7]
            })
            count =count + 1
            if(count == 1000):
                break

    return xy, txty, coefs

def loadLUTRaw(path):
    lut_raw = np.fromfile(path, dtype=float, sep=" ").reshape(267264,8)
    xy    = lut_raw[:,:2]
    txty  = lut_raw[:, 2:4]
    coefs = lut_raw[:, 4:8]

    return xy, txty, coefs

def groupIntoFeatures2(lutt):
    lutsize = len(lutt)
    xy, txty, coefs = np.zeros(lutsize),np.zeros(lutsize),np.zeros(lutsize)

    print(lutt[0])
    for i in range(0, lutsize):
        xy.append(lutt[i][:2])
        txty.append(lutt[i][2:4])
        coefs.append({
            "bl": lutt[i][4],
            "tl": lutt[i][5],
            "br": lutt[i][6],
            "tr": lutt[i][7]
        })

    return xy, txty, coefs

def convert(im):
    #'load look up table

    lutt = []
    lutsize = len(lutt)
    xy, txty, coefs = [],[],[]
    '''Load Lookup table'''
    if(len(lutt)<1):
        # blocksize = (lutsize + threadsize - 1) / threadsize;
        loadpath = "../luts/lut_seqImgs_(267264, 8).raw"
        xy, txty, coefs = loadLUTRaw(loadpath)

    '''Convert to equirectangular'''
    if (True):

        equirect_size  = (512,1024)
        equirect = equisolid2Equirect(im, im.shape[1], xy, txty, coefs, equirect_size )

        filepath = "../images/pano_shin.jpg"
        cv2.imwrite(filepath,equirect)#*255)

    else:
        print("Lookup Table is not successfully created")
        captureRunning = False


img = cv2.imread("../images/IMG_5680.JPG")

#Canon C70 + Canon 8mm~15mm
f = [440.2161, 440.2161]
c = [946.7238, 526.3891]

aperture = 180. * math.pi / 180.;
newsize = {
    "height": 1024,
    "width": 512
}
'''
lut = generateLUT(img, f, c, newsize, aperture)
'''

im = cv2.imread("../images/IMG_5680.JPG")
convert(im)