#from SimpleCV import Camera, VideoStream, Color, Display, Image, VirtualCamera
import math

import cv2
import numpy as np
import time


def pt2Vector(x,y, r,  aperture):
    theta  = math.atan2(y,x)
    phi    = r* aperture/2
    return theta, phi

def vector2LatLng(Px, Py, Pz):
    lng =  math.atan2(Py, Px)
    lat =  math.atan2(Pz, math.sqrt(math.pow(Px,2)+math.pow(Py,2)))
    return lat,lng

def latLng2EquiRectangular(lat,lng):
    x =  lng/math.PI
    y =  2 * lat / math.PI
    return x,y

def angle2Pt(theta, phi, r):
    x =  r * math.sin(phi) * math.cos(theta)
    y =  r * math.sin(phi) * math.sin(theta)
    z =  r * math.cos(phi)

    return x,y,z

def normaliseImg():
    ''''''


f = [440.2161, 440.2161]
c = [946.7238, 526.3891]
aperture = 190#180. * math.pi / 180.;
newsz = [513,1024,3]
#oldsz = np.array([512,512])
oldsz = np.array([512,512])
cnorm = oldsz/(2*oldsz[0])
pano_im = np.zeros(newsz)
aperture = 190/oldsz[0]#180. * math.pi / 180.;
r = 1
im = cv2.imread("../images/IMG_5680.jpg")
im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)

max = 0
mapped = []
mapped2 = []
map_src = []
radii = []
radii_min = 0
radII_max = 0

for y in range(0, oldsz[1]):
    if(y >= oldsz[1]/2):
        print("half")
    for x in range(0, oldsz[0]):
        x1,y1 = (x/oldsz[1]),(y/oldsz[0])
        r = math.dist(cnorm,[x1,y1])
        #radii.append(r)
        #radii_max  =  np.max(radii)
        #radii_min  =  np.min(radii)
        theta, phi = pt2Vector(x1,y1,r, aperture)
        px, py, pz = angle2Pt(theta, phi, r)
        lat,lng = vector2LatLng(px, py, pz)
        pano_x  =  lng / math.pi
        pano_y  =  2 * lat / math.pi

        pano_x1 = int(pano_x * newsz[1])
        pano_y1 = int(pano_y * newsz[0])

        if(pano_x1>=0 and pano_x1<newsz[1] and pano_y1>=0 and pano_y1<newsz[0]):

            pano_im[pano_y1, pano_x1] =  im[y,x]
            max = np.max(pano_im)
            row = [pano_x1, pano_y1, im[x,y][0], im[x,y][1], im[x,y][2]]
            row2 = [pano_x1, pano_y1]
            if row2 not in mapped2 :
                mapped.append(row)
                mapped2.append(row2)
                map_src.append([x,  y])
mapped2 = np.stack(mapped)
print(mapped)
pano_im =  (pano_im)
cv2.imwrite("../images/pano_paul10.jpg", pano_im)
cv2.imshow("ORIGINAL", im)
cv2.imshow("PANO", pano_im)
cv2.waitKey(0)



