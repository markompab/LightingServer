import numpy as np


def buildMap(Ws, Hs, Wd, Hd, R1, R2, Cx, Cy):
    Cx = 1950.2
    Cy = 2140.2;
    fx =  1615.4;
    fy =  1615.3
    map_x = np.zeros((Hd, Wd), np.float32)
    map_y = np.zeros((Hd, Wd), np.float32)
    for y in range(0, int(Hd - 1)):
        for x in range(0, int(Wd - 1)):
            r = (float(y) / float(Hd)) * (R2 - R1) + R1
            theta = (float(x) / float(Wd)) * 2.0 * np.pi
            xS = Cx + r * np.sin(theta)
            yS = Cy + r * np.cos(theta)

            phi = 0#np.arctan(pt.at < double > (0, 2), pt.at < double > (0, 0));
            x_src_norm = np.cos(phi);
            y_src_norm = np.sin(phi);

            xS = 2 * fx * x_src_norm * np.sin(theta / 2);
            yS = 2 * fy * y_src_norm * np.sin(theta / 2);

            map_x.itemset((y, x), int(xS))
            map_y.itemset((y, x), int(yS))

    return map_x, map_y


# 2 = r2
# center of the "donut"
Cx = 1950.2#2080 #vals[0][0]
Cy = 2080#vals[0][1]
# Inner donut radius
R1x = 500#vals[1][0]
R1y = 500#vals[1][1]
R1 = 100#1x - Cx
# outer donut radius
R2x = 1#vals[2][0]
R2y = 1#vals[2][1]
R2 = 2080#R2x - Cx
# our input and output image siZes
Wd = int(2.0 * ((R2 + R1) / 2) * np.pi)
Hd = int((R2 - R1))
Ws = 4160#img.width
Hs = 4160#img.height

xmap, ymap = buildMap(Ws, Hs, Wd, Hd, R1, R2, Cx, Cy)
np.savetxt("../luts/mapx.csv", xmap, fmt='%d')
np.savetxt("../luts/mapy.csv", ymap, fmt='%d')

print("MAP DONE!")
# do an unwarping and show it to us
#img = cv2.imread("imgs/sun2.jpg")

#result = unwarp(img, xmap, ymap)
#cv2.imwrite("imgs/test_equi2.jpg", result)