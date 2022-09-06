import os

# do the unwarping
import cv2
import numpy as np



def unwarp(img, xmap, ymap):
    ##output = cv2.remap(img.getNumpyCv2(), xmap, ymap, cv2.INTER_LINEAR)
    output = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR)
    #LINEAR)
    #output = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR)
    #result = Image(output, cv2image=True)
    return output

# SimpleCV/OpenCV video out was giving problems
# decided to output frames and convert using
# avconv / ffmpeg.

# I used these params for converting the raw frames to video
# avconv -f image2 -r 30 -v:b 1024K -i samples/lapinsnipermin/%03d.jpeg output.mpeg
i = 0
xmap  = np.loadtxt("../luts/mapx.csv", dtype=float)
ymap  = np.loadtxt("../luts/mapy.csv", dtype=float)
#impaths = os.listFiles("imgs","jpg")
impaths = ["images/IMG_5680.JPG"]
for impath in impaths:
    im =  cv2.imread(impath)
    result = unwarp(im, xmap, ymap)
    '''  # Once we get an image overlay it on the source
    derp = img.blit(result, (0, img.height - result.height))
    derp = derp.applyLayers()
    derp.save(disp)
    fname = "FRAME{num:05d}.png".format(num=i)
    derp.save(fname)
    '''
    # vs.writeFrame(derp)
    # get the next frame

    filename = "../images/pano_paul.jpg"
    #filename = "imgs/equi_{}".format(FUtils.getFileNm(impath))
    #img = cv2.imshow("test", result)
    cv2.imwrite(filename, im)
    break
    #cv2.waitKey(2000)
    i = i + 1
