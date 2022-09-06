import math
from math import *

import cv2
import numpy as np

PI    = math.pi
M_PI  = math.pi
TWOPI = math.pi * 2
SQRT2 = math.sqrt(2)
TRUE  = 1
FALSE = 0
DTOR  = PI/180
SQRT2 = 1.4142135624

XTILT = 0  # x
YROLL = 1  # y
ZPAN  = 2  # z

# The following are for obj generation
NOBJ = 160
SPHERE =     0
HEMISPHERE = 1
APERTURE   = 2

vars = {
     "fishwidth": 4096
   , "fishheight": 3456
   , "outwidth": 2048
   , "outheight": 1024
   , "antialias": 2
   , "antialias2": 4
   , "fishcenterx": 2084#-1
   , "fishcentery": 2078#-1
   , "fishradius": 1965
   , "fishradiusy": -1
   , "fishfov": 185#DTOR * 180 / 2.0
   , "hammer": False
   , "transform": None
   , "ntransform": 0
   , "longblend1": -1
   , "longblend2": -1
   , "latblend1": -1
   , "latblend2": -1
   , "latmax": 1e32
   , "longmax": 1e32
   , "rcorrection": False
   , "stmap16": False
   , "stmap32": False
   , "a1": 1
   , "a2": 0
   , "a3": 0
   , "a4": 0
   , "background": {
      "r": 128,
      "g": 128,
      "b": 128,
      "a": 255,
   }
}

vars["fishradius"] = 1965
vars["fishcenterx"] = 2084
vars["fishcentery"] = 2078
vars["fishfov"] = DTOR * 185 / 2

vars["transform"] = [{},{},{}]
# else if (strcmp(argv[i],"-z") == 0) {
vars["fishradius"] = 1965
vars["fishcenterx"] = 2084
vars["fishcentery"] = 2078
vars["fishfov"] = DTOR * 185 / 2

# else if (strcmp(argv[i],"-z") == 0) {
vars["transform"][0]["axis"] = ZPAN
vars["transform"][0]["value"] = DTOR * 120

# else if (strcmp(argv[i], "-x") == 0) {
vars["transform"][1]["axis"] = XTILT
vars["transform"][1]["value"] = DTOR * 90

vars["transform"][2]["axis"] = YROLL
vars["transform"][2]["value"] = DTOR * 90



def FindFishPixel(latitude,  longitude):

   p = [0,0,0]
   q = [0,0,0]
   theta,phi,r = 0,0,0

   # p is the ray from the camera position into the scene
   p[0] = math.cos(latitude) * math.cos(longitude)
   p[1] = math.cos(latitude) * math.sin(longitude)
   p[2] = math.sin(latitude)

   # Precompute sine and cosine transformation angles
   for j in range(2):#vars["ntransform"]):
      vars["transform"][j]["cvalue"] = cos(vars["transform"][j]["value"])
      vars["transform"][j]["svalue"] = sin(vars["transform"][j]["value"])

   # Apply transformation

   if vars["transform"][1]["axis"] == XTILT:
      q[0] =  p[0]
      q[1] =  p[1] * vars["transform"][1]["cvalue"] + p[2] * vars["transform"][1]["svalue"]
      q[2] = -p[2] * vars["transform"][1]["svalue"] + p[2] * vars["transform"][1]["cvalue"]

   if False:# vars["transform"][2]["axis"] == YROLL:
      q[0] =  p[0] * vars["transform"][2]["cvalue"] + p[2] * vars["transform"][2]["svalue"]
      q[1] =  p[1]
      q[2] = -p[0] * vars["transform"][2]["svalue"] + p[2] * vars["transform"][2]["cvalue"]

   if vars["transform"][0]["axis"] == ZPAN:
      q[0] =  p[0] * vars["transform"][0]["cvalue"] + p[1] * vars["transform"][0]["svalue"]
      q[1] = -p[0] * vars["transform"][0]["svalue"] + p[1] * vars["transform"][0]["cvalue"]
      q[2] =  p[2]

      p = q

   # Calculate fisheye polar coordinates
   theta = math.atan2(p[2],p[0]) # -pi ... pi
   phi = math.atan2(math.sqrt(p[0] * p[0] + p[2] * p[2]),p[1]) # 0 ... fov/2
   if (not vars["rcorrection"]) :
      r = phi / vars["fishfov"] # 0 .. 1
   else :
      r = phi * (vars["a1"] + phi * (vars["a2"] + phi * (vars["a3"] + phi * vars["a4"])))
   

   # Determine the u,v coordinate
	# Note this could be made more efficient if it weren't for the OBJ export
   fu = vars["fishcenterx"] + r * vars["fishradius"] * math.cos(theta)
   u = fu
   fu /= vars["fishwidth"]
   fv = vars["fishcentery"] + r * vars["fishradiusy"] * math.sin(theta)
   v = fv
   fv /= vars["fishheight"]

   if (r < 0 or r > 1 or phi > vars["fishfov"]):
      ''''''

   elif (u < 0 or fu < 0):
      u = 0
      fu = 0

   elif (u >= vars["fishwidth"] or fu >= 1):
      u = vars["fishwidth"]-1
      fu = 1

   elif (v < 0 or fv < 0) :
      v = 0
      fv = 0

   elif (v >= vars["fishheight"] or fv >= 1) :
      v = vars["fishheight"]-1
      fv = 1


   return  u,  v,  fu,  fv


def FindFishPixelCustom(latitude,  longitude):

   p = [0,0,0]
   q = [0,0,0]
   theta,phi,r = 0,0,0

   # p is the ray from the camera position into the scene
   p[0] = math.cos(latitude) * math.cos(longitude)
   p[1] = math.cos(latitude) * math.sin(longitude)
   p[2] = math.sin(latitude)

   # Precompute sine and cosine transformation angles
   for j in range(2):#vars["ntransform"]):
      vars["transform"][j]["cvalue"] = cos(vars["transform"][j]["value"])
      vars["transform"][j]["svalue"] = sin(vars["transform"][j]["value"])

   # Apply transformation

   if vars["transform"][1]["axis"] == XTILT:
      q[0] =  p[0]
      q[1] =  p[1] * vars["transform"][1]["cvalue"] + p[2] * vars["transform"][1]["svalue"]
      q[2] = -p[2] * vars["transform"][1]["svalue"] + p[2] * vars["transform"][1]["cvalue"]

   if False:# vars["transform"][2]["axis"] == YROLL:
      q[0] =  p[0] * vars["transform"][2]["cvalue"] + p[2] * vars["transform"][2]["svalue"]
      q[1] =  p[1]
      q[2] = -p[0] * vars["transform"][2]["svalue"] + p[2] * vars["transform"][2]["cvalue"]

   if vars["transform"][0]["axis"] == ZPAN:
      q[0] =  p[0] * vars["transform"][0]["cvalue"] + p[1] * vars["transform"][0]["svalue"]
      q[1] = -p[0] * vars["transform"][0]["svalue"] + p[1] * vars["transform"][0]["cvalue"]
      q[2] =  p[2]

      p = q

   # Calculate fisheye polar coordinates
   theta = math.atan2(p[2],p[0]) # -pi ... pi
   phi = math.atan2(math.sqrt(p[0] * p[0] + p[2] * p[2]),p[1]) # 0 ... fov/2
   if (not vars["rcorrection"]) :
      r = phi / vars["fishfov"] # 0 .. 1
   else :
      r = phi * (vars["a1"] + phi * (vars["a2"] + phi * (vars["a3"] + phi * vars["a4"])))

   r = 2* atan2(math.sqrt(pow(p[0],2)+pow(p[1],2)), p[2])/APERTURE
   theta = atan2(p[2], p[1])

   fx = vars["fishwidth"] * r * cos(theta)
   fy = vars["fishheight"] * r * sin(theta)

   return  int(fx), int(fy)


def MakeRemap(bn):

   i,j,ix,iy,u,v = 0,0,0,0,0
   x,y,longitude,latitude,fu,fv = 0,0,0,0,0
   ''' fname[256]
   FILE *fptrx = NULL,*fptry = NULL

   #print(fname,"%s_x.pgm",bn)
   print(fname,"fish2sphere_x.pgm")
   fptrx = fopen(fname,"w")
   print(fptrx,"P2\n%d %d\n65535\n",vars["outwidth,vars["outheight)

   #print(fname,"%s_y.pgm",bn)
   #print(fname,"fish2sphere_y.pgm")
   #ptry = fopen(fname,"w")
   #print(fptry,"P2\n%d %d\n65535\n",vars["outwidth,vars["outheight)
   '''
   for j in range(vars["outheight"]-1,0,-1):
   #for (j=vars["outheight-1j>=0j--) :
      for i  in range(0, vars["outwidth"]):
      #for (i=0i<vars["outwidthi++):
         ix = -1
         iy = -1

         #Normalised coordinates
         x = 2 * i / vars["outwidth"] - 1 # -1 to 1
         y = 2 * j / vars["outheight"] - 1 # -1 to 1

         # Calculate longitude and latitude
         longitude = x * M_PI     # -pi <= x < pi
         latitude = y * 0.5*M_PI     # -pi/2 <= y < pi/2

         # Find the corresponding pixel in the fisheye image
         if (FindFishPixel(latitude,longitude,u,v,fu,fv)):
            ix = u
            iy = vars["fishheight"]-1-v


   #fclose(fptrx)
   #fclose(fptry)\

def saveImg(path, im):
   '''

   print(fname,"%s.mtl",bn)
   fptr = fopen(fname, "w")
   print(fptr, "newmtl spheremtl\n")
   print(fptr, "Ka 0 0 0\n")
   print(fptr, "Kd 1 1 1\n")
   print(fptr, "Ks 0 0 0\n")
   print(fptr, "Ns 100\n")
   print(fptr, "map_Kd %s.jpg\n", bn)  # Assumes the original fisheye is a jpg
   '''

def extraOps():
   ''''''


def MakeObj(bn, thetype):
   i,j,i1,i2,i3,i4 = 0,0,0,0,0,0
   fu,fv,latitude,longitude =0,0,0,0
   p = {"x":0, "y":0, "z":0}
   u,v = 0,0

   # mtl file
   saveImg("fname", bn)

   # Vertices
   for j in range(0, NOBJ/2):
   #for j in range(0, NOBJ/2):
      if (thetype == HEMISPHERE):
         latitude = j * M_PI / NOBJ     # 0 ... FOV/2
      elif (thetype == APERTURE):
         latitude = -(vars["fishfov"]-0.5*M_PI) + 2 * j * (0.5*M_PI+vars["fishfov"]-0.5*M_PI) / NOBJ
      else:
         latitude = -0.5*M_PI + 2 * j * M_PI / NOBJ     # -pi/2 ... pi/2

   for i in range(0, NOBJ):
   #for (i=0i<=NOBJi++):
         longitude = -PI + i * TWOPI / NOBJ     # -pi ... pi
         p[0] = cos(latitude)*sin(longitude)
         p[1] = cos(latitude)*cos(longitude)
         p[2] = sin(latitude)
         print("v %lf %lf %lf\n",p[0],p[1],p[2])


   # Texture coordinates
   for j in range(0, NOBJ/2):
      if (thetype == HEMISPHERE):
         latitude = j * M_PI / NOBJ     # 0 ... FOV/2
      elif (thetype == APERTURE):
         latitude = -(vars["fishfov"]-0.5*M_PI) + 2 * j * (0.5*M_PI+vars["fishfov"]-0.5*M_PI) / NOBJ
      else:
         latitude = -0.5*M_PI + 2 * j * M_PI / NOBJ     # -pi/2 ... pi/2


      for i in range(0, NOBJ):
         longitude = -PI + i * TWOPI / NOBJ     # -pi ... pi
         FindFishPixel(latitude,longitude, u, v, fu, fv)



   # Normals, same as vertices

   for j in range(0, NOBJ/2):
      if (thetype == HEMISPHERE):
         latitude = j * M_PI / NOBJ     # 0 ... FOV/2
      elif (thetype == APERTURE):
         latitude = -(vars["fishfov"]-0.5*M_PI) + 2 * j * (0.5*M_PI+vars["fishfov"]-0.5*M_PI) / NOBJ
      else :
         latitude = -0.5*M_PI + 2 * j * M_PI / NOBJ     # -pi/2 ... pi/2

      for i in range(0, NOBJ):
      #for (i=0i<=NOBJi++):
         longitude = -PI + i * TWOPI / NOBJ     # -pi ... pi
         p[0] = cos(latitude)*sin(longitude)
         p[1] = cos(latitude)*cos(longitude)
         p[2] = sin(latitude)
         print("vn %lf %lf %lf\n",p[0],p[1],p[2])

   # Faces
   print("usemtl spheremtl\n")
   for j in range(0, NOBJ/2):
   #for (j=0j<NOBJ/2j++){
      for i  in range(0, NOBJ):
      #for (i=0i<NOBJi++):
         i1 = 1 + j * (NOBJ+1) + i
         i4 = 1 + j * (NOBJ+1) + i + 1
         i3 = 1 + (j+1) * (NOBJ+1) + i + 1
         i2 = 1 + (j+1) * (NOBJ+1) + i
         print(" %d/%d/%d %d/{}/{} {}/{}/{}\n".format(i1,i1,i1, i2,i2,i2, i3,i3,i3))
         print("f {}/{}/{} {}/{}/{} {}/{}/{}\n".format(i1,i1,i1, i3,i3,i3, i4,i4,i4))


def runConv():
   #inputformat = TGA
   makeobj = -1
   makeremap = False


   i, j, k, u, v, aj, ai, w, h, depth = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
   z, x, y, fu, fv = 0, 0, 0, 0, 0
   latblend  = 1
   longblend = 1
   fisheye = cv2.imread("../images/sample.jpg")#np.array((vars["fishwidth"],vars["fishheight"],3))
   fisheye = np.zeros_like(fisheye)
   fisheye[:,:500] = [255,255,255]
   spherical = np.zeros((vars["outheight"],vars["outwidth"],3))

   for j in range(vars["ntransform"]):
      vars["transform"][j]["cvalue"] = cos(vars["transform"][j].value);
      vars["transform"][j]["svalue"] = sin(vars["transform"][j].value);

   # Form the spherical map 
   #starttime = GetTime()
   for j in range(0,vars["outheight"]-2):
      for i in range(vars["outwidth"]-2):
         rsum = 0
         gsum = 0
         bsum = 0
   
         # Antialiasing loops
         for ai in range(0, vars["antialias"]):
            for aj in range(0, vars["antialias"]): # aj++):
               # Normalised coordinates
               x = 2 * (i + ai /  vars["antialias"]) /  vars["outwidth"] - 1 # -1 to 1
               y = 2 * (j + aj / vars["antialias"]) /  vars["outheight"] - 1 # -1 to 1

               if (vars["hammer"]) :
                  x *= SQRT2
                  y *= SQRT2
                  z = sqrt(1 - x * x / 4 - y * y / 4)
                  longitude = 2 * atan(x * z / (2 * z * z - 1))
                  latitude = asin(y * z)
               else :
                  longitude = x * M_PI # -pi <= x < pi
                  latitude = y * 0.5 * M_PI # -pi/2 <= y < pi/2

               if (longitude < -vars["longmax"] or longitude > vars["longmax"]):
                 continue

               if (latitude < -vars["latmax"] or latitude > vars["latmax"]):
                 continue

               if (vars["longblend1"] >= 0 and vars["longblend2"] >= 0):
                  longblend = (vars["longblend2"] - fabs(longitude)) / (vars["longblend2"] - vars["longblend1"])

                  if (longblend < 0):
                     longblend = 0

                  if (longblend > 1):
                     longblend = 1

               if (vars["latblend1"] >= 0 and vars["latblend2"] >= 0) :
                  latblend = (vars["latblend2"] - latitude) / (vars["latblend2"] - vars["latblend1"])

                  if (latblend < 0):
                     latblend = 0

                  if (latblend > 1):
                     latblend = 1

               # Clip
               if (vars["hammer"]):
                  if (x*x + y*y > 2):
                     continue

               # Find the corresponding pixel in the fisheye image
               # Sum over the supersampling set
               u, v, fu, fv = FindFishPixel(latitude, longitude)
               ix = int(u)
               iy = int(vars["fishheight"] - 1 - v)
               index = int(v * vars["fishwidth"] + u)
               v1 = int(index%vars["fishwidth"])#int(u*vars["fishheight"])
               u1 = int(index/vars["fishwidth"])#int(v*vars["fishwidth"])

               rsum += latblend * longblend * fisheye[ix][iy][0]#r
               gsum += latblend * longblend * fisheye[ix][iy][1]#g
               bsum += latblend * longblend * fisheye[ix][iy][2]#b
               '''
               rsum += latblend * longblend * fisheye[u1][v1][0]#r
               gsum += latblend * longblend * fisheye[u1][v1][1]#g
               bsum += latblend * longblend * fisheye[u1][v1][2]#b
               '''

         index = j * vars["outwidth"] + i
         val = np.array([rsum,gsum,bsum]) / vars["antialias2"]
         #spherical[j][i] = val

         '''
         spherical[index][0] = rsum / vars["antialias2"]
         spherical[index][1] = gsum / vars["antialias2"]
         spherical[index][2] = bsum / vars["antialias2"]'''

   cv2.imwrite("../images/pano_paul_spherical5.jpg", spherical)


def runConv2():
   # inputformat = TGA
   makeobj = -1
   makeremap = False

   i, j, k, u, v, aj, ai, w, h, depth = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
   z, x, y, fu, fv = 0, 0, 0, 0, 0
   latblend = 1
   longblend = 1
   fisheye = cv2.imread("../images/IMG_5680.jpg")  # np.array((vars["fishwidth"],vars["fishheight"],3))
   fisheye = np.zeros_like(fisheye)
   fisheye[:, :500] = [255, 255, 255]
   spherical = np.zeros((vars["outheight"], vars["outwidth"], 3))

   for j in range(vars["ntransform"]):
      vars["transform"][j]["cvalue"] = cos(vars["transform"][j].value);
      vars["transform"][j]["svalue"] = sin(vars["transform"][j].value);

   # Form the spherical map
   # starttime = GetTime()
   for j in range(vars["outheight"] - 1, 0, -1):
      # for (j=vars["outheight-1j>=0j--) :
      for i in range(0, vars["outwidth"]-1):
         # for (i=0i<vars["outwidthi++):
         ix = -1
         iy = -1

         # Normalised coordinates
         x = 2 * i / vars["outwidth"] - 1  # -1 to 1
         y = 2 * j / vars["outheight"] - 1  # -1 to 1

         # Calculate longitude and latitude
         longitude = x * M_PI  # -pi <= x < pi
         latitude = y * 0.5 * M_PI  # -pi/2 <= y < pi/2
         # Find the corresponding pixel in the fisheye image
         fx, fy = FindFishPixelCustom(latitude, longitude)

         #if(i == 1024):

         if(fx>=0 and fx<vars["outwidth"] and fy>=0 and fy<vars["outheight"]):
            spherical[j, i]  = fisheye[fy,fx]

         ''''''
         #print("x:{}, y:{}".format())

         #spherical[int(x*vars["outwidth"])-1][int(y*vars["outheight"])-1] =  fisheye[ix, iy]
   cv2.imwrite("../images/pano_paul_spherical7.jpg", spherical)
   cv2.imshow("test", spherical)
   cv2.waitKey(0)

#runConv()
runConv2()