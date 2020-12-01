import argparse
import warnings
from math import pi,cos,sin

import pyproj
import numpy as np
from path import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Options for trajectory visualization.')
parser.add_argument('--heading', type=float, default=0, help='initial heading angle of start point')
parser.add_argument('--gps', type=str, help='path to gps file', default="")
parser.add_argument('--dso', type=str, help='path to dso result', default="")
parser.add_argument('--stereo_dso', type=str, help='path to stereo dso result', default="")
parser.add_argument('--kitti', type=str, help='path to kitti ground truth', default="")
parser.add_argument('--abs_dso', type=str, help='path to abs dso result', default="")
parser.add_argument('--cnn_dso', type=str, help='path to cnn dso result', default="")
parser.add_argument('--odom_npys', type=str, help='path to evaluate pose odom result', default="")
parser.add_argument('--odom_scaled_npys', type=str, help='path to evaluate pose odom scaled result', default="")
args = parser.parse_args()

def filter(gps_data):
  gps_filtered = []
  for i,line in enumerate(gps_data):
    if ((line[1]>31) & (line[1]<32) & (line[2]<122) & (line[2]>121)): 
      print(line)
      gps_filtered.append(line)
  return np.array(gps_filtered)


def main():
  if args.gps != "":
    gps_data = np.loadtxt(Path(args.gps))
    gps_data = filter(gps_data)
    latitude  = gps_data[:,1]
    longitude = gps_data[:,2]

    p1 = pyproj.Proj(init='epsg:4326') 
    p2 = pyproj.Proj(init='epsg:32651') 

    x1, y1 = p1(longitude, latitude)
    x2, y2 = pyproj.transform(p1, p2, x1, y1)

    gps_x = x2 - x2[0]
    gps_y = y2 - y2[0]

    gps_x = gps_x[abs(gps_x) < 1000]
    gps_y = gps_y[abs(gps_y) < 1000]

    heading = args.heading * pi / 180
    print(gps_x.shape,type(gps_x))
    print(gps_y.shape,type(gps_y))
    rotate_gps_x = np.copy(gps_x) * cos(heading) - np.copy(gps_y) * sin(heading)
    rotate_gps_y = np.copy(gps_x) * sin(heading) + np.copy(gps_y) * cos(heading)

  if args.dso != "":
    mono_data = np.loadtxt(Path(args.dso))
    mono_x = mono_data[:,1]
    mono_y = mono_data[:,3]

  if args.abs_dso != "":
    abs_data = np.loadtxt(Path(args.abs_dso))
    abs_x = abs_data[:,1]
    abs_y = abs_data[:,3]

  if args.cnn_dso != "":
    cnn_data = np.loadtxt(Path(args.cnn_dso))
    cnn_x = cnn_data[:,1]
    cnn_y = cnn_data[:,3]

  if args.stereo_dso != "":
    stereo_data = np.loadtxt(Path(args.stereo_dso))
    stereo_x = stereo_data[:,4]
    stereo_y = stereo_data[:,12]

  if args.kitti != "":
    kitti_data = np.loadtxt(Path(args.kitti))
    kitti_data = np.reshape(kitti_data, (kitti_data.shape[0], 3, 4))
    kitti_x = kitti_data[:,0,3]
    kitti_y = kitti_data[:,2,3]
  
  if args.odom_npys !="":
    odom_poses = np.load(Path(args.odom_npys))
    odom_x = odom_poses[:,0]
    odom_y = odom_poses[:,2]

  if args.odom_scaled_npys !="":
    odom_poses = np.load(Path(args.odom_scaled_npys))
    odom_poses = np.reshape(odom_poses,(-1,3))
    print(odom_poses.shape)
    odom_scaled_x = odom_poses[:,0]
    odom_scaled_y = odom_poses[:,2]

  plt.figure()

  if args.gps != "":
    plt.plot(rotate_gps_x, rotate_gps_y, label="GPS")
  if args.dso != "":
    plt.plot(mono_x, mono_y, label="DSO", linewidth=1)
  if args.abs_dso != "":
    plt.plot(abs_x[:200], abs_y[:200], label="Abs DSO", linewidth=1)
  if args.cnn_dso != "":
    plt.plot(cnn_x, cnn_y, label="CNN DSO", linewidth=1)
  if args.stereo_dso != "":
    plt.plot(stereo_x, stereo_y, label="Stereo DSO", linewidth=1)
  if args.kitti != "":
    plt.plot(kitti_x, kitti_y, label="GT", linewidth=1)
    print(kitti_x)
  if args.odom_npys != "":
    print(odom_x)
    plt.plot(odom_x, odom_y, label="CNN odom", linewidth=1)
  if args.odom_scaled_npys != "":
    plt.plot(odom_scaled_x,odom_scaled_y, label="CNN scaled odom", linewidth=1)

  plt.axis('square')
  plt.legend()
  plt.savefig("traj_mech_loop2.png", bbox_inches='tight', dpi=300)


if __name__ == "__main__":
  main()