import airsim
import cv2
import numpy as np
import os
import setup_path 
import time
import argparse



# connect to the AirSim simulator 
client = airsim.CarClient()
client.confirmConnection()
client.reset()
client.enableApiControl(True)
car_controls = airsim.CarControls()

# create saving directory
tmp_dir = os.path.join("/harddisk_1/xuefeng_data/AirSim_data", "airsim_car")
print ("Saving images to %s" % tmp_dir)
try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

#set drive timing
parser = argparse.ArgumentParser(description='set driving time(s) for each segement of your path')
parser.add_argument('--iters', nargs = '+', type = int, default=[18,21,35,38,59,62])


def log_responses(responses,idx):

    for response_idx, response in enumerate(responses):
        filename = os.path.join(tmp_dir, "{}_{}_{}".format(idx,response.image_type,response_idx))

        if response.pixels_as_float:
            print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
            airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
        elif response.compress: #png format
            print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        else: #uncompressed array
            print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
            img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 3 channel image array H X W X 3
            cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png

def go_forward(it1,it2,turn,thres):
    for idx in range(it1,it2):
        # get state of the car
        car_state = client.getCarState()
        print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))
        pose_turned = client.simGetObjectPose("PhysXCar")
        _,_,yaw = airsim.utils.to_eularian_angles(pose_turned.orientation)
        print("yaw:",yaw)

        # go forward
        if (car_state.speed < 5.65):#guarantee that the car speed is 5m/s
            car_controls.throttle = 0.82
        else:
            car_controls.throttle = 0.38
        if turn=="forward":
            car_controls.steering = (thres-yaw)/10
            print("Go Forward")
        elif turn=="left":       
            if yaw>thres:
                car_controls.steering = -0.35
            else:
                car_controls.steering = -0.2
            print("Go Forward, Turn Left")
        else:
            car_controls.steering = 0.4
            print("Go Forward, Turn Right")
        client.setCarControls(car_controls)
        time.sleep(1)   # let car drive a bit
        # get camera images from the car
        responses = client.simGetImages([
            #airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, True), #depth in perspective projection
            airsim.ImageRequest("0", airsim.ImageType.Scene)]) #scene vision image in png format
            #airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGB array
        print('Retrieved images: %d', len(responses))
        log_responses(responses,idx)

def main():

    client.reset()
    args = parser.parse_args()
    go_forward(0, args.iters[0], "forward",0)
    go_forward(args.iters[0], args.iters[1],"left",-1.57)
    go_forward(args.iters[1], args.iters[2], "forward",-1.65)
    go_forward(args.iters[2], args.iters[3],"left",-3.14)
    go_forward(args.iters[3], args.iters[4], "forward",-3.3)
    go_forward(args.iters[4], args.iters[5],"left",-2.3)
    #restore to original state
    client.reset()
    client.enableApiControl(False)


if __name__=="__main__":
    main()






            
