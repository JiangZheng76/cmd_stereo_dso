bin_path="/home/sysu/cmd_ws/devel/lib/stereo_dso"
sequences=/media/sysu/200GB_Other/Dataset/Slam/kitti360/test_0/
calib=/home/sysu/cmd_ws/src/cmd-slam/vo/stereo-dso/cams/kitti360/camera0.txt

cd ${bin_path}

./dso_dataset files=${sequences}/ calib=${calib} preset=0 mode=1 quiet=1 nomt=1 