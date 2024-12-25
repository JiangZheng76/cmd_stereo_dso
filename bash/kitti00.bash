bin_path="/home/sysu/cmd_ws/devel/lib/stereo_dso"
sequences=/media/sysu/200GB_Other/Dataset/Slam/KITTI/sequences/00
calib=/home/sysu/cmd_ws/src/cmd-slam/vo/stereo-dso/cams/kitti/0_2/camera.txt

cd ${bin_path}

./dso_dataset files=${sequences} calib=${calib} preset=0 mode=1 quiet=1 nomt=1 

tail -f /dev/null