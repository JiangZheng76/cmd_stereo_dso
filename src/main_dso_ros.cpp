/**
 * This file is part of DSO.
 *
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

#define CODSV_MOD
#include <cv_bridge/cv_bridge.h>
#include <locale.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <boost/thread.hpp>
#include <thread>

#include "FullSystem/FullSystem.h"
#include "FullSystem/PixelSelector2.h"
#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "util/DatasetReader.h"
#include "util/NumType.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include "util/settings.h"

std::string vignette = "";
std::string gammaCalib = "";
std::string source = "";
std::string calib = "";
std::string topic0 = "";
std::string topic1 = "";
std::string groundtruth = "";
std::string gt_path = "";
std::vector<SE3> gt_pose;
int preset;
int quiet;
int nomt;
double lidar_range;
double rescale = 1;
double currentTimeStamp = 0;
bool reverse = false;
bool disableROS = false;
int start = 0;
int end = 100000;
bool prefetch = false;
float playbackSpeed =
    0;  // 0 for linearize (play as fast as possible, while sequentializing
        // tracking & mapping). otherwise, factor on timestamps.
bool preload = false;
bool useSampleOutput = false;
FullSystem *fullSystem = nullptr;
Undistort *undistorter0_;
Undistort *undistorter1_;
ImageFolderReader *reader;
ImageFolderReader *reader_right;

int mode = 0;

bool firstRosSpin = false;

using namespace dso;

void my_exit_handler(int s) {
  printf("Caught signal %d\n", s);

  exit(1);
}

void exitThread() {
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = my_exit_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

  firstRosSpin = true;
  while (true) pause();
}

void settingsDefault(int preset) {
  printf("\n=============== PRESET Settings: ===============\n");
  if (preset == 0 || preset == 1) {
    printf(
        "DEFAULT settings:\n"
        "- %s real-time enforcing\n"
        "- 2000 active points\n"
        "- 5-7 active frames\n"
        "- 1-6 LM iteration each KF\n"
        "- original image resolution\n",
        preset == 0 ? "no " : "1x");

    playbackSpeed = (preset == 0 ? 0 : 1);
    preload = preset == 1;
    setting_desiredImmatureDensity = 1500;
    setting_desiredPointDensity = 2000;
    setting_minFrames = 5;
    setting_maxFrames = 7;
    setting_maxOptIterations = 6;
    setting_minOptIterations = 1;

    setting_logStuff = false;
    setting_kfGlobalWeight =
        1.0;  // original is 1.0. 0.3 is a balance between speed and accuracy.
              // if tracking lost, set this para higher
    setting_maxShiftWeightT =
        0.04f * (640 + 480);  // original is 0.04f * (640+480); this para is
                              // depend on the crop size.
    setting_maxShiftWeightR =
        0.0f * (640 + 480);  // original is 0.0f * (640+480);
    setting_maxShiftWeightRT =
        0.02f * (640 + 480);  // original is 0.02f * (640+480);
  }

  if (preset == 2 || preset == 3) {
    printf(
        "FAST settings:\n"
        "- %s real-time enforcing\n"
        "- 800 active points\n"
        "- 4-6 active frames\n"
        "- 1-4 LM iteration each KF\n"
        "- 424 x 320 image resolution\n",
        preset == 0 ? "no " : "5x");

    playbackSpeed = (preset == 2 ? 0 : 5);
    preload = preset == 3;
    setting_desiredImmatureDensity = 600;
    setting_desiredPointDensity = 800;
    setting_minFrames = 4;
    setting_maxFrames = 6;
    setting_maxOptIterations = 4;
    setting_minOptIterations = 1;

    benchmarkSetting_width = 424;
    benchmarkSetting_height = 320;

    setting_logStuff = false;
  }

  printf("==============================================\n");
}

void parseArgument(char *arg) {
  int option;
  float foption;
  char buf[1000];

  if (1 == sscanf(arg, "sampleoutput=%d", &option)) {
    if (option == 1) {
      useSampleOutput = true;
      printf("USING SAMPLE OUTPUT WRAPPER!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "quiet=%d", &option)) {
    if (option == 1) {
      setting_debugout_runquiet = true;
      printf("QUIET MODE, I'll shut up!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "preset=%d", &option)) {
    settingsDefault(option);
    return;
  }

  if (1 == sscanf(arg, "rec=%d", &option)) {
    if (option == 0) {
      disableReconfigure = true;
      printf("DISABLE RECONFIGURE!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "noros=%d", &option)) {
    if (option == 1) {
      disableROS = true;
      disableReconfigure = true;
      printf("DISABLE ROS (AND RECONFIGURE)!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "nolog=%d", &option)) {
    if (option == 1) {
      setting_logStuff = false;
      printf("DISABLE LOGGING!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "reverse=%d", &option)) {
    if (option == 1) {
      reverse = true;
      printf("REVERSE!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "nogui=%d", &option)) {
    if (option == 1) {
      disableAllDisplay = true;
      printf("NO GUI!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "nomt=%d", &option)) {
    if (option == 1) {
      multiThreading = false;
      printf("NO MultiThreading!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "prefetch=%d", &option)) {
    if (option == 1) {
      prefetch = true;
      printf("PREFETCH!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "start=%d", &option)) {
    start = option;
    printf("START AT %d!\n", start);
    return;
  }
  if (1 == sscanf(arg, "end=%d", &option)) {
    end = option;
    printf("END AT %d!\n", start);
    return;
  }

  if (1 == sscanf(arg, "files=%s", buf)) {
    source = buf;
    printf("loading data from %s!\n", source.c_str());
    return;
  }
  if (1 == sscanf(arg, "groundtruth=%s", buf)) {
    gt_path = buf;
    printf("loading groundtruth from %s!\n", gt_path.c_str());
    return;
  }

  if (1 == sscanf(arg, "calib=%s", buf)) {
    calib = buf;
    printf("loading calibration from %s!\n", calib.c_str());
    return;
  }

  if (1 == sscanf(arg, "vignette=%s", buf)) {
    vignette = buf;
    printf("loading vignette from %s!\n", vignette.c_str());
    return;
  }

  if (1 == sscanf(arg, "gamma=%s", buf)) {
    gammaCalib = buf;
    printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
    return;
  }

  if (1 == sscanf(arg, "rescale=%f", &foption)) {
    rescale = foption;
    printf("RESCALE %f!\n", rescale);
    return;
  }

  if (1 == sscanf(arg, "speed=%f", &foption)) {
    playbackSpeed = foption;
    printf("PLAYBACK SPEED %f!\n", playbackSpeed);
    return;
  }

  if (1 == sscanf(arg, "save=%d", &option)) {
    if (option == 1) {
      debugSaveImages = true;
      if (42 == system("rm -rf images_out"))
        printf(
            "system call returned 42 - what are the odds?. This is only here "
            "to shut up the compiler.\n");
      if (42 == system("mkdir images_out"))
        printf(
            "system call returned 42 - what are the odds?. This is only here "
            "to shut up the compiler.\n");
      if (42 == system("rm -rf images_out"))
        printf(
            "system call returned 42 - what are the odds?. This is only here "
            "to shut up the compiler.\n");
      if (42 == system("mkdir images_out"))
        printf(
            "system call returned 42 - what are the odds?. This is only here "
            "to shut up the compiler.\n");
      printf("SAVE IMAGES!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "mode=%d", &option)) {
    mode = option;
    if (option == 0) {
      printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
    }
    if (option == 1) {
      printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
      setting_photometricCalibration = 0;
      setting_affineOptModeA =
          0;  //-1: fix. >=0: optimize (with prior, if > 0).
      setting_affineOptModeB =
          0;  //-1: fix. >=0: optimize (with prior, if > 0).
    }
    if (option == 2) {
      printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
      setting_photometricCalibration = 0;
      setting_affineOptModeA =
          -1;  //-1: fix. >=0: optimize (with prior, if > 0).
      setting_affineOptModeB =
          -1;  //-1: fix. >=0: optimize (with prior, if > 0).
      setting_minGradHistAdd = 3;
    }
    return;
  }

  printf("could not parse argument \"%s\"!!!!\n", arg);
}
void convertCompressImageToImage(sensor_msgs::CompressedImagePtr &compress_img,
                                 sensor_msgs::ImagePtr &img) {
  try {
    // 将压缩的图像消息转换为OpenCV格式
    cv::Mat image = cv::imdecode(cv::Mat(compress_img->data), 1);

    // 将OpenCV图像格式转换为图像消息
    img = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    img->header.stamp = compress_img->header.stamp;
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }
}

void getGroundtruth() {
  std::ifstream inf;
  inf.open(gt_path);
  std::string sline;
  std::getline(inf, sline);
  while (std::getline(inf, sline)) {
    std::istringstream ss(sline);
    Mat33 R;
    Vec3 t;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        ss >> R(i, j);
      }
      ss >> t(i);
    }
    SE3 temp(R, t);
    gt_pose.push_back(temp);
  }
  inf.close();
}
void imageMessageCallback(const sensor_msgs::ImageConstPtr &msg0,
                          const sensor_msgs::ImageConstPtr &msg1) {
  cv::Mat img0, img1;
  try {
    img0 = cv_bridge::toCvShare(msg0, "mono8")->image;
    img1 = cv_bridge::toCvShare(msg1, "mono8")->image;
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }

  // detect if a new sequence is received, restart if so
  if (currentTimeStamp > 0 &&
      fabs(msg0->header.stamp.toSec() - currentTimeStamp) > 10) {
    fullSystem->isLost = true;
  }
  currentTimeStamp = msg0->header.stamp.toSec();

  MinimalImageB minImg0((int)img0.cols, (int)img0.rows,
                        (unsigned char *)img0.data);
  // 去畸变
  ImageAndExposure *undistImg0 =
      undistorter0_->undistort<unsigned char>(&minImg0, 1, 0, 1.0f);
  undistImg0->timestamp = msg0->header.stamp.toSec();
  MinimalImageB minImg1((int)img1.cols, (int)img1.rows,
                        (unsigned char *)img1.data);
  ImageAndExposure *undistImg1 =
      undistorter1_->undistort<unsigned char>(&minImg1, 1, 0, 1.0f);
  auto t0 = std::chrono::steady_clock::now();
  static int incomingId = 0;
  fullSystem->addActiveFrame(undistImg0, undistImg1, incomingId++);
  auto t1 = std::chrono::steady_clock::now();
  // 记录时间
  // frame_tt_.push_back(t1 - t0);

  // reinitialize if necessary
  // initFailed在创建关键帧的时候进行设置
  if (fullSystem->initFailed || setting_fullResetRequested) {
    printf("RESETTING!\n");

    std::vector<IOWrap::Output3DWrapper *> wraps = fullSystem->outputWrapper;
    delete fullSystem;

    for (IOWrap::Output3DWrapper *ow : wraps) ow->reset();

    fullSystem = new FullSystem();
    fullSystem->setGammaFunction(reader->getPhotometricGamma());
    fullSystem->linearizeOperation = (playbackSpeed == 0);

    fullSystem->outputWrapper = wraps;

    setting_fullResetRequested = false;
    // setting_fullResetRequested=false;
  }
  if (fullSystem->isLost) {
    printf("LOST!!\n");
    exit(1);
  }

  incomingId++;
  delete undistImg0;
  delete undistImg1;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "stereo_dso");
  ros::NodeHandle nhPriv("~");

  /* *********************** required parameters ************************ */
  // stereo camera parameters
  if (!nhPriv.getParam("files", source) || !nhPriv.getParam("calib", calib) ||
      !nhPriv.getParam("preset", preset) || !nhPriv.getParam("mode", mode) ||
      !nhPriv.getParam("quiet", quiet) || !nhPriv.getParam("topic0", topic0) ||
      !nhPriv.getParam("topic1", topic1) || !nhPriv.getParam("nomt", nomt)) {
    ROS_ERROR("Fail to get sensor topics/params, exit.");
    std::cout << "source : " << source << std::endl;
    std::cout << "calib : " << calib << std::endl;
    std::cout << "preset : " << preset << std::endl;
    std::cout << "mode : " << mode << std::endl;
    std::cout << "quiet : " << quiet << std::endl;
    std::cout << "nomt : " << nomt << std::endl;
    std::cout << "topic0 : " << topic0 << std::endl;
    std::cout << "topic1 : " << topic1 << std::endl;
    return -1;
  } else {
    settingsDefault(preset);
    if (quiet == 1) {
      setting_debugout_runquiet = true;
    }
    if (nomt == 1) {
      multiThreading = false;
      printf("NO MultiThreading!\n");
    }
    // update mode
    if (mode == 0) {
      printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
    }
    if (mode == 1) {
      printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
      setting_photometricCalibration = 0;
      setting_affineOptModeA =
          0;  //-1: fix. >=0: optimize (with prior, if > 0).
      setting_affineOptModeB =
          0;  //-1: fix. >=0: optimize (with prior, if > 0).
    }
    if (mode == 2) {
      printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
      setting_photometricCalibration = 0;
      setting_affineOptModeA =
          -1;  //-1: fix. >=0: optimize (with prior, if > 0).
      setting_affineOptModeB =
          -1;  //-1: fix. >=0: optimize (with prior, if > 0).
      setting_minGradHistAdd = 3;
    }
    nhPriv.param<double>("lidar_range", lidar_range, 40.0);
    std::cout << "lidar_range : " << lidar_range << "\n";
    std::cout << "source : " << source << std::endl;
    std::cout << "calib : " << calib << std::endl;
    std::cout << "preset : " << preset << std::endl;
    std::cout << "mode : " << mode << std::endl;
    std::cout << "quiet : " << quiet << std::endl;
    std::cout << "nomt : " << nomt << std::endl;
    std::cout << "topic0 : " << topic0 << std::endl;
    std::cout << "topic1 : " << topic1 << std::endl;
    undistorter0_ =
        Undistort::getUndistorterForFile(calib, gammaCalib, vignette);
    undistorter1_ =
        Undistort::getUndistorterForFile(calib, gammaCalib, vignette);
  }

  nhPriv.param<std::string>("groundtruth", groundtruth);
  std::cout << "groundtruth : " << groundtruth << std::endl;

  // if(gt_path.size()>0)getGroundtruth();
  // hook crtl+C.
  boost::thread exThread = boost::thread(exitThread);

  // read the paramters
  reader =
      new ImageFolderReader(source + "/image_0", calib, gammaCalib, vignette);
  reader_right =
      new ImageFolderReader(source + "/image_1", calib, gammaCalib, vignette);
  reader->setGlobalCalibration();
  reader_right->setGlobalCalibration();

  if (setting_photometricCalibration > 0 &&
      reader->getPhotometricGamma() == 0) {
    printf(
        "ERROR: dont't have photometric calibation. Need to use commandline "
        "options mode=1 or mode=2 ");
    exit(1);
  }

  // read from a bag file
  std::string bag_path;
  bool is_cfe = true;
  nhPriv.param<std::string>("bag", bag_path, "");
  // nhPriv.param<bool>("is_cfe", is_cfe, true);
  // std::cout << "is_cfe: " << is_cfe << std::endl;
  /* ******************************************************************** */
  // 配置参数
  fullSystem = new FullSystem();
  fullSystem->setGammaFunction(reader->getPhotometricGamma());
  fullSystem->linearizeOperation = (playbackSpeed == 0);

  IOWrap::PangolinDSOViewer *viewer = 0;
  if (!disableAllDisplay) {
    viewer = new IOWrap::PangolinDSOViewer(wG[0], hG[0], true);
    fullSystem->outputWrapper.push_back(viewer);
  }
  // if(viewer != 0)
  //     viewer->run();

  // 利用这个播放topic
  if (!bag_path.empty()) {
    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);
    std::vector<std::string> topics = {topic0, topic1};
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    sensor_msgs::ImagePtr img0, img1;
    bool img0_updated(false), img1_updated(false);
    BOOST_FOREACH (rosbag::MessageInstance const m, view) {
      if (m.getTopic() == topic0) {
        img0 = m.instantiate<sensor_msgs::Image>();
#ifdef CODSV_MOD
        if (!img0) {
          sensor_msgs::CompressedImagePtr compress_img =
              m.instantiate<sensor_msgs::CompressedImage>();
          convertCompressImageToImage(compress_img, img0);
        }
#endif
        img0_updated = true;
      }
      if (m.getTopic() == topic1) {
        img1 = m.instantiate<sensor_msgs::Image>();
#ifdef CODSV_MOD
        if (!img1) {
          sensor_msgs::CompressedImagePtr compress_img =
              m.instantiate<sensor_msgs::CompressedImage>();
          convertCompressImageToImage(compress_img, img1);
        }
#endif
        img1_updated = true;
      }
      if (img0_updated && img1_updated) {
        assert(fabs(img0->header.stamp.toSec() - img1->header.stamp.toSec()) <
               0.1);
        // 图片传入入口
        imageMessageCallback(img0, img1);
        img0_updated = img1_updated = false;
      }
    }
    bag.close();
  } else {
    // 如果没有填bag的路径则是选择播放bag来实现
    // ROS subscribe to stereo images
    ros::NodeHandle nh;
    auto *cam0_sub =
        new message_filters::Subscriber<sensor_msgs::Image>(nh, topic0, 10000);
    auto *cam1_sub =
        new message_filters::Subscriber<sensor_msgs::Image>(nh, topic1, 10000);
    auto *sync = new message_filters::Synchronizer<
        message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                        sensor_msgs::Image>>(
        message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                        sensor_msgs::Image>(10),
        *cam0_sub, *cam1_sub);
    sync->registerCallback(boost::bind(&imageMessageCallback, _1, _2));
    ros::spin();
  }
  fullSystem->blockUntilMappingIsFinished();
  clock_t ended = clock();
  struct timeval tv_end;
  gettimeofday(&tv_end, NULL);

  // runthread.join();
  exit(0);

  for (IOWrap::Output3DWrapper *ow : fullSystem->outputWrapper) {
    ow->join();
    delete ow;
  }

  printf("DELETE FULLSYSTEM!\n");
  delete fullSystem;

  printf("DELETE READER!\n");
  delete reader;
  delete reader_right;
  delete undistorter0_;
  delete undistorter1_;
  printf("EXIT NOW!\n");
  return 0;

  return 0;
}
