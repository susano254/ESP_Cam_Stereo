#ifndef STEREO_GLASSES_H
#define STEREO_GLASSES_H

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>
#include <thread>

using namespace std;
using namespace Eigen;
using namespace cv;

class Stereo_Glasses {
	public:
		// Defining the dimensions of checkerboard
		int CHECKERBOARD[2]{8,5}; 
		bool terminateThreads = false;
		string LEFT_URL = "http://192.168.1.12";
		string RIGHT_URL = "http://192.168.1.13";
		const string STREAM = ":81/stream";
		const string CONTROL = "/control";
		string savePath = "/home/mahmoud/code/WalkEasy/ESP_Stereo_OpenCV/data/";
		VideoCapture videoCaptureLeft, videoCaptureRight, videoCapture;
		Mat frameLeft, frameRight, frame, rectified_left, rectified_right;
		Mat Left_Stereo_Map1, Left_Stereo_Map2;
		Mat Right_Stereo_Map1, Right_Stereo_Map2;
		double fx, fy, cx, cy, b;

		int k = 17;
		int blockSize = 9;
		int P1 = 8*9*blockSize*blockSize; //*blockSize;
		int P2 = 32*9*blockSize*blockSize; //*blockSize;
		int minDisparity = 0;
		int disparityFactor = 9;
		int numDisparities = 16*disparityFactor - minDisparity;
		Mat left_disp, right_disp, disparity, filteredDisparity, coloredDisparity;

		Ptr<StereoBM> leftMatcher = StereoBM::create(numDisparities, blockSize);
		Ptr<StereoMatcher>rightMatcher = ximgproc::createRightMatcher(leftMatcher);
		cv::Ptr<ximgproc::DisparityWLSFilter> wlsFilter = ximgproc::createDisparityWLSFilter(leftMatcher);

		Stereo_Glasses(){
			leftMatcher->setMinDisparity(minDisparity);
			leftMatcher->setNumDisparities(numDisparities);
			leftMatcher->setDisp12MaxDiff(1);
			leftMatcher->setPreFilterCap(63);
			leftMatcher->setUniquenessRatio(0);
			leftMatcher->setSpeckleWindowSize(0);
			leftMatcher->setSpeckleRange(32);


			// rightMatcher = ximgproc::createRightMatcher(leftMatcher);
			rightMatcher->setMinDisparity(minDisparity);
			rightMatcher->setNumDisparities(numDisparities);
			rightMatcher->setDisp12MaxDiff(1);
			rightMatcher->setSpeckleWindowSize(9);
			rightMatcher->setSpeckleRange(32);
		}

		Stereo_Glasses(string LEFT_URL, string RIGHT_URL){
			this->LEFT_URL = LEFT_URL;
			this->RIGHT_URL = RIGHT_URL;
		}
		void init();
		int run();
		int runOne();
		void calibrate();
		void stereoCalibrate();
		void runWithDepth();

		static void captureFrames(Stereo_Glasses *instance, VideoCapture& capture, Mat& frame, const string cam);
		void getDepthMap(Mat left, Mat right);
		void updateDisparity();
		void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);
};


#endif