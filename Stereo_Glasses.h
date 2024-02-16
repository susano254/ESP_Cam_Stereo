#ifndef STEREO_GLASSES_H
#define STEREO_GLASSES_H

#include <opencv2/opencv.hpp>
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
		Mat frameLeft, frameRight, frame;

		Stereo_Glasses(){}
		Stereo_Glasses(string LEFT_URL, string RIGHT_URL){
			this->LEFT_URL = LEFT_URL;
			this->RIGHT_URL = RIGHT_URL;
		}
		int run();
		int runOne();
		void calibrate();
		void stereoCalibrate();
		void getDepthMap(Mat left, Mat right, double fx, double fy, double cx, double cy, double b) ;

	private:
		static void captureFrames(Stereo_Glasses *instance, VideoCapture& capture, Mat& frame, const string cam);
		void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);
};


#endif