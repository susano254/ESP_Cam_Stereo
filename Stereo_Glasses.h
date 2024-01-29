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
		bool terminateThreads = false;
		string LEFT_URL = "http://192.168.1.12";
		string RIGHT_URL = "http://192.168.1.13";
		const string STREAM = ":81/stream";
		const string CONTROL = "/control";
		string savePath = "/home/mahmoud/code/cpp/ESP_Cam_Stereo/images/";
		VideoCapture videoCaptureLeft, videoCaptureRight;
		Mat frameLeft, frameRight, frame;

		Stereo_Glasses(){}
		Stereo_Glasses(string LEFT_URL, string RIGHT_URL){
			this->LEFT_URL = LEFT_URL;
			this->RIGHT_URL = RIGHT_URL;
		}
		int run();
		void calibrate();

	private:
		static void captureFrames(Stereo_Glasses *instance, VideoCapture& capture, Mat& frame, const string cam);
};


#endif