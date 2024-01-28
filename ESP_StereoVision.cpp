#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>

using namespace std;
using namespace Eigen;
using namespace cv;

int main(){
	VideoCapture videoCaptureLeft, videoCaptureRight;
	Mat frameLeft, frameRight, frame;

	const string LEFT_URL = "http://192.168.1.12";
	const string RIGHT_URL = "http://192.168.1.13";
	const string STREAM = ":81/stream";
	const string  CONTROL = "/control";



	//if stream is not active print error and return
	if(!videoCaptureLeft.open(LEFT_URL + STREAM)){
		cout << "Error opening left stream" << endl;
		return -1;
	}

	if(!videoCaptureRight.open(RIGHT_URL + STREAM)){
		cout << "Error opening right stream" << endl;
		return -1;
	}

	while(true){
		if(!videoCaptureLeft.read(frameLeft)){
			cout << "left Stream Stopped" << endl;
			waitKey();
		}
		if(!videoCaptureRight.read(frameRight)){
			cout << "Right Stream Stopped" << endl;
			waitKey();
		}

		hconcat(frameLeft, frameRight, frame);
		imshow("Stereo Stream", frame);

        if(cv::waitKey(1) >= 0) break;
	}

}
