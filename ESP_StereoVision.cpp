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

bool terminateThreads = false;

void captureFrames(VideoCapture& capture, Mat& frame, const string cam) {
    while (!terminateThreads) {
        if (!capture.read(frame)) {
            cout << cam + " Stream Stopped" << endl;
            waitKey();
            break;
        }
		cout << "captured: " << cam << " frame successfully" << endl;
    }
}

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



	thread leftThread(captureFrames, ref(videoCaptureLeft), ref(frameLeft), "Left");
	thread rightThread(captureFrames, ref(videoCaptureRight), ref(frameRight), "Right");

	while(true){
		if(!frameLeft.empty() && !frameRight.empty()){
			hconcat(frameLeft, frameRight, frame);
			imshow("Stereo Stream", frame);
		}

        if(cv::waitKey(1) >= 0){
			terminateThreads = true;

			leftThread.join();
			rightThread.join();

			break;
		} 
	}


	return 0;

}
