#include "Stereo_Glasses.h"



Stereo_Glasses::Stereo_Glasses(){}
Stereo_Glasses::Stereo_Glasses(string LEFT_URL, string RIGHT_URL){
	this->LEFT_URL = LEFT_URL;
	this->RIGHT_URL = RIGHT_URL;
}

int Stereo_Glasses::run(){
	//if stream is not active print error and return
	if(!videoCaptureLeft.open(LEFT_URL + STREAM)){
		cout << "Error opening left stream" << endl;
		return -1;
	}

	if(!videoCaptureRight.open(RIGHT_URL + STREAM)){
		cout << "Error opening right stream" << endl;
		return -1;
	}



	thread leftThread(captureFrames, this, ref(videoCaptureLeft), ref(frameLeft), "Left");
	thread rightThread(captureFrames, this,  ref(videoCaptureRight), ref(frameRight), "Right");

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



void Stereo_Glasses::captureFrames(Stereo_Glasses *instance, VideoCapture& capture, Mat& frame, const string cam) {
	while (!instance->terminateThreads) {
		if (!capture.read(frame)) {
			cout << cam + " Stream Stopped" << endl;
			waitKey();
			break;
		}
		cout << "captured: " << cam << " frame successfully" << endl;
	}
}