#include "Stereo_Glasses.h"


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


	int i = 0;
	while(true){
		if(!frameLeft.empty() && !frameRight.empty()){
			hconcat(frameLeft, frameRight, frame);
			imshow("Stereo Stream", frame);
		}

		char key = cv::waitKey(1);
        if (key >= 0) {
			switch (key) {
				case 'c':
					// Save the current frame to the specified path
					imwrite(savePath + "imageLeft" + to_string(i) + ".jpg", frameLeft);
					i++;

					std::cout << "Frame saved to: " << savePath << std::endl;
					break;

				case 'q':
					terminateThreads = true;
					break;
			}
        }

		if(terminateThreads){
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
	}
}

void Stereo_Glasses::calibrate() {
	// Defining the dimensions of checkerboard
	static int CHECKERBOARD[2]{8,5}; 

	// Creating vector to store vectors of 3D points for each checkerboard image
	vector<vector<Point3f> > objpoints;

	// Creating vector to store vectors of 2D points for each checkerboard image
	vector<vector<Point2f> > imgpoints;

	// Defining the world coordinates for 3D points
	vector<Point3f> objp;

	for(int i{0}; i<CHECKERBOARD[1]; i++) {
		for(int j{0}; j<CHECKERBOARD[0]; j++)
			objp.push_back(Point3f(j,i,0));
	}


	// Extracting path of individual image stored in a given directory
	vector<String> images;

	glob(savePath, images);

	Mat frame, gray;
	// vector to store the pixel coordinates of detected checker board corners 
	vector<Point2f> corner_pts;
	bool success;

	// Looping over all the images in the directory
	for(int i{0}; i<images.size(); i++) {
		frame = imread(images[i]);
		cvtColor(frame,gray,COLOR_BGR2GRAY);

		// Finding checker board corners
		// If desired number of corners are found in the image then success = true  
		success = findChessboardCorners(gray,Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_pts, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

		/*
			* If desired number of corner are detected,
			* we refine the pixel coordinates and display 
			* them on the images of checker board
		*/
		if(success) {
			TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001);

			// refining pixel coordinates for given 2d points.
			cornerSubPix(gray,corner_pts,Size(11,11), Size(-1,-1),criteria);

			// Displaying the detected corner points on the checker board
			drawChessboardCorners(frame, Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_pts,success);

			objpoints.push_back(objp);
			imgpoints.push_back(corner_pts);
		}

		imshow("image",frame);
		waitKey(0);

	}

	destroyAllWindows();

	Mat cameraMatrix,distCoeffs,R,T;

	/*
	* Performing camera calibration by 
	* passing the value of known 3D points (objpoints)
	* and corresponding pixel coordinates of the 
	* detected corners (imgpoints)
	*/
	calibrateCamera(objpoints, imgpoints,Size(gray.rows,gray.cols),cameraMatrix,distCoeffs,R,T);

	cout << "cameraMatrix : " << cameraMatrix << endl;
	cout << "distCoeffs : " << distCoeffs << endl;
	cout << "Rotation vector : " << R << endl;
	cout << "Translation vector : " << T << endl;
}