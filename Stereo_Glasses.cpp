#include "Stereo_Glasses.h"

using namespace std;
using namespace cv;


void Stereo_Glasses::init(){
	// Use the rotation matrixes for stereo rectification and camera intrinsics for undistorting the image
	// Compute the rectification map (mapping between the original image pixels and 
	// their transformed values after applying rectification and undistortion) for left and right camera frames
	Mat Left_Stereo_Map1, Left_Stereo_Map2;
	Mat Right_Stereo_Map1, Right_Stereo_Map2;

	float new_mtxL_arr[] = {
		601.4346923828125, 0, 309.1313368132032,
		0, 603.6869506835938, 253.085649887249,
		0, 0, 1
	};
	float new_mtxR_arr[] {
		608.9773559570312, 0, 303.3466741101074,
		0, 609.688720703125, 240.115708415542,
		0, 0, 1
	};
	float  distL_arr[] = {
		-0.07207857185555731, 0.3702109229807098, -0.001033895048862672, -0.002113478862088626, -0.7795669503389046
	};
	float  distR_arr[] = {
		-0.03710126850135417, -0.1562190279237428, -0.0006885435733324113, -0.003325590629556568, 0.6356048318908557
	};
	float  rect_l_arr[] = {
		0.9985343257075832, -0.0470054938467536, -0.0268269255026565,
		0.04681369496626913, 0.998873699986078, -0.007733656297477665,
		0.02716023466959338, 0.006466453768891886, 0.999610177333318
	};
	float  rect_r_arr[] = {
		0.9964467320530139, 0.0003576923945795528, -0.08422459401516917,
		0.0002410672480103206, 0.9999747741026808, 0.007098805876643879,
		0.08422500856308147, -0.007093885708354536, 0.9964215095621459
	};
	float  proj_mat_l_arr[] = {
		606.6878356933594, 0, 346.6963882446289, 0,
		0, 606.6878356933594, 247.1882648468018, 0,
		0, 0, 1, 0
	};
	float  proj_mat_r_arr[] = {
		606.6878356933594, 0, 346.6963882446289, -2245.23721291054,
		0, 606.6878356933594, 247.1882648468018, 0,
		0, 0, 1, 0
	};

	// mtxL = Mat(3, 3, CV_32FC1, cameraMatrixL);
	Mat new_mtxL = Mat(3, 3, CV_32FC1, new_mtxL_arr);
	Mat new_mtxR = Mat (3, 3, CV_32FC1, new_mtxR_arr);
	Mat distL = Mat (5, 1, CV_32FC1, distL_arr);
	Mat distR = Mat (5, 1, CV_32FC1, distR_arr);
	Mat rect_l = Mat (3, 3, CV_32FC1, rect_l_arr);
	Mat rect_r = Mat (3, 3, CV_32FC1, rect_r_arr);
	Mat proj_mat_l = Mat (3, 4, CV_32FC1, proj_mat_l_arr);
	Mat proj_mat_r = Mat (3, 4, CV_32FC1, proj_mat_r_arr);
	Mat grayL, grayR, rectFrameL, rectFrameR;



	// frameL = imread(savePath + "New Folder/imageLeft.jpg");
	Mat frameL = imread(savePath + "stereo/imageLeft70.jpg");
	cvtColor(frameL,grayL,COLOR_BGR2GRAY);
	// frameR = imread(savePath + "New Folder/imageRight.jpg");
	Mat frameR = imread(savePath + "stereo/imageRight70.jpg");
	cvtColor(frameR,grayR,COLOR_BGR2GRAY);

	cv::initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l, grayL.size(), CV_16SC2, Left_Stereo_Map1, Left_Stereo_Map2);
	cv::initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r, grayR.size(), CV_16SC2, Right_Stereo_Map1, Right_Stereo_Map2);
	cv::remap(grayL, rectFrameL, Left_Stereo_Map1, Left_Stereo_Map2, INTER_LINEAR);
	cv::remap(grayR, rectFrameR, Right_Stereo_Map1, Right_Stereo_Map2, INTER_LINEAR);

	Mat tempFrame1, tempFrame2;

    // Draw the lines on the second image
    cv::cvtColor(rectFrameL, rectFrameL, cv::COLOR_GRAY2BGR);
    cv::cvtColor(rectFrameR, rectFrameR, cv::COLOR_GRAY2BGR);

	hconcat(frameL, frameR, tempFrame1);
	imshow("Left and Right", tempFrame1);
	hconcat(rectFrameL, rectFrameR, tempFrame2);
	imshow("Rectified Frames", tempFrame2);
	waitKey(0);


	// Draw horizontal lines on the concatenated image
    int numLines = 30;  // Adjust the number of lines as needed
    int lineHeight = tempFrame1.rows / (numLines + 1);  // Evenly distribute lines
	vector<Scalar> colors = {Scalar(0, 255, 0), Scalar(255, 0, 0), Scalar(0, 0, 255)};
    for (int i = 1; i <= numLines; ++i) {
        int y = i * lineHeight;
        cv::line(tempFrame1, cv::Point(0, y), cv::Point(tempFrame1.cols, y), colors[i%3], 1);
        cv::line(tempFrame2, cv::Point(0, y), cv::Point(tempFrame2.cols, y), colors[i%3], 1);
    }

    // Display the concatenated image with horizontal lines
	cv::imshow("Left and Right with Lines", tempFrame1);
    cv::imshow("Rectified Frames with Lines", tempFrame2);
    cv::waitKey(0);


	double fx = new_mtxL.at<double>(0, 0);
	double fy = new_mtxL.at<double>(1, 1);
	double cx = new_mtxL.at<double>(0, 2);
	double cy = new_mtxL.at<double>(1, 2);
	double b = 10.29;


    cv::cvtColor(rectFrameL, rectFrameL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rectFrameR, rectFrameR, cv::COLOR_BGR2GRAY);


	getDepthMap(rectFrameL, rectFrameR);
	waitKey(0);
}


int Stereo_Glasses::run(){
	Mat grayL, grayR, frameDrawL, frameDrawR, frameDraw, rectFrameL, rectFrameR;
	bool successL, successR;
	vector<Point2f> corner_ptsL, corner_ptsR;

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


	Mat gridFrame;
	int i = 0;
	while(true){

		if(!frameLeft.empty() && !frameRight.empty()){
			cvtColor(frameLeft,grayL,COLOR_BGR2GRAY);
			cvtColor(frameRight,grayR,COLOR_BGR2GRAY);

			hconcat(frameLeft, frameRight, frame);
			frame.copyTo(gridFrame);
			// // Draw horizontal lines on the concatenated image
			// int numLines = 30;  // Adjust the number of lines as needed
			// int lineHeight = gridFrame.rows / (numLines + 1);  // Evenly distribute lines
			// vector<Scalar> colors = {Scalar(0, 255, 0), Scalar(255, 0, 0), Scalar(0, 0, 255)};
			// for (int j = 1; j <= numLines; ++j) {
			// 	int y = j * lineHeight;
			// 	cv::line(gridFrame, cv::Point(0, y), cv::Point(gridFrame.cols, y), colors[j%3], 1);
			// }
			imshow("Stereo Stream", frame);
			// imshow("Stereo Stream with Lines", gridFrame);

			cv::remap(grayL, rectFrameL, Left_Stereo_Map1, Left_Stereo_Map2, INTER_LINEAR);
			cv::remap(grayR, rectFrameR, Right_Stereo_Map1, Right_Stereo_Map2, INTER_LINEAR);
			getDepthMap(rectFrameL, rectFrameR);
		}









		char key = waitKey(1);
        if (key >= 0) {
			switch (key) {
				case 'c':
					frameLeft.copyTo(frameDrawL);
					frameRight.copyTo(frameDrawR);

					successL = findChessboardCorners(grayL,Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_ptsL, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
					successR = findChessboardCorners(grayR,Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_ptsR, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
					if(successL && successR){
						TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001);

						// refining pixel coordinates for given 2d points.
						cornerSubPix(grayL,corner_ptsL,Size(11,11), Size(-1,-1),criteria);
						cornerSubPix(grayR,corner_ptsR,Size(11,11), Size(-1,-1),criteria);

						// Displaying the detected corner points on the checker board
						drawChessboardCorners(frameDrawL, Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_ptsL,successL);
						drawChessboardCorners(frameDrawR, Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_ptsR,successR);
						hconcat(frameDrawL, frameDrawR, frameDraw);
						// imshow("Stereo cornersL", frameDrawL);
						// imshow("Stereo cornersR", frameDrawR);
						imshow("Stereo corners", frameDraw);
						char saveKey = waitKey(0);

						if(saveKey == 'x'){
							// Save the current frame to the specified path
							imwrite(savePath + "stereo/imageLeft" + to_string(i) + ".jpg", frameLeft);
							imwrite(savePath + "stereo/imageRight" + to_string(i) + ".jpg", frameRight);
							cout << i++ << ": Frame saved to: " << savePath + "stereo/" << endl;
						}

					}
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



void Stereo_Glasses::stereoCalibrate() {
	// Creating vector to store vectors of 3D points for each checkerboard image
	vector<vector<Point3f> > objpoints;

	// Creating vector to store vectors of 2D points for each checkerboard image
	vector<vector<Point2f> > imgpointsL, imgpointsR;

	// Defining the world coordinates for 3D points
	vector<Point3f> objp;
	for(int i{0}; i<CHECKERBOARD[1]; i++) {
		for(int j{0}; j<CHECKERBOARD[0]; j++)
		objp.push_back(Point3f(j,i,0));
	}


	// Extracting path of individual image stored in a given directory
	vector<String> imagesL, imagesR;

	// Path of the folder containing checkerboard images
	string pathL = savePath + "stereo/imageLeft*.jpg";
	string pathR = savePath + "stereo/imageRight*.jpg";

	glob(pathL, imagesL);
	glob(pathR, imagesR);

	Mat frameL, frameR, grayL, grayR;
	// vector to store the pixel coordinates of detected checker board corners 
	vector<Point2f> corner_ptsL, corner_ptsR;
	bool successL, successR;

	// Looping over all the images in the directory
	for(int i{0}; i<imagesL.size(); i++) {

		frameL = imread(imagesL[i]);
		cvtColor(frameL,grayL,COLOR_BGR2GRAY);

		frameR = imread(imagesR[i]);
		cvtColor(frameR,grayR,COLOR_BGR2GRAY);

		// Finding checker board corners
		// If desired number of corners are found in the image then success = true  
		successL = findChessboardCorners( grayL, Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_ptsL); // CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
		successR = findChessboardCorners( grayR, Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_ptsR); // CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

		/*
		* If desired number of corner are detected,
		* we refine the pixel coordinates and display 
		* them on the images of checker board
		*/
		if((successL) && (successR)) {
			TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001);

			// refining pixel coordinates for given 2d points.
			cornerSubPix(grayL,corner_ptsL,Size(11,11), Size(-1,-1),criteria);
			cornerSubPix(grayR,corner_ptsR,Size(11,11), Size(-1,-1),criteria);

			// Displaying the detected corner points on the checker board
			drawChessboardCorners(frameL, Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_ptsL,successL);
			drawChessboardCorners(frameR, Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_ptsR,successR);

			objpoints.push_back(objp);
			imgpointsL.push_back(corner_ptsL);
			imgpointsR.push_back(corner_ptsR);

			// Mat concatFrame;
			// hconcat(frameL, frameR, concatFrame);
			// imshow("chess corners", concatFrame);
			// waitKey(0);
		}

		cout << i << endl;
	}

	cv::destroyAllWindows();

	Mat mtxL,distL,R_L,T_L;
	Mat mtxR,distR,R_R,T_R;

	/*
		* Performing camera calibration by 
		* passing the value of known 3D points (objpoints)
		* and corresponding pixel coordinates of the 
		* detected corners (imgpoints)
	*/

	Mat new_mtxL, new_mtxR;
	int alpha = 0;

	// mtxL = Mat(3, 3, CV_32FC1, cameraMatrixL);
	// distL = Mat(5, 1, CV_32FC1, distCoeffsL);
	// mtxR = Mat(3, 3, CV_32FC1, cameraMatrixR);
	// distR = Mat(5, 1, CV_32FC1, distCoeffsR);
	

	// Calibrating left camera
	cv::calibrateCamera(objpoints, imgpointsL, grayL.size(), mtxL, distL, R_L, T_L);
	new_mtxL = cv::getOptimalNewCameraMatrix(mtxL, distL, grayL.size(), alpha, grayL.size(), 0);

	cout << "left Camera Matrix: " << mtxL << endl;
	cout << "left Camera New Matrix: " << new_mtxL << endl;
	cout << "left Camera distortion: " << distL << endl;

	// Calibrating right camera
	cv::calibrateCamera(objpoints, imgpointsR, grayR.size(), mtxR, distR, R_R, T_R);
	new_mtxR = cv::getOptimalNewCameraMatrix(mtxR, distR, grayR.size(), alpha, grayR.size(), 0);

	cout << "right Camera Matrix: " << mtxR << endl;
	cout << "right Camera New Matrix: " << new_mtxR << endl;
	cout << "right Camera distortion: " << distR << endl;

	// Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat 
	// are calculated. Hence intrinsic parameters are the same.
	Mat Rot, Trns, Emat, Fmat;

	int flag = 0;
	flag |= CALIB_FIX_INTRINSIC;


	// This step is performed to transformation between the two cameras and calculate Essential and 
	// Fundamenatl matrix
	cv::stereoCalibrate(objpoints, imgpointsL, imgpointsR, new_mtxL, distL, new_mtxR, distR, grayR.size(), Rot, Trns, Emat, Fmat, flag, TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 60, 0.5e-6));
	cout << "Rotation: " << Rot << endl;
	cout << "Translation: " << Trns << endl;
	cout << "Essential Matrix: " << Emat << endl;
	cout << "Fundamental Matrix: " << Fmat << endl;
	cout << "D left: " << distL << endl;
	cout << "D Right: " << distR << endl;

	Mat rect_l, rect_r, proj_mat_l, proj_mat_r, Q;

	// Once we know the transformation between the two cameras we can perform 
	// stereo rectification
	cv::stereoRectify(new_mtxL, distL, new_mtxR, distR, grayR.size(), Rot, Trns, rect_l, rect_r, proj_mat_l, proj_mat_r, Q, CALIB_ZERO_DISPARITY);
	cout << "Rectification Rotation Left: " << rect_l << endl;
	cout << "Rectification Rotation Right: " << rect_r << endl;
	cout << "Project of Left Camera: " << proj_mat_l << endl;
	cout << "Project of Right Camera: " << proj_mat_r << endl;
	cout << "Q Matrix: " << Q << endl;

	// Use the rotation matrixes for stereo rectification and camera intrinsics for undistorting the image
	// Compute the rectification map (mapping between the original image pixels and 
	// their transformed values after applying rectification and undistortion) for left and right camera frames

	initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l, grayL.size(), CV_16SC2, Left_Stereo_Map1, Left_Stereo_Map2);
	initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r, grayR.size(), CV_16SC2, Right_Stereo_Map1, Right_Stereo_Map2);

	// cout << "Left Map x" << Left_Stereo_Map1 << endl;
	// cout << "Left Map y" << Left_Stereo_Map2 << endl;
	// cout << "Right Map x" << Right_Stereo_Map1 << endl;
	// cout << "Right Map y" << Right_Stereo_Map2 << endl;

	FileStorage cv_file = FileStorage(savePath + "params_cpp.xml", FileStorage::WRITE);
	cv_file.write("Left_Stereo_Map_1",Left_Stereo_Map1);
	cv_file.write("Left_Stereo_Map_2",Left_Stereo_Map2);
	cv_file.write("Right_Stereo_Map_1",Right_Stereo_Map1);
	cv_file.write("Right_Stereo_Map_2",Right_Stereo_Map2);
	cv_file.release();


	cv:Mat rectFrameL, rectFrameR;
	// frameL = imread(savePath + "New Folder/imageLeft.jpg");
	frameL = imread(imagesL[70]);
	cvtColor(frameL,grayL,COLOR_BGR2GRAY);
	// frameR = imread(savePath + "New Folder/imageRight.jpg");
	frameR = imread(imagesR[70]);
	cvtColor(frameR,grayR,COLOR_BGR2GRAY);
	cv::remap(grayL, rectFrameL, Left_Stereo_Map1, Left_Stereo_Map2, INTER_LINEAR);
	cv::remap(grayR, rectFrameR, Right_Stereo_Map1, Right_Stereo_Map2, INTER_LINEAR);

	Mat tempFrame1, tempFrame2;

    // Draw the lines on the second image
    cv::cvtColor(rectFrameL, rectFrameL, cv::COLOR_GRAY2BGR);
    cv::cvtColor(rectFrameR, rectFrameR, cv::COLOR_GRAY2BGR);

	hconcat(grayL, grayR, tempFrame1);
	imshow("Left and Right", tempFrame1);
	hconcat(rectFrameL, rectFrameR, tempFrame2);
	imshow("Rectified Frames", tempFrame2);
	waitKey(0);


	// Draw horizontal lines on the concatenated image
    int numLines = 30;  // Adjust the number of lines as needed
    int lineHeight = tempFrame1.rows / (numLines + 1);  // Evenly distribute lines
	vector<Scalar> colors = {Scalar(0, 255, 0), Scalar(255, 0, 0), Scalar(0, 0, 255)};
    for (int i = 1; i <= numLines; ++i) {
        int y = i * lineHeight;
        cv::line(tempFrame1, cv::Point(0, y), cv::Point(tempFrame1.cols, y), colors[i%3], 1);
        cv::line(tempFrame2, cv::Point(0, y), cv::Point(tempFrame2.cols, y), colors[i%3], 1);
    }

    // Display the concatenated image with horizontal lines
	cv::imshow("Left and Right with Lines", tempFrame1);
    cv::imshow("Rectified Frames with Lines", tempFrame2);
    cv::waitKey(0);


	fx = new_mtxL.at<double>(0, 0);
	fy = new_mtxL.at<double>(1, 1);
	cx = new_mtxL.at<double>(0, 2);
	cy = new_mtxL.at<double>(1, 2);
	b = 10.29;


    cv::cvtColor(rectFrameL, rectFrameL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rectFrameR, rectFrameR, cv::COLOR_BGR2GRAY);

	// imwrite(savePath + "rectFrameL", rectFrameL);
	// imwrite(savePath + "rectFrameR", rectFrameR);


	getDepthMap(rectFrameL, rectFrameR);
	waitKey(0);
}

void onTrackbar(int, void* userdata) {
    Stereo_Glasses* stereoGlasses = static_cast<Stereo_Glasses*>(userdata);

	// Ensure blockSize is always odd and does not exceed 255
    stereoGlasses->blockSize = (stereoGlasses->blockSize % 2 == 0) ? stereoGlasses->blockSize + 1 : stereoGlasses->blockSize;
    stereoGlasses->blockSize = std::min(stereoGlasses->blockSize, 255);
    stereoGlasses->blockSize = std::max(5, stereoGlasses->blockSize);


    stereoGlasses->updateDisparity();
}


void Stereo_Glasses::updateDisparity() {
	P1 = 8*9*blockSize*blockSize; //*blockSize;
	P2 = 32*9*blockSize*blockSize; //*blockSize;
	numDisparities = 16*disparityFactor - minDisparity;

    leftMatcher->setBlockSize(blockSize);
    leftMatcher->setNumDisparities(numDisparities);

	rightMatcher->setBlockSize(blockSize);
    rightMatcher->setNumDisparities(numDisparities);

    leftMatcher->compute(rectified_left, rectified_right, left_disp);
    rightMatcher->compute(rectified_right, rectified_left, right_disp);

    wlsFilter->setLambda(5000.0);
    wlsFilter->setSigmaColor(1.5);

    normalize(left_disp, left_disp, 0, 255, cv::NORM_MINMAX, CV_8U);
    normalize(right_disp, right_disp, 0, 255, cv::NORM_MINMAX, CV_8U);

    wlsFilter->filter(left_disp, rectified_left, filteredDisparity, right_disp, Rect(), rectified_right);
    cv::applyColorMap(filteredDisparity, coloredDisparity, cv::COLORMAP_JET);
}


void Stereo_Glasses::getDepthMap(Mat left, Mat right) {
	left.copyTo(rectified_left);
	right.copyTo(rectified_right);

	Size kernel = Size(k, k);
	GaussianBlur(left, left, kernel, 0);
	GaussianBlur(right, right, kernel, 0);


	cv::namedWindow("Parameter Tuning", cv::WINDOW_NORMAL);

    cv::createTrackbar("BlockSize", "Parameter Tuning", &blockSize, 255, onTrackbar, this);
    cv::createTrackbar("disparityFactor", "Parameter Tuning", &disparityFactor, 255, onTrackbar, this);


	while (true) {
        updateDisparity();

		hconcat(left_disp, right_disp, disparity);
        cv::imshow("disparity", disparity);
        cv::imshow("filtered disparity", filteredDisparity);
        cv::imshow("Colored Disparity", coloredDisparity);
        cv::imshow("confidence map", wlsFilter->getConfidenceMap());

        int key = cv::waitKey(1);
        if (key == 27)  // Escape key to exit
            break;
    }


	// normalize(left_disp , left_disp, 0, 255, NORM_MINMAX, CV_8U);
	// normalize(right_disp , right_disp, 0, 255, NORM_MINMAX, CV_8U);


	// wlsFilter->filter(left_disp, left ,filteredDisparity, right_disp, Rect(), right);
	// hconcat(left_disp, right_disp, disparity);
	// cv::applyColorMap(filteredDisparity, coloredDisparity, cv::COLORMAP_JET);

	// cv::imshow("disparity", disparity);
	// cv::imshow("filtered disparity", filteredDisparity);
	// // cv::imshow("Colored Disparity", coloredDisparity);
	// cv::imshow("confidence map", wlsFilter->getConfidenceMap());
    // // showPointCloud(pointcloud);
}


















void Stereo_Glasses::showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {
    // //draw point cloud
    // //Generate point cloud
    // vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;
    // //If your machine is slow, please change the following v++ and u++ to v+=2, u+=2
    // for (int v = 0; v < left.rows; v++)
    //     for (int u = 0; u < left.cols; u++) {
    //         if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0) continue;

    //         Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0);//The first three dimensions are xyz, and the fourth dimension is color

    //         //Calculate the position of point based on the binocular model
    //         double x = (u - cx) / fx;
    //         double y = (v - cy) / fy;
    //         double depth = fx * b / (disparity.at<float>(v, u));
    //         point[0] = x * depth;
    //         point[1] = y * depth;
    //         point[2] = depth;

    //         pointcloud.push_back(point);
    //     }
    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);//sleep 5 ms
    }
    return;
}