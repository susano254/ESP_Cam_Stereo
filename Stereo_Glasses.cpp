#include "Stereo_Glasses.h"

using namespace std;
using namespace cv;



int Stereo_Glasses::runOne(){
	//if stream is not active print error and return
	if(!videoCapture.open(RIGHT_URL + STREAM)){
		cout << "Error opening stream" << endl;
		return -1;
	}

	thread thread(captureFrames, this, ref(videoCapture), ref(frame), "stream");

	int i = 0;
	while(true){
		if(!frame.empty()){
			imshow("Stereo Stream", frame);
		}

		char key = waitKey(1);
        if (key >= 0) {
			switch (key) {
				case 'c':
					// Save the current frame to the specified path
					imwrite(savePath + "right/imageRight" + to_string(i) + ".jpg", frame);
					i++;

					cout << "Frame saved to: " << savePath << endl;
					break;

				case 'q':
					terminateThreads = true;
					break;
			}
        }

		if(terminateThreads){
			thread.join();
			break;
		} 
	}
	return 0;
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


	int i = 0;
	while(true){
		if(!frameLeft.empty() && !frameRight.empty()){
			hconcat(frameLeft, frameRight, frame);
			// Draw horizontal lines on the concatenated image
			// int numLines = 30;  // Adjust the number of lines as needed
			// int lineHeight = frame.rows / (numLines + 1);  // Evenly distribute lines
			// vector<Scalar> colors = {Scalar(0, 255, 0), Scalar(255, 0, 0), Scalar(0, 0, 255)};
			// for (int i = 1; i <= numLines; ++i) {
			// 	int y = i * lineHeight;
			// 	cv::line(frame, cv::Point(0, y), cv::Point(frame.cols, y), colors[i%3], 1);
			// }
			imshow("Stereo Stream", frame);
		}

		char key = waitKey(1);
        if (key >= 0) {
			switch (key) {
				case 'c':
					// Save the current frame to the specified path
					imwrite(savePath + "stereo/imageLeft" + to_string(i) + ".jpg", frameLeft);
					imwrite(savePath + "stereo/imageRight" + to_string(i) + ".jpg", frameRight);
					i++;

					cout << "Frame saved to: " << savePath << endl;
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

void Stereo_Glasses::showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {

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

void Stereo_Glasses::calibrate() {

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

	// glob(savePath+ "left/imageLeft*.jpg", images);
	glob(savePath+ "right/imageRight*.jpg", images);

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
		cout << "image " << i << endl;
		waitKey(0);
	}

	cv::destroyAllWindows();

	Mat cameraMatrix,distCoeffs,R,T;

	/*
	* Performing camera calibration by 
	* passing the value of known 3D points (objpoints)
	* and corresponding pixel coordinates of the 
	* detected corners (imgpoints)
	*/
	double error = cv::calibrateCamera(objpoints, imgpoints,Size(gray.rows,gray.cols),cameraMatrix,distCoeffs,R,T);

	cout << "error : " << error << endl;
	cout << "cameraMatrix : " << cameraMatrix << endl;
	cout << "distCoeffs : " << distCoeffs << endl;
	cout << "Rotation vector : " << R << endl;
	cout << "Translation vector : " << T << endl;
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
		}

		cout << i << endl;
		// imshow("ImageL",frameL);
		// imshow("ImageR",frameR);
		// waitKey(0);
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

	// float cameraMatrix[] = {
	// 	614.8675967318162, 0, 308.0401953721954,
	// 	0, 614.864068654533, 239.466379271711,
	// 	0, 0, 1
	// };

	float cameraMatrixL[] = {
		607.313065591494, 0, 314.3565259270047,
		0, 605.1815436189786, 256.0093392116243,
		0, 0, 1
	};
	// float distCoeffsL[5] = { -0.05034197657097012, -0.1344318428062239, 0.0001064478496631177, -0.0009838564388245873, 1.062100776882182};
	float distCoeffsL[5] = {0};

	float cameraMatrixR[] = {
		615, 0, 308,
		0, 615, 240,
		0, 0, 1
	};
	// float distCoeffsR[5] = {-0.02578693747847143, -0.5278683009715619, -0.0008844399023631207, -0.002347492101437894, 2.029137482107601};
	float distCoeffsR[5] = {0};

	// float cameraMatrixL[] = {
	// 	607.313065591494, 0, 314.3565259270047,
	// 	0, 605.1815436189786, 256.0093392116243,
	// 	0, 0, 1
	// };
	// float cameraMatrixR[] = {
	// 	614.8675967318162, 0, 308.0401953721954,
	// 	0, 614.864068654533, 239.466379271711,
	// 	0, 0, 1

	// };
	// // float cameraMatrixR[] = {
	// // 	608.1660156003267, 0, 308.7438310585804,
	// // 	0, 607.2194197874154, 246.0961775610207,
	// // 	0, 0, 1
	// // };

	// float distCoeffs[] = { -0.02140906643521146, -0.09417490816520702, -0.001046617967208666, -0.000173185765608084, 0.3563403620045658};
	// // float distCoeffs[5] = { 0 };
	mtxL = Mat(3, 3, CV_32FC1, cameraMatrixL);
	distL = Mat(5, 1, CV_32FC1, distCoeffsL);
	mtxR = Mat(3, 3, CV_32FC1, cameraMatrixR);
	distR = Mat(5, 1, CV_32FC1, distCoeffsR);
	

	// Calibrating left camera
	// cv::calibrateCamera(objpoints, imgpointsL, grayL.size(), mtxL, distL, R_L, T_L);
	new_mtxL = cv::getOptimalNewCameraMatrix(mtxL, distL, grayL.size(), alpha, grayL.size(), 0);

	cout << "left Camera Matrix: " << mtxL << endl;
	cout << "left Camera New Matrix: " << new_mtxL << endl;

	// Calibrating right camera
	// cv::calibrateCamera(objpoints, imgpointsR, grayR.size(), mtxR, distR, R_R, T_R);
	new_mtxR = cv::getOptimalNewCameraMatrix(mtxR, distR, grayR.size(), alpha, grayR.size(), 0);

	// new_mtxL = mtxL;
	// new_mtxR = mtxR;
	distL = 0;
	distR = 0;


	cout << "right Camera Matrix: " << mtxR << endl;
	cout << "right Camera New Matrix: " << new_mtxR << endl;

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
	Mat Left_Stereo_Map1, Left_Stereo_Map2;
	Mat Right_Stereo_Map1, Right_Stereo_Map2;

	cv::initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l, grayL.size(), CV_16SC2, Left_Stereo_Map1, Left_Stereo_Map2);
	cv::initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r, grayR.size(), CV_16SC2, Right_Stereo_Map1, Right_Stereo_Map2);

	// cout << "Left Map x" << Left_Stereo_Map1 << endl;
	// cout << "Left Map y" << Left_Stereo_Map2 << endl;
	// cout << "Right Map x" << Right_Stereo_Map1 << endl;
	// cout << "Right Map y" << Right_Stereo_Map2 << endl;

	// FileStorage cv_file = FileStorage(savePath + "params_cpp.xml", FileStorage::WRITE);
	// cv_file.write("Left_Stereo_Map_x",Left_Stereo_Map1);
	// cv_file.write("Left_Stereo_Map_y",Left_Stereo_Map2);
	// cv_file.write("Right_Stereo_Map_x",Right_Stereo_Map1);
	// cv_file.write("Right_Stereo_Map_y",Right_Stereo_Map2);
	// cv_file.release();


	cv:Mat rectFrameL, rectFrameR;
	// frameL = imread(savePath + "New Folder/imageLeft.jpg");
	frameL = imread(imagesL[0]);
	cvtColor(frameL,grayL,COLOR_BGR2GRAY);
	// frameR = imread(savePath + "New Folder/imageRight.jpg");
	frameR = imread(imagesR[0]);
	cvtColor(frameR,grayR,COLOR_BGR2GRAY);
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
	getDepthMap(rectFrameL, rectFrameR, fx, fy, cx, cy, b);
}

void Stereo_Glasses::getDepthMap(Mat left, Mat right, double fx, double fy, double cx, double cy, double b) {
	int blockSize = 9;
	int P1 = 8*9*blockSize; //*blockSize;
	int P2 = 32*9*blockSize; //*blockSize;
	int minDisparity = 0;
	int disparityFactor = 6;
	int numDisparities = 16*disparityFactor - minDisparity;
    cv::Mat disparity_sgbm, disparity;

    // cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
	// 	minDisparity, 
	// 	numDisparities, 
	// 	blockSize, 
	// 	P1, 
	// 	P2, 
	// 	1,
	// 	 63, 
	// 	 15, 
	// 	 5, 
	// 	 32,
	// 	 StereoSGBM::MODE_SGBM
	// 	 );	//magic parameters
    // sgbm->compute(left, right, disparity_sgbm);


    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(numDisparities, blockSize);
    bm->compute(left, right, disparity_sgbm);

    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);

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
	Mat filteredDisparity;
	// cv::bilateralFilter(disparity, filteredDisparity, 15, 40,40);
	cv::imshow("disparity", disparity / numDisparities);
	// cv::imshow("filtered disparity", filteredDisparity / numDisparities);
    cv::waitKey(0);
    // //draw point cloud
    // showPointCloud(pointcloud);
}
