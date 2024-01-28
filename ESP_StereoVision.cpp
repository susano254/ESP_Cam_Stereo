#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>
#include <thread>

using namespace std;
#include "Stereo_Glasses.h"

int main(){
	Stereo_Glasses glasses;

	glasses.run();
	return 0;

}
