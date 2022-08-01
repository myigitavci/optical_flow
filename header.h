#ifndef HEADER_H
#define HEADER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>

#include <QCoreApplication>
#include<fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include "opencv2/stitching.hpp"
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, const Scalar& color);
void dense_optical_flow(string data_type,string blank_or_image,vector<string> filenames, size_t N);
void sparse_optical_flow(string data_type,vector<string> filenames, size_t N);
void draw_matched_features(string data_type,vector<string> filenames, size_t N);
vector<DMatch> find_matches(Mat img1, Mat img2,vector<KeyPoint> *keypoints1,vector<KeyPoint> *keypoints2);
Mat stitch_image(Mat image1, Mat image2, Mat H);
void stitch_images(string data_type,vector<string> filenames, size_t N);
vector<Scalar> random_colors();

#endif // HEADER_H
