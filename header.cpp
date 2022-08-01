// Function codes for EE_576 Project 5
// Mehmet Yiğit Avcı
// Bogazici University, 2022

#include "header.h"

// ref: https://github.com/yangwangx/denseFlow_gpu/blob/master/denseFlow_gpu.cpp
// this function draws vector field
void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, const Scalar& color)
{
    for (int y = 0; y < cflowmap.rows; y += step) {
        for (int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), color);
            circle(cflowmap, Point(x, y), 1.3, color, -1);

        }
    }
}

// ref: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
// this function draws dense optical flow for each consecutive data pairs in the dataset
void dense_optical_flow(string data_type,string blank_or_image,vector<string> filenames, size_t N)
{


    for (size_t k = 0; k < N-1; ++k)
    {
        // load images and convert to gray image
        cv::Mat previous = cv::imread(filenames[k]);
        cv::Mat previous2=previous;
        cv::Mat current = cv::imread(filenames[k+1]);
        cv::Mat current2=current;
        cvtColor(previous, previous, COLOR_BGR2GRAY);
        cvtColor(current, current, COLOR_BGR2GRAY);

        // calculate optical dense flow
        Mat flow(previous.size(), CV_32FC2);
        calcOpticalFlowFarneback(previous, current, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        // drawing vector fields on blank or the previous image
        Mat cflow;
        if (blank_or_image=="image")
        {
            cvtColor(previous, cflow, COLOR_GRAY2BGR);
            drawOptFlowMap(flow, cflow, 10, Scalar(10, 255, 0));
        }

        else if (blank_or_image=="blank")
        {
            cflow = Mat::zeros(flow.size(), CV_32F);
            drawOptFlowMap(flow, cflow, 10, Scalar(10, 255, 0));

        }

        imshow("Dense Optical Flow", cflow);

        //  int keyboard = waitKey(30);
        // if (keyboard == 'q' || keyboard == 27)
        //    break;
        waitKey(0);
    }

}
// create random colors
vector<Scalar> random_colors()
{
    // creating random colors
    vector<Scalar> colors;
    RNG rng;
    for(int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r,g,b));
    }
return colors;
}

// ref: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
// this function calculates sparse optical flow and draws on the images
void sparse_optical_flow(string data_type,vector<string> filenames, size_t N)
{
    vector<Scalar> colors=random_colors();

    // create a zeros mask
    vector<Point2f> p0, p1;
    cv::Mat previous = cv::imread(filenames[0]);
    Mat mask = Mat::zeros(previous.size(), previous.type());

    for (size_t k = 0; k < N-1; ++k)
    {
        // load images and convert to gray image
        cv::Mat previous = cv::imread(filenames[k]);
        cv::Mat previous2=previous;
        cv::Mat current = cv::imread(filenames[k+1]);
        cv::Mat current2=current;
        cvtColor(previous, previous, COLOR_BGR2GRAY);
        cvtColor(current, current, COLOR_BGR2GRAY);

        // calculate sparse optical flow
        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(previous, current, p0, p1, status, err, Size(15,15), 2, criteria);
        vector<Point2f> good_new;
        for(uint i = 0; i < p0.size(); i++)
        {
            // Select good points
            if(status[i] == 1) {
                good_new.push_back(p1[i]);
                // draw the tracks
                line(mask,p1[i], p0[i], colors[i], 2);
                circle(current2, p1[i], 3, colors[i], -1);
            }
        }
        Mat result;
        add(current2, mask, result);

        imshow("Sparse Optical flow", result);
        //  int keyboard = waitKey(30);
        // if (keyboard == 'q' || keyboard == 27)
        //    break;
        waitKey(0);

        p0 = good_new;

    }
}

// ref: https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
// this function calculates the SIFT features and draws matched features in consecutive image pairs
void draw_matched_features(string data_type,vector<string> filenames, size_t N)
{

    vector<KeyPoint> keypoints1,keypoints2;

    for (size_t k = 0; k < N-1; ++k)
    {
        cv::Mat previous = cv::imread(filenames[k]);
        cv::Mat current = cv::imread(filenames[k+1]);
        vector<DMatch> good_matches=find_matches(previous,current,&keypoints1,&keypoints2);

        //draw matches
        Mat img_matches;
        drawMatches( previous, keypoints1, current, keypoints2, good_matches, img_matches, Scalar::all(-1),
                     Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //show detected matches
        imshow("Good Matches", img_matches );
        waitKey();
    }
}

// this function looks at SIFT features and finds the good matches between two images
vector<DMatch> find_matches(Mat img1, Mat img2,vector<KeyPoint> *keypoints1,vector<KeyPoint> *keypoints2)
{
    //detect keypoints using SIFT and compute the descriptors
    int minHessian = 400;
    Ptr<SIFT> detector = SIFT::create( minHessian );
    Mat descriptors1, descriptors2;

    detector->detectAndCompute( img1, noArray(), *keypoints1, descriptors1 );
    detector->detectAndCompute( img2, noArray(), *keypoints2, descriptors2 );

    //match features
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );

    //find good matches
    const float ratio_thresh = 0.9f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    return good_matches;
}
// stitch two images with given homograpy matrix
Mat stitch_image(Mat image1, Mat image2, Mat H)
{

    cv::Mat result;
    warpPerspective(image1,result,H,cv::Size(image1.cols*2,image1.rows*2));
    cv::Mat half(result,cv::Rect(0,0,image2.cols,image2.rows));
    image2.copyTo(half);

    return result;
}

//ref: https://github.com/Manasi94/Image-Stitching
// This function stitchs all the images in the given dataset
void stitch_images(string data_type,vector<string> filenames, size_t N)
{

    cv::Mat previous = cv::imread(filenames[0]);
    cv::Mat current = cv::imread(filenames[1]);
    vector< Point2f > img1_points;
    vector< Point2f > img2_points;
    vector<KeyPoint> keypoints1,keypoints2;
    vector<DMatch> good_matches=find_matches(previous,current,&keypoints1,&keypoints2);
    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        img1_points.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
        img2_points.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
    }


    // find the homopraphy matrix
    Mat H = findHomography( img1_points, img2_points, RANSAC );
    Mat result=stitch_image(previous,current,H);
    for (size_t k = 1; k < N-1; ++k)
    {
        cv::Mat previous = result;
        cv::Mat current = cv::imread(filenames[k+1]);
        vector<DMatch> good_matches=find_matches(previous,current,&keypoints1,&keypoints2);

        vector< Point2f > img1_points;
        vector< Point2f > img2_points;

        for( int i = 0; i < good_matches.size(); i++ )
        {
            //-- Get the keypoints from the good matches
            img1_points.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
            img2_points.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
        }

        // find the homopraphy matrix
        Mat H = findHomography( img1_points, img2_points, RANSAC );
        result=stitch_image(result,current,H);
        Mat result_gray;
        cvtColor(result, result_gray, COLOR_BGR2GRAY);

        // finding the largest contour to remove the black region from image
        threshold(result_gray, result_gray,25, 255,THRESH_BINARY); //Threshold the gray
        vector<vector<Point> > contours2; // Vector for storing contour
        vector<Vec4i> hierarchy2;
        findContours( result_gray, contours2, hierarchy2,RETR_CCOMP, CHAIN_APPROX_SIMPLE ); // Find the contours in the image
        int largest_area2 = 0;
        Rect bounding_rect2;

        for( int i = 0; i< contours2.size(); i++ ) // iterate through each contour.
        {
            double a=contourArea( contours2[i],false);  //  Find the area of contour
            if(a>largest_area2){
                largest_area2=a;
                bounding_rect2=boundingRect(contours2[i]); // Find the bounding rectangle for biggest contour
            }

        }

        result = result(Rect(bounding_rect2.x, bounding_rect2.y, bounding_rect2.width, bounding_rect2.height));

        imshow("Stitched image", result );
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
        imwrite("../576_project5/stitched_result"+data_type+".bmp",result);
    }
}
