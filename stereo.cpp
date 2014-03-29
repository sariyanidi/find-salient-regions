#include "stereo.h"

/**
 * Initialize camera data;
 * camera matrix and distortion parameters
 *
 * @param double fx, fy - focal length, x,y
 * @param double cx, cy - displacement from optical axis
 * @param double k1, k2, p1, p2, k3 - distortion params
 */
Camera::Camera(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, double k3)
{
    M = cv::Mat::zeros(3,3,CV_64FC1); // camera matrix
    D = cv::Mat::zeros(5,1,CV_64FC1); // distortion coeffs

    M.at<double>(0,0) = fx;
    M.at<double>(0,2) = cx;
    M.at<double>(1,1) = fy;
    M.at<double>(1,2) = cy;
    M.at<double>(2,2) = 1.;

    SM = M.clone(); // scaled matrix

    D.at<double>(0,0) = k1;
    D.at<double>(1,0) = k2;
    D.at<double>(2,0) = p1;
    D.at<double>(3,0) = p2;
    D.at<double>(4,0) = k3;
}

/**
 * Useful function, all parameters so far are only valid for
 * 640x480 images. To work on smaller images one must change the
 * scale of the camera matrix!
 *
 * @param double scale
 */
void Camera::setScale(double scale)
{
    SM.at<double>(0,0) = scale*M.at<double>(0,0);
    SM.at<double>(0,1) = scale*M.at<double>(0,1);
    SM.at<double>(0,2) = scale*M.at<double>(0,2);
    SM.at<double>(1,0) = scale*M.at<double>(1,0);
    SM.at<double>(1,1) = scale*M.at<double>(1,1);
    SM.at<double>(1,2) = scale*M.at<double>(1,2);
}

/**
 * Constructor - Stereo Rig
 *
 * @param Camera* camL - ptr to left cam
 * @param Camera* camR - ptr to right cam
 * @param Mat R - Rotation bw cams
 * @param Mat T - Translation bw cams
 */
StereoRig::StereoRig(Camera *_camL, Camera *_camR, const cv::Mat &_R, const cv::Mat &_T, const cv::Size &_size)
    : camL(_camL), camR(_camR), R(_R), T(_T)
{
    cv::stereoRectify(camL->SM, camL->D, camR->SM, camR->D, _size, R, T, R1, R2, P1, P2, Q, 1.);
}

/**
 * Undistort and rectify an image given camera and rig data
 *
 * @param - camera params !
 * @return void
 */
cv::Mat StereoRig::getUndistortedRectified(const cv::Mat& D, const cv::Mat M, const cv::Mat R, const cv::Mat P, cv::Mat& src)
{
    using namespace cv;

    cv::Size imageSize = src.size();
    Mat view = src.clone();
    Mat rview, map1, map2;

    initUndistortRectifyMap(M, D, R, P, imageSize, CV_16SC2, map1, map2);

    if(!view.data)
        return view;
    //    undistort( view, rview, cameraMatrix, distCoeffs, cameraMatrix );
    remap(view, rview, map1, map2, INTER_LINEAR, BORDER_CONSTANT);

    return rview;
}

/**
 * Get depth from image coordinates and disparity data
 *
 * @param double xc - x coordinate on camera
 * @param double yc - y coordinate on camera
 * @param double d  - disparity obtained through stereo data
 * @return double depth - real world depth in mms
 */
double StereoRig::getDepth(double xc, double yc, double d) const
{
    cv::Mat x(4,1,CV_64FC1); // camera coordinates
    x.at<double>(0,0) = xc;
    x.at<double>(0,1) = yc;
    x.at<double>(0,2) = d;
    x.at<double>(0,3) = 1.0;

    cv::Mat X(4, 1, CV_64FC1); // real world coordinates
    X = Q*x;                   // use Q matrix of the stereo rig
    X = X/X.at<double>(3,0);

    return X.at<double>(2,0);  // return depth
}

/**
 * Get depth from image coordinates and disparity data
 *
 * @param double xc - x coordinate on camera
 * @param double yc - y coordinate on camera
 * @param double d  - disparity obtained through stereo data
 * @return double depth - real world depth in mms
 */
cv::Point3d StereoRig::get3DPoint(double xc, double yc, double d) const
{
    cv::Mat x(4,1,CV_64FC1); // camera coordinates
    x.at<double>(0,0) = xc;
    x.at<double>(0,1) = yc;
    x.at<double>(0,2) = d;
    x.at<double>(0,3) = 1.0;

    cv::Mat X(4, 1, CV_64FC1); // real world coordinates
    X = Q*x;                   // use Q matrix of the stereo rig
    X = X/X.at<double>(3,0);

    return cv::Point3d(X.at<double>(0,0), X.at<double>(1,0), X.at<double>(2,0));  // return depth
}
