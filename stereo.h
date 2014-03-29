#ifndef STEREO_H
#define STEREO_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

/**
 * Camera class
 * Class where camera data is kept and handled
 */
class Camera
{
public:
    Camera( double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, double k3 = 0 );

    // change intrinsic matrix scale to handle images of different sizes e.g. 320x240
    void setScale(double scale);

    // camera matrix
    cv::Mat M;

    // scaled camera matrix - needed for reduced image sizes etc
    cv::Mat SM;

    // distortion coefficients
    cv::Mat D;
};

/**
 * Class where Stereo rig data is evaluated and kept
 */
class StereoRig
{
public:
    // constructors
    StereoRig() : camL(NULL), camR(NULL) {}
    StereoRig(Camera* _camL, Camera* _camR, const cv::Mat& _R, const cv::Mat& _T, const cv::Size& _size);

    // is rig empty (initialized through default constructor?)
    bool isActive() const { return NULL != camL; }

    // undistort and rectify an image
    static cv::Mat getUndistortedRectified(const cv::Mat& D, const cv::Mat M, const cv::Mat R, const cv::Mat P, cv::Mat& src);

    // get 3d data using image coordinates and disparity value. See the book Learning OpenCV pg. 453
    double getDepth(double xc, double yc, double d) const;

    // get 3d point using image coordiante and disparity
    cv::Point3d get3DPoint(double xc, double yc, double d) const;

    // left and right camera
    Camera* camL;
    Camera* camR;

    // rotation matrix bw cams
    cv::Mat R;

    // translation matrix bw cams
    cv::Mat T;

    // matrices needed by cv::stereoRectify function
    cv::Mat R1, R2, P1, P2, Q;
};

#endif // STEREO_H
