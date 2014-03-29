#include "patchextractor.h"
#include "navigator.h"
#include "landmark.h"

#include <ctime>
#include <limits>
int main(int argc, char *argv[])
{
    /**/
    std::vector<std::string> sampleIms;
    sampleIms.push_back(std::string("./resources/00005.jpg"));
    sampleIms.push_back(std::string("./resources/8.jpg"));
    sampleIms.push_back(std::string("./resources/00006.jpg"));
    sampleIms.push_back(std::string("./resources/00007.jpg"));
    sampleIms.push_back(std::string("./resources/00008.jpg"));
    sampleIms.push_back(std::string("./resources/03.jpg"));
    sampleIms.push_back(std::string("./resources/9.jpg"));
    sampleIms.push_back(std::string("./resources/t.jpg"));
    sampleIms.push_back(std::string("./resources/1111.jpg"));


    clock_t t1 = clock();
    std::cout << "Detection took " << (double)(clock()-t1)/CLOCKS_PER_SEC << " seconds." << std::endl << std::endl;

    //cv::Mat im = cv::imread(sampleIm);
    //cv::VideoCapture capL(0);
    //cv::VideoCapture capR(1);


    // my rig
    /*
    Camera camL(884.619820246672475, 885.130452157858826,
                335.461342469849228, 225.605601852535898,
                -0.271168345926695, 0.171248086620693,
                0.000436448084702, 0.000333959090964);

    Camera camR(885.679500214111613, 885.587940988541959,
                308.631155554682550, 240.859136168840280,
                -0.272114096878774, -0.046570787534115,
                -0.001413873867930, 0.003539442941778);
        */
    Camera camL(800.6569, 796.9799,
                591.9070/2, 391.7431/2,
                0., 0.,
                0., 0.);

    Camera camR(795.4553, 795.0219,
                599.9589/2, 383.6223/2,
                0., 0.,
                0., 0.);

    cv::Mat R(3,3,CV_64FC1);
    cv::Mat T(3,1,CV_64FC1);

    /*
    // my rig
    R.at<double>(0,0) = 0.9967;
    R.at<double>(0,1) = -0.0440;
    R.at<double>(0,2) = 0.0677;
    R.at<double>(1,0) = 0.0449;
    R.at<double>(1,1) = 0.9989;
    R.at<double>(1,2) = -0.0110;
    R.at<double>(2,0) = -0.0671;
    R.at<double>(2,1) = 0.0140;
    R.at<double>(2,2) = 0.9976;

    T.at<double>(0,0) = -143.8316;
    T.at<double>(1,0) = -5.6844;
    T.at<double>(2,0) = -1.3908;
    */
/**/
    R.at<double>(0,0) = 0.9997;
    R.at<double>(0,1) = 0.0010;
    R.at<double>(0,2) = -0.0222;
    R.at<double>(1,0) = -0.0008;
    R.at<double>(1,1) = 0.9999;
    R.at<double>(1,2) = 0.0132;
    R.at<double>(2,0) = 0.0223;
    R.at<double>(2,1) = -0.0131;
    R.at<double>(2,2) = 0.9996;

    T.at<double>(0,0) = -112.7714;
    T.at<double>(1,0) = -1.5433;
    T.at<double>(2,0) = -0.1372;


    double scale = 1.;
    /*
    capL.set(CV_CAP_PROP_FRAME_WIDTH, 640*scale);
    capL.set(CV_CAP_PROP_FRAME_HEIGHT, 480*scale);
    capR.set(CV_CAP_PROP_FRAME_WIDTH, 640*scale);
    capR.set(CV_CAP_PROP_FRAME_HEIGHT, 480*scale);
    */
    camL.setScale(scale);
    camR.setScale(scale);


    StereoRig rig(&camL, &camR, R, T, cv::Size(512*scale,384*scale));
//    StereoRig rig(&camL, &camR, R, T, cv::Size(640*scale,480*scale));

    Navigator n(&rig);
//    n.runDetector(std::string("../../Documents/datasets/new_college/left/"), true);
    n.runDetector(std::string("../../Documents/datasets/Oxford_Ladybug_Panorama/"), true);
    return 0;
    //n.run(std::string("./resources/stereo_images_robotics_lab"));
//    n.runOnDir(std::string("/home/vangelrobot/Downloads/dataset_malaga_office2.3.7_20070326/dataset_malaga_office2.3.7_20070326_Images"));
//    n.runOnDir(std::string("/home/vangelrobot/Documents/datasets/Oxford_Stereo"));
//    n.runLoopClosureAssistant(true,std::string("/home/vangelrobot/Downloads/Images/left_side"));
    n.runLoopClosureAssistant(std::string("../../Documents/datasets/Oxford_Ladybug_Panorama/"), false);

    //n.run();
    //d.detect(capL);
    //d.detect(capL, capR, rig);
    /*return 0;
    for (uint i=0; i<sampleIms.size(); ++i)
    {
        cv::Mat im = cv::imread(sampleIms[i], 0);
        d.detect(im);
        if (cv::waitKey(10000) >= 0)
            continue;
    }*/

    /*
     Tek kamera, robotik lab parametreler:
     HESSIAN = 2500
     ALPHA = -1.0
     BETA = -0.02
     GAMMA = 0
     */

    /*
     Tek kamera, robotik lab parametreler:
     HESSIAN = 10
     ALPHA = -1.0
     BETA = -0.02
     GAMMA = 0
     DELTA = -10.0
     */
    // stereo datasını falan filan koy

    cv::waitKey(50000);

    return 0;
}
