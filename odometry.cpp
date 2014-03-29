#include "odometry.h"
#include <numeric>
#include <ctime>

/**
 * Take two observations; (previous and current), each having their
 * own KeyPoints along with their descriptors and 3D points.
 *
 *
 */
void Odometry::update(Observation *prevObs, Observation *curObs, cv::Mat& frameL)
{
    // match Previous To Current, vica versa
    std::vector<cv::DMatch> matchesRawPTC, matchesRawCTP;

    //! 1) match the keypoints between frames
    cv::FlannBasedMatcher matcher;
    matcher.match(prevObs->descriptors, curObs->descriptors, matchesRawPTC);
    matcher.clear();
    matcher.match(curObs->descriptors, prevObs->descriptors, matchesRawCTP);

    std::map<int,int> matchesPTC, matchesCTP;
    std::vector<unsigned char> matchesMask(matchesRawPTC.size(),0);

    for (size_t i=0; i<matchesRawPTC.size(); ++i)
        matchesPTC[matchesRawPTC[i].queryIdx] = matchesRawPTC[i].trainIdx;

    for (size_t i=0; i<matchesRawCTP.size(); ++i)
        matchesCTP[matchesRawCTP[i].queryIdx] = matchesRawCTP[i].trainIdx;

    size_t cnt=0;

    // match features in 2 directions
    typedef std::map<int,int>::const_iterator It_ii;
    for (It_ii it=matchesPTC.begin(); it != matchesPTC.end(); ++it, cnt++)
    {
        It_ii searchResult = matchesCTP.find(it->second);
        if (matchesCTP.end() == searchResult) // continue if feat does not exist in 2nd im
            continue;

        if (searchResult->first != it->second || searchResult->second != it->first)
            continue;

        // else
        matchesMask[cnt] = 1;
    }

    std::vector<int> queryIdxs, trainIdxs;
    queryIdxs.reserve(matchesRawPTC.size());
    trainIdxs.reserve(matchesRawPTC.size());

    for (uint i=0; i<matchesRawPTC.size(); i++) {
        queryIdxs.push_back(matchesRawPTC[i].queryIdx);
        trainIdxs.push_back(matchesRawPTC[i].trainIdx);
    }

    std::vector<cv::Point2f> ptsPrev, ptsCur;
    cv::KeyPoint::convert(prevObs->keyPoints, ptsPrev, queryIdxs);
    cv::KeyPoint::convert(curObs->keyPoints, ptsCur, trainIdxs);
    cv::Mat matPtsPrev(ptsPrev), matPtsCur(ptsCur);

    cv::Mat H = cv::findHomography(matPtsPrev, matPtsCur, matchesMask, CV_RANSAC, 5);

    for (int jj=0; jj<H.rows; ++jj)
    {
        for (int i=0; i<H.cols; ++i)
            std::cout << std::setprecision(1) << H.at<double>(jj,i) << " ";
        std::cout << std::endl;
    }

    // keyPoints which have been matched in stereo will be kept, mark their IDs
    std::vector<uint> keepIdx;
    keepIdx.reserve(prevObs->keyPoints.size());

    std::vector<double> weights;
    weights.reserve(prevObs->keyPoints.size());

    cv::Point3d motion(0.,0.,0.);
    size_t numP=0, numN=0;
    std::vector<cv::Point3d> motionPts;
    motionPts.reserve(prevObs->keyPoints.size());

    //! 1) gather allocation data for sparse representation
    for (uint i=0; i<prevObs->keyPoints.size(); ++i)
    {
        // eger noktanın derinligi yok ise devam et
        if (matchesPTC.end() == matchesPTC.find(i) || 0 == matchesMask[i])
            continue;

        // this kp has 3D information, keep it
        keepIdx.push_back(i);

        using std::pow;
        cv::Point2f ptBefore(ptsPrev[i]), ptAfter(ptsCur[matchesRawPTC[i].queryIdx]);
        double avgPxMov = std::sqrt(pow(ptBefore.x-ptAfter.x,2)+
                                    pow(ptBefore.y-ptAfter.y,2));

        cv::Point3d curPt = curObs->point3D(matchesPTC[i]);
        cv::Point3d prevPt = prevObs->point3D(i);

        // far points are not reliable!
        if (fabs(curPt.z) > 8000. || fabs(prevPt.z) > 8000.) continue;

        double w = pow(avgPxMov,2);
        weights.push_back(w);

        motionPts.push_back(curPt-prevPt);

        motion = w*(curPt-prevPt);

        //cv::circle(frameL, ptsPrev[i],3,CV_RGB(0,255,0),1);
        cv::circle(frameL, ptsCur[matchesRawPTC[i].queryIdx],3,CV_RGB(255,0,0),1);
        cv::line(frameL, ptBefore, ptAfter,CV_RGB(0,255,0),1);

        cv::Mat targetPt;
        perspectiveTransform(cv::Mat(std::vector<cv::Point2f>(1, ptBefore)), targetPt, H);

        cv::imshow("leftIm", frameL);

        if (0<motion.z) ++numP;
        else ++numN;
    }

    motion.x = 0.; motion.y = 0.; motion.z = 0.;

    std::stringstream ss;
    ss << "journey" << ".txt";
    std::ofstream outputFile(ss.str().c_str(), std::fstream::app);

    for (uint i=0; i<motionPts.size(); ++i)
    {
        if (numP > numN && motionPts[i].z < 0)
        {
            weights[i] = 0.;
            continue;
        }
        motion = motion + weights[i]*motionPts[i];
        outputFile << motionPts[i].z << '\t';
    }
    outputFile << '\n';

    outputFile.close();



    double totWeight = accumulate(weights.begin(), weights.end(), 0.);
    motion = (1./totWeight)*motion;
    std::cout << "Avg motion: " << std::setprecision(7) << motion.x << " " << motion.y << " " << motion.z << std::endl;
    std::cout << "num neg: " << numN << " | num pos: " << numP << std::endl;
}
