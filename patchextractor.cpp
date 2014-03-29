#include "patchextractor.h"
#include <iomanip>

const size_t PatchExtractor::MAX_ITERATIONS = 500000;
//const size_t PatchExtractor::MAX_ITERATIONS = 2000000;

/**
 * Operate on a single image.
 * 1) Extract observation from image, set it as the obs. of the evaluator
 * 2) Initialize detection
 * 3) Iterate until convergence
 *
 * @param Mat im - left image for visualization purposes
 * @param StereoRig rig - only needed for detections with stereo data
 * @param Mat imR - only needed for detections with stereo data
 * @return Rect
 */
Rect PatchExtractor::_extract(cv::Mat& im, Observation* obs)
{
    // set current observation
    eval.setObs(obs);

    RectInterval* fullSpace = new RectInterval(eval.obsPtr->width(), eval.obsPtr->height());
    sStack.push(fullSpace); // begin with full space

    size_t numIters = 0;

#ifdef INSPECTION_MODE
    //timing
    int64 t1 = cv::getTickCount();
#endif

    //! 3) iterate until convergence
    do {
        ++numIters;
    } while (NOT_CONVERGED == iterate() && numIters<PatchExtractor::MAX_ITERATIONS);


//    std::cout << " - " <<numIters << " iterations // ";

    Rect bestRect = sStack.top()->getAvgRect();

#ifdef INSPECTION_MODE
    // bakalim mantikli mesafeler var mi
    //obs->printDepths(bestRect);
#endif

    // free memory
    while (!sStack.empty())
    {
        delete sStack.top();
        sStack.pop();
    }


#ifdef INSPECTION_MODE
    //timing
//    std::cout << std::setprecision(4);
//    std::cout << std::fixed;
//    std::cout << "ESS search took " << (double)(cv::getTickCount()-t1)/cv::getTickFrequency() << " seconds." << std::endl;
#endif

    Rect result = eval.obsPtr->actualRect(bestRect);
    obs->recalcIntegrals(result); // remove patch from integral images and observation

    obs->drawFeaturesAndRect(im, result);

    return result;
}

/**
 * Wrapper method to use extract on video capture data
 *
 * @param VideoCapture
 * @return Rect
 */
void PatchExtractor::extract(cv::VideoCapture &cap)
{
    if (!cap.isOpened())
        return;

    cv::Mat frame;

    while (1)
    {
        cv::Mat tmp;
        cap >> tmp;
        cv::cvtColor(tmp,frame,CV_RGB2GRAY);
        Observation* obs = new Observation(frame);
        Rect r = _extract(frame, obs);

        delete obs;
        obs = NULL;

        if (cv::waitKey(10)>=0)
            break;
    }

    return;
}

/**
 * Wrapper method to use extract on stereo video capture data.
 * Images will be undistorted and rectified here before they are sent to
 * detection routine.
 *
 * @param VideoCapture
 * @return Rect
 */
void PatchExtractor::extract(cv::VideoCapture &capL, cv::VideoCapture &capR, const StereoRig& rig, double scale)
{
    if (!capL.isOpened() || !capR.isOpened())
        return;

    cv::Mat frameL, frameR;

    while (1)
    {
        cv::Mat tmp;
        capL >> tmp;
        cv::cvtColor(tmp,frameL,CV_RGB2GRAY);
        capR >> tmp;
        cv::cvtColor(tmp,frameR,CV_RGB2GRAY);

        // undistort and rectify to calculate depth later
        frameL = StereoRig::getUndistortedRectified(rig.camL->D, rig.camL->SM, rig.R1, rig.P1, frameL);
        frameR = StereoRig::getUndistortedRectified(rig.camR->D, rig.camR->SM, rig.R2, rig.P2, frameR);

        // black regions emerge after undistortion, crop image to avoid black regions
        double pad = 50*scale;
        frameL = frameL(cv::Rect(pad, pad/2., frameL.cols-pad*2, frameL.rows-pad*2));
        frameR = frameR(cv::Rect(pad, pad/2., frameR.cols-pad*2, frameR.rows-pad*2));

        Observation *obs;

        // if we are working with stereo data, hand the stereo rig data (R, T matrices)
        // into constructor, so the depths of feature points may be calculated
        if (rig.isActive())
            obs = new Observation(frameL, frameR, rig);
        else
            obs = new Observation(frameL);

        _extract(frameL, obs);
        delete obs;
        obs = NULL;
        if (cv::waitKey(10)>=0)
            break;
    }

    return;
}

/**
 * @brief a single iteration of ESS algorithm.
 *
 * To see ESS algorithm check Efficient Subwindow Search paper
 * of Lampert et. al in PAMI of Dec. 2009. The following steps
 * are accomplished in a single iteration:
 *
 * 1) Find most promising candidate region
 * 2) Check if converged
 * 3) Split most promising region if necessary
 * 4) Reinsert regions after calculating their upper bounds
 *
 * Code is mostly adopted from original ESS search code
 *
 * @return int /from PatchExtractor::State
 */
PatchExtractor::State PatchExtractor::iterate()
{
    //! step 1
    const RectInterval* curInterval = sStack.top();

    //! step 2
    const int splitIndex = curInterval->maxIndex();

    if (splitIndex < 0)
        return CONVERGED;

    sStack.pop(); // remove most promising, will be reinserted in two pieces

    RectInterval* newInterval0 = new RectInterval(curInterval);
    RectInterval* newInterval1 = new RectInterval(curInterval);

    const int& si = splitIndex;

    newInterval0->setHigh(si, (curInterval->getLow(si) + curInterval->getHigh(si))>>1);
    newInterval1->setLow(si, (curInterval->getLow(si) + curInterval->getHigh(si)+1)>>1);

//    std::cout << "{" << curInterval->maxIndex() << "} ";

    delete curInterval; curInterval = NULL;

    if (newInterval0->isLegal())
    {
        newInterval0->setUpper(eval.upper(newInterval0));
        sStack.push(newInterval0);
    }

    if (newInterval1->isLegal())
    {
        newInterval1->setUpper(eval.upper(newInterval1));
        sStack.push(newInterval1);
    }

    return NOT_CONVERGED;
}

