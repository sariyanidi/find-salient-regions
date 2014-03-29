#ifndef DETECTOR_H
#define DETECTOR_H

#include <string>
#include "observation.h"
#include "stereo.h"

/**
 * A class needed to keep RectangleInterval with the upper
 * in the top of a priority_queue, defined below this class
 */
class RectInterval_Comparison
{
public:
    bool operator() (const RectInterval* lhs, const RectInterval* rhs) const
        { return lhs->less(rhs); }
};

//! SearchStack: A stack to keep all RectIntervals
typedef std::priority_queue<const RectInterval*,
                            std::vector<const RectInterval*>,
                            RectInterval_Comparison>
                            SearchStack;

/**
 * The main ESS detector
 *
 * Use a SearchStack to store rectangles to search, extract
 * your ultimate target by following the most promising regions
 */
class PatchExtractor
{
public:
    friend class Navigator;

    // the state of the search
    enum State { CONVERGED = -1, NOT_CONVERGED = 0 };

    PatchExtractor() {}

    static const size_t MAX_ITERATIONS;

    // extract from image, given image matrix
    Rect extract(cv::Mat& im) { Observation* obs = new Observation(im); return _extract(im, obs); delete obs; obs = NULL;}

    // extract from video, given capture, dont output Rect just draw results
    void extract(cv::VideoCapture& cap);

    // extract from stereo video given a two cams and a stereo rig
    void extract(cv::VideoCapture& capL, cv::VideoCapture& capR, const StereoRig& rig, double scale = 1.0);

    // a single iteration of the ESS algorithm
    State iterate();

private:
    // implementation of extract method
    Rect _extract(cv::Mat& im, Observation* obs);

    // container to keep "RectInterval"s
    SearchStack sStack;

    // class to evaluate marginality score etc.
    Evaluator eval;

};

#endif // DETECTOR_H
