#ifndef OBSERVATION_H
#define OBSERVATION_H

#include <vector>
#include <set>
#include <map>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>
#include "vocabulary.h"
#include "stereo.h"

/**
 * @class Observation
 *
 * Observation vector, mostly used to keep features within a container.
 * Methods to extract observation provided within class too.
 */
class Observation
{
    friend class Navigator;
    friend class PatchExtractor;

public:
    // default constructor
    Observation() : isStereo(false) { init(); }

    // constructor - create from image matrix
    Observation(const cv::Mat& im) : isStereo(false) { init(); from(im); }

    // constructor - create from image matrix
    Observation(const cv::Mat& im, const cv::Mat& imR, const StereoRig& rig);

    // destructor
    ~Observation() { delete sparse; if (isStereo) delete points3D; points3D = NULL; }

    // create observation from image
    void from(const cv::Mat& im);

    // set depths of features using second image (other stereo pair) and the stereo rig data
    std::map<int, cv::Point3d> get3DPts(const cv::Mat& imL, const cv::Mat& imR, const StereoRig& rig);

    // draw features
    void drawFeaturesAndRect(cv::Mat& im, const Rect& r);

    // macros pointing to SparseRep of the class
    size_t width() const { return sparse->xValsVec.size(); }
    size_t height() const { return sparse->yValsVec.size(); }

    // visualize it
    void drawRect(cv::Mat& im,  const Rect& r);

    // get actual area of the feature
    double rectArea(int xl, int yl, int xh, int yh) const;

    // get the total marginality of a certain rectangle
    double totMarg(int xl, int yl, int xh, int yh) const;

    // get the total marginality of a certain rectangle
    double depthVar(int xl, int yl, int xh, int yh) const;

    // get the number of features falling into a rectangle
    size_t numFeats(int xl, int yl, int xh, int yh) const;

    // return actual coordinates of a rectangle
    Rect actualRect(const Rect& r) const;

    // return average of two rects, needed to throw best guess
    static Rect averageRect(const Rect* r1, const Rect* r2);

    // generic function, area from an integral image
    template <class T>
    static T areaFromII(const cv::Mat& II, const Rect* r);

    template <class T>
    static T areaFromII(const cv::Mat& II, int xl, int yl, int xh, int yh);

    // reset a region inside the integral image
    template <class T>
    static void resetRegion(cv::Mat& mat, const Rect& r);

    // keep the descriptors of this observation
    cv::Mat descriptors;

    // keypoints of the descriptors
    std::vector<cv::KeyPoint> keyPoints;

    // matches bw new observations and vocabulary
    std::vector<cv::DMatch> matches;

    // is observation derived from stereo data = do we have dept hinfo
    const bool isStereo;

    void recalcIntegrals(const Rect& r);

    /**
     * Represent features sparsely, this representation
     * will provide an efficient way to calculate marginality
     * scores etc. in a very efficient manner
     */
    class SparseRep
    {
        friend class Observation;
        friend class PatchExtractor;

    public:
        SparseRep() {}


        // the sets above are converted to vectors for fast access
        std::vector<size_t> xValsVec;
        std::vector<size_t> yValsVec;

    private:
        cv::Mat cumNumFeats;
        cv::Mat cumMargProbs;
        cv::Mat cumDepths;
        cv::Mat cumDepthVars;

        // temporary set, used to sort actual x,y values of features
        std::set<size_t> xVals;
        std::set<size_t> yVals;

        // temporary, words are kept in a multiset, compute frequency efficiently
        std::multiset<size_t> words;

        // convert std::set words to vector for fast access
        std::vector<double> wordFreqs;

        // add new x/y value for a new keypoint
        void insertX(const size_t x) { xVals.insert(x); }
        void insertY(const size_t y) { yVals.insert(y); }

        // After finding out the number of features and the unique
        // coordinates, we can initialize cv::Mat's.
        void initVars();

        // take features and fill matrices using them
        void fillWith(const std::vector<Feature>& features);
    };

    void printDepths(Rect r);

    cv::Point3d point3D(const size_t i) const { return (*points3D)[i]; }

    std::vector<Feature> features;
private:

    std::vector<cv::Point3d>* points3D;

    // keep sparse representation within
    SparseRep* sparse;

    // constructor commons
    void init() { sparse = new SparseRep(); if (isStereo) points3D = new std::vector<cv::Point3d>; }
};

/**
 * A class used to evaluate the marginality score of a certain window
 * I didn't want to call it Classifier because it isn't one.. :)
 *
 * Used to reveal marginality score of a rectangle, its cumulative
 * marginal probability or any other score we define.
 *
 * @todo so much... :)
 */
class Evaluator
{
public:
    // construct with constructor
    Evaluator() {}

    // coef.s used for the energy function, see their definition
    static const double ALPHA;
    static const double BETA;
    static const double GAMMA;
    static const double DELTA;

    // marginality score of a rect
    double rawMargScore(const Rect& r);

    // set current observation on which detection is performed
    void setObs(const Observation* _obsPtr) { obsPtr = _obsPtr; }
    void delObs() { obsPtr = NULL; }

    const Observation* obsPtr;

    // find upper bound of a certain marginality score within a region
    double upper(const RectInterval* ri) const;
};

/**
 * @todo DELETE or COMPLETELY MODERATE this function.
 */
inline void Observation::drawRect(cv::Mat& im, const Rect& r)
{
    cv::rectangle(im, r.cvStyle(), CV_RGB(0,255,0), 4);
}

/**
 * Inline function to return the area of a rectangular region
 *
 * @param int xl, yl, xh, yh
 * @return double
 * @todo inline this
 */
inline double Observation::rectArea(int xl, int yl, int xh, int yh) const
{
    /*
     * area: (x2-x1)*(y2-y1)
     * =====================
     * size_t x2 = sparse->xValsVec[xl];
     * size_t y2 = sparse->yValsVec[yl];
     * size_t x1 = sparse->xValsVec[xh];
     * size_t y1 = sparse->yValsVec[yh];
    */

    if (xh<xl || yh<yl)
        return 0.;

    return (sparse->xValsVec[xh]-sparse->xValsVec[xl])*
            (sparse->yValsVec[yh]-sparse->yValsVec[yl]);
}

/**
 * Inline function to return number of features within a rectangle
 *
 * @param int xl, yl, xh, yh
 * @return double
 * @todo inline this
 */
inline size_t Observation::numFeats(int xl, int yl, int xh, int yh) const
{
    return (size_t) Observation::areaFromII<double>(sparse->cumNumFeats, xl, yl, xh, yh);
}

/**
 * Inline function to return the cumulative marginality of a rectangular region
 *
 * @param int xl, yl, xh, yh
 * @return double
 * @todo inline this
 */
inline double Observation::totMarg(int xl, int yl, int xh, int yh) const
{
    return Observation::areaFromII<double>(sparse->cumMargProbs, xl, yl, xh, yh);
}

/**
 * Inline function to return the depth variance within a region.
 *
 * @param int xl, yl, xh, yh
 * @return double
 * @todo inline this
 */
inline double Observation::depthVar(int xl, int yl, int xh, int yh) const
{
    // variance = E[X^2]-(E[X])^2
    uint N = numFeats(xl, yl, xh, yh)+1; // add 1 to avoid divisions to zero
    double meanDepth = Observation::areaFromII<double>(sparse->cumDepths, xl, yl, xh, yh)/N;
    return Observation::areaFromII<double>(sparse->cumDepthVars, xl, yl, xh, yh)/N-(meanDepth*meanDepth);
}


/**
 * Return the average of two rectangles. Needed to compute at the best guess
 * which is the average of the inner and outer rectangle.
 *
 * @param const Rect *r1
 * @param const Rect *r2
 * @return Rect
 */
inline Rect Observation::averageRect(const Rect* r1, const Rect* r2)
{
    size_t x1 = round((double) (r1->x1+r2->x1)/2);
    size_t x2 = round((double) (r1->x2+r2->x2)/2);
    size_t y1 = round((double) (r1->y1+r2->y1)/2);
    size_t y2 = round((double) (r1->y2+r2->y2)/2);

    return Rect(x1, y1, x2-x1, y2-y1);
}

/**
 * A generic function used to evaluate the sum of a rectangular region
 * from an integral image.
 *
 * @param Cv::Mat II
 * @param int xl, yl, xh, yh
 * @return double
 */
template <class T>
        inline T Observation::areaFromII(const cv::Mat& II, int xl, int yl, int xh, int yh)
{
    size_t x1 = xl;
    size_t x2 = xh+1;  // add one to those coordinates, the integral
    size_t y1 = yl;    // image is padded from the left and top
    size_t y2 = yh+1;

    return (II.at<T>(y2,x2) - II.at<T>(y2,x1) - II.at<T>(y1,x2) + II.at<T>(y1,x1));
}

/**
 * A generic function used to evaluate the sum of a rectangular region
 * from an integral image.
 *
 * @param Cv::Mat II
 * @param Rect* r
 * @return double
 */
template <class T>
        inline T Observation::areaFromII(const cv::Mat& II, const Rect* r)
{
    size_t x1 = r->x1;
    size_t x2 = r->x2+1;  // add one to those coordinates, the integral
    size_t y1 = r->y1;    // image is padded from the left and top
    size_t y2 = r->y2+1;

    return (II.at<T>(y2,x2) - II.at<T>(y2,x1) - II.at<T>(y1,x2) + II.at<T>(y1,x1));
}


/**
 * Reset the integral matrix
 *
 * @param Mat - input image (matrix)
 * @param Rect - region to reset
 */
template <class T>
        inline void Observation::resetRegion(cv::Mat& mat, const Rect &r)
{
    T val = mat.at<T>(r.y1+1, r.x1+1);
    Rect* r2 = new Rect(r);
    T cumVal = Observation::areaFromII<double>(mat, &r);
    delete r2;
    for (int j=r.y1; j< mat.rows; ++j)
        for (int i=r.x1; i< mat.cols; ++i)
            mat.at<T>(j+1,i+1) -= mat.at<T>(r.y2+1, r.x2+1); // (i+1,j+1) because its integral image

    for (int j=r.y1; j<=r.y2; ++j)
        for (int i=r.x1; i<=r.x2; ++i)
            mat.at<T>(j+1,i+1) = val; // (i+1,j+1) because its integral image

    for (int j=r.y2; j<mat.rows; ++j)
        for (int i=r.x1; i<r.x2; ++i)
            mat.at<T>(j+1, i+1) -= mat.at<T>(r.y2, i+1);

    for (int j=r.y1; j<r.y2; ++j)
        for (int i=r.x1; i<mat.cols; ++i)
            mat.at<T>(j+1, i+1) -= mat.at<T>(j+1, r.x2);
}

/**
 * Find %upper bound% needed by Branch&Bound algorithm
 * My first trial for an upper bound on the marginality
 * relies on a logic like this:
 * Marginality is proportional to indiv. feat. marginality,
 * and inverse proportional to rectangular area and also
 * inv. proportional to number of feats within the rect:
 *
 * Marg ~ Indiv marginality of feats within rect
 * Marg 1/~ Area of the rectangle
 * Marg 1/~ Number of feates within rectangle
 *
 * @param RectInterval* ri
 * @return double
 */
inline double Evaluator::upper(const RectInterval* ri) const
{
    /*
    size_t numFeats = obsPtr->numFeats(ri->getHigh(RectInterval::LEFT),
                                       ri->getHigh(RectInterval::TOP),
                                       ri->getLow(RectInterval::RIGHT),
                                       ri->getLow(RectInterval::BOTTOM));
    */
    return ALPHA*obsPtr->totMarg(ri->getLow(RectInterval::LEFT), // marginality
                                 ri->getLow(RectInterval::TOP),
                                 ri->getHigh(RectInterval::RIGHT),
                                 ri->getHigh(RectInterval::BOTTOM))+
           BETA*obsPtr->rectArea(ri->getHigh(RectInterval::LEFT), // size constraint
                                 ri->getHigh(RectInterval::TOP),
                                 ri->getLow(RectInterval::RIGHT),
                                 ri->getLow(RectInterval::BOTTOM))+
           GAMMA*obsPtr->numFeats(ri->getHigh(RectInterval::LEFT), // limit on feat number
                                  ri->getHigh(RectInterval::TOP),
                                  ri->getLow(RectInterval::RIGHT),
                                  ri->getLow(RectInterval::BOTTOM));
           /*+
           (obsPtr->isStereo ? DELTA*obsPtr->depthVar(ri->getHigh(RectInterval::LEFT), // stereo variance constraint
                                                      ri->getHigh(RectInterval::TOP),
                                                      ri->getLow(RectInterval::RIGHT),
                                                      ri->getLow(RectInterval::BOTTOM))/(numFeats+1) : 0)*/;
}

#endif // OBSERVATION_H
