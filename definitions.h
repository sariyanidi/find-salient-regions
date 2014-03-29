#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <string>
#include <limits>
#include <cstdlib>
#include <opencv2/imgproc/imgproc.hpp>
#include <limits>
#include <queue>
#include <vector>

#define INSPECTION_MODE 1

/**
 * Paths to resource files
 */
namespace resources
{
    //const std::string vocabPath("../findmarginalregions/resources/FordVocabulary.oxv");
    //const std::string margsPath("../findmarginalregions/resources/FordMarginals.oxv");
    const std::string vocabPath("./resources/OxfordVocab_Surf_11k.oxv");
    const std::string margsPath("./resources/OxfordVocab_Marginals_11k.oxv");
}

/**
 * Configuration parameters
 */
namespace config
{
    //const size_t HESSIAN_THRESHOLD = 3500; //gostermelik-gecici
    //const size_t HESSIAN_THRESHOLD = 2500; // tek kamera için; fazla nitelik olunca alan çok büyük oluyor
    //static const size_t HESSIAN_THRESHOLD = 0;  // çift kamera için; çok nitelik lazım çoğu stereo eşleştirmede eleniyor zaten
//    const size_t HESSIAN_THRESHOLD = 1000;
    const size_t HESSIAN_THRESHOLD = 200;
//    const size_t HESSIAN_THRESHOLD = 500;
    const size_t FAST_THRESHOLD = 30;
//    const size_t HESSIAN_THRESHOLD = 750; // en son bu vardi
    static const size_t DESC_SIZE = 128;
    static const double EPS = 0.000001;//std::numeric_limits<double>::min();
    static const bool DEBUG = true;
    static const double PI = 3.14159265;
    static const uint MIN_OBJ_EDGE = 36;
}

/**
 * Namespace for common functions belonging to
 *
 */
namespace etc
{
}

/**
 * This Rect class is different from cv::Rect, x2 and y2 are
 * stored beside x1, y1, width and height
 */
class Rect
{
public:
    // constructor 1 - nice constructor
    Rect(const int _x1, const int _y1, const int _w, const int _h, const int _scaleKey = 0)
        : x1(_x1), y1(_y1), w(_w), h(_h), x2(x1+w), y2(y1+h), width(_w), height(_h), scaleKey(_scaleKey) {}

    // constructor 2 - initialize an invalid rectangle
    Rect(const int f) { if (f == -1) { x1=0; x2=0; width=0; height=0; w=0; h=0; scaleKey=0; }}

    // constructor 3 - copy from pointer
    Rect (const Rect* r) : x1(r->x1), y1(r->y1), w(r->w), h(r->h), x2(r->x2), y2(r->y2), scaleKey(r->scaleKey) {}

    // extend given rectangle by padding it with a given amount
    Rect extended(const cv::Size& sz, uint padAmountX=8, int padAmountY=-1) const;

    // return cv style rect for drawing etc.
    cv::Rect cvStyle() const { return cv::Rect(x1, y1, w, h); }

    // check if rectangle is well-defined
    bool legal() const { return w>0; }

    // intersection area of two rectangles
    static uint intersect(const Rect &r1, const Rect &r2);

    // intersection area of a rectangle and a given region
    static uint intersect(const Rect &r, const int x1, const int y1, const int x2, const int y2);

    // rectangle boundaries
    int x1, y1, w, h, x2, y2;
    uint width, height;

    //! @property used rarely but very functional, scale key of the rectangle
    int scaleKey;

    // combine rectangles
    static void combineDetections(const std::vector<Rect> &detections, std::vector<Rect> &resultVector, uint minDetects, bool isMin);

    //
    static Rect getStrongestCombination(const std::vector<Rect> &detections, const std::vector<double>& scaleMap, const uint minDetects);
};

/**
 * @class Image
 * Multiscale representation of a given image, includes the integral images
 */
class Image
{
public:
    // constructor
    Image(const cv::Mat& im, const std::vector<double>& scaleMap);

    // destructor
    ~Image();

    // variance of a given image region
    static double var(const cv::Mat& ii, const cv::Mat& ii2, const uint x1, const uint y1, const uint x2, const uint y2);

    //! @property resized images (at all scales)
    std::vector<cv::Mat*> ims;

    //! @property integral images at given scales
    std::vector<cv::Mat*> iis;

    //! @property squares of integral images at given scales
    std::vector<cv::Mat*> ii2s;

    // get image at ith scale
    cv::Mat* im(const uint& i) const { return ims[i]; }

    // get integral image at ith scale
    cv::Mat* ii(const uint& i) const { return iis[i]; }

    // get integral of square image at ith scale
    cv::Mat* ii2(const uint& i) const { return ii2s[i]; }
};

/**
 * An interval of rectangles is stored and manipulated in this class.
 *
 * The format of this rectangle information follows Lampert et al.'s
 * code for the study: Efficient Subwindow Search (ESS), PAMI 2009.
 *
 * @todo eliminate low[] and high[], use $outer and $inner instead
 */
class RectInterval
{
public:
    // initialize as full space
    RectInterval(const size_t _width, const size_t _height);

    // initialize using a current RectInterval
    RectInterval(const RectInterval* ri);

//    ~RectInterval() {}

    // return index of longest edge
    int maxIndex() const;

    // "less" is needed to compare upper bounds in priority_queue
    bool less(const RectInterval* other) const { return upper < other->upper; }

    // set boundaries
    void setHigh(const size_t i, const size_t val) { high[i] = val; }
    void setLow(const size_t i, const size_t val) { low[i] = val; }
    void setUpper(const double u) { upper = u; }

    // get boundaries
    size_t getHigh(const size_t i) const { return high[i]; }
    size_t getLow(const size_t i) const { return low[i]; }
    double getUpper() const { return upper; }

    // check if region is defined properly
    bool isLegal() const { return ((getLow(LEFT) <= getHigh(RIGHT)) && (getLow(TOP) <= getHigh(BOTTOM))); }

    // get the real rectangle, not the relative one
    Rect actualRectangle(const Rect* r) const;

    Rect getAvgRect() const;
    Rect getLargeRect() const;
    Rect getSmallRect() const;

    // goes clockwise beginninf from left
    enum Side { LEFT=0, TOP=1, RIGHT=2, BOTTOM=3 };

private:
    // this variable nomination follows ESS source code
    short low[4];
    short high[4];

    // upper marginality score bound of the rectangles in this set
    double upper;
};

#endif // DEFINITIONS_H
