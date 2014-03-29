#ifndef LANDMARKDETECTOR_H
#define LANDMARKDETECTOR_H

#include "definitions.h"
#include <fstream>
#include <iostream>

/**
 * @class PtFeature
 *
 * A single point feature consisting of two relative pixel coords
 * Used as a simple feature of a fern
 */
class PtFeature
{
public:
    // constructor
    PtFeature(const double _r, const double _l, const double _t, const double _b)
        : r(_r), l(_l), t(_t), b(_b), x1(0), x2(0), y1(0), y2(0) {}

    // initialize from another feature, to obtain scaled feature
    PtFeature(const PtFeature& ft, const double s, const uint w, const uint h)
        : r(ft.r), l(ft.l), t(ft.t), b(ft.b),
          x1(safeCoord(l*w*s)), x2(safeCoord(r*w*s)), y1(safeCoord(t*h*s)), y2(safeCoord(b*h*s)) {}

    //! @property relative coordinates of feature
    double r, l, t, b;

    //! @property absolute coords w.r.t window and scale
    uint x1, x2, y1, y2;

    // measure ft response on an image patch, CV_64F!!
    bool response(const cv::Mat& im, const uint offX, const uint offY) const { return im.at<double>(offY+y1, offX+x1) > im.at<double>(offY+y2, offX+x2); }

    // return safe coordinate value where image borders cannot be violated
    uint safeCoord(double value) const { int result=round(value)-1; return (result<0) ? 0 : result; }
};


/**
 * @class Fern
 *
 * A small tree consisting of several features, point features
 * in this case
 */
class Fern
{
public:
    // default constructor
    Fern() {}

    // add a single feature to Fern
    void addFeature(const double r, const double l, const double t, const double b)
            { feats.push_back(new PtFeature(r, l, t, b) ); }

    // add a single feature to Fern
    void addFeature(const PtFeature& ft, const double scale, const uint w, const uint h)
            { feats.push_back(new PtFeature(ft, scale, w, h) ); }

    // obtain scaled ferns               scale         width         height
    Fern* getAbsoluteScaled(const double s, const uint w, const uint h);

    // measure fern response to given patch
    uint response(const cv::Mat &im, const uint offX, const uint offY) const;

    // destructor - delete features
    ~Fern() { for (int i=feats.size()-1; i>=0; --i ) delete feats[i]; }

private:
    //! @property features of this fern
    std::vector<PtFeature*> feats;
};

/**
 * @class Classifier
 *
 * Fern classifier, store features
 */
class Classifier
{
    friend class Landmark;
public:
    // default constructor
    Classifier(const std::string& fileName);

    // clear classifier
    ~Classifier();

    // add a single fern to the classifier
    void addFern(Fern* f) { baseFerns.push_back(f); }

    bool good() const { return numT>0; }

private:
    //! @property base, unscaled ferns of the classifier
    std::vector<Fern*> baseFerns;

    //! @property number of features per fern
    uint numF;

    //! @property number of ferns in classifier
    uint numT;
};

/**
 * @class Landmark
 *
 * Detect a single landmark: Train it, update it and detect it
 * One object for each landmark
 */
class Landmark
{
public:
    //! @property initial and final scale to search object
//    static const double SCALE_START = 0.5;
    static const double SCALE_START = 1.0;
    static const double SCALE_STEP = 1.20;
    static const double SCALE_END = 1.70;
    static const double SEARCH_PAD = 2.;

    //! @property steps of scanning window method
    static const uint STEP_X = 3;
    static const uint STEP_Y = 3;
    static const uint NUM_MAX_UPDATES = 3;

    //! @property image width, fixed
    uint IM_W;

    //! @property image height, fixed
    uint IM_H;

    //! @property boundaries and size of the target object/window/landmark
    Rect objRect;

    //! @property ferns of all scales
    std::vector<std::vector<Fern*> > fernSets;

    //! @property associate the scale of each fern set (vector ferns) with a vector element
    std::vector<double> scaleMap;

    //! @property at which scales is this landmark available
    std::vector<bool> scaleFlag;

    //! @property minimum variance of the target window
    double minVar;

    // return binary pattern of given fern
    std::vector<uint> pattern(const cv::Mat& im, const uint offX, const uint offY, const uint scaleKey) const;

    // train the detector using an image
    void update(const Image& im, const Rect& cObjRect);

    // Constructor - the usual constructor
    Landmark(const Classifier* _cls, const Image& input, const Rect& _objRect, const bool crtParent, const bool crtChildren);

    // Copy constructor
    Landmark(const Landmark& landmark);

    // Destructor
    ~Landmark();

    // detect this landmark in image
    Rect detect(const Image& im, cv::Mat& drawHere);

    double getThr() const { return thrP; }

    // update model through a single patch
    void updateModel(const std::vector<uint>& patt, const bool isP);

    // measure ferns response to given PATTERN directly
    double response(const std::vector<uint>& patt) const;

    // is this landmark a parent or a child
    bool isParent() const { return NULL==0; }

private:
    // create ferns in all scales and convert coords to absolute from relative
    void createFernSets(const double scaleStart, const double scaleEnd, const double scaleStep);

    // measure ferns response to given image at given offset
    double response(const cv::Mat& im, const uint offX, const uint offY, const uint scaleKey) const;

    //! @property classifier info, we derive our specific features from the base features in here
    const Classifier* cls;

    //! @property number of positives falling to each pt feature
    std::vector< std::vector<uint> > numP;

    //! @property number of negatives falling to each pt feature
    std::vector< std::vector<uint> > numN;

    //! @property probability response for each pt feature
    std::vector< std::vector<double> > P;

    //! @property positive threshold
    double thrP;

    //! @property how many times has this model been updated?
    uint numUpdates;

    //! @property region to search this object
    Rect searchRegion;

    //! @property children of this landmark, used to detect the feature more precisely
    std::vector<Landmark*> children;

public:
    //! @property parent -> the landmark that must be detected before this
    Landmark* parent;
};

/**
 * Measure fern response for the given patch
 *
 * @param Mat im - input image
 * @param int offX - x offset to patch
 * @param int offY - y offset to patch
 * @return int - integer of max length 2^numF
 */
inline uint Fern::response(const cv::Mat &im, const uint offX, const uint offY) const
{
    uint val = 0;
    for (uint i=0; i<feats.size(); ++i)
        val |= feats[i]->response(im,offX,offY) << i;

    return val;
}

/**
 * Measure probability at given offset and scale NOT USED RIGTH NOW!
 *
 * @param Mat im - input image
 * @param uint offX - x offset to the patch
 * @param uint offY - y offset to the patch
 * @param uint scaleKey - scale key of the patch
 * @return double - fern response
 */
inline double Landmark::response(const cv::Mat &im, const uint offX, const uint offY, const uint scaleKey) const
{
    double val=0;

    // add the response of all trees
    for (uint i=0; i<fernSets[scaleKey].size(); ++i)
        val += P[i][fernSets[scaleKey][i]->response(im, offX, offY)];

    return val;
}

/**
 * Measure probability of given pattern
 *
 * @param vector<uint> - the given pattern
 * @return double - probability of this pattern
 */
inline double Landmark::response(const std::vector<uint>& patt) const
{
    double val=0;

    // add the response of all trees
    for (uint i=0; i<patt.size(); ++i)
        val += P[i][patt[i]];

    return val;
}

/**
 * Measure offset at given offset and scale
 *
 * @param Mat im - input image
 * @param uint offX - x offset to the patch
 * @param uint offY - y offset to the patch
 * @param uint scaleKey - scale key of the patch
 * @return vector<uint> - fern response
 */
inline std::vector<uint> Landmark::pattern(const cv::Mat &im, const uint offX, const uint offY, const uint scaleKey) const
{
    std::vector<uint> patt;
    patt.reserve(cls->numT);

    // add the response of all trees
    for (uint i=0; i<fernSets[scaleKey].size(); ++i)
        patt.push_back(fernSets[scaleKey][i]->response(im, offX, offY));

    return patt;
}


/**
 * Update object/window model, review the probability of each tree
 *
 * @param vector<uint> pattern
 * @param bool isP - is this a positive or a negative
 */
inline void Landmark::updateModel(const std::vector<uint>& patt, const bool isP)
{
    for (uint i=0; i<cls->numT; ++i)
    {
        (isP) ? numP[i][patt[i]] += 1 : numN[i][patt[i]] += 1;
        P[i][patt[i]] = (double) numP[i][patt[i]]/(numP[i][patt[i]]+numN[i][patt[i]]+config::EPS);
    }
}

#endif // LANDMARKDETECTOR_H
