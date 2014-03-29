#include "definitions.h"
#include <iostream>

uint Rect::intersect(const Rect &r1, const Rect &r2)
{
    using std::min;
    using std::max;

    if ((r1.x1 <= r2.x2) &&
        (r1.x2 >= r2.x1) &&
        (r1.y1 <= r2.y2) &&
        (r1.y2 >= r2.y1)   )
        return (min<uint>(r1.x2,r2.x2)-max<uint>(r1.x1,r2.x1))*(min<uint>(r1.y2,r2.y2)-max<uint>(r1.y1,r2.y1));
    else
        return 0;
}

uint Rect::intersect(const Rect &r, const int x1, const int y1, const int x2, const int y2)
{
    using std::min;
    using std::max;

    if ((r.x1 <= x2) &&
        (r.x2 >= x1) &&
        (r.y1 <= y2) &&
        (r.y2 >= y1)   )
        return (min<uint>(r.x2,x2)-max<uint>(r.x1,x1))*(min<uint>(r.y2,y2)-max<uint>(r.y1,y1));
    else
        return 0;
}

Rect Rect::extended(const cv::Size& sz, uint padAmountX, int padAmountY) const
{
    if (-1 == padAmountY)
        padAmountY = padAmountX;

    Rect d(*this);
    int maxL = x1;
    int maxR = sz.width-x2-1;
    int maxU = y1;
    int maxB = sz.height-y2-1;

    using std::min;

    // calculate the minimum rate that this box can extand
    int paddingX = min(maxL, maxR);
    if (paddingX > (int)padAmountX) paddingX = padAmountX;

    // calculate the minimum rate that this box can extand
    int paddingY = min(maxU, maxB);
    if (paddingY > (int)padAmountY) paddingY = padAmountY;

    d.x1 -= paddingX;
    d.y1 -= paddingY;
    d.x2 += paddingX;
    d.y2 += paddingY;
    d.width = d.x2-d.x1;
    d.height= d.y2-d.y1;
    d.w = d.width;
    d.h = d.height;

    return d;

}


// Combines multiple detections
void Rect::combineDetections(const std::vector<Rect> &detections, std::vector<Rect> &resultVector, uint minDetects, bool isMin)
{
    std::vector< std::vector<uint> > groups;	// groups[i][j] = index (in the detection vector) of the jth rectangle belonging to the ith disjoint set
    std::vector<uint> intGroups;			// Will store the indexes of groups that have overlapping rectangles with the rectangle in question

    // If there are no detections, return an empty vector
    if (detections.size() == 0) return;

    // Create the first group and add it the first rectangle
    std::vector<uint> v;		// Dummy vector for storing the vectors before adding to the group
    v.push_back(0);
    groups.push_back(v);

    for (uint i=1; i<detections.size(); i++) // For every other rectangle except the first one
    {
        intGroups.clear();
        for (uint j=0; j<groups.size(); j++)	// Look at each group
        {
            for (uint k=0; k<groups[j].size(); k++)	// Look at all rectangles in the current group
            {
                // If there is a rectangle in the current group that overlaps with the current rectangle
                // add that group to the overlapping groups vector
                uint area = Rect::intersect(detections[i],detections[groups[j][k]]);
                if (area > 0)
                {
                    // Check if the amount of overlap is greater than a fixed amount
                    // If it isn't, then don't count this as an intersection
                    double ratio1 = (double)area/(detections[i].width*detections[i].height);
                    double ratio2 = (double)area/(detections[groups[j][k]].width*detections[groups[j][k]].height);

                    using std::min; using std::max;

                    if (isMin)
                    {
                        if (min<double>(ratio1,ratio2) > 0.70f)
                        {
                            intGroups.push_back(j);
                            break;
                        }
                    } else
                    {
                        if (max<double>(ratio1,ratio2) > 0.75f)
                        {
                            intGroups.push_back(j);
                            break;
                        }
                    }
                }
            }
        }
        // If one or more groups overlap with the current rectangle, combine the groups
        // and add the current rectangle to the resulting group
        uint cnt = intGroups.size();
        if (cnt > 0)
        {
            groups[intGroups[0]].push_back(i); // Add the current rectangle to the first overlapping group
            for (uint j=cnt-1; j>0; j--)
            {
                // Copy the rectangles in the jth group to the first group
                for (uint k=0; k<groups[intGroups[j]].size(); k++)
                    groups[intGroups[0]].push_back(groups[intGroups[j]][k]);
                // Delete the jth group
                groups.erase(groups.begin( ) + intGroups[j]);
            }
        }
        else // If the current rectangle doesn't belong to an existing group, create a new group and add the rectangle to it
        {
            v.clear();
            v.push_back(i);
            groups.push_back(v);
        }
    }

    // Combine the detections to form a single detection for each group
    for (uint i=0; i<groups.size(); i++)
    {
        uint x1 = 0, y1 = 0, x2 = 0, y2 = 0, j;

        uint gSize = groups[i].size();

        if (gSize < minDetects) continue;

        for (j=0; j<gSize; j++)
        {
            x1 += detections[groups[i][j]].x1;
            y1 += detections[groups[i][j]].y1;
            x2 += detections[groups[i][j]].x2;
            y2 += detections[groups[i][j]].y2;
        }
        x1 /= j;
        y1 /= j;
        x2 /= j;
        y2 /= j;

        Rect r(x1,y1,x2-x1+1,y2-y1+1);
        //#pragma omp critical
        resultVector.push_back(r);
    }
}


/**
 * Take multiple detections and return only the strongest one
 *
 * @param vector<Rect> detections - Unc detectiosn
 * @return Rect - the combination of the detection group with the most detections
 */
Rect Rect::getStrongestCombination(const std::vector<Rect> &detections, const std::vector<double>& scaleMap, const uint minDetects)
{
    std::vector<Rect> resultVector;
    std::vector< std::vector<uint> > groups;	// groups[i][j] = index (in the detection vector) of the jth rectangle belonging to the ith disjoint set
    std::vector<uint> intGroups;			// Will store the indexes of groups that have overlapping rectangles with the rectangle in question

    // return an invalid rectangle if no detection exists
    if (detections.size() == 0)
        return Rect(-1);

    // Create the first group and add it the first rectangle
    std::vector<uint> v;		// Dummy vector for storing the vectors before adding to the group
    v.push_back(0);
    groups.push_back(v);

    for (uint i=1; i<detections.size(); i++) // For every other rectangle except the first one
    {
        intGroups.clear();
        for (uint j=0; j<groups.size(); j++)	// Look at each group
        {
            for (uint k=0; k<groups[j].size(); k++)	// Look at all rectangles in the current group
            {
                // If there is a rectangle in the current group that overlaps with the current rectangle
                // add that group to the overlapping groups vector
                uint area = Rect::intersect(detections[i],detections[groups[j][k]]);
                if (area > 0)
                {
                    // Check if the amount of overlap is greater than a fixed amount
                    // If it isn't, then don't count this as an intersection
                    double ratio1 = (double)area/(detections[i].width*detections[i].height);
                    double ratio2 = (double)area/(detections[groups[j][k]].width*detections[groups[j][k]].height);
                    if (std::max<double>(ratio1,ratio2) > 0.70)
                    {
                        intGroups.push_back(j);
                        break;
                    }
                }
            }
        }
        // If one or more groups overlap with the current rectangle, combine the groups
        // and add the current rectangle to the resulting group
        uint cnt = intGroups.size();
        if (cnt > 0)
        {
            groups[intGroups[0]].push_back(i); // Add the current rectangle to the first overlapping group
            for (uint j=cnt-1; j>0; j--)
            {
                // Copy the rectangles in the jth group to the first group
                for (uint k=0; k<groups[intGroups[j]].size(); k++)
                    groups[intGroups[0]].push_back(groups[intGroups[j]][k]);
                // Delete the jth group
                groups.erase(groups.begin( ) + intGroups[j]);
            }
        }
        else // If the current rectangle doesn't belong to an existing group, create a new group and add the rectangle to it
        {
            v.clear();
            v.push_back(i);
            groups.push_back(v);
        }
    }

    int largestGroupIdx = -1;
    uint largestGroupSize = 0;

    // Combine the detections to form a single detection for each group
    for (uint i=0; i<groups.size(); i++)
    {
        uint x1 = 0, y1 = 0, x2 = 0, y2 = 0, j;

        uint gSize = groups[i].size();

        if (gSize < minDetects) continue;

        std::vector<uint> scales(scaleMap.size(), 0);

        for (j=0; j<gSize; j++)
        {
            x1 += detections[groups[i][j]].x1;
            y1 += detections[groups[i][j]].y1;
            x2 += detections[groups[i][j]].x2;
            y2 += detections[groups[i][j]].y2;
            scales[detections[groups[i][j]].scaleKey]++;
        }

        uint maxVal = 0, maxScaleKey = 0;
        for (uint k=0; k<scales.size(); ++k)
        {
            if (scales[k]>maxVal) {
                maxVal = scales[k];
                maxScaleKey = k;
            }
        }

        x1 /= j;
        y1 /= j;
        x2 /= j;
        y2 /= j;

        Rect r(x1, y1, x2-x1+1, y2-y1+1, maxScaleKey);

        if (largestGroupSize < gSize)
        {
            largestGroupSize = gSize;
            if (largestGroupIdx != -1)
                resultVector.erase(resultVector.begin()+largestGroupIdx);
            largestGroupIdx = i;
            resultVector.push_back(r);
        }
    }

    if (resultVector.size()>0)
        return resultVector[0];

    // return an invalid rectangle if no strong combination exists
    return Rect(-1);
}


/**
 * Constructor
 *
 * @param Mat im - input image
 * @param vector<double> - expected scales
 */
Image::Image(const cv::Mat& im, const std::vector<double>& scaleMap)
{
    ims.reserve(scaleMap.size());
    iis.reserve(scaleMap.size());
    ii2s.reserve(scaleMap.size());

    cv::Mat imclone = im.clone();

    double scaleRatio = 1.;
    for (uint s=0; s<scaleMap.size(); ++s)
    {
        scaleRatio = scaleMap[s]/((s==0) ? 1 : scaleMap[s-1]);
        cv::Mat *prevIm = (s==0) ? &imclone : ims[s-1];
        cv::Mat *resized = new cv::Mat(round(prevIm->rows/scaleRatio), round(prevIm->cols/scaleRatio),prevIm->type());
        cv::resize(*prevIm, *resized, resized->size(),0,0);
        cv::Mat *ii = new cv::Mat(resized->rows+1, resized->cols+1, resized->type());
        cv::Mat *ii2 = new cv::Mat(resized->rows+1, resized->cols+1, resized->type());
        cv::integral(*resized, *ii, *ii2);

        ims.push_back(resized);
        iis.push_back(ii);
        ii2s.push_back(ii2);
    }
}

/**
 * Efficient computation of variance of a given image region through integral images.
 *
 * @param Mat ii - integral of input image
 * @param Mat ii2 - integral of squares of input image
 * @param x1, y2, x2, y2 - bounds of the region to compute the variance
 * @return double - the variance
 */
double Image::var(const cv::Mat& ii, const cv::Mat& ii2, const uint x1, const uint y1, const uint x2, const uint y2)
{
    double area = (x2-x1)*(y2-y1);
    double muX = (ii.at<double>(y2,x2)
                  -ii.at<double>(y2,x1)
                  -ii.at<double>(y1,x2)
                  +ii.at<double>(y1, x1))/area;
    double muX2 = (ii2.at<double>(y2,x2)
                    -ii2.at<double>(y2,x1)
                    -ii2.at<double>(y1,x2)
                    +ii2.at<double>(y1, x1))/area;

    return muX2-muX*muX;
}

/**
 * Clear dirt
 */
Image::~Image()
{
    for (uint i=0; i<ims.size(); ++i)
    {
        delete ims[i];
        delete iis[i];
        delete ii2s[i];
    }
}

/**
 * Construct rectangular interval
 */
RectInterval::RectInterval(const size_t _width, const size_t _height)
{
    setUpper(std::numeric_limits<double>::max());

    //! goes clockwise beginning from left
    setLow(LEFT, 1); // left low
    setLow(TOP, 1); // top low
    setLow(RIGHT, 1); // right low
    setLow(BOTTOM, 1); // bottom low

    setHigh(LEFT, _width-1);  // left high
    setHigh(TOP, _height-1);  // top high
    setHigh(RIGHT, _width-1);  // right high
    setHigh(BOTTOM, _height-1); // bottom high
}

/**
 * An alternative constructor
 * Like a copy constructor but takes a pointer instead a reference.
 *
 * @param RectInterval* ri
 */
RectInterval::RectInterval(const RectInterval *ri)
{
    for (size_t i=0; i<4; ++i)
    {
        setLow(i, ri->getLow((i)));
        setHigh(i, ri->getHigh((i)));
    }
}

/**
 * Return the average rectangle for this interval
 *
 * @return Rect
 */
Rect RectInterval::getAvgRect() const
{
    size_t x1 = ((low[LEFT]+high[LEFT])>>1)-1;
    size_t y1 = ((low[TOP]+high[TOP])>>1)-1;
    size_t x2 = ((low[RIGHT]+high[RIGHT])>>1)-1;
    size_t y2 = ((low[BOTTOM]+high[BOTTOM])>>1)-1;

    return Rect(x1, y1, x2-x1, y2-y1);
}

/**
 * Return the Rect representing the large interval of this RectInterval
 *
 * @return Rect
 */
Rect RectInterval::getLargeRect() const
{
    size_t x1 = low[LEFT];
    size_t y1 = low[TOP];
    size_t x2 = high[RIGHT];
    size_t y2 = high[BOTTOM];

    return Rect(x1,y1,x2-x1,y2-y1);
}
/**
 * Return the Rect representing the small interval of this RectInterval
 *
 * @return Rect
 */
Rect RectInterval::getSmallRect() const
{
    size_t x1 = high[LEFT];
    size_t y1 = high[TOP];
    size_t x2 = low[RIGHT];
    size_t y2 = low[BOTTOM];

    return Rect(x1,y1,x2-x1,y2-y1);
}

/**
 * Return the edge where the interval width of this rectangular
 * interval is the largest.
 * Method adopted from the actual ESS code of Lampert et al
 *
 * @return int
 */
int RectInterval::maxIndex() const
{
    int splitIndex = -1;
    int maxWidth = 0;

    for (size_t i=0; i<4; ++i)
    {
        int intervalWidth = high[i] - low[i];

        if (intervalWidth > maxWidth)
        {
            splitIndex = i;
            maxWidth = intervalWidth;
        }
    }

    return splitIndex;
}

