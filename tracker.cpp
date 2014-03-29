#include "tracker.h"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

uint TrackItem::counter = 0; //! keep number of TrackItem instances
uint TrackItem::maxId = 1; //! use this each time an id is assigned to a new TrackItem

/**
 * A TrackItem becomes active only if it is tracked for
 * at least Tracker::NUM_MIN_DETECTIONS times
 *
 * @return bool
 */
bool TrackItem::isActive() const
{
    return numActiveFrames > Tracker::NUM_MIN_DETECTIONS;
}

/**
 * Update an active item using new rectangle
 * Assuming that this TrackItem is active in this frame
 *
 * @param  Rect&
 * @return void
 */
void TrackItem::update(const Rect& d)
{
    ++numActiveFrames;
    numInactiveFrames = 0;

    // Tracking 4 points
    cv::Mat measurement(kalman.M, 1, CV_32F);
    measurement.at<float>(0,0) = (float) d.x1;
    measurement.at<float>(1,0) = (float) d.y1;
    measurement.at<float>(2,0) = (float) d.x2;
    measurement.at<float>(3,0) = (float) d.y2;

    kalman.filter->predict();
//    const cv::Mat& statePost = kalman.filter->correct(measurement);

    // update rectangle
    setRectFrom(kalman.filter->correct(measurement));
}

/**
 * Update an non-active item, an item which is not
 * detected in this frame.
 * Return false if item is inactive for long time, true otherwise
 *
 * @return bool
 */
bool TrackItem::update()
{
//    std::cout << "Tracking ... " << numInactiveFrames << std::endl;

    if (++numInactiveFrames >= Tracker::NUM_MAX_INACTIVE_FRAMES || !isActive())
        return false;

    const cv::Mat& statePre = kalman.filter->predict();

    // update rectangle
    setRectFrom(statePre);

    return true;
}

/**
 * Update rectangle from the most recent state.
 *
 * @param  Mat& state
 * @return void
 */
void TrackItem::setRectFrom(const cv::Mat &state)
{
    dRect.x1 = state.at<float>(0,0);
    dRect.y1 = state.at<float>(1,0);
    dRect.x2 = state.at<float>(2,0);
    dRect.y2 = state.at<float>(3,0);

    dRect.width = dRect.x2-dRect.x1;//2*halfWin;
    dRect.height = dRect.y2-dRect.y1;//2*halfWin;
}

/**
 * Kalman constructor: Parameters of the filter are set in here.
 * These parameters have a direct effect on the behaviour pf the filter.
 */
TrackItem::Kalman::Kalman(const Rect& initRect)
{
    N = 8; // dimension of trans. matrix
    M = 4; // length of measurement

    // setup kalman filter with a Model Matrix, a Measurement Matrix and no control vars
    filter = new cv::KalmanFilter(N, M, 0);

    // transitionMatrix is eye(n,n) by default
    filter->transitionMatrix.at<float>(0,4) = 0.067f; // dt=0.04, stands for the time
    filter->transitionMatrix.at<float>(1,5) = 0.067f; // betweeen two video frames in secs.
    filter->transitionMatrix.at<float>(2,6) = 0.067f;
    filter->transitionMatrix.at<float>(3,7) = 0.067f;

    // measurementMatrix is zeros(n,p) by default
    filter->measurementMatrix.at<float>(0,0) = 1.0f;
    filter->measurementMatrix.at<float>(1,1) = 1.0f;
    filter->measurementMatrix.at<float>(2,2) = 1.0f;
    filter->measurementMatrix.at<float>(3,3) = 1.0f;

    using cv::Scalar;

    // assign a small value to diagonal coeffs of processNoiseCov
    cv::setIdentity(filter->processNoiseCov, Scalar::all(1e-2)); // 1e-2

    // Measurement noise is important, it defines how much can we trust to the
    // measurement and has direct effect on the smoothness of tracking window
    // - increase this tracking gets smoother
    // - decrease this and tracking window becomes almost same with detection window
    cv::setIdentity(filter->measurementNoiseCov, Scalar::all(1e-3)); // 1e-1
    cv::setIdentity(filter->errorCovPost, Scalar::all(1));

    // we are tracking 4 points, thus having 4 states: corners of rectangle
    filter->statePost.at<float>(0,0) = initRect.x1;
    filter->statePost.at<float>(1,0) = initRect.y1;
    filter->statePost.at<float>(2,0) = initRect.x2;
    filter->statePost.at<float>(3,0) = initRect.y2;
}


/**
 * Destructor
 * Release memory. Delete detector and all items
 */
Tracker::~Tracker()
{
    typedef std::map<uint, TrackItem*>::iterator TiIter;

    for (TiIter it = items.begin(); it != items.end(); ++it)
        drop(it->first);
}

/**
 * Take new detections and update the whole items list.
 * Processes are distributed to some internal methods.
 *
 * @param  vector<Rect>& freshDetects - incoming detections
 * @return void
 */
void Tracker::updateWith(std::vector<Rect> freshDetects)
{
    // a flag map keeping the state of each item: updated or not
    std::map<uint, bool> flagActive;

    // 1) update whatever you matchs
    flagActive = updateActiveItems(freshDetects);

    // 2) add remaining rectangles ass new items
    this->addNewItems(freshDetects);

    // 3) update unmatched items, drop them if necessary
    this->updateInactiveItems(flagActive);
}

/**
 * Take new detections and update the ones matched with the
 * existing items.
 * WARNING! the detections which are not matched are removed
 * from the vector.
 *
 * @param  vector<Rect>& freshDetects - incoming detections
 * @return std::map<uint,bool> - a flag for each item stating whether item is updated or not
 */
std::map<uint,bool> Tracker::updateActiveItems(std::vector<Rect>& freshDetects)
{
    std::map<uint, bool> flagActive;

    typedef std::map<uint, TrackItem*>::iterator TiIter;

    for (TiIter it=items.begin(); it != items.end(); ++it)
        flagActive[it->first] = false;

    // associate rects to items
    for (int i=freshDetects.size()-1; i>=0; --i)
    {
        int maxIdx = -1;
        double maxArea = 0.;

        for (TiIter it=items.begin(); it != items.end(); ++it)
        {
            if (flagActive[it->first]) // update an item only once
                continue;

            uint area = Rect::intersect(freshDetects[i], it->second->dRect);

            // check if they intersect
            if (area > 0)
            {
                double ratio1 = (double)area/(freshDetects[i].width*freshDetects[i].height);
                double ratio2 = (double)area/(it->second->dRect.width*it->second->dRect.height);

                // try to find the best/largest intersection between rects
                if (std::max<double>(ratio1,ratio2) > maxArea)
                {
                    maxArea = std::max<double>(ratio1,ratio2);
                    maxIdx = it->first;
                }
            }
        }

        // is the best good enough?
        if (maxArea > 0.40)
        {
            // if it is, freshDetects[i] is assumed to stand for the
            // TrackedItem with id = maxIdx
            flagActive[maxIdx] = true;
            items[maxIdx]->update(freshDetects[i]);
            freshDetects.erase(freshDetects.begin()+i);
        }
    }

    // this will be piped to updateInactiveItems()
    return flagActive;
}

/**
 * Take a list including a flag for each TrackItem: updated or not.
 * Update each inactive item
 *
 * @param  map<uint,bool>& flagActive
 * @return void
 */
void Tracker::updateInactiveItems(const std::map<uint,bool>& flagActive)
{
    typedef std::map<uint, bool>::const_iterator FlagIter;

    for (FlagIter it = flagActive.begin(); it != flagActive.end(); ++it )
    {
        // skip if item is active at this frame
        if (it->second)
            continue;

        // drop item if it's inactive for long
        if (!items[it->first]->update())
            drop(it->first);
    }
}

/**
 * Add remaining unmatched freshRects as new TrackItems
 *
 * @param  vector<Rect>
 * @return void
 */
void Tracker::addNewItems(std::vector<Rect>& freshDetects)
{
    for (uint i=0; i<freshDetects.size(); ++i)
        add(new TrackItem(freshDetects[i]));
}
