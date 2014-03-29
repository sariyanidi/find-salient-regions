#ifndef TRACKER_H
#define TRACKER_H

#include "definitions.h"
#include "opencv2/video/tracking.hpp"

#include <map>

/**
     * Class to keep a track item.
     * Duty of this class:
     * - Update tracking rectangle
     * - Keep and give tracking statistics
     *
     * @author evangelos sariyanidi / sariyanidi[at]gmail[dot]com
     * @date april 2011
     */
class TrackItem
{
public:
    // count instances + assign new
    TrackItem(const Rect& d) : id(maxId++),
        numInactiveFrames(0), numActiveFrames(1), tStart(cv::getTickCount()), kalman(d),
        dRect(d.x1, d.y1, d.width, d.height) {}

    // decrease num of instances on destruct
    ~TrackItem() { --counter; if (!isActive()) --maxId; }

    // update active item with rect
    void update(const Rect& dRect);

    // update inactive item
    bool update();

    // time passed since the tracking this (in secs.)
    double uptime() const { return (double)(cv::getTickCount()-tStart)/cv::getTickFrequency(); }

    // dont begin tracking immediately, begin when ...
    bool isActive() const;

    /**
         * Wrap class cv::KalmanFilter like this to handle it prettier.
         */
    class Kalman
    {
    public:
        friend class TrackItem;

        Kalman(const Rect& initRect);
        ~Kalman() { delete filter; }

    private:
        uint N; //! dimension of transition matrix: NxN
        uint M; //! length of measurement vector

        cv::KalmanFilter* filter;
    };

    //! @property unique id of track item
    const uint id;

    //! @property number of inactive frames - use to drop track if needed
    unsigned short numInactiveFrames;

    //! @property number of active frames - just record data
    uint numActiveFrames;

private:
    // set detection item
    void setRectFrom(const cv::Mat& state);

    //! @property count TrackItem instances
    static uint counter;

    //! @property use to assign a new id
    static uint maxId;

    //! @property tick count at Track init. time of this item
    unsigned long long tStart;

    //! @property anything kalman is inside here
    Kalman kalman;

public:
    //! @property rect to draw, most up-to-date item position
    Rect dRect;
};


/**
     * Tracker class written according to a "kind of" decorator pattern:
     * Take a detector and wrap it with this Tracker
     *
     * @todo convert FaceDetector* to Detector* after Detectors are abstracted
     * @todo update totalTime()
     * @author evangelos sariyanidi / sariyanidi[at]gmail[dot]com
     * @date april 2011
     */
class Tracker
{
public:

    // construct tracker using a detector
    Tracker() :tStart(cv::getTickCount()), numFrames(0) {}

    // in destructor delete detector and all track items
    ~Tracker();

    // Track items on video
    void onVideo();

    // add/drop items
    void add(TrackItem* ti) { items.insert(std::pair<uint, TrackItem*>(ti->id, ti)); }
    void drop(uint id) { delete items[id]; items.erase(id); }

    // update items with fresh detections
    void updateWith(std::vector<Rect> freshDetects);

    // record regarding tracker
    uint numItems() const { return items.size(); }
    double totalTime() const { return (double)(cv::getTickCount()-tStart)/cv::getTickFrequency(); }

    //! @property allowed num of inactive frames, drop tracking if this number exceeded
    static const unsigned short NUM_MAX_INACTIVE_FRAMES = 6;

    //! @property minimum number of detections before start to track an item
    static const unsigned short NUM_MIN_DETECTIONS = 0;

public:
    //! @property items being tracked -> associate each item with its id
    std::map<uint, TrackItem*> items;

    //! @property tick count of Tracker initialization time
    unsigned long tStart;

    //! @property total number of frames run
    unsigned long numFrames;

    // see definition of Tracker::updateItems() for comments of these:
    std::map<uint, bool>
            updateActiveItems(std::vector<Rect>& freshDetects);
    void updateInactiveItems(const std::map<uint, bool>& flagUpdated);
    void addNewItems(std::vector<Rect>& freshDetects);
};

#endif // TRACKER_H
