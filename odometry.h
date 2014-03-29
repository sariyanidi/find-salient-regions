#ifndef ODOMETRY_H
#define ODOMETRY_H

#include "observation.h"

class Odometry
{
public:
    Odometry() {}
    void update(Observation *prevObs, Observation* curObs, cv::Mat& frameL);
};

#endif // ODOMETRY_H
