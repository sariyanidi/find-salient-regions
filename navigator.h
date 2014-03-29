#ifndef NAVIGATOR_H
#define NAVIGATOR_

#include "patchextractor.h"
#include "odometry.h"
#include "landmark.h"

class Navigator
{
public:
    Navigator(StereoRig* _rig) : rig(_rig) {}
    void runOnVideo();
    void runOnDir(const std::string& dirName );
    void runLoopClosureAssistant(const std::string &dirName, const bool saveVid=false);
    void runDetector(const std::string &dirName, const bool saveVid=false);
    ~Navigator() {}
    //enum ST_STATUS { SATISFIED, UNCERTAIN, UNSATISFIED };
    uint numDetected(const std::vector<Landmark>& landmark) const;
private:
    StereoRig* rig;
    PatchExtractor patchExtractor;
    Odometry odometry;
};

#endif // NAVIGATOR_H
