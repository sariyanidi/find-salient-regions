#include "navigator.h"
#include <QtCore/QDir>
#include "tracker.h"
#include <fstream>

uint Navigator::numDetected(const std::vector<Landmark> &landmark) const
{
    uint _numDetected=0;
    for (uint i=0; i<landmark.size(); ++i)
    {

    }

    return _numDetected;
}

/**
 * Run odometry estimator and loop closure detector on video data
 */
void Navigator::runOnVideo()
{
    double scale = 1.0;
    cv::VideoCapture capL(0);
    cv::VideoCapture capR(1);

    capL.set(CV_CAP_PROP_FRAME_WIDTH, 640*scale);
    capL.set(CV_CAP_PROP_FRAME_HEIGHT, 480*scale);
    capR.set(CV_CAP_PROP_FRAME_WIDTH, 640*scale);
    capR.set(CV_CAP_PROP_FRAME_HEIGHT, 480*scale);

    rig->camL->setScale(scale);
    rig->camR->setScale(scale);

    if (!capL.isOpened() || !capR.isOpened())
        return;

    cv::Mat frameL, frameR;
    Observation *curObs = NULL, *prevObs = NULL;

    while (1)
    {
        cv::Mat tmp;
        capL >> tmp;
        cv::cvtColor(tmp,frameL,CV_RGB2GRAY);
        capR >> tmp;
        cv::cvtColor(tmp,frameR,CV_RGB2GRAY);

        /**/
        // undistort and rectify to calculate depth later
        frameL = StereoRig::getUndistortedRectified(rig->camL->D, rig->camL->SM, rig->R1, rig->P1, frameL);
        frameR = StereoRig::getUndistortedRectified(rig->camR->D, rig->camR->SM, rig->R2, rig->P2, frameR);

        // black regions emerge after undistortion, crop image to avoid black regions
        double pad = 50*scale;
        frameL = frameL(cv::Rect(pad, pad/2., frameL.cols-pad*2, frameL.rows-pad*2));
        frameR = frameR(cv::Rect(pad, pad/2., frameR.cols-pad*2, frameR.rows-pad*2));

        curObs = new Observation(frameL, frameR, *rig);
        prevObs = (NULL == prevObs) ? new Observation(frameL, frameR, *rig) : prevObs;

        patchExtractor._extract(frameL, curObs);
        odometry.update(prevObs, curObs, frameL);

        delete prevObs;
        prevObs = curObs;
        if (cv::waitKey(10)>=0)
            break;
    }

    delete curObs; // clear
    return;
}

void Navigator::runLoopClosureAssistant(const std::string &dirName, const bool saveVid)
{
    // read files at source dir384
    QDir dir(QString(dirName.c_str()));
    dir.setNameFilters(QStringList("*.jpg"));

    QStringList imagesList = dir.entryList(QDir::NoDotAndDotDot | QDir::Files);

    std::vector<Landmark> landmarks;

    Observation *obs = NULL;

    Tracker t;

    // save video output if necessary
    cv::VideoWriter vid;
    if (saveVid) {
        std::stringstream ss; ss << "vid" << cv::getTickCount() << random() << ".avi";
        vid.open(ss.str(), CV_FOURCC('D', 'I', 'V', 'X'), 12, cv::Size(1024,309));
    }

    for (uint i=0; i<(size_t)imagesList.size(); ++i)
    {
        std::cout << i << "- ";
        //if (i%10!=0) continue;
        cv::Mat frame, frame2;
        cv::Mat tmp = cv::imread(dir.filePath(imagesList.at(i)).toStdString());

        cv::cvtColor(tmp, frame, CV_RGB2GRAY);
        frame2 = frame.clone();

        int64 t1 = cv::getTickCount();
        obs = new Observation(frame);
        int64 t2 = cv::getTickCount();

        std::cout << "SURF extraction took " << (double)(t2-t1)/cv::getTickFrequency() << std::endl;

        if (obs->keyPoints.size() == 0)
            continue;

        Rect r = patchExtractor._extract(frame, obs);
        //cv::rectangle(tmp, r.cvStyle(), CV_RGB(255,255,255),1);
        double wthRatio = (double) r.width/r.height; // width-to-height ratio:

        std::vector<Rect> detections;
        //        if (wthRatio<12. && wthRatio>1./12) // very narrow rectangles usually point to unsuccesfull detections
        detections.push_back(r);
        detections.push_back(patchExtractor._extract(frame, obs));
        detections.push_back(patchExtractor._extract(frame, obs));
        detections.push_back(patchExtractor._extract(frame, obs));

        // update the tracker
        t.updateWith(detections);

        // draw detection rects
        //for (uint j=0; j<detections.size(); ++j)
        //    cv::rectangle(tmp, detections[0].cvStyle(), CV_RGB(255,0,0),2);

        // draw tracking rects
        typedef std::map<uint, TrackItem*>::const_iterator TiIter;
        for (TiIter it=t.items.begin(); it != t.items.end(); ++it)
        {
            Rect d = it->second->dRect;
            Rect dNew = d.extended(frame.size(),10);

            cv::Rect r(d.x1, d.y1, d.width, d.height);
            cv::Rect dr = dNew.cvStyle();

            // draw bold if item is being tracked
            if (it->second->isActive())
            {
                if (it->second->numInactiveFrames == 0 && it->second->numActiveFrames > 2)
                    cv::rectangle(tmp, dr, CV_RGB(0,255,0), 2);

                std::stringstream ss1, ss2, ss3;

                ss1 << "id: " << it->first;
                ss2 << "Uptime: " << it->second->uptime();
                ss3 << "# of dead frames: " << it->second->numInactiveFrames;

                //cv::putText(tmp, ss1.str().c_str(), cv::Point(d.x1,d.y2), cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0), 2);
                //cv::putText(tmp, ss2.str().c_str(), cv::Point(d.x1,d.y2+20), cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0), 2);
            }
            else
            {
                // draw thin otherwise
                //cv::rectangle(tmp, r, cv::Scalar::all(255),1);
            }
        }

        // draw SURF features
#ifdef INSPECTION_MODE
        typedef std::vector<Feature>::const_iterator FtIter;
        for (FtIter iter = obs->features.begin(); iter != obs->features.end(); ++iter)
            cv::circle(tmp, cv::Point(iter->x, iter->y), 2, CV_RGB(255,0,255), 1);
#endif

	delete obs;
	obs = NULL; 

        // save video or display image
        if (saveVid)
            vid << tmp;
        else {
            cv::imshow("Output", tmp);
            if (cv::waitKey(10)>=0) break;
        }
    }

    return;
}

/**
 * @todo scaleMap olayina guzel bir cozum bul
 * @todo detection'lari disari dondur, dongude cizdir
 */
void Navigator::runDetector(const std::string &dirName, const bool saveVid)
{
    // read files at source dir384
    QDir dir(QString(dirName.c_str()));
    dir.setNameFilters(QStringList("*.jpg"));

    QStringList imagesList = dir.entryList(QDir::NoDotAndDotDot | QDir::Files);

    double sigma = 1.5;
    uint w = 9;
    cv::Mat im = cv::imread("../../Documents/datasets/Oxford_Ladybug_Panorama/PanoStitchOutput_LisaNewCollegeNov3_0001.jpg");
    cv::Mat sasa;
    cv::cvtColor(im, sasa, CV_RGB2GRAY);
    cv::GaussianBlur(sasa, sasa, cv::Size(w, w), sigma, sigma);
    sasa.convertTo(sasa,CV_64F);
    //    Rect obj(520, 83, 200, 60);

    Rect obj(629, 113, 97, 24);
    //    Rect obj(625, 109, 108, 32);

    //    Landmark ld("./resources/classifier.ferns", sasa, sasa.size(), obj.extended(sasa.size(), 0.75*obj.width));

    Classifier c("./resources/classifier.ferns");
    std::vector<double> scaleMap;
    double scale = Landmark::SCALE_START;
    while (scale <= Landmark::SCALE_END) {
        scaleMap.push_back(scale);
        scale *= Landmark::SCALE_STEP;
    }

    Tracker t;
    Tracker tFresh;

    Image inp(sasa, scaleMap);
    cv::VideoWriter vid;
    if (saveVid) {
        std::stringstream ss; ss << "vid" << cv::getTickCount() << random() << ".avi";
        vid.open(ss.str(), CV_FOURCC('D', 'I', 'V', 'X'), 12, cv::Size(1024,309));
    }

//    const int NUM_FEATS = 1;
    const int NUM_FEATS = 7;
    std::vector<Landmark*> landmarks;

    std::ofstream recFile("doysa_FAST.txt");
    for (uint i=0; i<(size_t)imagesList.size(); ++i)
    {
        std::stringstream ss5;
        ss5 << "./fastOutput/" << std::setw(4) << std::setfill('0') << i << ".jpg";
        int64 t1 = cv::getTickCount();
        int64 tStrt = t1;
        cv::Mat frame;
        cv::Mat tmp = cv::imread(dir.filePath(imagesList.at(i)).toStdString());

        std::cout << "Frame #" << i << std::endl;
        std::cout << "------------------------------------------" << std::endl;
        cv::cvtColor(tmp, frame, CV_RGB2GRAY);

        std::cout << std::setprecision(6) << std::fixed;

        // Extract obeervation
        Observation* obs = new Observation(frame);
        std::cout << "- Observation extracted in " << (double)(cv::getTickCount()-t1)/cv::getTickFrequency() << " secs.\n";

        frame.convertTo(frame, CV_64F);

        cv::GaussianBlur(frame, frame, cv::Size(w, w), sigma, sigma);

        // Prepare input image
        t1 = cv::getTickCount();
        Image input(frame, scaleMap);
        std::cout << "- Image prepared in " << (double)(cv::getTickCount()-t1)/cv::getTickFrequency() << " secs.\n";

        std::vector<Rect> landmarkCandidates;
        std::vector<Rect> curDetects;

        t1 = cv::getTickCount();
        for (int ii=0; ii<1;/*ii<NUM_FEATS-(int)t.items.size();*/ ++ii)
        {
            Rect newRect = patchExtractor._extract(tmp, obs);
            if (newRect.width<config::MIN_OBJ_EDGE || newRect.height<config::MIN_OBJ_EDGE)
                continue;

            landmarkCandidates.push_back(newRect);
        }

        tFresh.updateWith(landmarkCandidates);


        typedef std::map<uint, TrackItem*>::const_iterator TiIter;
        for (TiIter it=t.items.begin(); it != t.items.end(); ++it)
        {
            Rect d(it->second->dRect);

            for (int k=landmarkCandidates.size()-1; k>=0; --k)
            {
                Rect now = landmarkCandidates[k];

                uint inter = Rect::intersect(d, now);
                double area1 = (double)inter/(d.width*d.height);
                double area2 = (double)inter/(now.width*now.height);

                if (max<double>(area1,area2)>=0.2)
                    landmarkCandidates.erase(landmarkCandidates.begin()+k);
            }
        }

        typedef std::vector<Rect>::iterator Fit;
        Fit itF1 = landmarkCandidates.begin();

        while (itF1!=landmarkCandidates.end())
        {
            Fit itF2 = landmarkCandidates.begin();
            while (itF2!=landmarkCandidates.end())
            {
                if (itF2==itF1)
                {
                    ++itF2;
                    continue;
                }
                uint inter = Rect::intersect(*itF1, *itF2);
                double area1 = (double)inter/(itF1->width*itF1->height);
                double area2 = (double)inter/(itF2->width*itF2->height);

                if (max<double>(area1,area2)>=0.2)
                    itF2=landmarkCandidates.erase(itF2);

                if (landmarkCandidates.end()!=itF2)
                    ++itF2;
            }
            ++itF1;
        }

        std::cout << "- Patches extracted in " << (double)(cv::getTickCount()-t1)/cv::getTickFrequency() << " secs.\n";

        t1 = cv::getTickCount();
        for (uint jj=0; jj<landmarks.size(); ++jj)
        {
            curDetects.push_back(landmarks[jj]->detect(input, tmp));
            Rect& thisRect = curDetects.back();
            if (!thisRect.legal())
            {
                curDetects.pop_back();
                continue;
            }

            std::stringstream ss; ss<<jj;
            cv::Point pt((thisRect.x1+thisRect.x2)/2., (thisRect.y1+thisRect.y2)/2.);
            cv::putText(tmp, ss.str().c_str(),pt,CV_FONT_NORMAL, 1,CV_RGB(255,0,255));

            recFile << ss.str() << '\t';

            for (int k=landmarkCandidates.size()-1; k>=0; --k)
            {
                Rect& thisCan = landmarkCandidates[k];
                uint inter = Rect::intersect(thisCan, thisRect);
                double area1 = (double) inter/(thisRect.width*thisRect.height);
                double area2 = (double) inter/(thisCan.width*thisCan.height);

                if (max<double>(area1,area2)>=0.2)
                    landmarkCandidates.erase(landmarkCandidates.begin()+k);
            }
        }
        std::cout << "- Landmarks ( "<< landmarks.size() <<" ) detected in " << (double)(cv::getTickCount()-t1)/cv::getTickFrequency() << " secs.\n";

        uint total=0;
        t1 = cv::getTickCount();
        typedef std::map<uint, TrackItem*>::const_iterator TiIter;


        for (uint jj=0; jj<landmarkCandidates.size(); ++jj)
        {
            if (curDetects.size()>=NUM_FEATS)
                break;
            total++;
            landmarks.push_back(new Landmark(&c, input, landmarkCandidates[jj], true, true));
            cv::rectangle(tmp, landmarkCandidates[jj].cvStyle(), CV_RGB(255,0,255));
        }

        std::cout << "- New landmarks ( "<< total <<" ) learned and detected in " << (double)(cv::getTickCount()-t1)/cv::getTickFrequency() << " secs.\n";

        t.updateWith(curDetects);

        for (TiIter it=t.items.begin(); it != t.items.end(); ++it)
        {
            Rect d(it->second->dRect);

            // draw bold if item is being tracked
            if (it->second->isActive())
            {
//                if (it->second->numInactiveFrames == 0 && it->second->numActiveFrames > 1)
//                    cv::rectangle(tmp, d.cvStyle(), CV_RGB(255,255,0), 9);
            }
        }

        //vid << tmp;

        std::cout << "Everything took " << (double)(cv::getTickCount()-tStrt)/cv::getTickFrequency() << "secs. " << std::endl << std::endl;

        cv::imwrite(ss5.str(), tmp);
        delete obs;
        cv::imshow("rezult", tmp);
        cv::waitKey(10);
        recFile << std::endl;
    }

    for (uint l=0; l<landmarks.size(); ++l)
        delete landmarks[l];
    return;

}


void Navigator::runOnDir(const std::string &dirName)
{
    double scale = 1.;

    // read files at source dir384
    QDir dirLeft(QString(dirName.c_str())), dirRight(QString(dirName.c_str()));
    // my images
    //    dirLeft.setNameFilters(QStringList("left*.png"));
    //    dirRight.setNameFilters(QStringList("right*.png"));
    //        dirLeft.setNameFilters(QStringList("img_stereo_1_left_*.jpg"));
    //        dirRight.setNameFilters(QStringList("img_stereo_1_right_*.jpg"));
    dirLeft.setNameFilters(QStringList("left*.jpg"));
    dirRight.setNameFilters(QStringList("right*.jpg"));

    QStringList leftImagesList = dirLeft.entryList(QDir::NoDotAndDotDot | QDir::Files);
    QStringList rightImagesList = dirRight.entryList(QDir::NoDotAndDotDot | QDir::Files);

    Observation *curObs = NULL, *prevObs = NULL;

    for (uint i=800; i<(size_t)leftImagesList.size(); ++i)
    {
        if (i%4!=0) continue;

        cv::Mat frameL, frameR, frameLb, frameRb;
        cv::Mat tmpL = cv::imread(dirLeft.filePath(leftImagesList.at(i)).toStdString());
        cv::Mat tmpR = cv::imread(dirRight.filePath(rightImagesList.at(i)).toStdString());

        cv::cvtColor(tmpL,frameLb,CV_RGB2GRAY);
        cv::cvtColor(tmpR,frameRb,CV_RGB2GRAY);

        std::cout << dirLeft.filePath(leftImagesList.at(i)).toStdString() << std::endl;
        cv::resize(frameLb, frameL, cv::Size(), scale, scale);
        cv::resize(frameRb, frameR, cv::Size(), scale, scale);

        // undistort and rectify to calculate depth later
        // bunu sadece bumblebee ile çekilmiş kameralar için ortadan kaldırıyoruz
        /*
        frameL = StereoRig::getUndistortedRectified(rig->camL->D, rig->camL->SM, rig->R1, rig->P1, frameL);
        frameR = StereoRig::getUndistortedRectified(rig->camR->D, rig->camR->SM, rig->R2, rig->P2, frameR);
        */
        // black regions emerge after undistortion, crop image to avoid black regions
        double pad = 50*scale;
        frameL = frameL(cv::Rect(pad, pad/2., frameL.cols-pad*2, frameL.rows-pad*2));
        frameR = frameR(cv::Rect(pad, pad/2., frameR.cols-pad*2, frameR.rows-pad*2));

        curObs = new Observation(frameL, frameR, *rig);

        //        curObs = new Observation(frameL);
        prevObs = (NULL == prevObs) ? new Observation(frameL, frameR, *rig) : prevObs;
        //curObs = new Observation(frameL);

        if (prevObs->keyPoints.size() == 0 || curObs->keyPoints.size() == 0)
            continue;

        odometry.update(prevObs, curObs, frameL);

        cv::imshow("leftIm", frameL);
        cv::imshow("rightIm", frameR);

        delete prevObs;
        prevObs = curObs;
        if (cv::waitKey(10)>=0)
            break;
    }

    delete curObs; // clear
    return;
}
