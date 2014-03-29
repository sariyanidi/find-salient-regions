#include "landmark.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <cstdlib>
#include <iomanip>

/**
 * Create a scaled copy where absolute coordinates exist.
 *
 * @param double s - scale of the feature
 * @param uint w - width of the target landmark
 * @param uint h - height of the target landmark
 * @return Fern* - the new Fern feature with absolute coordinates in the given scale
 */
Fern* Fern::getAbsoluteScaled(const double s, const uint w, const uint h)
{
    Fern* f = new Fern;
    for (uint i=0; i<feats.size(); ++i)
        f->addFeature(*(feats[i]),s,w,h);
    return f;
}

/**
 * Default constructor, read classifier through file
 *
 * @param string - the classifier file
 */
Classifier::Classifier(const std::string &fileName)
{
    std::ifstream clsFile(fileName.c_str());

    if (!clsFile.is_open()) {
        std::cerr << "Classifier file (" << fileName << ") does not exist!";
        numT = 0;
        return;
    }

    // read number of features and baseFerns
    clsFile >> numF >> numT;

    // fill each feature
    for (uint i=0; i<numT; ++i)
    {
        Fern* f = new Fern();
        for (uint j=0; j<numF; ++j) {
            double r, l, t, b;
            clsFile >> r >> l >> t >> b;
            f->addFeature(r,l,t,b);
        }
        addFern(f);
    }
}

/**
 * Clear everything
 *
 */
Classifier::~Classifier()
{
    for (int i=baseFerns.size()-1; i>=0; --i )
        delete baseFerns[i];
}

/**
 * Initialize a detector for a certain landmark. Assuming that the width and height
 * of the original box are known.
 *
 * @param string fileName - name of the classifier file
 * @param uint w - width of the target window/object
 * @param uint h - height of the target window/object
 * @todo objRect'i duzgun baslat
 */
Landmark::Landmark(const Classifier* _cls, const Image& input, const Rect& _objRect, const bool crtParent, const bool crtChildren)
    :
    IM_W(input.ii(0)->size().width),
    IM_H(input.ii(0)->size().height),
    objRect(_objRect),
    cls(_cls),
    numP(cls->numT, std::vector<uint>(std::pow(2,cls->numF), 0)), // numP[numT][numF]
    numN(cls->numT, std::vector<uint>(std::pow(2,cls->numF), 0)), // numN[numT][numF]
    P(cls->numT, std::vector<double>(std::pow(2,cls->numF), 0)),  // P[numT][numF]
    thrP(0.5*cls->numT),
    numUpdates(0),
    searchRegion(objRect.extended(cv::Size(IM_W, IM_H), objRect.width, objRect.height))
{
    // return false if classifier is invalid
    if (!cls->good())
        return;

    if (crtParent)
        parent = new Landmark(cls, input, objRect.extended(input.ii(0)->size(), objRect.width*0.4), false, false);
    else
        parent = NULL;

    Rect& o = objRect;
    std::cout << o.x1 << " " << o.x2 << " " << o.y1 << " " << o.y2 << std::endl;
    if (crtChildren) {
        for (uint i=0; i<4; ++i)
        {
            uint row = i/4. >= 0.5;
            uint col = i%2;
            std::cout << row << " " << col << std::endl;
            Rect r(o.x1+o.width*col*.5, o.y1+o.height*row*.5, o.width/2.,o.height/2.);
            std::cout << r.x1 << " " << r.x2 << " " << r.y1 << " " << r.y2 << std::endl;
//            children.push_back(new Landmark(cls, input, r, false, false));
        }
    } else {
        for (uint i=0; i<4; ++i)
            children.push_back(NULL);
    }

    createFernSets(SCALE_START, SCALE_END, SCALE_STEP);
    update(input, objRect);
}

/**
 * We have pointers in the class, we need a copy constructor.
 *
 * @param Landmark
 */
Landmark::Landmark(const Landmark &l) :
        IM_W(l.IM_W),
        IM_H(l.IM_H),
        objRect(l.objRect),
        cls(l.cls),
        numP(l.numP),
        numN(l.numN),
        P(l.P),
        thrP(l.thrP),
        numUpdates(l.numUpdates),
        searchRegion(l.searchRegion)
{
    IM_W = l.IM_W;
    IM_H = l.IM_H;

    if (l.parent != NULL)
        parent = new Landmark(*(l.parent));
    else
        parent = NULL;

    double scaleStart = SCALE_START;
    double scaleStep = SCALE_STEP;
    double scaleEnd = SCALE_END;

    fernSets.clear();
    double scale = scaleStart;
    while (scale<scaleEnd)
    {
        // skip if target is larger than window
        if ( scale*objRect.width >= IM_W
             || scale*objRect.height >= IM_H
             || scale*objRect.width < config::MIN_OBJ_EDGE
             || scale*objRect.height < config::MIN_OBJ_EDGE )
            scaleFlag.push_back(false);
        else
            scaleFlag.push_back(true);

        fernSets.push_back(std::vector<Fern*>());
        fernSets.back().reserve(cls->numF);
        scaleMap.push_back(scale);

        // modify fern sets to fit to this landmark/object
        for (uint i=0; i<cls->numT; ++i)
            fernSets.back().push_back(cls->baseFerns[i]->getAbsoluteScaled(scale, objRect.width, objRect.height));

        scale *= scaleStep;
    }
}

/**
 * Create fern sets for all candidade scales
 *
 * @param double scaleStart - initials scale
 * @param double scaleEnd - final scale
 * @param double scaleStep - step bw two subsequent scales
 * @return void
 */
void Landmark::createFernSets(const double scaleStart, const double scaleEnd, const double scaleStep)
{
    fernSets.clear();

    double scale = scaleStart;
    while (scale<scaleEnd)
    {
        // skip if target is larger than window
        if ( scale*objRect.width >= IM_W
             || scale*objRect.height >= IM_H
             || scale*objRect.width < config::MIN_OBJ_EDGE
             || scale*objRect.height < config::MIN_OBJ_EDGE )
            scaleFlag.push_back(false);
        else
            scaleFlag.push_back(true);

        fernSets.push_back(std::vector<Fern*>());
        fernSets.back().reserve(cls->numF);
        scaleMap.push_back(scale);

        // modify fern sets to fit to this landmark/object
        for (uint i=0; i<cls->numT; ++i)
            fernSets.back().push_back(cls->baseFerns[i]->getAbsoluteScaled(scale, objRect.width, objRect.height));

        scale *= scaleStep;
    }
}

/**
 * Update positives if patch is spatially close to original window.
 *
 * @param Mat im - input image
 * @param Rect cObjRect - current object rect -> new estimation about object location
 * @todo objeyi multiscale egitmeyi duzelt!
 * @return void
 */
void Landmark::update(const Image &im, const Rect& cObjRect)
{
    // allow patch to be updated for limited times
    if (numUpdates>=NUM_MAX_UPDATES) {
        //std::cout << "Reached max updates!" << std::endl;
        return;
    } else
        ++numUpdates;

    // minimum target variance is the half variance of the window
    minVar = Image::var(*im.ii(0), *im.ii2(0), cObjRect.x1, cObjRect.y1, cObjRect.x2, cObjRect.y2)/2;
    //        uint stepX = (round(objRect.width/12.)>STEP_X) ? round(objRect.width/12.):STEP_X;
    //        uint stepY = (round(objRect.height/12.)>STEP_Y) ? round(objRect.height/12.):STEP_Y;
    //    uint stepX = STEP_X;
    //    uint stepY = STEP_Y;
    uint stepX = 2;
    uint stepY = 2;

    // timing
    int64 t1 = cv::getTickCount();
    for (uint s=0; s<scaleMap.size(); ++s)
    {
        if (!scaleFlag[s])
            continue;

        double sVal = scaleMap[s];
        uint xStrt = 0;
        uint yStrt = 0;
        //        uint xEnd = im.im(s)->cols-objRect.width-1;
        //        uint yEnd = im.im(s)->rows-objRect.height-1;
        uint xEnd = (int)im.im(s)->cols-(int)objRect.width-1<0?0:im.im(s)->cols-objRect.width-1;//im.im(s)->cols-width;
        uint yEnd = (int)im.im(s)->rows-(int)objRect.height-1<0?0:im.im(s)->rows-objRect.height-1;//im.im(s)->rows-height;

        Rect sObjRect = Rect(cObjRect.x1/sVal, cObjRect.y1/sVal, cObjRect.width/sVal, cObjRect.height/sVal);

        for (uint j=yStrt; j<yEnd; j+=stepY/*STEP_Y*/)
        {
            for (uint i=xStrt; i<xEnd; i+=stepX/*STEP_X*/)
            {
                // if the variation is smaller than...
                if (minVar > Image::var(*im.ii(s), *im.ii2(s), i, j, i+objRect.width, j+objRect.height)) continue;

                int x2 = i+cObjRect.width;
                int y2 = j+cObjRect.height;

                int inter = Rect::intersect(sObjRect, i, j, x2, y2);

                double area1 = (double) inter/(sObjRect.width*sObjRect.height);
                double area2 = (double) inter/(cObjRect.width*cObjRect.height);

                using std::min;

                // this is a negative if it is far enough
                if (min<double>(area1, area2)<0.4 && min<double>(area1, area2)>=0.00) {
                    std::vector<uint> patt = pattern(*im.im(s),i,j,0);
                    if (response(patt) >= thrP)
                        updateModel(patt, false);
                }
                else if (min<double>(area1, area2)>0.80) { // positive in this case
                    std::vector<uint> patt = pattern(*im.im(s),i,j,0);
                    if (response(patt) <= thrP)
                        updateModel(patt, true);
                }
            } // for x
        } // for y
    } // for scale

    int64 t2 = cv::getTickCount();
    //std::cout << "Learning: " << (double)(t2-t1) / cv::getTickFrequency()  << std::endl;
}

/**
 * Detect landmark in given image
 * 1) Scan the image extract detections
 * 2) Combine overlapping detections and assume that the strongest
 *    combination is the actual landmark if it is strong enough
 * 3) Update the model with the new detection, add irrelevant patches
 *    to negative set
 *
 * @param Image - input image to scan
 * @return void
 */
Rect Landmark::detect(const Image &im, cv::Mat& drawHere)
{
    Rect* roi;
    Rect _roi = (parent != NULL) ? parent->detect(im, drawHere) : Rect(-1);
    //    Rect *roi = &searchRegion;
    roi = (parent == NULL) ? &searchRegion : &_roi;

    if (!roi->legal())
        return Rect(-1);

    std::vector<Rect> detections; // uncombined detections

    uint stepX = (round(objRect.width/8.)>STEP_X) ? round(objRect.width/8.):STEP_X;
    uint stepY = (round(objRect.height/8.)>STEP_Y) ? round(objRect.height/8.):STEP_Y;
    //uint stepX = 2;
    //uint stepY = 2;
    //uint stepX = STEP_X
    //uint stepY = STEP_Y;

    //! 1) Scan input image and detect rectangles
    int64 t1 = cv::getTickCount();
    for (uint s=0; s<scaleMap.size(); ++s)
    {
        if (!scaleFlag[s])
            continue;

        double sVal = scaleMap[s];
        uint xStrt = roi->x1/sVal;//0;
        uint yStrt = roi->y1/sVal;//0;
        //        uint xStrt = 0;
        //        uint yStrt = 0;
        uint width = objRect.width/sVal;
        uint height = objRect.height/sVal;
        uint xEnd = (int)roi->x2/sVal-(int)objRect.width<0?0:(int)roi->x2/sVal-(int)objRect.width;//im.im(s)->cols-width;
        uint yEnd = (int)roi->y2/sVal-(int)objRect.height<0?0:(int)roi->y2/sVal-(int)objRect.height;//im.im(s)->rows-height;
        //        uint xEnd = ((int)im.im(s)->cols-(int)objRect.width>=0)?(int)im.im(s)->cols-(int)objRect.width:0;
        //        uint yEnd = ((int)im.im(s)->rows-(int)objRect.height>=0)?(int)im.im(s)->rows-(int)objRect.height:0;

        for (uint j=yStrt; j<yEnd; j+=stepY/*STEP_Y*/)
        {
            for (uint i=xStrt; i<xEnd; i+=stepX/*STEP_X*/)
            {
                // if the variance is smaller than...
                if (minVar > Image::var(*im.ii(s), *im.ii2(s), i, j, i+objRect.width, j+objRect.height))
                    continue;

                // push rect to detections if it exceeds the threshold
                if (response(pattern(*im.im(s),i,j,0)) >= thrP)
                {
                    cv::Rect r(i*sVal, j*sVal, objRect.width*sVal, objRect.height*sVal);
                    //cv::rectangle(drawHere, r, CV_RGB(255,255,255), 1);
                    detections.push_back(Rect(i*sVal, j*sVal, objRect.width*sVal, objRect.height*sVal, s));
                }
            } // x
        } // y
    } // scale

    //! 2) Combine overlapping detections, the strongest combionation is assumed to be the actual landmark
    Rect landmark = Rect::getStrongestCombination(detections, scaleMap, 1);

    int64 t2 = cv::getTickCount();
    //    std::cout << std::setprecision(5);
    //    std::cout << std::fixed;
    //    std::cout << "detection: " <<(double)(t2-t1) / cv::getTickFrequency() << std::endl;

    if (!landmark.legal()) // if a strong enough rect has not been found
        return landmark;

    //! 3-a) Update the model with the new detection
    if (response(pattern(*im.im(0),landmark.x1,landmark.y1,0)) >= thrP)
        update(im, landmark);

    //! 3-b) Update negative model with false positives

    if (numUpdates > NUM_MAX_UPDATES)
    {/*
        for (uint i=0; i<detections.size(); ++i )
        {
            Rect& r = detections[i];
            double sVal = scaleMap[r.scaleKey];

            int inter = Rect::intersect(r, landmark);

            double area1 = (double) inter/(landmark.width*landmark.height);
            double area2 = (double) inter/(r.width*r.height);

            // if detection irrelevant add them to negative patterns
            if (std::max<double>(area1, area2)<0.05)
            {
                // find the original coordinate by dividing to scale again
                std::vector<uint> patt = pattern(*im.im(r.scaleKey),round(r.x1/sVal),round(r.y1/sVal),r.scaleKey);

                // update negative set with false detections
                //                if (response(patt)>=thrP)
                //                    updateModel(patt, false);
            }
        }*/
    } else {
        /*
        // update with false positives emerging from out of ROI
        for (uint s=0; s<scaleMap.size(); ++s)
        {
            double sVal = scaleMap[s];
            uint xStrt = 0;
            uint yStrt = 0;
            uint width = objRect.width/sVal;
            uint height = objRect.height/sVal;
            uint xEnd = im.im(s)->cols-width;
            uint yEnd = im.im(s)->rows-height;

            uint roiXStrt = roi->x1/sVal;//0;
            uint roiYStrt = roi->y1/sVal;//0;
            uint roiXEnd = roi->x2/sVal-width;//im.im(s)->cols-width;
            uint roiYEnd = roi->y2/sVal-height;//im.im(s)->rows-height;



            for (uint j=yStrt; j<yEnd; j+=STEP_X)
            {
                for (uint i=xStrt; i<xEnd; i+=STEP_Y)
                {
                    // if the variance is smaller than...
                    if (minVar > Image::var(*im.ii(s), *im.ii2(s), i, j, i+width, j+height))
                        continue;

                    // continue if region is inside ROI
                    if (i>=roiXStrt && i<=roiXEnd &&
                        j>=roiYStrt && j<=roiYEnd)
                        continue;

                    std::vector<uint> patt = pattern(*im.im(s),i,j,0);
                    // push rect to detections if it exceeds the threshold
                    if (response(patt) >= thrP) {
                        cv::rectangle(drawHere, cv::Rect(i, j, width, height), CV_RGB(255,255,0),4);
                        updateModel(patt, false);
                    }
                } // x
            } // y
        } // scale
        */
    }
    if (NULL!=parent)
        cv::rectangle(drawHere, landmark.cvStyle(), CV_RGB(0,255,0), 2);
    return landmark;
}

/**
 * Destructor
 * Release Memory
 */
Landmark::~Landmark()
{
    if (NULL != parent)
        delete parent;

    for (uint j=0; j<fernSets.size(); ++j)
        for (uint i=0; i<fernSets[j].size(); ++i)
        {
        delete fernSets[j][i];
    }

    for (uint i=0; i<children.size(); ++i)
        if (children[i] != NULL)
            delete children[i];
}
