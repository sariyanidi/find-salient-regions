#include "observation.h"
#include <algorithm>
#include <numeric>

/**
 * Constructor - Stereo data
 * Extract observation using a pair of %RECTIFIED% stereo images and the stereo rig.
 * The difference of stereo observation is that each has word has also depth information
 * and words whose depth cannot be extracted are eliminated.
 * Another important difference is that the %keyPoints% and %descriptors% properties of
 * stereo observations are filtered at the end of this constructor. This is done in order
 * to keep only points with 3D data to use in Visual Odometry extraction.
 *
 * @param Mat im - left image
 * @param Mat imR - right image
 * @param Mat StereoRig - stereo rig, giving the relation bw two images
 */
Observation::Observation(const cv::Mat& im, const cv::Mat& imR, const StereoRig& rig)
    : isStereo(true)
{
    init();

    // initialize keypoint and descriptor extractors separately
    cv::Ptr<cv::FeatureDetector> detector = new cv::SurfFeatureDetector(config::HESSIAN_THRESHOLD,6,7);
    cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = new cv::SurfDescriptorExtractor(6, 7, true);

    // detect keypoints and extract features
    detector->detect(im, keyPoints);
    descriptorExtractor->compute(im, keyPoints, descriptors);

    // convert features to words
    Vocabulary::instance()->translateAll(descriptors, matches);

    // clear first
    this->features.clear();

    // get points which have depth information
    std::map<int, cv::Point3d> tmpPoints3D = get3DPts(im, imR, rig);

#ifdef INSPECTION_MODE
    //timing
    clock_t t1 = clock();

    //timing
    //std::cout << "Features extracted in " << (double)(clock()-t1)/CLOCKS_PER_SEC;
    //std::cout << " seconds" << std::endl;
    //std::cout << "--------------------------------------" << std::endl;

    //timing
    t1 = clock();
#endif

    features.reserve(keyPoints.size());

    // keyPoints which have been matched in stereo will be kept, mark their IDs
    std::vector<uint> keepIdx;
    keepIdx.reserve(keyPoints.size());

    //! 1) gather allocation data for sparse representation
    for (uint i=0; i<keyPoints.size(); ++i)
    {
        // eger noktanın derinligi devam et
        if (tmpPoints3D.end() == tmpPoints3D.find(i))
            continue;

        keepIdx.push_back(i); // this kp has 3D information, keep it

        size_t curX = round(keyPoints[i].pt.x);
        size_t curY = round(keyPoints[i].pt.y);
        size_t wordId = matches[i].trainIdx;

        // insert word to multiset, compute their frequency later
        sparse->words.insert(wordId);

        // these will be used to allocate sparse matrix correctly
        this->sparse->insertX(curX);
        this->sparse->insertY(curY);

        features.push_back(Feature(wordId, curX, curY, -tmpPoints3D[i].z/1000.)); // add depth in meters
    }

    // fill the sparse matrix which is used to efficiently compute marginality
    this->sparse->fillWith(features);

    std::vector<cv::KeyPoint> tmpKeyPts(keyPoints);
    cv::Mat tmpDescriptors(descriptors);
    /**/
    // filter keyPoints and descriptors, re-fill them
    keyPoints.clear();
    keyPoints.reserve(keepIdx.size());
    descriptors = cv::Mat::zeros(keepIdx.size(), descriptors.cols, descriptors.type());
    points3D->clear();

    // now copy the filtered keyPoints
    for (uint i=0; i<keepIdx.size(); ++i)
    {
        keyPoints.push_back(tmpKeyPts[keepIdx[i]]);
        points3D->push_back(tmpPoints3D[keepIdx[i]]);
        for (int j=0; j<tmpDescriptors.cols; ++j)
            descriptors.at<float>(i,j) = tmpDescriptors.at<float>(keepIdx[i], j);
    }
#ifdef INSPECTION_MODE
    //timing
    //std::cout << "Features are converted to words in " << (double)(clock()-t1)/CLOCKS_PER_SEC;
    //std::cout << " seconds" << std::endl;
    //std::cout << "------------------------------------------------" << std::endl;
#endif

}

void Observation::recalcIntegrals(const Rect &r)
{
    for (uint i=0; i<features.size(); ++i)
    {
        Feature& f = features[i];
        if (f.x<=r.x2 && f.x>r.x1 && f.y<r.y2 && f.y>r.y1)
            features.erase(features.begin()+i);
    }

    sparse->fillWith(features);
}

/**
 * Extract features from single image, save 'em to class container.
 * 1) Pass through all features to allocate space for sparse data matrices
 * 2) Fill sparce matrices, see Observation::Sparse class
 * 3) Fill an auxiliary matrix cv::mat Observation::Sparse::numFeats
 *
 * @param cv::Mat im
 * @return void
 */
void Observation::from(const cv::Mat& im)
{
     //initialize keypoint and descriptor extractors separately
    cv::Ptr<cv::FeatureDetector> detector = new cv::SurfFeatureDetector(config::HESSIAN_THRESHOLD);
//    cv::Ptr<cv::FeatureDetector> detector = new cv::FastFeatureDetector(config::FAST_THRESHOLD);

    cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = new cv::SurfDescriptorExtractor(4, 2, true);

    // detect keypoints and extract features
    detector->detect(im, keyPoints);
    descriptorExtractor->compute(im, keyPoints, descriptors);

    // convert features to words
    Vocabulary::instance()->translateAll(descriptors, matches);

    // clear first
    this->features.clear();

#ifdef INSPECTION_MODE
    //timing
    clock_t t1 = clock();

    //timing
    //std::cout << "Features extracted in " << (double)(clock()-t1)/CLOCKS_PER_SEC;
    //std::cout << " seconds" << std::endl;
    //std::cout << "--------------------------------------" << std::endl;
#endif

    features.reserve(keyPoints.size());

    //! 1) gather allocation data for sparse representation
    for (size_t i=0; i<keyPoints.size(); ++i)
    {
        size_t curX = round(keyPoints[i].pt.x);
        size_t curY = round(keyPoints[i].pt.y);
        size_t wordId = matches[i].trainIdx;

        // insert word to multiset, compute their frequency later
        sparse->words.insert(wordId);

        // these will be used to allocate sparse matrix correctly
        this->sparse->insertX(curX);
        this->sparse->insertY(curY);

        features.push_back(Feature(wordId, curX, curY));
    }

#ifdef INSPECTION_MODE
    //timing
    //std::cout << "Features are converted to words in " << (double)(clock()-t1)/CLOCKS_PER_SEC;
    //std::cout << " seconds" << std::endl;
    //std::cout << "------------------------------------------------" << std::endl;
#endif

    this->sparse->fillWith(features);

    detector.release();
    descriptorExtractor.release();
}

/**
 * Get the depths of the features and eliminate those who are not matched.
 * Optionally, get the 3D pts of
 * with the one given to this->fromImage() method or constructor.
 *
 * Image is not saved to class by default, it will consume lots
 * of memory! This method is only used for DEBUGGING
 *
 * @param string imPath
 * @return void
 */
std::map<int, cv::Point3d> Observation::get3DPts(const cv::Mat& imL, const cv::Mat& imR, const StereoRig& rig)
{
    cv::Ptr<cv::FeatureDetector> detector = new cv::SurfFeatureDetector(config::HESSIAN_THRESHOLD);
    cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = new cv::SurfDescriptorExtractor(4, 2, true);

    std::map<int, cv::Point3d> tmpPoints3D;

    std::vector<cv::KeyPoint> keyPointsR;
    cv::Mat descriptorsR;

    // detect keypoints and extract feaures
    detector->detect(imR, keyPointsR);
    descriptorExtractor->compute(imR, keyPointsR, descriptorsR);

    // use a new matcher to match features from left im to right im
    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches, matchesCrossCheck;

    // match in both directions: left image to right vice versa
    matcher.match(descriptors, descriptorsR, matches);
    matcher.clear();
    matcher.match(descriptorsR, descriptors, matchesCrossCheck);

    std::vector<char> matchesMask(matches.size(),0);

    std::map<int,int> matchesLTR, matchesRTL;
    for (size_t i=0; i<matches.size(); ++i)
        matchesLTR[matches[i].queryIdx] = matches[i].trainIdx;

    for (size_t i=0; i<matchesCrossCheck.size(); ++i)
        matchesRTL[matchesCrossCheck[i].queryIdx] = matchesCrossCheck[i].trainIdx;

    size_t cnt=0;
    typedef std::map<int,int>::const_iterator It_ii;
    for (It_ii it=matchesLTR.begin(); it != matchesLTR.end(); ++it, cnt++)
    {
        It_ii searchResult = matchesRTL.find(it->second);
        if (matchesRTL.end() == searchResult) // continue if feat does not exist in 2nd im
            continue;
        if (searchResult->second != it->first)
            continue;

        // else
        matchesMask[cnt] = 1;
    }

    // keep only matches on the same row (epipolar constraint) and having a positive disparity value
    for (int i=matches.size()-1; i>=0; --i)
    {
        if (matchesMask[i]==0)
            continue;
        double disparity = keyPoints[matches[i].queryIdx].pt.x-keyPointsR[matches[i].trainIdx].pt.x;
        if (disparity < 0 || 10 < std::abs(keyPoints[matches[i].queryIdx].pt.y-keyPointsR[matches[i].trainIdx].pt.y))
            matchesMask[i] = 0;
        else
            tmpPoints3D[matches[i].queryIdx] = rig.get3DPoint(keyPoints[matches[i].queryIdx].pt.x,  // negate the negative sign, we want
                                                              keyPoints[matches[i].queryIdx].pt.y,  // positive depth values
                                                              disparity);
    }

#ifdef INSPECTION_MODE
    // buralar derinligi gostermek icin, gecici
    cv::Mat di = imL.clone();

    typedef std::map<int, cv::Point3d>::const_iterator It;
    for (It it=tmpPoints3D.begin(); it != tmpPoints3D.end(); ++it)
    {
        std::stringstream ss;
        ss << std::setprecision(2) << -it->second.z/1000.;
        cv::Point pt(keyPoints[it->first].pt.x, keyPoints[it->first].pt.y);
        cv::circle(di, pt, 2, cv::Scalar::all(255), 1);
        cv::putText(di, ss.str(), pt, cv::FONT_HERSHEY_PLAIN, 0.5, cv::Scalar::all(255));
    }
    cv::imshow("disparities", di);

    cv::Mat drawImg;
    cv::drawMatches(imL, keyPoints, imR, keyPointsR, matches, drawImg, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask);
    cv::imshow("matches", drawImg);
    /**/
#endif

    return tmpPoints3D;
}

/**
 * Draw features on image. The image path must be the same
 * with the one given to this->fromImage() method or constructor.
 *
 * Image is not saved to class by default, it will consume lots
 * of memory! This method is only used for DEBUGGING
 *
 * @param string imPath
 * @param Rect& r
 * @return void
 */
void Observation::drawFeaturesAndRect(cv::Mat& im, const Rect& r)
{/*
    cv::Mat newIm;
    if (im.channels() == 1)
        cv::cvtColor(im, newIm,CV_GRAY2RGB);
    else
        newIm = im.clone();
        */
#ifdef INSPECTION_MODE
    for (std::vector<Feature>::const_iterator iter = features.begin();
    iter != features.end(); ++iter)
    {
        std::stringstream ss;
        ss << std::setprecision(2) << iter->pMarg();

        cv::Point pt(iter->x, iter->y);

        cv::circle(im, pt, 2, CV_RGB(255,0,255), 1);
        //cv::putText(im, ss.str(), pt, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar::all(255));
    }
#endif
    drawRect(im, r);
    //cv::imshow("output", newIm); // cv::waitKey() must be performed after this
}

/**
 * Once xVals and yVals sets are filled we don't need these sets anymore.
 * This function does:
 * 1) We can safely convert them to a vector
 * 2) And then initialize
 *
 * @return void
 */
void Observation::SparseRep::initVars()
{
    wordFreqs.assign(Vocabulary::instance()->size(), 0);

    xValsVec.reserve(xVals.size());
    yValsVec.reserve(yVals.size());

    std::copy(xVals.begin(), xVals.end(), std::back_inserter(xValsVec));
    std::copy(yVals.begin(), yVals.end(), std::back_inserter(yValsVec));

    xVals.clear();
    yVals.clear();

    // these will be marginal probabilities so save 'em like that
    cumNumFeats = cv::Mat::zeros(yValsVec.size()+1, xValsVec.size()+1, CV_64FC1);
    cumMargProbs = cv::Mat::zeros(yValsVec.size()+1, xValsVec.size()+1, CV_64FC1);
    cumDepths = cv::Mat::zeros(yValsVec.size()+1, xValsVec.size()+1, CV_64FC1);
    cumDepthVars = cv::Mat::zeros(yValsVec.size()+1, xValsVec.size()+1, CV_64FC1);
}

/**
 * Do the following:
 * 1) Allocate space for arrays
 * 2) Fill those arrays with the provided features
 *
 * @param vector<Feature>
 * @return void
 */
void Observation::SparseRep::fillWith(const std::vector<Feature> &features)
{
    initVars(); //! 1) Allocate space for arrays

    cv::Mat tmpNumFeats(yValsVec.size(), xValsVec.size(), CV_64FC1, cv::Scalar::all(0));
    cv::Mat tmpMargProbs(yValsVec.size(), xValsVec.size(), CV_64FC1, cv::Scalar::all(0));
    cv::Mat tmpDepths(yValsVec.size(), xValsVec.size(), CV_64FC1, cv::Scalar::all(0));

    //! 2) fill sparse data
    for (std::vector<Feature>::const_iterator it = features.begin(); it != features.end(); ++it)
    {
        size_t xPos = std::distance(xValsVec.begin(), std::find(xValsVec.begin(), xValsVec.end(), it->x));
        size_t yPos = std::distance(yValsVec.begin(), std::find(yValsVec.begin(), yValsVec.end(), it->y));

        // compute word frequency, use for tf-idf
        wordFreqs[it->wordId] = (double)words.count(it->wordId)/features.size();

        tmpNumFeats.at<double>(yPos, xPos) = 1;
        tmpMargProbs.at<double>(yPos, xPos) = log(it->invFreq()*wordFreqs[it->wordId]); // tf-idf
        //tmpMargProbs.at<double>(yPos, xPos) = it->pMarg() == 0 ? 0 : log(it->pMarg());
        tmpDepths.at<double>(yPos, xPos) = it->depth;
    }

    // integral images for efficient computation
    cv::integral(tmpNumFeats, cumNumFeats);
    cv::integral(tmpMargProbs, cumMargProbs);
    cv::integral(tmpDepths, cumDepths, cumDepthVars);
}

/**
 * Return the actual coordinates of a relative rectangle
 *
 * @param const Rect& r
 * @return Rect
 */
Rect Observation::actualRect(const Rect& r) const
{
    size_t x1 = sparse->xValsVec[r.x1];
    size_t x2 = sparse->xValsVec[r.x2];
    size_t y1 = sparse->yValsVec[r.y1];
    size_t y2 = sparse->yValsVec[r.y2];

    return Rect(x1, y1, x2-x1, y2-y1);
}

void Observation::printDepths(Rect r)
{
    std::cout << "toplam feat:" <<  Observation::areaFromII<double>(sparse->cumNumFeats, r.x1, r.y1, r.x2, r.y2) << std::endl;
    int numFeats = 0;
    double totDepth = 0;
    for (int j=0; j<sparse->cumDepths.rows; ++j)
    {
        if (j<r.y1 || j>r.y2)
            continue;
        for (int i=0; i<sparse->cumDepths.cols; ++i)
        {
            if (i<r.x1 || i>r.x2)
                continue;

            if (!(i<r.x2 && i>r.x1 && j<r.y2 && j>r.y1))
                continue;

            if (1 != Observation::areaFromII<double>(sparse->cumNumFeats, i, j, i, j))
                continue;
            double depth = Observation::areaFromII<double>(sparse->cumDepths, i, j, i, j);
            totDepth += depth;
            ++numFeats;
            //std::cout << "derinlik #"<< ++numFeats <<": "<< depth << std::endl;
        }
    }
    double meanDepth = totDepth/numFeats;
    std::cout <<  "ortalama: " << meanDepth << std::endl;
    numFeats = 0;
    for (int j=0; j<sparse->cumDepths.rows; ++j)
    {
        if (j<r.y1 || j>r.y2)
            continue;
        for (int i=0; i<sparse->cumDepths.cols; ++i)
        {
            if (i<r.x1 || i>r.x2)
                continue;

            if (!(i<r.x2 && i>r.x1 && j<r.y2 && j>r.y1))
                continue;

            if (1 != Observation::areaFromII<double>(sparse->cumNumFeats, i, j, i, j))
                continue;
            double depth = Observation::areaFromII<double>(sparse->cumDepths, i, j, i, j);
            totDepth += depth;
            std::cout << "derinlik sapmasi #"<< ++numFeats <<": "<< depth-meanDepth << std::endl;
        }
    }
    std::cout << "Diyor ki; " <<  depthVar(r.x1,r.y1,r.x2,r.y2) << std::endl;
}

/**
 * Coefficients which are used in the energy function where
 * the marginality of a rectangular region is computed.
 */
/*
const double Evaluator::ALPHA = -1.0; // coef. of marginality, ~ with area of the region
const double Evaluator::BETA = -0.025;//-0.015; // coef. of area 1/~ with area
const double Evaluator::GAMMA = 0;//0.1;//0.6;    // negatif olacak, küçüldükçe alanda daha az nitelik seçilecek
const double Evaluator::DELTA = 5;//-7.5;
*/

const double Evaluator::ALPHA = -1.0; // coef. of marginality, ~ with area of the region
const double Evaluator::BETA = -0.020; // coef. of area 1/~ with area
//const double Evaluator::BETA = -0.015; // coef. of area 1/~ with area
//const double Evaluator::BETA = -0.025; // coef. of area 1/~ with area
//const double Evaluator::GAMMA = 0.0;    // coef. of number of feats, 1/~ with the number of features
//const double Evaluator::GAMMA =-0.0000000000001;    // coef. of number of feats, 1/~ with the number of features
const double Evaluator::GAMMA =-0.00000000000001;    // coef. of number of feats, 1/~ with the number of features
const double Evaluator::DELTA = 0;

//const double Evaluator::ALPHA = 1.0; // coef. of marginality, ~ with area of the region
//const double Evaluator::BETA = -0.0035; // coef. of area 1/~ with area
//const double Evaluator::GAMMA = 0;    // coef. of number of feats, 1/~ with the number of features
