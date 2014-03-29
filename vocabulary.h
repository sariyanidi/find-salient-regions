// vocabulary.h: Definition of Vocabulary class (i.e. codebook of FAB-MAP)
#ifndef VOCABULARY_H
#define VOCABULARY_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <exception>
#include <stdexcept>
#include "opencv2/features2d/features2d.hpp"
#include "definitions.h"

/**
 * Class where words are stored along with their
 * 1) IDs
 * 2) Marginal Probabilities
 * 3) Descriptors
 *
 * There are also helper methods to find out to which word corresponds
 * a feature etc.
 *
 * @todo inline singleton method instance()
 */
class Vocabulary
{
public:
    // singleton instance
    static Vocabulary* vocabInstance;

    // singleton instance
    static Vocabulary *instance();

    // construct a vocabulary through words along with their marg. probabilities
    Vocabulary(const std::string& vocabPath, const std::string& margsPath);

    // macro returning vocab size
    size_t size() const { return _size; }

    // marginal probability of a feature
    double pMarg(size_t wordId) const { return marginals[wordId]; }

    // inverse frequency of a word, use for tf-idf
    double invFreq(size_t wordId) const { return invFreqs[wordId]; }

    // translate all features to vocabulary words
    void translateAll(const cv::Mat& descriptors, std::vector<cv::DMatch>& matches) { matcher.match(descriptors, words, matches); /*matcher.clear();*/ }

    // save vocabulary as an opencv file
    bool writeCvStyle( const std::string& filename, const cv::Mat& vocabulary );

private:
    // load vocabulary
    void loadVocab(const std::string& vocabPath);

    // load marginal probabilities
    void loadMargs(const std::string& margsPath);

    // store marginal probabilities
    std::vector<double> marginals;

    // store inverse frequencies for tf-idf
    std::vector<double> invFreqs;

    // number of words
    size_t _size;

    // descriptors of words
    cv::Mat words;

    // the matcher, this will cluster words
    cv::FlannBasedMatcher matcher;
};

/**
 * Hold a single featrue along with its coordinates wihtin the image and
 * the corresponding word.
 *
 * Use Vocabulary to get the marginal probability of this feature
 */
class Feature
{
public:
    // id of corresponding word
    size_t wordId;

    // x and y coordinates of this feature
    size_t x, y;

    // depth data of each feature, how far is it from camera
    double depth;

    // default constructor
    Feature() {}

    // constructor
    Feature(const size_t _wordId, const size_t _x, const size_t _y, const double _depth = -1 )
       : wordId(_wordId), x(_x), y(_y), depth(_depth) {}

    // marginal probability of this word
    double pMarg() const { return Vocabulary::instance()->pMarg(wordId); }

    // inverse probability of word, needed for tf-idf
    double invFreq() const { return Vocabulary::instance()->invFreq(wordId); }
};
#endif // VOCABULARY_H
