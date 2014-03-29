// vocabulary.cpp: Definition of Vocabulary class (i.e. codebook of FAB-MAP)
#include "vocabulary.h"

Vocabulary *Vocabulary::vocabInstance = 0;

/**
 * Constructor, load vocabulary and marginals
 */
Vocabulary::Vocabulary(const std::string& vocabPath, const std::string& margsPath)
{
    loadVocab(vocabPath);
    loadMargs(margsPath);
}

bool Vocabulary::writeCvStyle(const std::string &filename, const cv::Mat &vocabulary)
{
    std::cout << "Saving vocabulary..." << std::endl;
    cv::FileStorage fs( filename, cv::FileStorage::WRITE );
    if( fs.isOpened() )
    {
        fs << "vocabulary" << vocabulary;
        return true;
    }
    return false;
}

/**
 * Macro for singleton instance();
 *
 * @return Vocabulary*
 */
Vocabulary *Vocabulary::instance()
{
    if (!vocabInstance)
        vocabInstance = new Vocabulary(resources::vocabPath, resources::margsPath);

    return vocabInstance;
}

/**
 * Load vocabulary, get the id of each word and its descriptor
 *
 * vocabulary is Oxford style, it is actually the vocabulary of Cummins
 * in Fab-Map, consisting of 11000 words of 128 dim. features
 *
 * @param string vocabPath
 * @return void
 */
void Vocabulary::loadVocab(const std::string &vocabPath)
{
    std::ifstream vocabFile(vocabPath.c_str());

    if (!vocabFile.is_open())
        throw std::invalid_argument("Vocabulary file could not be opened!");

    std::string strSize;
    getline(vocabFile, strSize); // first line of the file is vocab size: "WORDS:XXXXX\n"


    // chomp string to read vocab size
    _size = atoi(strSize.substr(strSize.find(':')+1, strSize.size()-strSize.find(':')-2).c_str());

    std::string tmpString;
    char tmpChar;

    // skip second line in vocab file
    getline(vocabFile, tmpString);

    words = cv::Mat::zeros(this->size(), config::DESC_SIZE, CV_32F);

/**/
    // read each word and save it along with its descriptor
    for (size_t j=0; j<this->size() && vocabFile.good(); ++j)
    {
        getline(vocabFile, tmpString); // skip label ("WORD:XXXXX\n")

        // read descriptor
        float d;
        for (size_t i=0; i<config::DESC_SIZE; ++i)
        {
            vocabFile >> d;
            words.at<float>(j,i) = d; // Fill words
        }

        vocabFile >> tmpChar; // skip '\n'
    }

    vocabFile.close();

    // add all words and train vocabulary
    matcher.add(std::vector<cv::Mat>(1, words));
    matcher.train();
}

/**
 * Load marginal probability of the words in vocabulary.
 * This file is provided by Cummins' FabMap too.
 *
 * @param string
 * @return void
 */
void Vocabulary::loadMargs(const std::string &margsPath)
{
    std::ifstream margsFile(margsPath.c_str());

    if ( !margsFile.is_open() )
        throw std::invalid_argument("Marginals file could not be opened!");

    float d;
    char tmpChar; // used to skip '\n'

    marginals.reserve(this->size());
    invFreqs.reserve(this->size()); // use for tf-idf score

    // Read marginal probability of each word
    for (size_t i=0; i<this->size(); i++)
    {
        margsFile >> d >> tmpChar;
        marginals.push_back(d);
        invFreqs.push_back(log(1./d));
    }

    margsFile.close();
}
