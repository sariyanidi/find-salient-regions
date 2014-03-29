======================================
    Indoor Vocabulary for FAB-MAP
======================================

To use this vocabulary, just drop in beside the exisiting vocabulary in 

(FAB_MAP_BASE)/Resources/Vocabularies

Then, change WordMaker_Config.moos and FabMap_Config.moos
so that "VocabPath" and "VocabName" specify this new vocabulary.

In WordMaker_Config.moos, you will also need to change SURFThres to
  SURFThres = 4.0
This lowers the blob-response threshold at which SURF detects keypoints. The deafult outdoor value doesn't tend to produce enough keypoints indoors.

If you've already generated Surf and Words files for a set of images, you'll need to regenerate them with the new SURFThresh and the new vocabulary.

======================================
Last updated by mjc on 2008/6/20 