#-------------------------------------------------
#
# Project created by QtCreator 2011-03-23T21:53:57
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = FindMarginalRegions
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    vocabulary.cpp \
    observation.cpp \
    patchextractor.cpp \
    definitions.cpp \
    stereo.cpp \
    odometry.cpp \
    navigator.cpp \
    tracker.cpp \
    landmark.cpp


HEADERS += \
    vocabulary.h \
    definitions.h \
    observation.h \
    patchextractor.h \
    stereo.h \
    odometry.h \
    navigator.h \
    tracker.h \
    landmark.h


INCLUDEPATH  = /usr/local/include/opencv  # make sure that all folders and libraries
INCLUDEPATH += /home/vangelrobot/git/findmarginalregions/ann/include  # are located correctly

LIBS += -lopencv_core -lopencv_highgui -lopencv_features2d -lopencv_calib3d -lopencv_video -lopencv_flann -lopencv_imgproc
