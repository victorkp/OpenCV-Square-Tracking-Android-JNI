/*
 * Processor.h
 *
 * Chris McClanahan
 *
 *  modified android-opencv example
 */

#ifndef PROCESSOR_H_
#define PROCESSOR_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

#include "image_pool.h"

#include <android/log.h>
#define MSG(msg,...) do { __android_log_print(ANDROID_LOG_DEBUG, "viewercv", __FILE__":%d(%s) " msg "\n", __LINE__, __FUNCTION__, ##__VA_ARGS__); } while (0)

#define DETECT_FAST     2
#define DETECT_SQUARES  3
#define DETECT_SOBEL    4
#define DETECT_HDR      5
#define DETECT_HISTEQ   6
#define DETECT_NEON     7
#define DETECT_VIVID    8

class Processor {

    cv::FastFeatureDetector fastd;

    std::vector<cv::KeyPoint> keypoints;
    vector<vector<Point2f> > imagepoints;

    cv::Mat K;
    cv::Mat distortion;
    cv::Size imgsize;

    int _mode;
    int _take_pic;

public:

    Processor();

    virtual ~Processor();

    void detectAndDrawFeatures(int idx, image_pool* pool);

    void drawText(int idx, image_pool* pool, const char* text);

    void detectAndDrawContours(int idx, image_pool* pool);

    void runSobel(int idx, image_pool* pool);

    void runHistEq(int idx, image_pool* pool);

    void setMode(int mode);

    void snapPic();

    void runHDR(int idx, image_pool* pool, int skip);

    void saveJpg(Mat& img);

    void runNEON(int input_idx, image_pool* pool, int var);

    void runVivid(int input_idx, image_pool* pool, int var);

};

#endif /* PROCESSOR_H_ */
