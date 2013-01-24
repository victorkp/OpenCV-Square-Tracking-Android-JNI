/*
 * Processor.cpp
 *
 * Chris McClanahan
 *
 *  highly modified android-opencv example
 */

#include "Processor.h"
#include <sys/stat.h>

using namespace cv;

int __mode = 0; // hack
int __take_pic = 0; // hack


Processor::Processor() :
    fastd(20 /*threshold*/, true /*nonmax_suppression*/), _mode(0), _take_pic(0) {
    __mode = _mode;
    __take_pic = _take_pic;
}

Processor::~Processor() {

}

void Processor::saveJpg(Mat& img) {
    if (_take_pic) {
        const char* p = "/sdcard/ViewerCV/img_";
        char f[128];
        sprintf(f, "%s%d%d.jpg", p, getCPUTickCount(), getTickCount());
        MSG("%s", f);
        imwrite(f, img);
        _take_pic = 0;
    }
}

// entry point
void Processor::detectAndDrawFeatures(int input_idx, image_pool* pool) {
    FeatureDetector* fd = 0;
    fd = &fastd;


    Mat greyimage;
    pool->getGrey(input_idx, greyimage);
    Mat* img = pool->getImage(input_idx);
    if (!img || greyimage.empty() || fd == 0) {
        return;
    }

    keypoints.clear();
    fd->detect(greyimage, keypoints);

    for (vector<KeyPoint>::const_iterator it = keypoints.begin();
            it	!= keypoints.end(); ++it) {
        circle(*img, it->pt, 3, cvScalar(255, 0, 255, 0));
    }

    saveJpg(*img);
}

void Processor::drawText(int i, image_pool* pool, const char* ctext) {
    string text = ctext;
    int fontFace = FONT_HERSHEY_COMPLEX_SMALL;
    double fontScale = .8;
    int thickness = .5;

    Mat img = *pool->getImage(i);

    int baseline = 0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;

    // center the text
    Point textOrg((img.cols - textSize.width) / 2, (img.rows - textSize.height * 2));

    // draw the box
    rectangle(img, textOrg + Point(0, baseline), textOrg +
              Point(textSize.width, -textSize.height), Scalar(0, 0, 255), CV_FILLED);
    // ... and the baseline first
    line(img, textOrg + Point(0, thickness), textOrg +
         Point(textSize.width, thickness), Scalar(0, 0, 255));

    // then put the text itself
    putText(img, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
}

/* =================================================================== */
int THRESH = 4;
int MEAN = 5;

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
float angle(Point pt1, Point pt2, Point pt0) {
    float dx1 = pt1.x - pt0.x;
    float dy1 = pt1.y - pt0.y;
    float dx2 = pt2.x - pt0.x;
    float dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) /
           sqrtf((dx1 * dx1 + dy1 * dy1) *
                 (dx2 * dx2 + dy2 * dy2) + 1e-9);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
void findSquares(const Mat& image, vector<vector<Point> >& squares) {
    Mat pyr, timg, gray0(image.size(), CV_8U), gray;
    vector < vector<Point> > contours;
    int maxcontourlen = image.cols * image.rows / 2;
    squares.clear();

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols / 2, image.rows / 2));
    pyrUp(pyr, timg, image.size());

#if 0
    // find squares in every color plane of the image
    for (int c = 0; c < 3; ++c) {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);
#else
    {
        cvtColor(timg, gray0, CV_RGB2GRAY);
#endif

        //		imshow("gray", gray0);

        // auto adapt to average pixel value
        Scalar mean = cv::mean(gray0);
        MEAN = mean[0];
        float meanlo = 0.33f * mean[0];
        float meanmd = 0.66f * mean[0];
        float meanhi = 1.25f * mean[0];

        // lower/raise thresholds
        switch (__mode) {
        case 0:
            break;
        case 1:
            meanlo *= 0.8f;
            meanmd *= 0.8f;
            meanhi *= 0.8f;
            break;
        case 2:
            meanlo *= 1.1f;
            meanmd *= 1.1f;
            meanhi *= 1.1f;
            break;
        default:
            break;
        }
        meanhi = meanhi > 255 ? 255 : meanhi;

        // try several threshold levels
        //        int level = THRESH; {
        for (int level = 0; level < 5; ++level) {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            switch (level) {
            case 0:
                // apply Canny. Take the upper threshold and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, MEAN, 5);
                // dilate canny output to remove potential holes between edge segments
                dilate(gray, gray, Mat(), Point(-1, -1));
                //				imshow("gray0", gray);
                break;
            case 1:
                gray = gray0 >= (int) meanlo;
                //				imshow("gray1", gray);
                break;
            case 2:
                // gray = gray0 >= (int) MEAN;
                adaptiveThreshold(gray0, gray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 0);
                erode(gray, gray, Mat(), Point(-1, -1), 1);
                dilate(gray, gray, Mat(), Point(-1, -1), 1);
                //				imshow("gray2", gray);
                break;
            case 3:
                gray = gray0 >= (int) meanhi;
                //				imshow("gray3", gray);
                break;
            case 4:
                gray = gray0 >= (int) meanmd;
                //			 	imshow("gray4", gray);
                break;
            default:
                break;
            }

            // find contours and store them all as a list
            findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
            vector<Point> approx;

            // test each contour
            for (size_t i = 0; i < contours.size(); ++i) {
                // approximate contour with accuracy proportional to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(
                                 contours[i]), true) * 0.025f, true);

                // square contours should have:
                //  4 vertices after approximation
                //  relatively large area (to filter out noisy contours)
                //  relatively not too large (to filter out screen edge)
                //  and be convex.
                float contour_area = fabs(contourArea(Mat(approx)));
                if ((approx.size() == 4) && contour_area > 100 &&
                        contour_area < maxcontourlen && isContourConvex(Mat(approx))) {
                    float maxCosine = 0;

                    for (int j = 2; j < 5; ++j) {
                        // find the maximum cosine of the angle between joint edges
                        float cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small (all angles are ~90 degree)
                    //   then save vertices to resultant sequence
                    if (maxCosine < 0.3) {
                        squares.push_back(approx);
                    }
                }
            }
        }
    }
}

// the function draws all the squares in the image
void draw_contours(Mat& image, const vector<vector<Point> >& squares) {
    for (size_t i = 0; i < squares.size(); ++i) {
        const Point* p = &squares[i][0];
        int n = (int) squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0, 255, 0), 3, CV_AA);
    }
}

/* =================================================================== */

// entry point
void Processor::detectAndDrawContours(int input_idx, image_pool* pool) {
    Mat* img;
    img = pool->getImage(input_idx);
    if (!img) {
        return;
    }

    vector < vector<Point> > squares;
    findSquares(*img, squares);
    draw_contours(*img, squares);

    saveJpg(*img);
}

/* Adapted from:                                                            */
/*   TEXAS INSTRUMENTS, INC.                                                */
/*   IMGLIB  DSP Image/Video Processing Library                             */
void _sobelFilter(uchar* input, int width, int height, uchar* output, int negative) {
    int H, O, V, i;
    uchar i00, i01, i02;
    uchar i10,      i12;
    uchar i20, i21, i22;
    int w = width;
    int numpx = width * (height - 1);
    for (i = 0; i < numpx - 1; ++i) {
        i00 = input[i    ] ; i01 = input[i    +1] ; i02 = input[i    +2] ;
        i10 = input[i+  w] ;                        i12 = input[i+  w+2] ;
        i20 = input[i+2*w] ; i21 = input[i+2*w+1] ; i22 = input[i+2*w+2] ;
        H = -  i00 - 2 * i01 -  i02 +
            +  i20 + 2 * i21 +  i22;
        V = -     i00  +     i02
            - 2 * i10  + 2 * i12
            -     i20  +     i22;
        O = abs(H) + abs(V);
        if (O > 255) { O = 255; }
        if (negative) { O = 255 - O; }
        output[i + 1] = (uchar) O;
    }
}

// entry point
void Processor::runSobel(int input_idx, image_pool* pool) {
    Mat gray;
    pool->getGrey(input_idx, gray);
    Mat* img;
    img = pool->getImage(input_idx);

    if (!img) {
        return;
    }

    Mat temp(img->size(), CV_8UC1);
    uchar* in = (uchar*) gray.data;
    uchar* out = (uchar*) temp.data;
    _sobelFilter(in, img->cols, img->rows, out, _mode);
    cvtColor(temp, *img, CV_GRAY2RGB); // 3 channel for display

    saveJpg(*img);
}

// entry point
void Processor::runHistEq(int input_idx, image_pool* pool) {
    Mat gray;
    pool->getGrey(input_idx, gray);
    Mat* img;
    img = pool->getImage(input_idx);

    if (!img) {
        return;
    }

    if (_mode == 0) { equalizeHist(gray, gray); }
    cvtColor(gray, *img, CV_GRAY2RGB); // 3 channel for display

    saveJpg(*img);
}


// entry point
void Processor::setMode(int mode) {
    _mode = mode;
    __mode = mode; // hack
}

// entry point
void Processor::snapPic() {
    _take_pic = (_take_pic == 0);
    __take_pic = (__take_pic == 0); // hack
}

/**
 * Tone Mapping
 */
#include "tmo.h"
int imgcnt = 0;
Mat imgbuff[3];
// entry point
void Processor::runHDR(int input_idx, image_pool* pool, int skip) {
    Mat* img;
    img = pool->getImage(input_idx);
    if (!img) {
        return;
    }

    if (skip) {
        img = NULL; // dont refresh screen
        pool->addImage(0, img);
        return;
    }

    imgbuff[imgcnt] = Mat::zeros(img->size().height , img->size().width , CV_32FC3);
    img->convertTo(imgbuff[imgcnt], CV_32FC3); // queue up 3 images

    ++imgcnt;

    if (imgcnt > 2) {
        imgcnt = 0;

        Mat hdr = Mat::zeros(img->size(), CV_32FC3);
#if 2
        // 3 hdr
        makehdr3log(&(imgbuff[0]), &(imgbuff[1]), &(imgbuff[2]), &(hdr));
#else
        // 2 hdr
        makehdr2log(&(imgbuff[0]), &(imgbuff[2]), &(hdr));
#endif
        imgbuff[0].release();
        imgbuff[1].release();
        imgbuff[2].release();

        /////////////////////////////////////////////////////////////
        // tonemapping
        /////////////////////////////////////////////////////////////
        if (_mode == 2) {

            Mat luv(hdr.rows, hdr.cols, CV_32FC3);
            cvtColor(hdr, luv, CV_RGB2YCrCb);

            vector<Mat> lplanes;
            split(luv, lplanes);

            Mat Y(hdr.rows, hdr.cols, CV_32FC1);
            lplanes[0].convertTo(Y, CV_32FC1);

            vector<Mat> cplanes;
            split(hdr, cplanes);
            Mat R(hdr.rows, hdr.cols, CV_32FC1);
            cplanes[0].convertTo(R, CV_32FC1);
            Mat G(hdr.rows, hdr.cols, CV_32FC1);
            cplanes[1].convertTo(G, CV_32FC1);
            Mat B(hdr.rows, hdr.cols, CV_32FC1);
            cplanes[2].convertTo(B, CV_32FC1);

            // tonemapping params
            bool bcg = false;
            int itmax = 200; // higher = looks better but runs slower
            float tol = 2e-3;
            int cols = hdr.cols;
            int rows = hdr.rows;
            float contrast = (_mode != 2) ? -0.20 : 0.20; // contrast control
            float saturation = 1.1; // color control
            float detail = 2; // texture control

            float* fR = (float*) R.data;
            float* fG = (float*) G.data;
            float* fB = (float*) B.data;
            float* fY = (float*) Y.data;

            MSG("tonemapping...");

            tmo_mantiuk06_contmap(cols, rows, fR, fG, fB, fY,
                                  contrast, saturation, detail, bcg, itmax, tol);

            // combine channels
            Mat rgb[] = { R, G, B };
            merge(rgb, 3, hdr);
            hdr *= 245;

            MSG("done.");
            R.release();
            G.release();
            B.release();

            hdr.convertTo(*img, img->type()); // display hdr

        } else if (_mode == 1) {

            float _exposure = 1.0f;
            float _tmo_sval = 1.1f * _mode;

            //hdr /= 255.f;
            //pow(hdr,_exposure, hdr);// * 0.7f + 0.15f;
            //hdr *= 255.f;
            cv::exp(hdr, hdr);

            Mat xyz(hdr.rows, hdr.cols, CV_32FC3);
            cvtColor(hdr, xyz, CV_RGB2XYZ);

            vector<Mat> lplanes;
            split(xyz, lplanes);
            Mat X(hdr.rows, hdr.cols, CV_32FC1);
            Mat Y(hdr.rows, hdr.cols, CV_32FC1);
            Mat Z(hdr.rows, hdr.cols, CV_32FC1);
            lplanes[0].convertTo(X, CV_32FC1);
            lplanes[1].convertTo(Y, CV_32FC1);
            lplanes[2].convertTo(Z, CV_32FC1);
            Mat localcontrast(hdr.rows, hdr.cols, CV_32FC1);

            // blur-scale Y channel
            Mat imgY = Y;
            Mat blurredY; double sigmaY = 1;
            bilateralFilter(imgY, blurredY, 0, 0.1 * 255.f, 2);
            GaussianBlur(blurredY, blurredY, Size(), sigmaY, sigmaY);
            divide(imgY, blurredY, localcontrast);

            Mat scale(hdr.rows, hdr.cols, CV_32FC1);
            pow(localcontrast, _tmo_sval, scale);
            multiply(X, scale, X);
            multiply(Y, scale, Y);
            multiply(Z, scale, Z);

            Mat rgb[] = {X, Y, Z};
            merge(rgb, 3, hdr);
            cvtColor(hdr, hdr, CV_XYZ2RGB);

            // sharpen image using "unsharp mask" algorithm
            Mat temp = hdr;
            Mat blurred; double sigma = 7, threshold = 0, amount = 0.3f;
            GaussianBlur(temp, blurred, Size(), sigma, sigma);
            Mat sharpened = temp * (1 + amount) + blurred * (-amount);
            // Mat lowContrastMask = abs(temp - blurred) < threshold;
            Mat diff = abs(temp - blurred);
            Mat lowContrastMask; inRange(diff, -255, -1, lowContrastMask);
            temp.copyTo(sharpened, lowContrastMask);
            hdr = sharpened;

            hdr.convertTo(*img, img->type()); // display hdr

        }  else {

            cv::exp(hdr, hdr);
            Mat xyz(hdr.rows, hdr.cols, CV_32FC3);
            cvtColor(hdr, xyz, CV_RGB2XYZ);

            vector<Mat> lplanes;
            split(xyz, lplanes);
            Mat X(hdr.rows, hdr.cols, CV_32FC1);
            Mat Y(hdr.rows, hdr.cols, CV_32FC1);
            Mat Z(hdr.rows, hdr.cols, CV_32FC1);
            lplanes[0].convertTo(X, CV_32FC1);
            lplanes[1].convertTo(Y, CV_32FC1);
            lplanes[2].convertTo(Z, CV_32FC1);

            float bias = 0.975;  // 0.85;
            int width = hdr.cols;
            int height = hdr.rows;
            Mat L(hdr.rows, hdr.cols, CV_32FC1);
            float* fY = (float*)Y.data;
            float* fL = (float*)L.data;

            MSG("tonemapping...");

            tmo_drago03(width, height,
                        fY,
                        fL,
                        bias);

            Mat scale(hdr.rows, hdr.cols, CV_32FC1);
            divide(L, Y, scale);
            multiply(X, scale, X);
            multiply(Y, scale, Y);
            multiply(Z, scale, Z);

            Mat rgb[] = {X, Y, Z};
            merge(rgb, 3, hdr);
            hdr = hdr * 245;
            cv::min(hdr, 255, hdr);

            MSG("done.");
            X.release();
            Y.release();
            Z.release();

            hdr.convertTo(*img, img->type());
        }

    } else {
        img = NULL; // don't refresh screen
        pool->addImage(0, img);
    }

    saveJpg(*img);
}


void Processor::runNEON(int input_idx, image_pool* pool, int var) {

    Mat* img;
    img = pool->getImage(input_idx);

    if (!img) {
        return;
    }

    Mat temp = *img;
    cv::morphologyEx(temp, temp, cv::MORPH_GRADIENT, cv::Mat());
    temp *= (_mode + 1);

    temp.convertTo(*img, img->type());
    saveJpg(*img);
}


void Processor::runVivid(int input_idx, image_pool* pool, int var) {

    Mat* img;
    img = pool->getImage(input_idx);

    if (!img) {
        return;
    }

    if (_mode == 1) {
        Mat hdr = *img;

        float _exposure = 1.0f;
        float _tmo_sval = 1.1f * _mode;

        //hdr /= 255.f;
        //pow(hdr,_exposure, hdr);// * 0.7f + 0.15f;
        //hdr *= 255.f;
        //cv::exp(hdr, hdr);

        Mat xyz(hdr.rows, hdr.cols, CV_32FC3);
        cvtColor(hdr, xyz, CV_RGB2XYZ);

        vector<Mat> lplanes;
        split(xyz, lplanes);
        Mat X(hdr.rows, hdr.cols, CV_32FC1);
        Mat Y(hdr.rows, hdr.cols, CV_32FC1);
        Mat Z(hdr.rows, hdr.cols, CV_32FC1);
        lplanes[0].convertTo(X, CV_32FC1);
        lplanes[1].convertTo(Y, CV_32FC1);
        lplanes[2].convertTo(Z, CV_32FC1);
        Mat localcontrast(hdr.rows, hdr.cols, CV_32FC1);

        // blur-scale Y channel
        Mat imgY = Y;
        Mat blurredY; double sigmaY = 1;
        bilateralFilter(imgY, blurredY, 0, 0.1 * 255.f, 2);
        GaussianBlur(blurredY, blurredY, Size(), sigmaY, sigmaY);
        divide(imgY, blurredY, localcontrast);

        Mat scale(hdr.rows, hdr.cols, CV_32FC1);
        pow(localcontrast, _tmo_sval, scale);
        multiply(X, scale, X);
        multiply(Y, scale, Y);
        multiply(Z, scale, Z);

        Mat rgb[] = {X, Y, Z};
        merge(rgb, 3, hdr);
        cvtColor(hdr, hdr, CV_XYZ2RGB);

        // sharpen image using "unsharp mask" algorithm
        Mat temp = hdr;
        Mat blurred; double sigma = 7, threshold = 0, amount = 0.3f;
        GaussianBlur(temp, blurred, Size(), sigma, sigma);
        Mat sharpened = temp * (1 + amount) + blurred * (-amount);
        // Mat lowContrastMask = abs(temp - blurred) < threshold;
        Mat diff = abs(temp - blurred);
        Mat lowContrastMask; inRange(diff, -255, -1, lowContrastMask);
        temp.copyTo(sharpened, lowContrastMask);
        hdr = sharpened;

        hdr.convertTo(*img, img->type()); // display hdr
    } else {

        Mat temp = *img;
        Mat blurred; double sigma = 7, threshold = 0, amount = (_mode + 1);
        GaussianBlur(temp, blurred, Size(), sigma, sigma);
        Mat sharpened = temp * (1 + amount) + blurred * (-amount);
        Mat diff = abs(temp - blurred);
        Mat lowContrastMask; inRange(diff, -255, -1, lowContrastMask);
        temp.copyTo(sharpened, lowContrastMask);
        temp = sharpened;

        temp.convertTo(*img, img->type());

    }

    saveJpg(*img);
}



