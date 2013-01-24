#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

using namespace std;
using namespace cv;

extern "C" {
	JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial3_Sample3Native_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba);
	JNIEXPORT jint JNICALL Java_org_opencv_samples_tutorial3_Sample3Native_FindSquares(JNIEnv*, jobject, jlong addrRgba, jint draw);
	double angle(Point pt1, Point pt2, Point pt0);

	JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial3_Sample3Native_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba) {
		Mat& mGr = *(Mat*)addrGray;
		Mat& mRgb = *(Mat*)addrRgba;
		vector<KeyPoint> v;

		FastFeatureDetector detector(50);
		detector.detect(mGr, v);
		for( unsigned int i = 0; i < v.size(); i++ ) {
			const KeyPoint& kp = v[i];
			circle(mRgb, Point(kp.pt.x, kp.pt.y), 10, Scalar(255,0,0,255));
		}
	}



	JNIEXPORT jint JNICALL Java_org_opencv_samples_tutorial3_Sample3Native_FindSquares(JNIEnv*, jobject, jlong addrRgba, jint draw)
		{
			Mat& image = *(Mat*)addrRgba;

			int thresh = 50, N = 4;
			int found = 0;

			Mat pyr, timg, gray0(image.size(), CV_8U), gray;

			// down-scale and upscale the image to filter out the noise
			pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
			pyrUp(pyr, timg, image.size());
			vector<vector<Point> > contours;

			// find squares in every color plane of the image
			for( int c = 1; c < 3; c++ )
			{

				int ch[] = {c, 0};
				mixChannels(&timg, 1, &gray0, 1, ch, 1);

				// try several threshold levels
				for( int l = 0; l < N; l++ )
				{
					// hack: use Canny instead of zero threshold level.
					// Canny helps to catch squares with gradient shading
					if( l == 0 )
					{
						// apply Canny. Take the upper threshold from slider
						// and set the lower to 0 (which forces edges merging)
						Canny(gray0, gray, 0, thresh, 5);
						// dilate canny output to remove potential
						// holes between edge segments
						dilate(gray, gray, Mat(), Point(-1,-1));
					}
					else
					{
						// apply threshold if l!=0:
						//     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
						gray = gray0 >= (l+1)*255/N;
					}

					// find contours and store them all as a list
					findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

					vector<Point> approx;

					// test each contour
					for( size_t i = 0; i < contours.size(); i++ )
					{
						// approximate contour with accuracy proportional
						// to the contour perimeter
						approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

						// square contours should have 4 vertices after approximation
						// relatively large area (to filter out noisy contours)
						// and be convex.
						// Note: absolute value of an area is used because
						// area may be positive or negative - in accordance with the
						// contour orientation
						if( approx.size() == 4 &&
							fabs(contourArea(Mat(approx))) > 1000 &&
							isContourConvex(Mat(approx)) )
						{
							double maxCosine = 0;

							for( int j = 2; j < 5; j++ )
							{
								// find the maximum cosine of the angle between joint edges
								double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
								maxCosine = MAX(maxCosine, cosine);
							}

							// if cosines of all angles are small
							// (all angles are ~90 degree) then write quandrange
							// vertices to resultant sequence
							if( maxCosine < 0.3 ){
								/*
								circle(image, approx[0], 10, Scalar(255,0,0,255));
								circle(image, approx[1], 10, Scalar(255,0,0,255));
								circle(image, approx[2], 10, Scalar(255,0,0,255));
								circle(image, approx[3], 10, Scalar(255,0,0,255));
								rectangle(image, approx[0], approx[2], Scalar(0,255,0,255), 5, 4, 0);
								*/

								//Center of this rectangle
								int x = (int) ((approx[0].x + approx[1].x + approx[2].x + approx[3].x)/4.0);
								int y = (int) ((approx[0].y + approx[1].y + approx[2].y + approx[3].y)/4.0);

								if((int)draw){
									//outline
									line(image, approx[0], approx[1], Scalar(0,255,0,255), 2, 4, 0);
									line(image, approx[1], approx[2], Scalar(0,255,0,255), 2, 4, 0);
									line(image, approx[2], approx[3], Scalar(0,255,0,255), 2, 4, 0);
									line(image, approx[3], approx[0], Scalar(0,255,0,255), 2, 4, 0);
									//center
									circle(image, Point(x,y), 1, Scalar(255,0,0,255));
								}

								found = 1;
								jint result = (jint) found;
								return result;
							}
						}
					}
				}
			}
			jint result = (jint) found;
			return result;
		}



	double angle(Point pt1, Point pt2, Point pt0) {
		double dx1 = pt1.x - pt0.x;
		double dy1 = pt1.y - pt0.y;
		double dx2 = pt2.x - pt0.x;
		double dy2 = pt2.y - pt0.y;
		return (dx1 * dx2 + dy1 * dy2)
				/ sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
	}

}
