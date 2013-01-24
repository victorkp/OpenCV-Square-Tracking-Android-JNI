package org.opencv.samples.tutorial3;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

public class Sample3Native extends Activity implements CvCameraViewListener {
	private static final String TAG = "OCVSample::Activity";

	private Mat mRgba;
	private Mat mGrayMat;
	private CameraBridgeViewBase mOpenCvCameraView;
	private int mDraw = 1;

	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
				Log.i(TAG, "OpenCV loaded successfully");

				// Load native library after(!) OpenCV initialization
				System.loadLibrary("native_sample");

				mOpenCvCameraView.enableView();
			}
				break;
			default: {
				super.onManagerConnected(status);
			}
				break;
			}
		}
	};

	public Sample3Native() {
		Log.i(TAG, "Instantiated new " + this.getClass());
	}

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		Log.i(TAG, "called onCreate");
		super.onCreate(savedInstanceState);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		setContentView(R.layout.tutorial3_surface_view);

		mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial4_activity_surface_view);
		mOpenCvCameraView.setCvCameraViewListener(this);
	}

	@Override
	public void onPause() {
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
		super.onPause();
	}

	@Override
	public void onResume() {
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this,
				mLoaderCallback);
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		menu.add("Draw Outlines");
		return true;
	}

	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		if (mDraw == 1) {
			mDraw = 0;
		} else {
			mDraw = 1;
		}
		return true;
	}

	public void onDestroy() {
		super.onDestroy();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	public void onCameraViewStarted(int width, int height) {
		mRgba = new Mat(height, width, CvType.CV_8UC4);
		mGrayMat = new Mat(height, width, CvType.CV_8UC1);
	}

	public void onCameraViewStopped() {
		mRgba.release();
		mGrayMat.release();
	}

	public Mat onCameraFrame(Mat inputFrame) {
		long start = System.currentTimeMillis();
		
		Size originalSize = inputFrame.size();

		Imgproc.resize(inputFrame, mRgba, new Size(320, 180));
		FindSquares(mRgba.getNativeObjAddr(), mDraw);

		if (mDraw == 1) {
			Imgproc.resize(mRgba, inputFrame, originalSize);
		}
		
		long end = System.currentTimeMillis();
		Log.d("Frame time", "" + (end - start) + " ms");

		return inputFrame;
	}

	public native void FindFeatures(long matAddrGr, long matAddrRgba);

	public native int FindSquares(long matAddrRgba, int draw);

}
