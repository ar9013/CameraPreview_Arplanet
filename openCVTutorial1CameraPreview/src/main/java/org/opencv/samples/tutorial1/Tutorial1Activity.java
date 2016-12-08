package org.opencv.samples.tutorial1;

import java.io.IOException;

import javax.crypto.spec.GCMParameterSpec;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.TextView;



public class Tutorial1Activity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "OCVSample::Activity";
    
	// FlagDraw
	private boolean flagDraw =false;
	private String FLAGDRAW = "FLAGDRAW";

    private CameraBridgeViewBase mOpenCvCameraView;
    private TextView imgTilte, imgDisp;
    
    
    private boolean              mIsJavaCamera = true;
    private MenuItem             mItemSwitchCamera = null;
    
 // A key for storing the index of the active image size.
 	private static final String STATE_IMAGE_SIZE_INDEX = "imageSizeIndex";

 	// Keys for storing the indices of the active filters.
 	private static final String STATE_IMAGE_DETECTION_FILTER_INDEX = "imageDetectionFilterIndex";
    
 // The filters.
 	private ImageDetectionFilter[] mImageDetectionFilters;
 	
 	// The indices of the active filters.
 	private int mImageDetectionFilterIndex;
 	
 	// Target found index.
 	private int foundTargetIndex;
 	
 // The index of the active image size.
 	private int mImgSizeIndex;

 	 
 	

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                
                	
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    
                    final ImageDetectionFilter chengpo;
    				try {
    					chengpo = new ImageDetectionFilter(Tutorial1Activity.this,
    							R.drawable.chengpo);
    				} catch (IOException e) {
    					Log.e(TAG, "Failed to load drawable: " + "chengpo");
    					e.printStackTrace();
    					break;
    				}
                    
    				final ImageDetectionFilter chiayi;
    				try {
    					chiayi = new ImageDetectionFilter(
    							Tutorial1Activity.this,
    							R.drawable.chiayi);
    				} catch (IOException e) {
    					Log.e(TAG, "Failed to load drawable: "
    							+ "chiayi");
    					e.printStackTrace();
    					break;
    				}

    				final ImageDetectionFilter summer_street;
    				try {
    					summer_street = new ImageDetectionFilter(
    							Tutorial1Activity.this,
    							R.drawable.summer_street);
    				} catch (IOException e) {
    					Log.e(TAG, "Failed to load drawable: "
    							+ "summer_street");
    					e.printStackTrace();
    					break;
    				}
    		
    				mImageDetectionFilters = new ImageDetectionFilter[] {summer_street, chengpo, chiayi };
    				  
                 break;
                 
                 
                default:
                
                    super.onManagerConnected(status);
                break;
            }
        }
    };

    public Tutorial1Activity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        if (savedInstanceState != null) {
			
			mImageDetectionFilterIndex = savedInstanceState.getInt(
					STATE_IMAGE_DETECTION_FILTER_INDEX, 0);
			mImgSizeIndex = savedInstanceState
					.getInt(STATE_IMAGE_SIZE_INDEX, 0);
			
		} else {
			
			mImgSizeIndex = 0;
			mImageDetectionFilterIndex = 0;
			
		}
        
        setContentView(R.layout.tutorial1_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_java_surface_view);
        
        imgTilte = (TextView) findViewById(R.id.imgTitle);
        imgDisp = (TextView) findViewById(R.id.imgDisp);

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setMaxFrameSize(320, 240);

        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_4, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
    	final Mat rgba = inputFrame.rgba();

		// Apply the active filters.
    	// 透過 for 迴圈一直套用 , 一直找直到 flagDraw 為 true 停止
   for(mImageDetectionFilterIndex = 0 ; mImageDetectionFilterIndex < mImageDetectionFilters.length;mImageDetectionFilterIndex++){
    		
    		if(mImageDetectionFilterIndex == mImageDetectionFilters.length){
    			mImageDetectionFilterIndex = 0;
   		}

    		mImageDetectionFilters[mImageDetectionFilterIndex].apply(rgba, rgba);	
        	flagDraw = mImageDetectionFilters[mImageDetectionFilterIndex].isTargetFound();
        	
        	Log.d(FLAGDRAW, "flagDraw : "+ mImageDetectionFilterIndex);
        	Log.d(FLAGDRAW, "flagDraw : "+ flagDraw);
        	

        		if(flagDraw){
        		    
        			foundTargetIndex = mImageDetectionFilterIndex;
        			//mImageDetectionFilters[foundTargetIndex].apply(rgba, rgba);	
            		//flagDraw = mImageDetectionFilters[mImageDetectionFilterIndex].targetFound();
            		
            		Log.d(FLAGDRAW, "!!!!flagDraw : "+ foundTargetIndex);
            		Log.d(FLAGDRAW, "!!!!flagDraw : "+ flagDraw);


//            		下面這些事讓 設定 Handler，透過 handler 顯示文字訊息在手機上。
//                  switch (foundTargetIndex) {
//            		case 2:
//						// 設定文字說明
//						Thread chiayi = new Thread(new Runnable() {
//
//							@Override
//							public void run() {
//								
//								mHandler.sendEmptyMessage(2);  
//								
//							}
//							
//						});
//						chiayi.start();
//						
//						break;
//            		
//            		
//					case 1:
//						// 設定文字說明
//						Thread chengpo = new Thread(new Runnable() {
//
//							@Override
//							public void run() {
//								
//								mHandler.sendEmptyMessage(1);  
//								
//							}
//							
//						});
//						chengpo.start();
//						
//						break;
//						
//					case 0:
//						// 設定文字說明
//						Thread summer_street = new Thread(new Runnable() {
//
//							@Override
//							public void run() {
//								
//								mHandler.sendEmptyMessage(0);  
//								
//							}
//							
//						});
//						summer_street.start();
//						break;
//
//					}
            		
            		
        		}
    		
    		}
		return rgba;
    }
    
//    Handler mHandler = new Handler() {  
//        @Override  
//        public void handleMessage(Message msg) {  
//        	 if(msg.what == 2) {  
//             	imgTilte.setText("嘉義街外");  
//             	imgDisp.setText("藝術家： 陳澄波"+"\n"+"年代： 1927");  
//             }  
//        	
//            if(msg.what == 1) {  
//            	imgTilte.setText("廟口");  
//            	imgDisp.setText("藝術家： 陳澄波");  
//            }  
//            
//            if(msg.what == 0){
//            	imgTilte.setText("夏日街景");  
//            	imgDisp.setText("藝術家： 陳澄波"+"\n"+"年代： 1927");  
//            }
//            super.handleMessage(msg);  
//        }  
//    };

	
   

}
