package org.opencv.samples.tutorial1;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;

import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;

import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.KeyPoint;

import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.Highgui;
import org.opencv.utils.Converters;


import android.content.Context;
import android.content.Intent;
import android.util.Log;

import static android.R.attr.height;
import static android.R.attr.width;

public final class ImageDetectionFilter {

    private boolean targetFound =false;

    private String Tag = "ImageDetectionFilter";

    // 參考 圖片
    private final Mat RImage;

    // MatOfKeyPoint 需要 設定這個 類別
    private final MatOfKeyPoint RKeypoints = new MatOfKeyPoint();

    private final Mat RDescriptors = new Mat();

    // 參考 圖片的 四個角
    private final Mat RCorners = new Mat(4, 1, CvType.CV_32FC2);
    List<KeyPoint> RKeypointsList = new LinkedList<KeyPoint>();



    // Features of the scene (the current frame).
    private final MatOfKeyPoint SKeypoints = new MatOfKeyPoint();

    private final Mat SDescriptors = new Mat();

    private final Mat mSceneCorners = new Mat(4, 1, CvType.CV_32FC2);


    // Tentative corner coordinates detected in the scene, in
    // pixels.
    private final Mat mCandidateSceneCorners = new Mat(4, 1, CvType.CV_32FC2);
    // The good detected corner coordinates, in pixels, as integers.
    private final MatOfPoint mIntSceneCorners = new MatOfPoint();
    List<KeyPoint> SKeypointsList = new LinkedList<KeyPoint>();

    List<Point> transPoints = new ArrayList<Point>();

    // A grayscale version of the scene.
    private final Mat SGraySrc = new Mat();
    // Tentative matches of scene features and reference features.
    //private final MatOfDMatch mMatches = new MatOfDMatch();

    // A feature detector, which finds features in images.
    private final FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);

    // A descriptor extractor, which creates descriptors of
    // features.
    private final DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);


    // A descriptor matcher, which matches features based on their
    // descriptors.
    private final DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);


    // 劃線的顏色
    private final Scalar lineColor = new Scalar(0, 255, 0);

    double reprojectionThreshold=3.0;

    // 比對結果
    List<MatOfDMatch> matches=new ArrayList<MatOfDMatch>();
    List<DMatch> resultMatches=new ArrayList<DMatch>();

    DMatch bestMatch = null, betterMatch = null;

    private Mat dstCorner ;


    // ImageDetectionFilter 辨識 image 工作都寫在這裡。
    public ImageDetectionFilter(final Context context, final int referenceImageResourceID) throws IOException {

        // 設定 ORB , Freak 參數。
        FilterSetting();

        // 載入 drawable資料夾中的圖片。
        RImage = Utils.loadResource(context, referenceImageResourceID, Highgui.CV_LOAD_IMAGE_COLOR);


        final Mat RImageGray = new Mat();
        Imgproc.cvtColor(RImage, RImageGray, Imgproc.COLOR_BGR2GRAY);  // 參考圖片 灰階
        Imgproc.cvtColor(RImage, RImage, Imgproc.COLOR_BGR2RGBA); // 參考圖片 轉換 RGBA

        // 記錄原圖的四個點
        RCorners.put(0, 0, new double[] {0.0, 0.0});
        RCorners.put(1, 0, new double[] {RImageGray.cols(), 0.0});
        RCorners.put(2, 0, new double[] {RImageGray.cols(), RImageGray.rows()});
        RCorners.put(3, 0, new double[] {0.0, RImageGray.rows()});


        detector.detect(RImageGray, RKeypoints);  // 取出 ORB 特徵點 ， 存放到 RKeypoints。
        extractor.compute(RImageGray, RKeypoints, RDescriptors); // 取出 freak 特徵描述

        // 轉換  成為 List<Point>

        RKeypointsList = RKeypoints.toList();

        Log.d(Tag,"RKeypoints = "+RKeypoints.size());
        Log.d(Tag,"RDescriptors = "+RDescriptors.size());
        Log.d(Tag,"RKeypointsList = "+RKeypointsList.size());

        RImageGray.release();
    }

    // 套用 手機的影像
    public void apply(final Mat src, final Mat dst) {

        Log.d(Tag,"apply Start");

        // 設定 ORB , Freak 參數。
        FilterSetting();

        // 將 螢幕的圖片 轉成灰階
        Imgproc.cvtColor(src, SGraySrc, Imgproc.COLOR_RGBA2GRAY);

        // 偵測 特徵點
        detector.detect(SGraySrc,  SKeypoints);

        Log.d(Tag,"SKeypoints = "+SKeypoints.size());


        // 製作 特徵描述
        extractor.compute(SGraySrc, SKeypoints, SDescriptors);
        Log.d(Tag,"SDescriptors = "+SDescriptors.size());

        // 轉換  成為 List<Point>
        SKeypointsList = SKeypoints.toList();

        // 用 knn Math 參考圖特徵描述 與 螢幕畫面特徵描述 將結果放到 matches
        descriptorMatcher.knnMatch(SDescriptors, RDescriptors, matches,2);
        Log.d(Tag,"apply match = "+matches.size());

        // 設定 minRatio
        float minRatio = 1.f / 1.2f;

        try {
            // 計算 matches 結果的距離
            for (int k=0; k<matches.size(); k++){
                bestMatch = matches.get(k).toArray()[0];
                betterMatch =  matches.get(k).toArray()[1];
                float distanceRatio = bestMatch.distance / betterMatch.distance;

                if (distanceRatio < minRatio){
                    // 當發現 距離 比 minRatio 還要近的點 加入 resultMatches
                    resultMatches.add(bestMatch);
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            e.printStackTrace();
        }

        // 印出 resultMatches 長度
        Log.d(Tag,"apply resultMatches = "+resultMatches.size());

        Log.d(Tag,"apply match END");

        Log.d(Tag,"apply findSceneCorners");


        // 做 Homography 與 perspectiveTransform
        findSceneCorners();

        Log.d(Tag,"apply draw");
        // 畫出框框
        draw(src, dst);
    }



    private void findSceneCorners() {

        ArrayList<Point>   matchedHitPoints = new ArrayList<Point>(resultMatches.size());
        ArrayList<Point> matchedQueryPoints = new ArrayList<Point>(resultMatches.size());

//        for (int m = 0; m < resultMatches.size(); m++){
//
//            if(resultMatches.get(m).queryIdx >= 0 && resultMatches.get(m).queryIdx < RKeypointsList.size() ) {
//                matchedQueryPoints.add(RKeypointsList.get(resultMatches.get(m).queryIdx).pt);
//                matchedHitPoints.add(SKeypointsList.get(resultMatches.get(m).trainIdx).pt);
//            }
//            else
//            {
//                continue;
//            }
//
////            if(resultMatches.get(m).trainIdx >= 0 && resultMatches.get(m).trainIdx < SKeypointsList.size() ) {
////                matchedHitPoints.add(SKeypointsList.get(resultMatches.get(m).trainIdx).pt);
////            }
//        }

        if(resultMatches.isEmpty()){
            return;
        }

        if(resultMatches.size() <4){
            return;
        }


            for(int i = 0 ;i< resultMatches.size() ;i++) {

                DMatch match = resultMatches.get(i);

               if(bestMatch.trainIdx >= 0 && bestMatch.trainIdx < RKeypointsList.size()){

                   matchedQueryPoints.add(RKeypointsList.get(match.trainIdx).pt);

               }
                if(bestMatch.queryIdx >= 0 && bestMatch.queryIdx < SKeypointsList.size()){
                    try {
                        matchedHitPoints.add(SKeypointsList.get(match.queryIdx).pt);
                    }catch (IndexOutOfBoundsException e){
                        e.printStackTrace();
                    }
                }
            }

        if (matchedQueryPoints.size() < 4 || matchedHitPoints.size() < 4) {
            // There are too few good points to find the homography.
            return;
        }


        Log.d(Tag,"matchedQueryPoints = "+matchedQueryPoints.size());
        Log.d(Tag,"matchedHitPoints = "+matchedHitPoints.size());

        matches.clear();
        resultMatches.clear();


        //filter Matches By Homography
        MatOfPoint2f RMat2f = new MatOfPoint2f();
        MatOfPoint2f SMat2f = new MatOfPoint2f();

        RMat2f.fromList(matchedQueryPoints);
        SMat2f.fromList(matchedHitPoints);

        // 清空 內容
        matchedQueryPoints.clear();
        matchedHitPoints.clear();


        double reprojectionThreshold=3.0;


        //Mat mask = new Mat();


            Mat homography = Calib3d.findHomography(RMat2f, SMat2f, Calib3d.LMEDS, reprojectionThreshold);

            RMat2f.release();
            SMat2f.release();

            Core.perspectiveTransform(RCorners, mCandidateSceneCorners, homography);

        //perspectiveTransform

        mCandidateSceneCorners.convertTo(mIntSceneCorners, CvType.CV_32S);
//
        if (Imgproc.isContourConvex(mIntSceneCorners)) {
            // The corners form a convex polygon, so record them as
            // valid scene corners.
            mCandidateSceneCorners.copyTo(mSceneCorners);
            targetFound = true;
        }

    }

    protected void draw(final Mat src, final Mat dst) {

//        if (dst != src) {
//            src.copyTo(dst);
//        }

         // Outline the found target in green.
        Core.line(dst, new Point(mSceneCorners.get(0, 0)), new Point(mSceneCorners.get(1, 0)), lineColor, 1);
        Core.line(dst, new Point(mSceneCorners.get(1, 0)), new Point(mSceneCorners.get(2, 0)), lineColor, 1);
        Core.line(dst, new Point(mSceneCorners.get(2, 0)), new Point(mSceneCorners.get(3, 0)), lineColor, 1);
        Core.line(dst, new Point(mSceneCorners.get(3, 0)), new Point(mSceneCorners.get(0, 0)), lineColor, 1);

//        Core.line(dst, transPoints.get(0), transPoints.get(1), lineColor, 1);
//        Core.line(dst, transPoints.get(1), transPoints.get(2), lineColor, 1);
//        Core.line(dst, transPoints.get(2), transPoints.get(3), lineColor, 1);
//        Core.line(dst, transPoints.get(3), transPoints.get(4), lineColor, 1);

    }


    public boolean isTargetFound() {

        return targetFound;
    }


    // 設定 ORB detector , freak Descriptor 參數
    private void FilterSetting(){
        File temp;
        try {
            // 設定 detector 參數
            temp = File.createTempFile("tempFile", ".tmp");
            String orbSettings = "%YAML:1.0\nname: \"Feature2D.ORB\"\nWTA_K: 2\nedgeThreshold: 31\nfirstLevel: 0\nnFeatures: 3500 \nnLevels: 8 \npatchSize: 31\nscaleFactor: 1.20\nscoreType: 0\n";
            FileWriter writer = new FileWriter(temp, false);
            writer.write(orbSettings);
            writer.close();

            // 讀取 detector 參數
            detector.read(temp.getPath());

            // 設定 freak 參數
            String freakSettings = "%YAML:1.0 \npatternScale: 22.5 \nnOctaves: 4 \norientationNormalized : True \nscaleNormalized : True\n";

            writer = new FileWriter(temp, false);
            writer.write(freakSettings);
            writer.close();

            // 讀取 freak 參數
            extractor.read(temp.getPath());
            temp.deleteOnExit();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }


}
