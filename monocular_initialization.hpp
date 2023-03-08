/*************************************************************************************
 * Copyright 2023 lanxin
 *************************************************************************************/

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// 设置关键参数，可以自己改动
#define MAX_REPROJECT_ERROR 5.0
#define MIN_TRIANGLE_ANGLE 2.0
#define MIN_INIT_3DPOINT_NUM 100
#define MEDIAN_TRIANGLE_ANGLE 3.0

/**
 * @brief 用ORB特征点进行特征匹配，筛选出正确的匹配对
 *
 * @param img_1                         输入图1
 * @param img_2                         输入图2
 * @param keypoints_1                   图像1中关键点的像素坐标
 * @param keypoints_2                   图像2中关键点的像素坐标
 * @param matches                       图像1,2中特征点的匹配结果
 */
void feature_match(const Mat &img_1, const Mat &img_2,
                   std::vector<KeyPoint> &keypoints_1,
                   std::vector<KeyPoint> &keypoints_2,
                   std::vector<DMatch> &matches);

/**
 * @brief 用特征匹配对进行位姿估计
 *
 * @param keypoints_1                   图像1中关键点的像素坐标
 * @param keypoints_2                   图像2中关键点的像素坐标
 * @param points1                       图像1转化为vector形式的特征点
 * @param points2                       图像2转化为vector形式的特征点
 * @param matches                       图像1,2中特征点的匹配结果
 * @param K                             相机内参矩阵
 * @param inlierE 经过E筛选后的特征点是否是离群点，内点记为1，离群点记为0
 * @param R                             旋转矩阵
 * @param t                             平移向量
 */
void pose_estimation_2d2d(const vector<KeyPoint> &keypoints_1,
                          const vector<KeyPoint> &keypoints_2,
                          vector<Point2d> &points1, vector<Point2d> &points2,
                          const std::vector<DMatch> &matches, const Mat K,
                          Mat &inlierE, Mat &R, Mat &t);

/**
 * @brief 特征匹配点三角化，检查是否满足初始化条件
 *
 * @param pts_1                         图像1转化为vector形式的特征点
 * @param pts_2                         图像2转化为vector形式的特征点
 * @param R                             旋转矩阵
 * @param t                             平移向量
 * @param K                             相机内参矩阵
 * @param points                        所有三角化点的三维坐标
 * @param inlierPts 记录三角化点是否通过筛选，内点记为1，离群点记为0
 * @return int                          成功三角化点数目
 */
int triangulation(const vector<Point2d> &pts_1, const vector<Point2d> &pts_2,
                  const Mat &R, const Mat &t, const Mat &K,
                  vector<Point3d> &points, Mat &inlierPts);

/**
 * @brief 世界坐标三维点变换到像素坐标
 *
 * @param p                             三维点
 * @param R                             旋转矩阵
 * @param t                             平移向量
 * @param K                             相机内参矩阵
 * @return Point2d                      变换后的图像像素坐标
 */
Point2d world2pixel(const Point3d &p, const Mat &R, const Mat &t, const Mat &K);
