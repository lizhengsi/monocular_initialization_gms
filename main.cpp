
/**
 * @brief 单目相机SLAM初始化
 * @details 提取特征点，特征匹配和筛选，位姿估计，三角化， 环境：ubuntu 18.04,
 * OpenCV 3.4.6
 * @author 小六
 * @version v1.0
 * @date 2021-01-10
 * @license 引用请注明来自公众号：计算机视觉life，ORBSLAM2课程作业题
 */

#include "monocular_initialization.hpp"

int main(int argc, char **argv) {

  if (argc != 2) {
    cout << "usage: monocular_initialization imageDirectory" << endl;
    return 1;
  }

  // 相机内参
  Mat K =
      (Mat_<double>(3, 3) << 520.9, 0.0, 325.1, 0, 521.0, 249.7, 0.0, 0.0, 1.0);

  // 获取目录下所有图像名称
  vector<String> imageNames;
  String imageDirectory = argv[1];
  glob(imageDirectory, imageNames);

  // 读取第一帧，设置为世界坐标系原点
  Mat imageInit = imread(imageNames[0], CV_LOAD_IMAGE_COLOR);
  // 开始遍历后续帧，寻找满足初始化条件的帧
  for (int i = 1; i < imageNames.size(); ++i) {

    Mat imageCurr = imread(imageNames[i], CV_LOAD_IMAGE_COLOR);
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;

    // 进行ORB特征匹配
    feature_match(imageInit, imageCurr, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // 估计两张图像间运动 R, t
    Mat R, t, inliers;
    vector<Point2d> points1;
    vector<Point2d> points2;
    pose_estimation_2d2d(keypoints_1, keypoints_2, points1, points2, matches, K,
                         inliers, R, t);

    // 判断匹配数目是否满足初始化3D点基本要求
    int num_valid_matches = countNonZero(inliers);
    if (num_valid_matches < MIN_INIT_3DPOINT_NUM) {
      continue;
    }

    // 根据估计的位姿来三角化三维点，判断是否符合初始化条件
    vector<Point3d> points3d;
    int num = triangulation(points1, points2, R, t, K, points3d, inliers);
    if (num < MIN_INIT_3DPOINT_NUM) {
      continue;
    }

    // 初始化成功后显示信息
    printf("经过E筛选，有效匹配对还有 %d / %d \n", num_valid_matches,
           matches.size());
    printf("成功从第 %d 帧初始化！ 获得有效三维点 %d 个!\n", i, num);

    // 初始化成功后，显示有效三维点对应的特征匹配点对
    Mat img1_plot = imageInit.clone();
    Mat img2_plot = imageCurr.clone();
    for (int i = 0; i < points1.size(); i++) {
      if (inliers.at<int>(i)) { // 显示初始化三维点中的内点对应的二维匹配点
        // green
        cv::circle(img1_plot, points1[i], 2, cv::Scalar(0, 255, 0), 2);
        cv::circle(img2_plot, points2[i], 2, cv::Scalar(0, 255, 0), 2);
      } else { // 显示初始化三维点中的外点对应的二维匹配点
        // red
        cv::circle(img1_plot, points1[i], 2, cv::Scalar(0, 0, 255), 2);
        cv::circle(img2_plot, points2[i], 2, cv::Scalar(0, 0, 255), 2);
      }
    }
    cv::imshow("img 1 features, green-inliers, red-outliers", img1_plot);
    cv::imshow("img 2 features, green-inliers, red-outliers", img2_plot);

    // 显示初始化三维点重投影后的图像点
    Mat img2_project_inlier = imageCurr.clone();
    Mat img2_project_outlier = imageCurr.clone();
    for (int i = 0; i < points1.size(); i = i + 3) {
      if (inliers.at<int>(i)) { // 显示内点重投影
        Point2d projected_pt2 = world2pixel(points3d[i], R, t, K);
        cv::circle(img2_project_inlier, projected_pt2, 2, cv::Scalar(255, 0, 0),
                   2); // project point, blue
        cv::circle(img2_project_inlier, points2[i], 2, cv::Scalar(0, 255, 0),
                   2); // match point, green
      } else {         // 显示外点重投影
        Point2d projected_pt2 = world2pixel(points3d[i], R, t, K);
        cv::circle(img2_project_outlier, projected_pt2, 2,
                   cv::Scalar(0, 0, 255), 2); // project point, red
        cv::circle(img2_project_outlier, points2[i], 2, cv::Scalar(0, 255, 255),
                   2); // match point, yellow
      }
    }
    cv::imshow("img2_project_inlier, blue-project point, green-match point",
               img2_project_inlier);
    cv::imshow("img2_project_outlier, red-project point, yellow-match point",
               img2_project_outlier);
    cv::waitKey();

    break;
  }

  return 0;
}
