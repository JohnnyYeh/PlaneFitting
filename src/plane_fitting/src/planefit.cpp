#include <ros/ros.h>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <opencv/cv.h>

#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/LaserScan.h>
#include <iostream>
#include <vector>

using namespace std;
#define POINTSNUM 10
int main(int argc, char ** argv)
{
        //ros参数
        ros::init(argc,argv,"planefit");
        ros::NodeHandle nh("~");

        ros::Publisher pubPlanePoints = nh.advertise<sensor_msgs::PointCloud2>("/plane_points",2);
        ros::Publisher pubPlanePointsNoise = nh.advertise<sensor_msgs::PointCloud2>("/plane_points_noise",2);
        ros::Publisher pubPlanePointsFit = nh.advertise<sensor_msgs::PointCloud2>("/plane_points_fit",2);
        //平面参数
        double a = 1.2;
        double b = 1.1;
        double c = 2.0;
        double d = 0.5;
        
        double xmin = -5.0;
        double xmax = 5.0;
       
        int blocksize =  POINTSNUM*POINTSNUM;
        double delta = (xmax - xmin)/POINTSNUM;
        Eigen::Matrix<double,POINTSNUM*POINTSNUM,3> PointsXYZ;
        Eigen::Matrix<double,POINTSNUM*POINTSNUM,3> PointsXYZNoise;
        Eigen::Matrix<double,POINTSNUM*POINTSNUM,3> PointsXYZFit;
        pcl::PointCloud<pcl::PointXYZ>::Ptr Points;
        Points.reset(new pcl::PointCloud<pcl::PointXYZ>());

        pcl::PointCloud<pcl::PointXYZ>::Ptr PointsNoise;
        PointsNoise.reset(new pcl::PointCloud<pcl::PointXYZ>());

        pcl::PointCloud<pcl::PointXYZ>::Ptr PointsFit;
        PointsFit.reset(new pcl::PointCloud<pcl::PointXYZ>());
        
        int i = 0;

        cv::RNG rng(12345);
        for (double x = xmin;x< xmax;x+=delta ){
                if(i >= blocksize) break;
                for(double y = xmin;y< xmax;y+=delta){
                        if(i >= blocksize) break;
                        double z = -(d+a*x+b*y)/c;
                        PointsXYZ(i,0) = x;
                        PointsXYZ(i,1) = y;
                        PointsXYZ(i,2) = z;

                        PointsXYZNoise(i,0) = x;
                        PointsXYZNoise(i,1) = y;
                        PointsXYZNoise(i,2) = z+rng.gaussian(0.5);
                        pcl::PointXYZ point;
                        point.x = x;
                        point.y = y;
                        point.z = z;
                        Points->push_back(point);
                        
                        point.z =  PointsXYZNoise(i,2);
                        PointsNoise->push_back(point);
                        // ROS_WARN("(%f,%f,%f)",x,y,z);
                        i++;
                        
                }
        }
       //平面拟合
        Eigen::Vector3d pmeans = PointsXYZNoise.colwise().mean();
        Eigen::Matrix<double,POINTSNUM*POINTSNUM,3> centroid = PointsXYZNoise.rowwise() - pmeans.transpose();
        
        //SVD
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(centroid,Eigen::ComputeThinU|Eigen::ComputeThinV);
        Eigen::Matrix3d V = svd.matrixV();
        Eigen::Vector3d normal;
        normal << V (0,2),V (1,2),V (2,2);
        ROS_WARN("svd v =(%f,%f,%f)",normal(0),normal(1),normal(2));
        // cout << "svd v = " << normal << endl;

        Eigen::MatrixXd cov = (centroid.adjoint()*centroid);
       
        // eigenvector
        cv::Mat matCov;
        cv::Mat matEig;
        cv::Mat matVec;
        matCov = cv::Mat(3,3,CV_64F,cv::Scalar::all(0));
        matEig = cv::Mat(1,3,CV_64F,cv::Scalar::all(0));
        matVec = cv::Mat(3,3,CV_64F,cv::Scalar::all(0));
        cv::eigen2cv(cov,matCov);
        cv::eigen(matCov,matEig,matVec);
        
        // cv 是行优先
        ROS_WARN("eigv =(%f,%f,%f)",matVec.at<double>(2,0),matVec.at<double>(2,1),matVec.at<double>(2,2));
       
       // eigenvector with Eigen
       Eigen::EigenSolver<Eigen::MatrixXd> es(cov);

       Eigen::Matrix3d esD = es.pseudoEigenvalueMatrix();
       Eigen::Matrix3d esV= es.pseudoEigenvectors();

       cout << esV <<endl;
       ROS_WARN("Eigen =(%f,%f,%f)", esV(0,2), esV(1,2), esV(2,2));


       //测试拟合效果
       double d_fit = -normal.transpose()*pmeans;
       i = 0;
       double a_fit = normal(0);
       double b_fit = normal(1);
       double c_fit = normal(2);
        for (double x = xmin;x< xmax;x+=delta ){
                if(i >= blocksize) break;
                for(double y = xmin;y< xmax;y+=delta){
                        if(i >= blocksize) break;
                        double z = -(d_fit+a_fit*x+b_fit*y)/c_fit;
                
                        pcl::PointXYZ point;
                        point.x = x;
                        point.y = y;
                        point.z = z;
                        PointsFit->push_back(point);
                        // ROS_WARN("(%f,%f,%f)",x,y,z);
                        i++;
                }
        }

       //发布点云数据
        sensor_msgs::PointCloud2 planeCloudTemp;
        pcl::toROSMsg(*Points,planeCloudTemp);
        // ROS_WARN("POINT SIZE = %d",Points->size());
        planeCloudTemp.header.frame_id = "/map";
        planeCloudTemp.header.stamp = ros::Time::now();

        sensor_msgs::PointCloud2 planeCloudNoise;
        pcl::toROSMsg(*PointsNoise,planeCloudNoise);
        // ROS_WARN("POINT SIZE = %d",Points->size());
        planeCloudNoise.header.frame_id = "/map";
        planeCloudNoise.header.stamp = ros::Time::now();

        sensor_msgs::PointCloud2 planeCloudFit;
        pcl::toROSMsg(*PointsFit,planeCloudFit);
        // ROS_WARN("POINT SIZE = %d",Points->size());
        planeCloudFit.header.frame_id = "/map";
        planeCloudFit.header.stamp = ros::Time::now();


        ros::Rate loop_rate(50);
        while (ros::ok())
        {
                pubPlanePoints.publish(planeCloudTemp);
                pubPlanePointsNoise.publish(planeCloudNoise);
                pubPlanePointsFit.publish(planeCloudFit);
                ros::spinOnce();//处理订阅话题的所有回调函数callback()，
                loop_rate.sleep(); //休眠，休眠时间由loop_rate()设定
        }


        return 0;
}