/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include "pointcloudmapping.h"
#include "YOLOv3SE.h"
#include "segmentation.h"
#include <Eigen/Geometry>
#include <KeyFrame.h>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <ctime>
#include <pcl/surface/gp3.h>

#include <pcl/surface/poisson.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/poisson.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/projection_matrix.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "Converter.h"
#include <boost/make_shared.hpp>

#ifdef GPU
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#endif

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <unistd.h>
#include <time.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

//VTK include needed for drawing graph lines
#include <vtkPolyLine.h>





using namespace cv;
using namespace std;
using namespace pcl;
using namespace APC;
using namespace pcl::io;
using namespace pcl::console;
typedef pcl::PointXYZI PointTypeIO;
typedef pcl::PointXYZINormal PointTypeFull;


int pub_port = 6666;
#define NUM 5


int k = 500;
int min_size = 500;





PointCloudMapping::PointCloudMapping(double resolution_) {

    this->resolution = resolution_;
    voxel.setLeafSize(resolution, resolution, resolution);
    globalMap = boost::make_shared<PointCloud>();
    viewerThread = make_shared<thread>(bind(&PointCloudMapping::viewer, this));

}
void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}


void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{


    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;

    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back( kf );


    colorImgs.push_back( color.clone());

    depthImgs.push_back( depth.clone() );

    keyFrameUpdated.notify_one();
}




pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    PointCloud::Ptr tmp( new PointCloud() );
    // point cloud is null ptr
    for ( int m=0; m<depth.rows; m+=3)
    {
        for ( int n=0; n<depth.cols; n+=3 )
        {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d>10)
                //尝试加上距离滤波 d>5
                continue;
            PointT p;
            p.z = d;
            p.x = ( n - kf->cx) * p.z / kf->fx;
            p.y = ( m - kf->cy) * p.z / kf->fy;

            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];

//            if(p.b == p.g || p.b == p.r)
//                continue;


            tmp->points.push_back(p);
        }
    }

    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;

    cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return cloud;
}



//void PointCloudMapping::viewer()
//{
//    pcl::visualization::CloudViewer viewer("viewer");
//    while(1)
//    {
//        {
//            unique_lock<mutex> lck_shutdown( shutDownMutex );
//            if (shutDownFlag)
//            {
//                break;
//            }
//        }
//        {
//            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
//            keyFrameUpdated.wait( lck_keyframeUpdated );
//        }
//
//        // keyframe is updated
//        size_t N=0;
//        {
//            unique_lock<mutex> lck( keyframeMutex );
//            N = keyframes.size();
//        }
//
//        for ( size_t i=lastKeyframeSize; i<N ; i++ )
//        {
//            PointCloud::Ptr prep = generatePointCloud( keyframes[i], colorImgs[i], depthImgs[i] );
//            PointCloud::Ptr p = regionGrowingSeg(prep);
//            *globalMap += *p;
//        }
//        //PointCloud::Ptr tmp(new PointCloud());
//        voxel.setInputCloud( globalMap );
////        voxel.filter( *tmp );
////        globalMap->swap( *tmp );
//        viewer.showCloud( globalMap );
//        cout << "show global map, size=" << globalMap->points.size() << endl;
//        lastKeyframeSize = N;
//    }
//
// }

// int PointCloudMapping::Init(string &weight_file, string  &config_file ){
//     //YOLOv3 detector;
// 	torch::DeviceType device_type;

//     if (torch::cuda::is_available() ) {        
//         device_type = torch::kCUDA;
//     } else {
//         device_type = torch::kCPU;
//     }
//     torch::Device device(device_type);

//     // input image size for YOLO v3
//     input_image_size = 416;

//     Darknet net(config_file, &device);

//     map<string, string> *info = net.get_net_info();

//     info->operator[]("height") = std::to_string(input_image_size);

//     std::cout << "loading weight ..." << endl;
//     net.load_weights(weight_file);
//     std::cout << "weight loaded ..." << endl;
    
//     net.to(device);

//     torch::NoGradGuard no_grad;
//     net.eval();

//     std::cout << "start to inference ..." << endl;
    
//     return 0;
// }

void PointCloudMapping::viewer()
{
     
    std::vector<cv::Scalar> colors;
    string line, number;
    ifstream f("color.txt");
    if (!f.good())
    {
        cout << "Cannot open file" << endl;
        exit(-1);
    }
    torch::DeviceType device_type;

    if (torch::cuda::is_available() ) {        
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    // input image size for YOLO v3
    input_image_size = 416;

    //Darknet net("yolov3-tiny.cfg", &device);
    Darknet net("yolov3.cfg", &device);

    map<string, string> *info = net.get_net_info();

    info->operator[]("height") = std::to_string(input_image_size);

    std::cout << "loading weight ..." << endl;
    //net.load_weights("yolov3-tiny.weights");
    net.load_weights("yolov3.weights");
    std::cout << "weight loaded ..." << endl;
    
    net.to(device);

    torch::NoGradGuard no_grad;
    net.eval();

    std::cout << "start to inference ..." << endl;


    vector<int> v;
    while(std::getline(f, line))
    {
        istringstream is(line);

        while(std::getline(is, number, ','))
        {
            v.push_back(atoi(number.c_str()));
        }
        colors.push_back(cv::Scalar(v[0],v[1],v[2]));
        v.clear();
    }
//    for (int i = 0; i < 80; i++) {
//        colors.push_back(cv::Scalar(rand() % 127 + 128, rand() % 127 + 128, rand() % 127 + 128));
//    }

    //std::cout<<"detector ready"<<std::endl;
    //Init("./yolov3.weights", "./yolov3.cfg", "./coco.names");
   // detector.Create("yolov3.weights", "yolov3.cfg", "coco.names");
    //sleep(3);
   // std::cout<<"detector created"<<std::endl;
    pcl::visualization::CloudViewer viewer("viewer");

    while(1)
    {
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }

        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }


        // keyframe is updated
        size_t N=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N = keyframes.size();
        }

        std::cout<<"key frames joined pointcloud start"<<std::endl;
        {

            for (size_t i=lastKeyframeSize; i<N ; i++) {

                // cv::Mat tmp_color = colorImgs[i];
                // char img[20];
                // sprintf(img, "%s%d%s", "img/image", i, ".png");  //sprintf将后面的字符串存储在img中
                // cv::imwrite(img,tmp_color);
                // //PointCloud::Ptr pre_p = generatePointCloud(keyframes[i], colorImgs[i], depthImgs[i]);
                // //PointCloud::Ptr p = regionGrowingSeg(pre_p);
                // //*globalMap += *pre_p;
                // //sleep(3);
                // cv::Mat img_tmp_color = cv::imread(img);
                // std::vector<BoxSE> boxes = detector.Detect(img_tmp_color, 0.5F);
                // //continue;
                // int n = boxes.size();
                // for (int i = 0; i < n; i++) {
                //     if(boxes[i].m_class_name == "person")
                //     {
                //         cv::rectangle(img_tmp_color, boxes[i].tl(), boxes[i].br(), colors[80], -1, 4);
                //     }
                //     //cv::putText(img, detector.Names(box.m_class), box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1.0, colors[box.m_class], 2);
                //     //cv::rectangle(img, box, colors[box.m_class], 2);
                //     cv::rectangle(img_tmp_color, boxes[i].tl(), boxes[i].br(), colors[boxes[i].m_class], -1, 4);
                // }
                cv::Mat origin_image, resized_image;

                origin_image = colorImgs[i];
                
                //cv::cvtColor(origin_image, resized_image,  cv::COLOR_BGR2RGB);
                cv::resize(origin_image, resized_image, cv::Size(input_image_size, input_image_size));

                cv::Mat img_float;
                resized_image.convertTo(img_float, CV_32F, 1.0/255);

                auto img_tensor = torch::from_blob(img_float.data, {1, input_image_size, input_image_size, 3}).to(device);
                img_tensor = img_tensor.permute({0,3,1,2});

                auto start = std::chrono::high_resolution_clock::now();
                
                auto output = net.forward(img_tensor);
                
                // filter result by NMS 
                // class_num = 80
                // confidence = 0.6
                auto result = net.write_results(output, 80, 0.6, 0.4);

                auto end = std::chrono::high_resolution_clock::now();

                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); 

                // It should be known that it takes longer time at first time
                std::cout << "inference taken : " << duration.count() << " ms" << endl; 

                if (result.dim() == 1)
                {
                    std::cout << "no object found" << endl;
                }
                else
                {
                    int obj_num = result.size(0);

                    std::cout << obj_num << " objects found" << endl;

                    float w_scale = float(origin_image.cols) / input_image_size;
                    float h_scale = float(origin_image.rows) / input_image_size;

                    result.select(1,1).mul_(w_scale);
                    result.select(1,2).mul_(h_scale);
                    result.select(1,3).mul_(w_scale);
                    result.select(1,4).mul_(h_scale);

                    auto result_data = result.accessor<float, 2>();
            /*	for(int i=0;i<result.size(0);i++){
                    for(int j=0;j<result.size(1);j++){
                    std::cout<<"data_"<<i<<":"<<result_data[i][j]<<" ";
            }
                    std::cout<<std::endl;
            }*/

                    for (int i = 0; i < result.size(0) ; i++)
                    {
                //std::string str=std::to_string(result_data[i][5]);
                //cv::putText(origin_image, str, cv::Point(result_data[i][1], result_data[i][2])-cv::Point(0,5), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);            
                //cv::rectangle(origin_image, cv::Point(result_data[i][1], result_data[i][2]), cv::Point(result_data[i][3], result_data[i][4]), cv::Scalar(0, 0, 255), -1, 1, 0);
                cv::rectangle(origin_image, cv::Point(result_data[i][1], result_data[i][2]), cv::Point(result_data[i][3], result_data[i][4]), colors[result_data[i][7]], -1, 1, 0);
                    }

                }

                PointCloud::Ptr surf_p = generatePointCloud(keyframes[i], origin_image, depthImgs[i]);
                //PointCloud::Ptr p = RegionGrowingSeg(surf_p);
                //PointCloud::Ptr surf_p = generatePointCloud(keyframes[i],colorImgs[i], depthImgs[i]);
                *globalMap += *surf_p;

            }


        }

        //voxel.setInputCloud( globalMap );

        viewer.showCloud( globalMap );
//      pcl::PointCloud<pcl::PointXYZ> ply_file;
//      PointCloudXYZRGBAtoXYZ(*globalMap,ply_file);

        cout << "show global map, size=" << globalMap->points.size() << endl;
        lastKeyframeSize = N;

    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PCDWriter pcdwriter;
    pcdwriter.write<pcl::PointXYZRGBA>("global_color.pcd", *globalMap);//write global point cloud map and save to a pcd file
    cpf_seg(globalMap);
    //detector.Release();
    /*
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PCDWriter pcdwriter;
    pcdwriter.write<pcl::PointXYZRGBA>("global.pcd", *globalMap);//write global point cloud map and save to a pcd file
    pcl::io::savePCDFile("pointcloud.pcd", *globalMap );


    pcl::PointCloud<pcl::PointXYZ> ply_file;
    PointCloudXYZRGBAtoXYZ(*globalMap,ply_file);
    pcl::PLYWriter plywriter;
    plywriter.write<pcl::PointXYZ>("test.ply",ply_file,true);

     */

    //poisson_reconstruction(globalMap);
    final_process();
    //exit(0);


}


void PointCloudMapping::final_process()
{
    vector<int> vbottle;
    vector<int> vteddy;
    vector<int> vkeyboard;
    vector<int> vmouse;
    vector<int> vmonitor;
    vector<int> vchair;
    vector<int> vstr;


    PointCloudT::Ptr tgt (new PointCloudT);
    pointcloudL::Ptr ll (new pointcloudL);

    pcl::io::loadPCDFile<pcl::PointXYZRGBA>("global_color.pcd",*tgt);
    pcl::io::loadPCDFile<pcl::PointXYZL>("segmentation.pcd",*ll);

    PointCloudT Final1 = *tgt;
    pcl::PointCloud<pcl::PointXYZL> lcloud = *ll;
    //cout <<lcloud.size()<<endl;
    //cout<<"1:"<<Final2.points.size()<<"/n"<<"2:"<<Final2.points.size();
    static int sum=0;
    for(int i=0;i< Final1.points.size()-20;i++)

    {

        pcl::PointXYZRGBA basic_point;
        pcl::PointXYZL basic_pointl;
        basic_point.x = Final1.points[i].x;
        basic_point.y = Final1.points[i].y;
        basic_point.z = Final1.points[i].z;
        basic_pointl.x = lcloud.points[i].x;
        basic_pointl.y = lcloud.points[i].y;
        basic_pointl.z = lcloud.points[i].z;
        basic_pointl.label=lcloud.points[i].label;
        basic_point.rgba = Final1.points[i].rgba;
        uint32_t rgba = Final1.points[i].rgba;
        int r =rgba >> 16 & 0x0000ff;
        int g = rgba >> 8 & 0x0000ff;
        int b = rgba & 0x0000ff;
        //cout <<basic_point.rgba<<endl;
        // for(int j=0;j<200;j++){
        //     if(lcloud.points[i+j].label==lcloud.points[i].label){
        //         sum++;
        //     }
        // }


        //if(b==0&&g==254&&r==124&&sum>15){//0,252,124
        //if(b==0&&g==0&&r==255&&sum>150){//0,252,124
        if(b==0&&g==0&&r==255){//0,252,124
            //cout <<r<<endl;//monitor
            vmonitor.push_back(i);
            lcloud[i].label = 300;

        }
        //if(b==240&&g==176&&r==0&&sum>150){//203,192,255
        if(b==240&&g==176&&r==0){//203,192,255
            //cout <<r<<endl;//mouse
            vmouse.push_back(i);
            lcloud[i].label = 400;

        }
        //if(b==80&&g==208&&r==146&&sum>150){//128,249,237
         if(b==80&&g==208&&r==146){//128,249,237
            //cout <<r<<endl;//keyboard
            vkeyboard.push_back(i);
            lcloud[i].label = 500;

        }
        //if(b==160&&g==48&&r==112&&sum>150){//50,205,50
        if(b==160&&g==48&&r==112){//50,205,50
            //cout <<r<<endl;//teddy bear
            vteddy.push_back(i);
            lcloud[i].label = 600;


        }
        //if(b==83&&g==103&&r==191&&sum>150){//240,160,120
        if(b==83&&g==103&&r==191){//240,160,120
            //cout <<r<<endl;//cup
            vchair.push_back(i);
            lcloud[i].label = 700;

        }
        //if(b==0&&g==255&&r==255&&sum>150){
        if(b==0&&g==255&&r==255){
            vbottle.push_back(i);
            lcloud[i].label = 800;

        }
        // else{
        //     vstr.push_back(i);
        //     lcloud[i].label = 900;
        // }

    }


    pcl::PCDWriter pcdwriter;
    pcdwriter.write<pcl::PointXYZL>("semantic_map.pcd", lcloud);//write global
    cout <<"finish semantic labelling,please see details in semantic_map.pcd"<<endl;

}
void PointCloudMapping::obj2pcd(const std::string& inputFilename, const std::string& outputFilename)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;

    // Input stream
    std::ifstream is(inputFilename.c_str());

    // Read line by line
    for(std::string line; std::getline(is, line); )
    {
        std::istringstream in(line);

        std::string v;
        in >> v;
        if (v != "v") continue;

        // Read x y z
        float x, y, z;
        in >> x >> y >> z;
        cloud.push_back(pcl::PointXYZ(x, y, z));
    }

    is.close();

    // Save to pcd file
    pcl::io::savePCDFileBinaryCompressed(outputFilename, cloud);
}


void PointCloudMapping::PointXYZRGBAtoXYZ(const pcl::PointXYZRGBA& in,
                                pcl::PointXYZ& out)
{
    out.x = in.x; out.y = in.y; out.z = in.z;
}

void PointCloudMapping::PointXYZLtoXYZ(const pcl::PointXYZL& in,
                                          pcl::PointXYZ& out)
{
    out.x = in.x; out.y = in.y; out.z = in.z;

}

void PointCloudMapping::PointCloudXYZRGBAtoXYZ(const pcl::PointCloud<pcl::PointXYZRGBA>& in,
                            pcl::PointCloud<pcl::PointXYZ>& out)
{
    out.width = in.width;
    out.height = in.height;
    for (size_t i = 0; i < in.points.size(); i++)
    {
        pcl::PointXYZ p;
        PointXYZRGBAtoXYZ(in.points[i],p);
        out.points.push_back (p);
    }
}



void PointCloudMapping::PointXYZRGBtoXYZRGBA(const pcl::PointXYZRGB& in,
                                pcl::PointXYZRGBA& out)
{
    out.x = in.x; out.y = in.y; out.z = in.z;
    out.r = in.r; out.g = in.g; out.b = in.z; out.a =0;

}

void PointCloudMapping::PointCloudXYZRGBtoXYZRGBA(const pcl::PointCloud<pcl::PointXYZRGB>& in,
                            pcl::PointCloud<pcl::PointXYZRGBA>& out)
{
    out.width = in.width;
    out.height = in.height;
    for (size_t i = 0; i < in.points.size(); i++)
    {
        pcl::PointXYZRGBA p;
        PointXYZRGBtoXYZRGBA(in.points[i],p);
        out.points.push_back (p);
    }
}


void PointCloudMapping::poisson_reconstruction(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr object_cloud)
{

    pcl::PointCloud<PointXYZRGB>::Ptr cloud(new pcl::PointCloud<PointXYZRGB>());
    pcl::copyPointCloud(*object_cloud, *cloud);
    pcl::PointCloud<PointXYZRGB>::Ptr filtered(new pcl::PointCloud<PointXYZRGB>());
    PassThrough<PointXYZRGB> filter;
    filter.setInputCloud(cloud);
    filter.filter(*filtered);
    cout << "passthrough filter complete" << endl;

    cout << "begin normal estimation" << endl;
    NormalEstimationOMP<PointXYZRGB, Normal> ne;//计算点云法向
    ne.setNumberOfThreads(8);//设定临近点
    ne.setInputCloud(filtered);
    ne.setRadiusSearch(0.01);//设定搜索半径
    Eigen::Vector4f centroid;

    compute3DCentroid(*filtered, centroid);//计算点云中心
    ne.setViewPoint(centroid[0], centroid[1], centroid[2]);//将向量计算原点置于点云中心

    pcl::PointCloud<Normal>::Ptr cloud_normals (new pcl::PointCloud<Normal>());
    ne.compute(*cloud_normals);
    cout << "normal estimation complete" << endl;
    cout << "reverse normals' direction" << endl;

//将法向量反向
    for(size_t i = 0; i < cloud_normals->size(); ++i)
    {
        cloud_normals->points[i].normal_x *= -1;
        cloud_normals->points[i].normal_y *= -1;
        cloud_normals->points[i].normal_z *= -1;
    }

//融合RGB点云和法向
    cout << "combine points and normals" << endl;
    pcl::PointCloud<PointXYZRGBNormal>::Ptr cloud_smoothed_normals(new pcl::PointCloud<PointXYZRGBNormal>());
    concatenateFields(*filtered, *cloud_normals, *cloud_smoothed_normals);

//泊松重建
    cout << "begin poisson reconstruction" << endl;
    Poisson<PointXYZRGBNormal> poisson;
    //poisson.setDegree(2);
    poisson.setDepth(8);
    poisson.setSolverDivide (6);
    poisson.setIsoDivide (6);

    poisson.setConfidence(false);
    poisson.setManifold(false);
    poisson.setOutputPolygons(false);

    poisson.setInputCloud(cloud_smoothed_normals);
    PolygonMesh mesh;
    poisson.reconstruct(mesh);

    cout << "finish poisson reconstruction" << endl;

//给mesh染色
    pcl::PointCloud<PointXYZRGB> cloud_color_mesh;
    pcl::fromPCLPointCloud2(mesh.cloud, cloud_color_mesh);

    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud (cloud);
    // K nearest neighbor search
    int K = 5;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    for(int i=0;i<cloud_color_mesh.points.size();++i)
    {
        uint8_t r = 0;
        uint8_t g = 0;
        uint8_t b = 0;
        float dist = 0.0;
        int red = 0;
        int green = 0;
        int blue = 0;
        uint32_t rgb;

        if ( kdtree.nearestKSearch (cloud_color_mesh.points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
        {
            for (int j = 0; j < pointIdxNKNSearch.size (); ++j)
            {

                r = cloud->points[ pointIdxNKNSearch[j] ].r;
                g = cloud->points[ pointIdxNKNSearch[j] ].g;
                b = cloud->points[ pointIdxNKNSearch[j] ].b;

                red += int(r);
                green += int(g);
                blue += int(b);
                dist += 1.0/pointNKNSquaredDistance[j];

                std::cout<<"red: "<<int(r)<<std::endl;
                std::cout<<"green: "<<int(g)<<std::endl;
                std::cout<<"blue: "<<int(b)<<std::endl;
                cout<<"dis:"<<dist<<endl;
            }
        }
        cloud_color_mesh.points[i].r = int(red/pointIdxNKNSearch.size ()+0.5);
        cloud_color_mesh.points[i].g = int(green/pointIdxNKNSearch.size ()+0.5);
        cloud_color_mesh.points[i].b = int(blue/pointIdxNKNSearch.size ()+0.5);
    }
    toPCLPointCloud2(cloud_color_mesh, mesh.cloud);
    io::savePLYFile("object_mesh.ply", mesh);
}


void PointCloudMapping::cpf_seg(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr input_cloud_ptr)
{
    APC::Segmentation seg;
    APC::Config config;
    config.noise_threshold = 0.01; // 1cm
    config.voxel_resolution = 0.008f;// 0.8cm
    config.seed_resolution = 0.08f; // 8cm
    config.min_plane_area = 0.01f; // m^2;
    config.max_curvature = 0.01;
    seg.setConfig(config);
    pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
    pcl::VoxelGrid<PointT> sor;
    sor.setInputCloud(input_cloud_ptr);
    sor.setLeafSize (0.005f, 0.005f, 0.005f);
    sor.filter(*cloud_filtered);
    std::cerr << "Number of points after filtered " << cloud_filtered->size() << std::endl;
    seg.setPointCloud(input_cloud_ptr);
    seg.doSegmentation();
    pcl::PointCloud<pcl::PointXYZL>::Ptr segmented_cloud_ptr;
    segmented_cloud_ptr = seg.getSegmentedPointCloud();
    bool save_binary_pcd = false;
    pcl::io::savePCDFile ("segmentation.pcd", *segmented_cloud_ptr, save_binary_pcd);
    cout <<"finish segmentation,please see details in segmentation.pcd"<<endl;
}


pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::CEC(const std::string& file_name) {

    pcl::PointCloud<PointTypeIO>::Ptr cloud_in(new pcl::PointCloud<PointTypeIO>), cloud_out(
            new pcl::PointCloud<PointTypeIO>);
    pcl::PointCloud<PointTypeFull>::Ptr cloud_with_normals(new pcl::PointCloud<PointTypeFull>);
    pcl::IndicesClustersPtr clusters(new pcl::IndicesClusters), small_clusters(
            new pcl::IndicesClusters), large_clusters(new pcl::IndicesClusters);
    pcl::search::KdTree<PointTypeIO>::Ptr search_tree(new pcl::search::KdTree<PointTypeIO>);

    pcl::io::loadPCDFile (file_name, *cloud_in);
    pcl::VoxelGrid<PointTypeIO> vg;

    vg.setInputCloud (cloud_in);
    vg.setLeafSize (80.0, 80.0, 80.0);
    vg.setDownsampleAllData (true);
    vg.filter (*cloud_out);
    pcl::copyPointCloud (*cloud_out, *cloud_with_normals);
    pcl::NormalEstimation<PointTypeIO, PointTypeFull> ne;
    ne.setInputCloud (cloud_out);
    ne.setSearchMethod (search_tree);
    ne.setRadiusSearch (300.0);
    ne.compute (*cloud_with_normals);
    pcl::ConditionalEuclideanClustering<PointTypeFull> cec (true);
    cec.setInputCloud (cloud_with_normals);
    //cec.setConditionFunction (&customRegionGrowing);
    cec.setClusterTolerance (500.0);
    cec.setMinClusterSize (cloud_with_normals->points.size () / 1000);
    cec.setMaxClusterSize (cloud_with_normals->points.size () / 5);
    cec.segment (*clusters);
    cec.getRemovedClusters (small_clusters, large_clusters);

    // Using the intensity channel for lazy visualization of the output
    for (int i = 0; i < small_clusters->size (); ++i)
        for (int j = 0; j < (*small_clusters)[i].indices.size (); ++j)
            cloud_out->points[(*small_clusters)[i].indices[j]].intensity = -2.0;
    for (int i = 0; i < large_clusters->size (); ++i)
        for (int j = 0; j < (*large_clusters)[i].indices.size (); ++j)
            cloud_out->points[(*large_clusters)[i].indices[j]].intensity = +10.0;
    for (int i = 0; i < clusters->size (); ++i)
    {
        int label = rand () % 8;
        for (int j = 0; j < (*clusters)[i].indices.size (); ++j)
            cloud_out->points[(*clusters)[i].indices[j]].intensity = label;
    }
}

pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::ECE(PointCloud::Ptr cloud)
{
    //std::time_t start = std::time(nullptr);
    // std::cout << "PointCloud before filtering has: " << cloud->points.size() << " data points." << std::endl;
    pcl::VoxelGrid<PointT> vg;
    PointCloud::Ptr cloud_filtered (new PointCloud);
    PointCloud::Ptr cloud_f (new PointCloud);
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.01f, 0.01f, 0.01f);
    vg.filter(* cloud_filtered);
    // std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size() << " data points." << std::endl;
    // std::cout << "Wall time passed: "
    //           << std::difftime(std::time(nullptr), start) << " s.\n";

    pcl::SACSegmentation<PointT> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    PointCloud::Ptr cloud_plane (new PointCloud ());
    pcl::PCDWriter writer;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.02);

    int nr_points = (int) cloud_filtered->points.size();
    while(cloud_filtered->points.size() > 0.3 * nr_points)
    {
        seg.setInputCloud(cloud_filtered);
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size() ==0)
        {
            std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
            break;
        }

        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud (cloud_filtered);
        extract.setIndices(inliers);
        extract.setNegative(false);

        extract.filter(*cloud_plane);
        std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

        extract.setNegative(true);
        extract.filter (*cloud_f);
        *cloud_filtered = *cloud_f;
    }

    // Creating the kdTree object for the search method of the extraction
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(0.02); //2cm
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(25000);
    ec.setInputCloud(cloud_filtered);
    ec.extract(cluster_indices);
    PointCloud::Ptr cloud_cluster (new PointCloud);

    //int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {

        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end();++pit)
            cloud_cluster->points.push_back (cloud_filtered->points[*pit]);
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // std::cout << "PointCloud representing the Cluster: "<< cloud_cluster->points.size() << "data points." << std::endl;
        // std::stringstream ss;
        // ss << "cloud_cluster_"<< j << ".pcd";
        // writer.write<PointT> (ss.str(),*cloud_cluster,false);
        // j++;
    }
    return cloud_cluster;
}

pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::cylinderSeg(PointCloud::Ptr cloud)
{
    pcl::PassThrough<PointT> pass;
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
    pcl::ExtractIndices<PointT> extract;
    pcl::ExtractIndices<pcl::Normal> extract_normals;
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);

    pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<PointT>::Ptr cloud_filtered2 (new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
    pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients), coefficients_cylinder (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices), inliers_cylinder (new pcl::PointIndices);

  // Build a passthrough filter to remove spurious NaNs
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0, 1.5);
    pass.filter (*cloud_filtered);
    std::cerr << "PointCloud after filtering has: " << cloud_filtered->points.size () << " data points." << std::endl;

  // Build a passthrough filter to remove spurious NaNs
    ne.setSearchMethod (tree);
    ne.setInputCloud (cloud_filtered);
    ne.setKSearch (50);
    ne.compute (*cloud_normals);
  // Create the segmentation object for the planar model and set all the parameters

    seg.setOptimizeCoefficients(true);
    seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight(0.1);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.03);
    seg.setInputCloud(cloud_filtered);
    seg.setInputNormals(cloud_normals);
    // Obtain the plane inliers and coefficients
    seg.segment (*inliers_plane, *coefficients_plane);
    std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

    // Extract the planar inliers from the input cloud
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers_plane);
    extract.setNegative (false);
    pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT> ());
    extract.filter (*cloud_plane);

     //Remove the planar inliers, extract the rest
//     extract.setNegative (true);
//     extract.filter (*cloud_filtered2);
//     extract_normals.setNegative (true);
//     extract_normals.setInputCloud (cloud_normals);
//     extract_normals.setIndices (inliers_plane);
//     extract_normals.filter (*cloud_normals2);
//
//     // Create the segmentation object for cylinder segmentation and set all the parameters
//     seg.setOptimizeCoefficients (true);
//     seg.setModelType (pcl::SACMODEL_CYLINDER);
//     seg.setMethodType (pcl::SAC_RANSAC);
//     seg.setNormalDistanceWeight (0.1);
//     seg.setMaxIterations (10000);
//     seg.setDistanceThreshold (0.05);
//     seg.setRadiusLimits (0, 0.1);
//     seg.setInputCloud (cloud_filtered2);
//     seg.setInputNormals (cloud_normals2);
//
//     // Obtain the cylinder inliers and coefficients
//     seg.segment (*inliers_cylinder, *coefficients_cylinder);
//     std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;

//     // Write the cylinder inliers to disk
//     extract.setInputCloud (cloud_filtered2);
//     extract.setIndices (inliers_cylinder);
//     extract.setNegative (false);
//     pcl::PointCloud<PointT>::Ptr cloud_cylinder (new pcl::PointCloud<PointT> ());
//     extract.filter (*cloud_cylinder);

    return cloud_plane;
}


pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::regionGrowingSeg(PointCloud::Ptr cloud_in)
{

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  PointCloudXYZRGBAtoXYZ(*cloud_in,*cloud);
  pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (cloud);
  normal_estimator.setKSearch (50);
  normal_estimator.compute (*normals);

  pcl::IndicesPtr indices (new std::vector <int>);
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 1.0);
  pass.filter (*indices);

  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize (50);
  reg.setMaxClusterSize (1000000);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (30);
  reg.setInputCloud (cloud);
  //reg.setIndices (indices);
  reg.setInputNormals (normals);
  reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold (1.0);

  std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters);
//
//  std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;
//  std::cout << "First cluster has " << clusters[0].indices.size () << " points." << endl;
//  std::cout << "These are the indices of the points of the initial" <<
//    std::endl << "cloud that belong to the first cluster:" << std::endl;
  int counter = 0;
  while (counter < clusters[0].indices.size ())
  {
    std::cout << clusters[0].indices[counter] << ", ";
    counter++;
    if (counter % 10 == 0)
      std::cout << std::endl;
  }

  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
//  pcl::visualization::CloudViewer viewer ("Cluster viewer");
//  viewer.showCloud(colored_cloud);
//  while (!viewer.wasStopped ())
//  {
//  }

  pcl::PointCloud <pcl::PointXYZRGBA>::Ptr cloud_out (new pcl::PointCloud <pcl::PointXYZRGBA>);
  PointCloudXYZRGBtoXYZRGBA(*colored_cloud, *cloud_out);
  return cloud_out ;
}


pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::colorRegionGrowingSeg(pcl::PointCloud <pcl::PointXYZRGB>::Ptr  cloud)
{

    pcl::search::Search <pcl::PointXYZRGB>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZRGB> > (new pcl::search::KdTree<pcl::PointXYZRGB>);;

    pcl::IndicesPtr indices (new std::vector <int>);
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.0, 1.0);
    pass.filter (*indices);

    pcl::RegionGrowingRGB<pcl::PointXYZRGB> reg;
    reg.setInputCloud (cloud);
    reg.setIndices (indices);
    reg.setSearchMethod (tree);
    reg.setDistanceThreshold (10);
    reg.setPointColorThreshold (6);
    reg.setRegionColorThreshold (5);
    reg.setMinClusterSize (600);

    std::vector <pcl::PointIndices> clusters;
    reg.extract (clusters);

    pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();


    pcl::PointCloud <pcl::PointXYZRGBA>::Ptr cloud_out (new pcl::PointCloud <pcl::PointXYZRGBA>);
    PointCloudXYZRGBtoXYZRGBA(*colored_cloud, *cloud_out);
    return cloud_out;
}



bool
enforceIntensitySimilarity (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
    if (fabs (point_a.intensity - point_b.intensity) < 5.0f)
        return (true);
    else
        return (false);
}

bool
enforceCurvatureOrIntensitySimilarity (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
    Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
    if (fabs (point_a.intensity - point_b.intensity) < 5.0f)
        return (true);
    if (fabs (point_a_normal.dot (point_b_normal)) < 0.05)
        return (true);
    return (false);
}

bool customRegionGrowing (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
    Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
    if (squared_distance < 10000)
    {
        if (fabs (point_a.intensity - point_b.intensity) < 8.0f)
            return (true);
        if (fabs (point_a_normal.dot (point_b_normal)) < 0.06)
            return (true);
    }
    else
    {
        if (fabs (point_a.intensity - point_b.intensity) < 3.0f)
            return (true);
    }
    return (false);
}
