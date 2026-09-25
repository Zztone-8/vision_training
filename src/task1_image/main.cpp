/* 作业要求 
1. 
读图与颜⾊转换：检查读图是否成功；输出灰度图以及 H、S、V 单通道图。(4)
2. 
滤波对⽐：实现均值、⾼斯、中值滤波，记录核尺⼨和标准差等参数，对⽐花瓣边缘与细节的变化。(3)
3. 
红⾊提取：使⽤ HSV 双区间阈值⽣成红⾊掩膜，记录阈值。红⾊、⻩⾊边缘和阴影的处理效果应分别说明。(1)
4. 
形态学与轮廓：
对掩膜分别展⽰腐蚀、膨胀、开运算、闭运算的效果；选择合适结果提取外轮廓，按⾯积筛选，在原图上绘制轮廓和外接矩形，
并在结果图或总 README 中标明筛选后的轮廓⾯积。(5)
5. 
绘制与变换：在原图副本上绘制圆、矩形和⽂字；绕图像中⼼旋转 35°；单独裁剪原图左上 1/4，即原宽、原⾼各取⼀半。(3)

框选对象是红⾊连通区域，不要求⼀个框恰好对应⼀朵完整的花。共16张图
*/

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
int step_5_assignment(const Mat& img)
{
    int h =img.rows;
    int w =img.cols;
    Mat roi = img(Rect(0,0,w/2,h/2)).clone();//剪裁
    Point2f center(w/2.0,h/2.0);//定义旋转中心
    Mat M= getRotationMatrix2D(center, 35,1.0);//中心，角度，缩放比
    Mat rotated;
    warpAffine(img, rotated, M,img.size());
    Mat text = img.clone();                             
    rectangle(text, Point(100, 100), Point(300, 200), Scalar(0, 255, 0), 2);
    circle(text, Point(400, 400), 80, Scalar(0, 0, 255), 3);
    putText(text, "NO HOMEWORK!!!", Point(90, 90),
            FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);

    imwrite("./result/task1_images/drawing.png",text);
    imwrite("./result/task1_images/rotated_35deg.png",rotated);
    imwrite("./result/task1_images/crop_top_left.png",roi);
    return 0;

}


int step_4_assignment(const Mat& img)
{
    Mat gray, binary;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, 128, 255,THRESH_BINARY);
    Mat kernel =getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat dilated, eroded, opened, closed;
    dilate(binary, dilated, kernel);
    erode(binary, eroded, kernel);
    morphologyEx(binary, opened, MORPH_OPEN, kernel);
    morphologyEx(binary, closed, MORPH_CLOSE, kernel);

   
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, 128, 255,THRESH_BINARY);
    std::vector<std::vector<cv::Point>>contours;
    findContours(binary ,contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat result=img.clone();
    for(size_t i=0; i< contours.size();i++)
    {
        Rect box = boundingRect(contours[i]);
        if(box.area()<500) continue;
        drawContours(result ,contours,(int)i,Scalar(0,255,0),2);
        rectangle(result,box,Scalar(0,0,255),2);
        putText(result,"Area: "+std::to_string(box.area()),
                    Point(box.x, box.y-5),
                    FONT_HERSHEY_SIMPLEX, 0.5,
                    Scalar(255,255,255),1);
    }
    imwrite("./result/task1_images/close.png",closed);
    imwrite("./result/task1_images/dilate.png",dilated);
    imwrite("./result/task1_images/erode.png",eroded);
    imwrite("./result/task1_images/open.png",opened);
    imwrite("./result/task1_images/contours_boxs.png",result);
    return 0;
}

int step_3_assignment(const Mat& img)
{
    Mat hsv, maskLow, maskHigh, mask;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    inRange(hsv,Scalar(0, 100, 100), Scalar(10, 255, 255),maskLow);
    inRange(hsv, Scalar(170, 100, 100),Scalar(180, 255, 255),maskHigh);
    bitwise_or(maskLow, maskHigh, mask);
    imwrite("./result/task1_images/red_mask.png",mask);
    return 0;

}



int step_2_assignment(const Mat& img)
{
    Mat meanImg, gaussianImg, medianImg;
    blur(img, meanImg, Size(5,5));img,
    GaussianBlur(img, gaussianImg, Size(5,5), 1.5);
    medianBlur(img, medianImg, 5);
 
    imwrite("./result/task1_images/mean_filter.png",meanImg);
    imwrite("./result/task1_images/gaussian_filter.png",gaussianImg);
    imwrite("./result/task1_images/median_filter.png",medianImg);

    return 0;
}

// 读图与颜⾊转换：检查读图是否成功；输出灰度图以及 H、S、V 单通道图。(4)
int step_1_assignment(const Mat& img)
{
    //输出灰度图
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    // imshow("Gray", gray);
    imwrite("./result/task1_images/gray.png", gray);

    //输出H、S、V单通道图
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    // 拆分通道
    std::vector<Mat> channels;
    // channels[0]=H, channels[1]=S, channels[2]=V
    split(hsv, channels);   

    Mat H = channels[0];
    Mat S = channels[1];
    Mat V = channels[2];

    // 显示
    // imshow("H通道", H);
    // imshow("S通道", S);
    // imshow("V通道", V);
    // waitKey(0);

    imwrite("./result/task1_images/hsv_h.png", H);
    imwrite("./result/task1_images/hsv_s.png", S);
    imwrite("./result/task1_images/hsv_v.png", V);
    
    return 0;
}


int main() {

    // 统一读一次图
    Mat img = imread("./resources/test_image.jpg");
    if (img.empty()) {
        std::cerr << "Cannot read image!\n";
        return 1;
    }

    cout << "step 1 begining \n" ;
    step_1_assignment(img);   
    cout << "step 1 finished \n";
    cout << "step 2 begining \n" ;
    step_2_assignment(img);   
    cout << "step 2 finished \n";
    cout << "step 3 begining \n";
    step_3_assignment(img);
    cout << "step 3 finished \n";
    cout << "step 4 begining \n";
    step_4_assignment(img);
    cout << "step 4 finished \n";
     cout << "step 5 begining \n";
    step_5_assignment(img);
    cout << "step 5 finished \n";

    
    return 0;
}
