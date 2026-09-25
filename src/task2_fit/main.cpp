/*
1. 识别⻘⾊⽬标，提取其中⼼坐标，计算并展开⻆度。
2. 选择上述⼀种⽅式估计参数；优化或求解⽅法不限，说明⽅法、约束，
以及适⽤时的初值与求解状态。
3. 在同⼀图中展⽰观测点与对应拟合曲线，绘制残差图，⾄少报告 RMSE。
⻆度拟合报告⻆度误差，⻆速度拟合报告⻆速度误差，
并写明单位、有效样本数及参与计算的帧范围。
4. 输出估计的⻆速度曲线及带识别标记的视频
*/
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <ceres/ceres.h>

using namespace cv;
using namespace std;
using namespace ceres;


// 存储计算出来的运动点坐标（x，y）；已知视频文件总共1440帧

cv::Point2d targetCenterPoint[1442];    //目标中心点坐标（像素）
double angleRaw[1442];                  //atan2 原始角，范围（-pi， pi)
double angleUnwrapped[1442];            //展开角（连续、无正负pi的跳变)


double para_A = 0;
double para_b = 0;
double para_phi = 0;
double para_omega = 0;


// 在 HSV 图像中提取青色目标质心；找不到返回 false
static bool extractCyanCenter(const cv::Mat& hsv,
                              const cv::Mat& kernel,
                              cv::Point2d& center)
{
    // 1) 青色掩码：OpenCV 中 H 范围 0~179，青色约在 90 附近
    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(85, 120, 120), cv::Scalar(100, 255, 255), mask);

    // 2) 开运算去掉椒盐噪声
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    // 3) 找轮廓，取面积最大的连通域作为目标
    vector<vector<Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return false;
    }
    size_t best = max_element(contours.begin(), contours.end(),
                              [](const vector<Point>& a, const vector<Point>& b) {
                                  return cv::contourArea(a) < cv::contourArea(b);
                              }) - contours.begin();
    if (cv::contourArea(contours[best]) < 20.0) {  // 太小的误检丢弃
        return false;
    }

    // 4) 矩法求质心
    cv::Moments m = cv::moments(contours[best]);
    if (m.m00 <= 0) {
        return false;
    }
    center = cv::Point2d(m.m10 / m.m00, m.m01 / m.m00);

    return true;
}

//生成一个视频文件
int makeNewVideo()
{
    cv::VideoCapture cap;
    cap.open("./resources/task_2.mp4");
    if (!cap.isOpened()) {
        cerr << "cannot open video\n";
        return 0;
    }

    // 输出帧率尽量与源一致，取不到则按 60fps
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0 || fps > 1000) {
        fps = 60.0;
    }
    int width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::VideoWriter writer;
    writer.open("./result/task2_fit/tracking_overlay.mp4",
                cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        cerr << "cannot open video writer\n";
        return 0;
    }

    cv::Mat frame;
    int frameNumber = 0;
    while (cap.read(frame)) {
        double dt = 1.0/60;
        // ω(t) = 1.35 + 0.5499*sin(1.6498*t + 0.7)，t = frameNumber/60 秒
        // double t = (frameNumber * dt + (frameNumber+1) * dt) / 2;
        double t = frameNumber * dt ;
        double result = para_b + para_A * sin(para_omega * t + para_phi);
        int x, y;
        double tht;

        // 左上角标注："{framenumber} {result}"
        char buf[128];
        snprintf(buf, sizeof(buf), "frameIdx=%d", frameNumber);
        cv::putText(frame, buf, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

        snprintf(buf, sizeof(buf), "angular_velocity=%.4f", result);
        cv::putText(frame, buf, cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);


        //按照公式计算点位置
        // θ(t) = θ0 + b*t + (A/Ω)*(cos(φ) − cos(Ωt+φ))
        //x(t)=x0 +r*cos(θt)
        //y(t)= y0 -r*cos(θt)

        tht = angleUnwrapped[0] + para_b*t + (para_A/para_omega)*(cos(para_phi) - cos(para_omega*t + para_phi));

        x = 480 + 220*cos(tht);
        y = 360 - 220*sin(tht);
        // 去一个8像素的红色点，分帧能看到总是在青色小球中的，说明拟合没有问题
        circle(frame, Point(x, y), 8, Scalar(0, 0, 255), -1);

        // ===== 新增：红色小球中心坐标标注 =====
        char coordBuf[64];
        snprintf(coordBuf, sizeof(coordBuf), "(%d, %d)", x, y);

        int    fontFace  = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.6;
        int    thick     = 2;
        int    baseline  = 0;
        cv::Size ts = cv::getTextSize(coordBuf, fontFace, fontScale, thick, &baseline);

        // 默认放在小球右上角，避免压住小球本身
        int tx = x + 15;
        int ty = y - 15;

        // 越界保护：右边放不下就放左边，上边放不下就放下边
        if (tx + ts.width > width - 5)  tx = x - 15 - ts.width;
        if (ty - ts.height < 5)         ty = y + 15 + ts.height;
        tx = std::max(5, std::min(tx, width  - ts.width - 5));
        ty = std::max(ts.height + 5, std::min(ty, height - 5));

        cv::putText(frame, coordBuf, cv::Point(tx, ty),
                    fontFace, fontScale, cv::Scalar(0, 0, 255), thick);
        // =====================================

        writer.write(frame);

        // imshow("residuals", frame);
        // waitKey(0);


        frameNumber++;
    }

    writer.release();
    cap.release();
    cout << "tracking_overlay.mp4 written, frames: " << frameNumber << "\n";

    return 0;
}

// 识别⻘⾊⽬标，提取其中⼼坐标，计算并展开⻆度。
// 将获取的数据存储到全局变量中
int identfiyTarget()
{
    cv::VideoCapture cap;
    cap.open("./resources/task_2.mp4");
    if (!cap.isOpened()) {
        cerr << "cannot open video\n";
        return 0;
    }
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat frame;
    cv::Point2d rotCenter(480.0, 360.0);
    
    int frameIdx = 0;    // 帧号，从0到1439

    // 一帧一帧读
    while (cap.read(frame)) 
    {
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        cv::Point2d center;
        if (extractCyanCenter(hsv, kernel, center)) 
        {
            // 存储检测目标的检测中心点
            targetCenterPoint[frameIdx] = center;

            
            // 计算每一帧的角; 图像 y 轴向下，取负使逆时针为正
            angleRaw[frameIdx] = atan2(-(center.y-rotCenter.y), center.x - rotCenter.x);

            // 假设是第0帧
            if (frameIdx == 0)
            {
                angleUnwrapped[frameIdx] = angleRaw[frameIdx];
            }
            else
            {
                
                double delta = angleRaw[frameIdx] - angleRaw[frameIdx-1];
                // 处理回绕：如果相邻帧的差值超过 pi，说明发生了跳变
                if (delta > CV_PI)  delta -= 2 * CV_PI;
                if (delta < -CV_PI) delta += 2 * CV_PI;
                angleUnwrapped[frameIdx] = angleUnwrapped[frameIdx-1] + delta;

            }
            // cout << "frame idx :"<< frameIdx << " pos = " << targetCenterPoint[frameIdx] << " angleRaw = " << angleRaw[frameIdx] << " angleUnRaw = " << angleUnwrapped[frameIdx] << "\n"; 
        }
        else
        {
            cout << "搜素中心点失败了\n";
        }

        frameIdx++;
        if (frameIdx >= 1440)
        {
            break;
        }
    }
    
    return 0;
}

// 检验中心点检测的质量，统计计算半径的最大和最小值
int verifyComputeResult()
{
    //统计计算半径的最小值和最大值
    double redius_max = -1.0;
    double redius_min = 300.0;
    int i;

    // 旋转中心（白色圆点）
    cv::Point2d rotCenter(480.0, 360.0);
    cout << "旋转的中心点：x="<<rotCenter.x << " y:"<<rotCenter.y<<"\n";

    for (i = 0; i < 1440; i++)
    {
        double dx = std::abs(targetCenterPoint[i].x - rotCenter.x);  // 直角边1
        double dy = std::abs(targetCenterPoint[i].y - rotCenter.y);  // 直角边2
        double r = std::hypot(dx, dy);

        if (redius_max < r) 
        {
            redius_max = r;
        }
        
        if (redius_min > r)
        {
            redius_min = r;
        }

    }

    cout << "统计出青色小球中心，距离运动圆心半径的范围：[" << redius_min << ", " << redius_max << "]\n";
    return 0;
}

struct SineResidual {
    SineResidual(double t, double omega) : t_(t), omega_(omega) {}
    template <typename T>
        bool operator()(const T* const params, T* residual) const {
            T prediction = params[1] + params[0] * ceres::sin(params[2] * T(t_) + params[3]);
            residual[0] = T(omega_) - prediction;
            return true;
        }
    private:
        const double t_;
        const double omega_;
};

//Ceres 非线性拟合
int usingCeresFitting()
{
    double best_omega_init = 0.5;
    double best_cost = 1e30;
    double b = 1.5;
    double A = 0.5;
    double phase = 0;
    double t = 0;
    double dt = 1.0/60.0;
    double omega;
    
    // 已知总共运行圈数不大于5圈，时间24秒
    // Omega 从 0.5 到 5.0，步长 0.01 
    for (double omega_test = 0.5; omega_test <= 5.0; omega_test += 0.1) 
    {
        // ω(t) = b+A*sin(Ωt+φ)
        

        // 手动算一遍所有残差的平方和
        double cost = 0;
        for (size_t i = 0; i < 1439; i++) 
        {            
            double pred; 
            double r;
            
        
            //t是时间，去1/60的中心时间
            //t = (i * dt + (i+1) * dt) / 2;
            t = i * dt ;
            pred = b + A * sin(omega_test * t + phase);
            omega = (angleUnwrapped[i+1] - angleUnwrapped[i]) / dt;
            r = omega - pred;
            cost += r * r;
        }
        if (cost < best_cost) {
            best_cost = cost;
            best_omega_init = omega_test;
        }
    }
    std::cout << "粗扫 Omega : " << best_omega_init 
              << " (cost=" << best_cost << ")" << std::endl;

    // 用粗扫的结果作为 Omega 的初值，A 和 b 也可以根据数据给个合理的初值
    double params[4] = {0.5, 1.5, best_omega_init, 0.0};    
    ceres::Problem problem;   
    for (size_t i = 0; i <1439; i++ ) 
    {
        //t = (i * dt + (i+1) * dt) / 2;
        t = i * dt ;
        omega = (angleUnwrapped[i+1] - angleUnwrapped[i]) / dt;
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<SineResidual, 1, 4>(new SineResidual(t, omega));

        problem.AddResidualBlock(cost_function, nullptr, params);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);


    // cout << summary.BriefReport() << "\n";
    cout << "估计的 A     = " << params[0] << "\n";
    cout << "估计的 b     = " << params[1] << "\n";
    cout << "估计的 Omega = " << params[2] << "\n";
    // 把 phi 归一化到 [-pi, pi)
    double phi = params[3];
    while (phi >= M_PI) phi -= 2 * M_PI;
    while (phi < -M_PI) phi += 2 * M_PI;
    cout << "估计的 phi   = " << phi << "\n";

    para_A = params[0];
    para_b = params[1];
    para_omega = params[2];
    para_phi = phi;


    return 0;

}


// putText 不支持旋转：先画到小图上，整图逆时针转 90°，再贴回 canvas
// center 是旋转后文字的中心位置；要求 canvas 背景为白色
static void putVerticalText(cv::Mat& canvas, const std::string& text,
                            cv::Point center, double fontScale,
                            cv::Scalar color, int thickness)
{
    const int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    int baseline = 0;
    cv::Size ts = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);

    // 1) 小画布：白底横排文字，四周留 3px 边距
    cv::Mat txt(cv::Size(ts.width + 6, ts.height + baseline + 6), CV_8UC3,
                cv::Scalar(255, 255, 255));
    cv::putText(txt, text, cv::Point(3, ts.height + 3), fontFace, fontScale,
                color, thickness);

    // 2) 逆时针旋转 90°（自下往上读，y 轴标签惯例）：transpose + 上下翻转
    cv::Mat rot;
    cv::transpose(txt, rot);
    cv::flip(rot, rot, 0);

    // 3) 按中心点对齐贴回 canvas，只覆盖非白色（文字）像素，越界自动裁剪
    int x0 = center.x - rot.cols / 2;
    int y0 = center.y - rot.rows / 2;
    cv::Rect dst(x0, y0, rot.cols, rot.rows);
    dst &= cv::Rect(0, 0, canvas.cols, canvas.rows);
    cv::Rect src(dst.x - x0, dst.y - y0, dst.width, dst.height);
    cv::Mat mask;
    cv::compare(rot(src), cv::Scalar(255, 255, 255), mask, cv::CMP_NE);
    rot(src).copyTo(canvas(dst), mask);
}

//产生带文字的报表图片
int makeReport()
{
    int W = 900, H = 570;
    //拟合图
    Mat canvas(H, W, CV_8UC3, Scalar(255, 255, 255)); 

    // 边距：左100，右100，上50，下50，中间是绘图区
    int margin_left = 100, margin_right = 100, margin_top = 50, margin_bottom = 50;
    int plot_w = W - margin_left - margin_right;  
    int plot_h = H - margin_top - margin_bottom;  

    double t_min = 0.0, t_max = 24.0;
    double w_min = 0.0, w_max = 2.5;
    double t;
    double omega;
    double dt = 1.0/60.0;

    // 数据坐标 (t 秒, omega rad/s) → 图像像素坐标，两处画图统一走这里
    auto toPixel = [&](double tt, double ww) -> Point {
        int px = margin_left + (int)((tt - t_min) / (t_max - t_min) * plot_w);
        int py = (H - margin_bottom) - (int)((ww - w_min) / (w_max - w_min) * plot_h); // 图像 y 轴向下
        return Point(px, py);
    };

    line(canvas, Point(margin_left, margin_top), Point(margin_left, H - margin_bottom), Scalar(0,0,0), 2); // y轴
    line(canvas, Point(margin_left, H - margin_bottom), Point(W - margin_right, H - margin_bottom), Scalar(0,0,0), 2); // x轴
    putText(canvas, "t (s)", Point(W/2, H-10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,0), 2);
    // y 轴标签：竖排，中心放在 y 轴左侧、绘图区垂直中间
    putVerticalText(canvas, "omega (rad/s)", Point(30, margin_top + plot_h / 2), 0.7, Scalar(0,0,0), 2);

    for (size_t i = 0; i < 1439; i++)
    {
        //t = (i * dt + (i+1) * dt) / 2;
        t = i * dt ;
        omega = (angleUnwrapped[i+1] - angleUnwrapped[i]) / dt;
        Point p = toPixel(t, omega);   // 直接传数据坐标，换算统一在 toPixel 里
        circle(canvas, p, 3, Scalar(255, 0, 0), -1);
    }

    vector<Point> fit_points;
    for (double t = t_min; t <= t_max; t += 0.05) {
        double pred = para_b + para_A * sin(para_omega * t + para_phi);
        fit_points.push_back(toPixel(t, pred));
    }
    // 用 polylines 一次性画整条折线
    polylines(canvas, fit_points, false, Scalar(0, 0, 255), 2); // BGR：(0,0,255) 是红色


    string info = "A=" + to_string(para_A) + ", b=" + to_string(para_b) 
                + ", Omega=" + to_string(para_omega) + ", phi=" + to_string(para_phi);
    putText(canvas, info, Point(margin_left, margin_top - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,0,0), 1);



    imwrite("./result/task2_fit/fit_comparison.png", canvas);
    cout << "Saved fit_comparison.png" << endl;
    // imshow("Fit Comparison", canvas);
    // waitKey(0);

    //残差图
    Mat canvas_res(H, W, CV_8UC3, Scalar(255, 255, 255));

    double r_min = -0.2, r_max = 0.2;

    auto toPixelRes = [&](double t, double res) {
        int x = margin_left + (int)((t - t_min) / (t_max - t_min) * plot_w);
        int y = H - margin_bottom - (int)((res - r_min) / (r_max - r_min) * plot_h);
        return Point(x, y);
    };

    
    // 画轴
    line(canvas_res, Point(margin_left, margin_top), Point(margin_left, H-margin_bottom), Scalar(0,0,0), 2);
    line(canvas_res, Point(margin_left, H-margin_bottom), Point(W-margin_right, H-margin_bottom), Scalar(0,0,0), 2);
    putText(canvas_res, "time t (s)", Point(W/2, H-10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,0), 2);
    // putText(canvas_res, "Residual", Point(10, H/2), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,0), 2);
    putVerticalText(canvas_res, "residual (rad/s)", Point(30, margin_top + plot_h / 2), 0.7, Scalar(0,0,0), 2);
    int zero_y = H / 2;
    line(canvas_res, Point(100, zero_y), Point(W - 100, zero_y), Scalar(0, 0, 255), 2);

    double redius_max = 0;
    double redius_min = 1000;
    // 画残差
    for (size_t i = 0; i < 1439; i++) {
        
        double pred;
        double res;
        //t =  (i * dt + (i+1) * dt) / 2;
        t = i * dt;
        pred = para_b + para_A * sin(para_omega * t + para_phi);
        omega = (angleUnwrapped[i+1] - angleUnwrapped[i]) / dt;
        res = omega - pred;
        if (res > redius_max)
        {
            redius_max = res;
        }
        if (res < redius_min)
        {
            redius_min = res;
        }

        circle(canvas_res, toPixelRes(t, res), 3, Scalar(255, 0, 0), -1);
    }    

    info = "redius = [" + to_string(redius_min) + ",  " + to_string(redius_max) 
                + "]";
    putText(canvas_res, info, Point(margin_left, margin_top - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,0,0), 1);


    imwrite("./result/task2_fit/residuals.png", canvas_res);
    cout << "Saved residuals.png" << endl;
    // imshow("residuals", canvas_res);
    // waitKey(0);

    // 画角速度
    Mat canvas_omega(H, W, CV_8UC3, Scalar(255, 255, 255));

    line(canvas_omega, Point(margin_left, margin_top), Point(margin_left, H-margin_bottom), Scalar(0,0,0), 2);
    line(canvas_omega, Point(margin_left, H-margin_bottom), Point(W-margin_right, H-margin_bottom), Scalar(0,0,0), 2);

    // 画观测角速度的折线（蓝色）
    vector<Point> obs_line;
    for (size_t i = 0; i < 1439; i++) {
        //t =  (i * dt + (i+1) * dt) / 2;
        t = i* dt;
        omega = (angleUnwrapped[i+1] - angleUnwrapped[i]) / dt;


        int x = margin_left + (int)((t - t_min) / (t_max - t_min) * plot_w);
        int y = H - margin_bottom - (int)((omega - w_min) / (w_max - w_min) * plot_h);
        
        
        obs_line.push_back(Point(x,y));
    }
    polylines(canvas_omega, obs_line, false, Scalar(255, 0, 0), 1);

    // 叠加拟合曲线（红色）
    vector<Point> fit_line;
    for (double t = t_min; t <= t_max; t += 0.05) {
        double pred = para_b + para_A * sin(para_omega * t + para_phi);
        int x = margin_left + (int)((t - t_min) / (t_max - t_min) * plot_w);
        int y = H - margin_bottom - (int)((pred - w_min) / (w_max - w_min) * plot_h);


        fit_line.push_back(Point(x, y));
    }
    polylines(canvas_omega, fit_line, false, Scalar(0, 0, 255), 2);

    info = "The red line represents the ideal curve, while the blue line represents the fitted curve";
    putText(canvas_omega, info, Point(margin_left-50, margin_top - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,0,0), 1);

    imwrite("./result/task2_fit/angular_velocity.png", canvas_omega);
    cout << "angular_velocity.png" << endl;
    // imshow("residuals", canvas_res);
    // waitKey(0);


    return 0;

}

// 选择上述⼀种⽅式估计参数；优化或求解⽅法不限，说明⽅法、约束，
// 以及适⽤时的初值与求解状态。
int main()
{
    int i;
    ofstream ofs("./result/task2_fit/task2_obs.csv");   // 逐帧观测结果，供检查与拟合
    ofs << "frame,x_px,y_px,angle_raw_rad,angle_unwrap_rad, omega\n";
    //识别⻘⾊⽬标，提取其中⼼坐标，计算并展开⻆度。
    identfiyTarget();

    // 检验得到数据
    verifyComputeResult();

    // 将计算的1440组数据存起来放在csv文件中
    for (i = 0; i < 1440; i++)
    {
        // 最后一帧不计算omega
        double dt = 1.0/60.0;
        double omega;

        if (i < 1439)
        {
            omega = (angleUnwrapped[i+1] - angleUnwrapped[i]) / dt;
        }
        else
        {
            omega = 0.0;
        }

        ofs << i << ',' << targetCenterPoint[i].x << ',' << targetCenterPoint[i].y << ','
                << angleRaw[i] << ',' << angleUnwrapped[i] << ',' << omega <<'\n';
    }
    
    // 使用ceres做非线性拟合
    usingCeresFitting();


    // 生成报告
    makeReport();

    //生成带角速度标注的叠加视频
    makeNewVideo();
    // cout << "makeNewVideo is finished\n";

    return 0;
}