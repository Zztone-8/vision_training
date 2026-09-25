#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <map>
#include <sstream>
#include <iomanip>

using namespace cv;
using namespace std;

// 检测到的圆形靶标（由至少两个同心圆组成的发光图案：外圆 + 内圆 + 中心）
struct CircleTarget {
    cv::Point2f center;
    float radius;
    float score;
};

// 沿半径方向统计同心圆笔画的个数（径向剖面）。
// 对每个半径 rho，在圆周上均匀采样 60 个点，计算落在橙色笔画上的比例（覆盖率）；
// 覆盖率连续达标的半径区间记为一个"笔画环带"。
// 单一圆环（如风车臂末端的装饰圈）只有 1 个环带；
// 真靶标（外圆 + 内圆 + 中心点）有 >=2 个环带。
static int countConcentricRings(const cv::Mat& maskRaw, const cv::Point2f& c, float r)
{
    const double th = 0.3; // 环带判定阈值：该圆周上橙色像素占比
    int bands = 0;
    bool inBand = false;
    for (double rho = 1.0; rho <= r * 1.25; rho += 0.5) {
        int total = 0, lit = 0;
        for (int k = 0; k < 60; ++k) {
            const double a = 2.0 * CV_PI * k / 60.0;
            const int x = cvRound(c.x + rho * std::cos(a));
            const int y = cvRound(c.y + rho * std::sin(a));
            if (x < 0 || x >= maskRaw.cols || y < 0 || y >= maskRaw.rows) continue;
            ++total;
            if (maskRaw.at<uchar>(y, x)) ++lit;
        }
        const double cov = total ? double(lit) / total : 0.0;
        if (cov >= th && !inBand) { ++bands; inBand = true; }
        else if (cov < th) { inBand = false; }
    }
    return bands;
}

// 检测一帧图像中的圆形靶标。
// 思路：靶标是发光的橙色同心圆图案（至少 2 个同心圆）
// -> 先做 HSV 橙色阈值，闭运算补全笔画断点，
// 再提取"孔洞"轮廓（被圆笔画围住的封闭内部区域），
// 用半径 / 圆形度 / 填充率筛出圆形，
// 然后对候选做同心圆计数，环带数 <2 的（单一圆环）忽略，最后做去重（NMS）。
std::vector<CircleTarget> detectCircleTargets(const cv::Mat& frame)
{
    std::vector<CircleTarget> targets;

    cv::Mat hsv, maskLo, maskHi, maskRaw, mask;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    // 阈值放得较松：暗场里靶标笔画偏暗偏细（V 低至 50 左右），
    // 太严格会导致圆环笔画断裂、闭合孔洞检测失败
    cv::inRange(hsv, cv::Scalar(0,   80, 50), cv::Scalar(30,  255, 255), maskLo);
    cv::inRange(hsv, cv::Scalar(166, 80, 50), cv::Scalar(180, 255, 255), maskHi);
    cv::bitwise_or(maskLo, maskHi, maskRaw);
    maskRaw.copyTo(mask);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);

    std::vector<CircleTarget> candidates;
    for (size_t i = 0; i < contours.size(); ++i) {
        // RETR_CCOMP 下 hierarchy[i][3] != -1 的是"孔洞"轮廓（圆环内部的封闭区域）
        if (hierarchy[i][3] == -1) continue;
        const std::vector<cv::Point>& c = contours[i];
        if (c.size() < 6) continue;

        cv::Point2f ctr;
        float r = 0.f;
        cv::minEnclosingCircle(c, ctr, r);
        if (r < 10.f || r > 90.f) continue;   // 靶标像素半径范围（远小近大）

        const double area = cv::contourArea(c);
        const double perim = cv::arcLength(c, true);
        if (perim <= 0.0) continue;
        const double circularity = 4.0 * CV_PI * area / (perim * perim); // 越接近1越圆
        const double fill = area / (CV_PI * r * r);                      // 孔洞接近其外接圆
        if (circularity < 0.55 || fill < 0.70) continue;

        // 只保留由至少两个同心圆组成的靶标，单一圆环忽略
        if (countConcentricRings(maskRaw, ctr, r) < 2) continue;

        candidates.push_back({ ctr, r, float(circularity * fill) });
    }

    // 去重：同一靶标内外边界或多个候选重叠时只保留得分最高的一个
    std::sort(candidates.begin(), candidates.end(),
              [](const CircleTarget& a, const CircleTarget& b) { return a.score > b.score; });
    for (const CircleTarget& t : candidates) {
        bool dup = false;
        for (const CircleTarget& k : targets) {
            const double d = cv::norm(t.center - k.center);
            if (d < 0.8 * std::max(t.radius, k.radius)) { dup = true; break; }
        }
        if (!dup) targets.push_back(t);
    }
    return targets;
}

// 箭头链检测结果（HUD 中指向靶标的 ">>>>" 雪佛龙图标）
struct ArrowInfo {
    cv::Point2f center;                    // 箭头链质心
    std::vector<cv::Point2f> points;       // 链上全部像素点（用于画外接框）
    int blobCount = 0;                     // 聚类内连通块数（分立的雪佛龙个数）
    double angleDeg = 0.0;                 // 箭头质心 -> 靶标圆点 连线与图像水平轴的夹角
                                           // （数学约定：x 轴向右、y 轴向上，逆时针为正，单位度）
    const CircleTarget* target = nullptr;  // 箭头指向的靶标
};

// 检测指向某个靶标的箭头链。
// 思路：雪佛龙笔画是亮橙色小glyph（H4-18、S>=150、V>=130，与靶环红色 H~5、
// 黄色臂条 H~20 不同），先用颜色阈值并抹掉靶标自身（半径 2.2r 内），
// 把环带 2.3r-7r 内的碎片按质心距离 <28px 聚成链；
// 对每个聚类做直线拟合（fitLine）：长 35-160px、垂直厚度细长、
// 且链的长轴方向要正对靶心（与"质心->靶心"向量夹角 <=28 度），
// 由此排除 "R" 字母、能量立方支架等噪声。
static bool detectArrowChain(const cv::Mat& frame, const CircleTarget& target, ArrowInfo& info)
{
    const cv::Point2f& tc = target.center;
    const float tr = target.radius;

    cv::Mat hsv, m;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(4, 150, 130), cv::Scalar(18, 255, 255), m);
    cv::circle(m, tc, cvRound(tr * 2.2f), cv::Scalar(0), -1); // 抹掉靶标圆环与刻度

    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(m, labels, stats, centroids, 8);

    struct Blob { std::vector<cv::Point2f> pts; double cx, cy; int area; };
    std::vector<Blob> blobs;
    for (int i = 1; i < n; ++i) {
        const int x = stats.at<int>(i, cv::CC_STAT_LEFT), y = stats.at<int>(i, cv::CC_STAT_TOP);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        const int a = stats.at<int>(i, cv::CC_STAT_AREA);
        if (a < 8 || a > 3000 || w > 130 || h > 130) continue;
        const double cx = centroids.at<double>(i, 0), cy = centroids.at<double>(i, 1);
        const double d = std::hypot(cx - tc.x, cy - tc.y);
        if (d < tr * 2.3 || d > tr * 7.0) continue;           // 距靶心的环带范围
        if (w <= 16 && h <= 16 && a > 0.85 * w * h) continue; // 实心小圆点不是箭头笔画
        Blob b; b.cx = cx; b.cy = cy; b.area = a; b.pts.reserve(a);
        for (int yy = y; yy < y + h; ++yy) {
            const int* row = labels.ptr<int>(yy);
            for (int xx = x; xx < x + w; ++xx)
                if (row[xx] == i) b.pts.emplace_back(static_cast<float>(xx), static_cast<float>(yy));
        }
        blobs.push_back(std::move(b));
    }
    if (blobs.empty()) return false;

    // 并查集聚类：质心距离 <28px 的碎片归为同一条链
    std::vector<int> par(blobs.size());
    for (size_t i = 0; i < par.size(); ++i) par[i] = (int)i;
    std::function<int(int)> find = [&](int u) {
        while (par[u] != u) { par[u] = par[par[u]]; u = par[u]; }
        return u;
    };
    for (size_t i = 0; i < blobs.size(); ++i)
        for (size_t j = i + 1; j < blobs.size(); ++j)
            if (std::hypot(blobs[i].cx - blobs[j].cx, blobs[i].cy - blobs[j].cy) < 28.0)
                par[find((int)i)] = find((int)j);
    std::map<int, std::vector<const Blob*>> groups;
    for (size_t i = 0; i < blobs.size(); ++i) groups[find((int)i)].push_back(&blobs[i]);

    double bestScore = -1e18;
    ArrowInfo best;
    for (auto& kv : groups) {
        const auto& g = kv.second;
        std::vector<cv::Point2f> P;
        int A = 0;
        for (const Blob* b : g) { P.insert(P.end(), b->pts.begin(), b->pts.end()); A += b->area; }
        if (P.size() < 20) continue;
        if (g.size() > 1 && A < 120) continue;
        if (g.size() == 1 && A < 250) continue;

        cv::Point2f c(0.f, 0.f);
        for (const auto& p : P) c += p;
        c *= 1.f / (float)P.size();

        // 最小二乘直线拟合作为主轴
        cv::Vec4f line;
        cv::fitLine(P, line, cv::DIST_L2, 0, 1e-4, 1e-4);
        const cv::Point2f axis(line[0], line[1]);
        const cv::Point2f perp(-axis.y, axis.x);
        double tmin = 1e18, tmax = -1e18, smin = 1e18, smax = -1e18;
        for (const auto& p : P) {
            const double t = (p.x - c.x) * axis.x + (p.y - c.y) * axis.y;
            const double s = (p.x - c.x) * perp.x + (p.y - c.y) * perp.y;
            tmin = std::min(tmin, t); tmax = std::max(tmax, t);
            smax = std::max(smax, s); smin = std::min(smin, s);
        }
        const double L = tmax - tmin, Wd = smax - smin;
        if (L < 35.0 || L > 160.0) continue;                  // 链长范围
        if (Wd > 0.45 * L + 8.0 || Wd > 34.0) continue;       // 细长：垂直厚度受限
        if (g.size() == 1) {                                  // 粘连成整块的链也要形状细长
            const cv::RotatedRect rr = cv::minAreaRect(P);
            const float mn = std::min(rr.size.width, rr.size.height);
            const float mx = std::max(rr.size.width, rr.size.height);
            if (mn < 1.f || mx / std::max(mn, 1.f) < 2.2f) continue;
        }

        const cv::Point2f dc = tc - c;
        const double nd = cv::norm(dc);
        if (nd < tr * 1.5) continue;
        const double cosang = std::abs(axis.x * dc.x + axis.y * dc.y) / nd;
        const double ang = std::acos(std::min(1.0, cosang)) * 180.0 / CV_PI;
        if (ang > 28.0) continue;                             // 长轴必须正对靶心
        const double ddist = std::hypot(c.x - tc.x, c.y - tc.y);
        if (ddist < tr * 2.3 || ddist > tr * 7.0) continue;

        const double score = g.size() * 10.0 + A / 200.0 - ang;
        if (score > bestScore) {
            bestScore = score;
            best.center = c;
            best.points = std::move(P);
            best.blobCount = (int)g.size();
            // 夹角：x 轴向右、y 轴向上（图像 y 向下，故取反）
            best.angleDeg = std::atan2(-(double)(tc.y - c.y), (double)(tc.x - c.x)) * 180.0 / CV_PI;
            best.target = &target;
        }
    }
    if (bestScore <= -1e17) return false;
    info = best;
    return true;
}

// 检测 HUD 中的橙色字母 "R" 图标（与箭头链同色，约 20x18 的实心字形）
struct LetterR { cv::Point2f center; float w; float h; };

static bool detectLetterR(const cv::Mat& frame, LetterR& letter)
{
    cv::Mat hsv, m;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(4, 150, 130), cv::Scalar(18, 255, 255), m);

    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(m, labels, stats, centroids, 8);

    bool found = false;
    LetterR best;
    int bestArea = 0;
    for (int i = 1; i < n; ++i) {
        const int x = stats.at<int>(i, cv::CC_STAT_LEFT), y = stats.at<int>(i, cv::CC_STAT_TOP);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        const int a = stats.at<int>(i, cv::CC_STAT_AREA);
        // R 是近方形小字形：尺寸、填充率、长宽比过滤掉圆环/箭头/支架等
        if (w < 12 || w > 40 || h < 12 || h > 40) continue;
        if (a < 150) continue;
        if (double(a) / (w * h) < 0.55) continue;          // 圆环/雪佛龙填充率 ~0.3-0.5
        if (float(w) / h < 0.5f || float(w) / h > 1.8f) continue;
        // 恰好一个内孔（字母 R 的圈），且孔有一定面积
        const cv::Mat blob = (labels(cv::Rect(x, y, w, h)) == i);
        std::vector<std::vector<cv::Point>> bc;
        std::vector<cv::Vec4i> bh;
        cv::findContours(blob, bc, bh, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
        int nHoles = 0; double maxHole = 0;
        for (size_t j = 0; j < bc.size(); ++j) {
            if (bh[j][3] != -1) { ++nHoles; maxHole = std::max(maxHole, cv::contourArea(bc[j])); }
        }
        if (nHoles != 1 || maxHole < 5.0) continue;
        if (a > bestArea) {
            bestArea = a;
            best.center = cv::Point2f(float(centroids.at<double>(i, 0)), float(centroids.at<double>(i, 1)));
            best.w = float(w); best.h = float(h);
            found = true;
        }
    }
    if (!found) return false;
    letter = best;
    return true;
}

// 在图上标注箭头链，并给出"箭头质心 -> 靶标圆点"连线相对图像水平轴的夹角
static void annotateArrowChain(cv::Mat& out, const cv::Mat& frame,
                               const std::vector<CircleTarget>& targets)
{
    ArrowInfo best;
    const CircleTarget* bestTgt = nullptr;
    double bestDist = 1e18;
    for (const CircleTarget& t : targets) {
        ArrowInfo a;
        if (detectArrowChain(frame, t, a)) {
            const double d = cv::norm(a.center - t.center);
            if (d < bestDist) { best = a; bestTgt = &t; bestDist = d; }
        }
    }
    if (bestTgt == nullptr) return;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << best.angleDeg;
    const std::string txt = "arrow->target: " + oss.str() + " deg";

    // 用蓝框框出箭头链，蓝线连到靶标圆点，并标出夹角
    if (best.points.size() >= 5) {
        std::vector<cv::Point2f> box(4);
        cv::minAreaRect(best.points).points(box.data());
        std::vector<cv::Point> ibox;
        for (const auto& p : box) ibox.emplace_back(cv::Point(cvRound(p.x), cvRound(p.y)));
        cv::polylines(out, ibox, true, cv::Scalar(255, 0, 0), 2);
    }
    cv::circle(out, best.center, 4, cv::Scalar(255, 0, 255), -1);
    cv::line(out, best.center, bestTgt->center, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
    const cv::Point2f mid = (best.center + bestTgt->center) * 0.5f;
    cv::putText(out, txt, mid + cv::Point2f(8, -8), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);

    cv::putText(out, "arrow found, angle to horizontal = " + oss.str() + " deg",
                cv::Point(10, 75), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
}

// 输入主循环读取的原始帧及其帧号（从 0 开始），输出标记了圆形靶标与箭头的图像
cv::Mat markCircleTargets(const cv::Mat& frame, int frameIndex)
{
    cv::Mat out = frame.clone();
    std::vector<CircleTarget> targets = detectCircleTargets(frame);

    annotateArrowChain(out, frame, targets);

    // 标注字母 R：白色圆圈框住 + 标出坐标
    {
        LetterR R;
        if (detectLetterR(frame, R)) {
            const int rr = cvRound(0.5f * std::hypot(R.w, R.h)) + 4;
            cv::circle(out, R.center, rr, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            std::ostringstream oss;
            oss << "(" << cvRound(R.center.x) << "," << cvRound(R.center.y) << ")";
            cv::putText(out, "R " + oss.str(),
                        R.center + cv::Point2f(-20, -rr - 6),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            cv::putText(out, "R Flag: 1", cv::Point(10, 100),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        }
    }

    for (const CircleTarget& t : targets) {
        const cv::Point2f c = t.center;
        const int r = cvRound(t.radius);
        cv::circle(out, c, r, cv::Scalar(0, 255, 0), 2);            // 标记靶标圆环
        cv::line(out, c + cv::Point2f(-8, 0), c + cv::Point2f(8, 0), cv::Scalar(0, 255, 0), 2); // 中心十字
        cv::line(out, c + cv::Point2f(0, -8), c + cv::Point2f(0, 8), cv::Scalar(0, 255, 0), 2);
        cv::putText(out, "T r=" + std::to_string(r),
                    c + cv::Point2f(r + 4, -r),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
    cv::putText(out, "frame: " + std::to_string(frameIndex),
                cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    cv::putText(out, "targets: " + std::to_string(targets.size()),
                cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    
    // if (targets.size() > 0)
    // {
    //     std::cout << "target find in frame: " << frameIndex <<"t" << targets.size() << "\n";
    // }
   
    return out;
}

int main()
{
    cv::VideoCapture cap;
    cv::Mat frame;
    int width, height;
    double fps;
    int frameIndex ;
    cap.open("./resources/task_3.mp4");
    if (!cap.isOpened()) {
        cerr << "cannot open video\n";
        return 0;
    }

    fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0 || fps > 1000) {
        fps = 60.0;
    }
    width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::VideoWriter writer;
    writer.open("./result/task3_windmill/task_3/recognition_overlay.mp4",
                cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        cerr << "cannot open video writer\n";
        return 0;
    }


    // 逐帧做检测
    frameIndex = 0;
    while (cap.read(frame))
    {
        //检测并标记每一帧图像里的圆形靶标，并生成新的mp4文件
        cv::Mat marked = markCircleTargets(frame, frameIndex);
        writer << marked;
        ++frameIndex;
    }

    cap.release();
    writer.release();


    cap.open("./resources/task_4.mp4");
    if (!cap.isOpened()) {
        cerr << "cannot open video\n";
        return 0;
    }

    fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0 || fps > 1000) {
        fps = 60.0;
    }
    width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    // cv::VideoWriter writer;
    writer.open("./result/task3_windmill/task_4/recognition_overlay.mp4",
                cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        cerr << "cannot open video writer\n";
        return 0;
    }


    // 逐帧做检测
    frameIndex = 0;
    while (cap.read(frame))
    {
        //检测并标记每一帧图像里的圆形靶标，并生成新的mp4文件
        cv::Mat marked = markCircleTargets(frame, frameIndex);
        writer << marked;
        ++frameIndex;
    }

    cap.release();
    writer.release();



    return 0;

}
