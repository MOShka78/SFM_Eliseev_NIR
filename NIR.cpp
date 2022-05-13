#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/img_hash.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/freetype.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/core.hpp"
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;


void extract_features(
    vector<string>& image_names,
    vector<vector<KeyPoint>>& key_points_for_all,
    vector<Mat>& descriptor_for_all,
    vector<vector<Vec3b>>& colors_for_all
    )
{
    key_points_for_all.clear();
    descriptor_for_all.clear();
    Mat image;
 
    // Считываем изображение, получаем характерные точки изображения и сохраняем
    double hessianThreshold=400; // Минимальный гессенский порог может быть выбран между 300 ~ 500
    int nOctaves = 4; // 4 средних в четырех масштабах
    int nOctaveLayers = 3; // указывает количество слоев в каждом масштабе

    Ptr<Feature2D> sift = cv::xfeatures2d::SURF::create(hessianThreshold,nOctaves,nOctaveLayers);
    for (vector<string>::iterator it = image_names.begin(); it != image_names.end(); ++it)
    {
        image = imread(*it);
        if (image.empty()) {continue;}
 
        vector<KeyPoint> key_points;
        Mat descriptor;
        // Иногда возникает ошибка выделения памяти
        sift->detectAndCompute(image, noArray(), key_points, descriptor);
 
        // Слишком мало характерных точек, исключить изображение
        if (key_points.size() <= 5) continue;
 
        key_points_for_all.push_back(key_points);
        descriptor_for_all.push_back(descriptor);
 
        vector<Vec3b> colors(key_points.size());
        for (int i = 0; i < static_cast<int>(key_points.size()); ++i)
        {
            Point2f& p = key_points[i].pt;
            colors[i] = image.at<Vec3b>(p.y, p.x);
        }
        colors_for_all.push_back(colors);
    }
}

void match_features(Mat& query, Mat& train, vector<DMatch>& matches)
{
    vector<vector<DMatch>> knn_matches;
    BFMatcher matcher(NORM_L2);
    matcher.knnMatch(query, train, knn_matches, 2);
 
    // Получение минимального расстояния совпадения, которое удовлетворяет тесту соотношения
    float min_dist = FLT_MAX;
    for (int r = 0; r < static_cast<int>(knn_matches.size()); ++r)
    {
        if (knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance)
            {continue;}
 
        float dist = knn_matches[r][0].distance;
        if (dist < min_dist) min_dist = dist;
    }
 
    matches.clear();
    for (size_t r = 0; r < knn_matches.size(); ++r)
    {
        // Исключаем точки, не удовлетворяющие тесту соотношения, и точки со слишком большим расстоянием совпадения
        if (
            knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance ||
            knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
            )
            {continue;}
 
        // Сохраняем совпадающие точки
        matches.push_back(knn_matches[r][0]);
    }
}

void get_matched_points(
	vector<KeyPoint>& p1,
	vector<KeyPoint>& p2,
	vector<DMatch> matches,
	vector<Point2f>& out_p1,
	vector<Point2f>& out_p2
)
{
	out_p1.clear();
	out_p2.clear();
	for (int i = 0; i < static_cast<int>(matches.size()); ++i)
	{
		out_p1.push_back(p1[matches[i].queryIdx].pt);
		out_p2.push_back(p2[matches[i].trainIdx].pt);
	}
}

void get_matched_colors(
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
)
{
	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < static_cast<int>(matches.size()); ++i)
	{
		out_c1.push_back(c1[matches[i].queryIdx]);
		out_c2.push_back(c2[matches[i].trainIdx]);
	}
}

bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask)
{
    // Получить фокусное расстояние и координаты оптического центра камеры (координаты первичной точки) в соответствии с внутренней матрицей параметров
    double focal_length = 0.5*(K.at<double>(0) + K.at<double>(4));
    Point2d principle_point(K.at<double>(2), K.at<double>(5));
 
    // Получение собственной матрицы в соответствии с точками совпадения, использование RANSAC для дальнейшего устранения точек несовпадения
    Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
    if (E.empty()) {return false;}
 
    double feasible_count = countNonZero(mask);
    // Для RANSAC, когда количество выбросов больше 50%, результат ненадежный
    if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6)
        {return false;}
 
    // Разложить собственную матрицу, чтобы получить относительное преобразование
    int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);
 
    // Количество точек перед двумя камерами одновременно должно быть достаточно большим
    if (((double)pass_count) / feasible_count < 0.7)
        {return false;}
 
    return true;
}

void maskout_points(vector<Point2f>& p1, Mat& mask)
{
	vector<Point2f> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}

void maskout_colors(vector<Vec3b>& p1, Mat& mask)
{
	vector<Vec3b> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}

void reconstruct(Mat& K, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure)
{
    // Матрица проекции двух камер [R T], triangulatePoints поддерживает только тип float

    Mat proj1;
    proj1 = Mat::eye(3, 4, CV_32FC1);
    Mat proj2;
    proj2 = Mat::eye(3, 4, CV_32FC1);
 
    proj1(Range(0, 3), Range(0, 3)) = Mat::eye(3, 3, CV_32FC1);
    proj1.col(3) = Mat::zeros(3, 1, CV_32FC1);
 
    R.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
    T.convertTo(proj2.col(3), CV_32FC1);
 
    Mat fK;
    K.convertTo(fK, CV_32FC1);
    proj1 = fK*proj1;
    proj2 = fK*proj2;
 
    // Реконструкция триангуляции
    triangulatePoints(proj1, proj2, p1, p2, structure);
}

void save_structure(string file_name, vector<Mat>& rotations, vector<Mat>& motions, Mat& structure, vector<Vec3b>& colors)
{
	int n = (int)rotations.size();

	FileStorage fs(file_name, FileStorage::WRITE);
	fs << "Camera Count" << n;
	fs << "Point Count" << structure.cols;

	fs << "Rotations" << "[";
	for (int i = 0; i < n; ++i)
	{
		fs << rotations[i];
	}
	fs << "]";

	fs << "Motions" << "[";
	for (int i = 0; i < n; ++i)
	{
		fs << motions[i];
	}
	fs << "]";

	fs << "Points" << "[";
	for (int i = 0; i < structure.cols; ++i)
	{
		Mat_<float> c = structure.col(i);
		c /= c(3);
		fs << Point3f(c(0), c(1), c(2));
	}
	fs << "]";

	fs << "Colors" << "[";
	for (size_t i = 0; i < colors.size(); ++i)
	{
		fs << colors[i];
	}
	fs << "]";

	fs.release();
}

int main(int argc, char** argv){
    
    vector<string> image_names;
    vector<vector<KeyPoint>> key_points_for_all;
    vector<Mat> descriptor_for_all;
    vector<vector<Vec3b>> colors_for_all;
    vector<DMatch> matches;
	vector<Point2f> p1, p2;
	vector<Vec3b> c1, c2;

	Mat R, T;	
	Mat mask;	

    Mat K(Matx33d(
		1626.37, 0, 1482.19,
		0, 1626.37, 2057.07,
		0, 0, 1));
    
    const int max = 35;
    int num = 0;
    string fn[max];
    string name = "../photo1/exper";
    string format = ".jpg";

    while (num<max){
        fn[num] = name + to_string(num) + format;
        num+=1;
    }

    for (size_t count = 0; count < max; count++){
        image_names.push_back(fn[count]);
        cout << fn[count] << endl;
    }

	double time_ = static_cast<double>(getTickCount());
	double time0 = 0;

	extract_features(image_names, key_points_for_all, descriptor_for_all, colors_for_all);

	time0 = ((double)getTickCount() - time_) / getTickFrequency();
	cout << "Извлечение признаков требует времени = " <<time0<< endl;

	match_features(descriptor_for_all[0], descriptor_for_all[1], matches);

	time0 = ((double)getTickCount() - time_) / getTickFrequency();
	cout << "Сопоставление характеристик требует времени = " << time0 << endl;

	get_matched_points(key_points_for_all[0], key_points_for_all[1], matches, p1, p2);

	time0 = ((double)getTickCount() - time_) / getTickFrequency();
	cout << "Получение совпадающих точек занимает врмени = " << time0 << endl;

	get_matched_colors(colors_for_all[0], colors_for_all[1], matches, c1, c2);

	time0 = ((double)getTickCount() - time_) / getTickFrequency();
	cout << "Получение совпадений цветов и точек занимает врмени = " << time0 << endl;

	find_transform(K, p1, p2, R, T, mask);

	Mat structure;
	maskout_points(p1, mask);

	maskout_points(p2, mask);

	reconstruct(K, R, T, p1, p2, structure);

	vector<Mat> rotations = { Mat::eye(3, 3, CV_32FC1), R };
	vector<Mat> motions = { Mat::zeros(3, 1, CV_32FC1), T };

	maskout_colors(c1, mask);

	save_structure("/home/vadim/NIR/struc.yaml", rotations, motions, structure, c1);

	time0 = ((double)getTickCount() - time_) / getTickFrequency();
	cout << "Сохранение структуры = " << time0 << endl;
    
    return 0;
}