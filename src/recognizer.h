#pragma once
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <iostream>
#include <map>
#include <vector>
#include <string>
using namespace cv;
using namespace std;

// Функция для обработки и бинаризации изображения
Mat preprocessImage(const Mat& image) ;
// Функция для извлечения контуров
vector<Rect> extractContours(const Mat& binaryImage);
// Функция для распознавания текста с использованием Tesseract OCR
pair<string, float> recognizeCharacter(const Mat& letter, tesseract::TessBaseAPI& ocr);
// Функция поворота изображения
Mat rotateImage(const Mat& image, double angle);
// Функция для распознавания буквы с учётом поворотов
string recognizeCharacterWithRotation(Mat letter, tesseract::TessBaseAPI& ocr);
// Основной обработчик API
string processImage(const string& imagePath);
