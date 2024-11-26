#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <iostream>
#include <map>
#include <vector>

using namespace cv;
using namespace std;

// Функция для обработки и бинаризации изображения
Mat preprocessImage(const Mat& image) {
    Mat gray, binary, test;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
    resize(binary, test, Size(720, 720));
    imshow("Контуры на изображении", test);
    waitKey(0);

    return binary;
}

// Функция для извлечения контуров
vector<Rect> extractContours(const Mat& binaryImage) {
    vector<vector<Point>> contours;
    
    // Извлекаем контуры на бинаризованном изображении
    findContours(binaryImage, contours, RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    vector<Rect> boundingBoxes;
    // Для каждого контура вычисляем ограничивающий прямоугольник
    for (const auto& contour : contours) {
        boundingBoxes.push_back(boundingRect(contour));
    }

    

    
    return boundingBoxes;
}

// Функция для распознавания текста с использованием Tesseract OCR
string recognizeCharacter(const Mat& letter, tesseract::TessBaseAPI& ocr) {
    ocr.SetImage(letter.data, letter.cols, letter.rows, 1, letter.step);
    ocr.SetSourceResolution(100); // Задаем DPI для лучшего распознавания
    return string(ocr.GetUTF8Text());
}

Mat rotateImage(const Mat& image, double angle) {
    Point2f center(image.cols / 2.0, image.rows / 2.0);
    Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0);
    Mat rotatedImage;
    warpAffine(image, rotatedImage, rotationMatrix, image.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(255, 255, 255));  // Белый фон
    return rotatedImage;
}

string recognizeCharacterWithRotation(Mat letter, tesseract::TessBaseAPI& ocr) {
    string result;
    double angle = 0.0;
    
    // Пробуем распознать букву, начиная с 0 градусов и вращая на 2 градуса
    for (angle = 0; angle < 360; angle += 12) {
        // Поворачиваем изображение
        Mat rotatedLetter = rotateImage(letter, angle);
        
        // Пробуем распознать букву
        result = recognizeCharacter(rotatedLetter, ocr);

        // Если распознана буква (условие может быть изменено в зависимости от требований)
        if (!result.empty() && result != " " && result != "\n") {
            break; // Если результат удовлетворительный, выходим
        }
    }

    return result;
}

int main() {
    // Загружаем изображение
    string imagePath = "/home/thaul/test_task/materials/123.jpg"; // Замените на путь к вашему изображению
    Mat image = imread(imagePath);

    if (image.empty()) {
        cerr << "Не удалось загрузить изображение!" << endl;
        return -1;
    }

    // Предобработка изображения
    Mat binaryImage = preprocessImage(image);

    // Извлечение контуров
    vector<Rect> boundingBoxes = extractContours(binaryImage);

    // Инициализация Tesseract
    tesseract::TessBaseAPI ocr;
    if (ocr.Init(NULL, "ell", tesseract::OEM_LSTM_ONLY)) {
        cerr << "Не удалось инициировать греческий язык" << endl;
        return -1;
    }

    // Настройки для распознавания символов
    ocr.SetVariable("tessedit_char_whitelist", "μσπλ"); // Указываем допустимые буквы

    // Словарь для подсчета частоты букв
    map<string, int> letterCounts;
        cout << boundingBoxes.size();
    // Распознаем каждую букву
    for (const auto& box : boundingBoxes) {
        Mat letter = binaryImage(box); // Вырезаем область буквы
        string recognizedLetter= recognizeCharacterWithRotation(letter, ocr);

        // Удаляем лишние пробелы или символы
        recognizedLetter.erase(remove_if(recognizedLetter.begin(), recognizedLetter.end(), ::isspace), recognizedLetter.end());

        if (!recognizedLetter.empty()) {
            letterCounts[recognizedLetter]++;
        }
    }
    


    // Вывод результата
    cout << "{";
    for (const auto& [letter, count] : letterCounts) {
        cout << "\"" << letter << "\": " << count << ", ";
    }
    cout << "}" << endl;

    // Освобождаем ресурсы
    ocr.End();
    return 0;
}
