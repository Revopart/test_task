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
    imshow(binary, "sd");
    waitkey(0);
    return binary;
}

// Функция для извлечения контуров
vector<Rect> extractContours(const Mat& binaryImage) {
    vector<vector<Point>> contours;
    findContours(binaryImage, contours, RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    vector<Rect> boundingBoxes;
    for (const auto& contour : contours) {
        boundingBoxes.push_back(boundingRect(contour));
    }
    return boundingBoxes;
}

// Функция для распознавания текста с использованием Tesseract OCR
pair<string, float> recognizeCharacter(const Mat& letter, tesseract::TessBaseAPI& ocr) {
    ocr.SetImage(letter.data, letter.cols, letter.rows, 1, letter.step);
    ocr.SetSourceResolution(100); // Устанавливаем разрешение источника
    string text = string(ocr.GetUTF8Text());
    float confidence = ocr.MeanTextConf(); // Получаем среднюю вероятность
    return {text, confidence};
}
 //Функция поворота буквы вокруг центра
Mat rotateImage(const Mat& image, double angle) {
    Point2f center(image.cols / 2.0, image.rows / 2.0);
    Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0);
    Mat rotatedImage;
    warpAffine(image, rotatedImage, rotationMatrix, image.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(255, 255, 255));  // Белый фон
    return rotatedImage;
}

// Функция для распознавания буквы с учетом всех вращений
string recognizeCharacterWithRotation(Mat letter, tesseract::TessBaseAPI& ocr) {
    string bestResult = "n/a"; // На случай, если не найдется символ из ТЗ
    float bestConfidence = -1;
    double angleStep = 12; 
    for (double angle = 0; angle < 360; angle += angleStep) {
        Mat rotatedLetter = rotateImage(letter, angle);
        auto [result, confidence] = recognizeCharacter(rotatedLetter, ocr);
        if (!result.empty() && result.length() == 1 && confidence > bestConfidence) {
            bestResult = result;
            bestConfidence = confidence;
        }
    }

    cout << "Лучший результат: " << bestResult << " с вероятностью: " << bestConfidence << "%" << endl;
    return bestResult;
}

int main() {
    string imagePath = "/home/vinylbro/test_task/materials/123.jpg"; // Замените на путь к вашему изображению
    Mat image = imread(imagePath);
    if (image.empty()) {
        cerr << "Не удалось загрузить изображение!" << endl;
        return -1;
    }
    Mat binaryImage = preprocessImage(image);
    vector<Rect> boundingBoxes = extractContours(binaryImage);
    tesseract::TessBaseAPI ocr;
    if (ocr.Init(NULL, "ell", tesseract::OEM_LSTM_ONLY)) {
        cerr << "Не удалось инициировать греческий язык" << endl;
        return -1;
    }
    ocr.SetVariable("tessedit_char_whitelist", "μσπλ"); // Указываю буквы из ТЗ
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
