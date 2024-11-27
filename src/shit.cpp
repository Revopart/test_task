#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <httplib.h>
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <fstream>

using namespace cv;
using namespace std;
using namespace httplib;

// Функция для обработки и бинаризации изображения
Mat preprocessImage(const Mat& image) {
    Mat gray, binary;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
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
    ocr.SetSourceResolution(100);
    string text = string(ocr.GetUTF8Text());
    float confidence = ocr.MeanTextConf();
    return {text, confidence};
}

// Функция поворота изображения
Mat rotateImage(const Mat& image, double angle) {
    Point2f center(image.cols / 2.0, image.rows / 2.0);
    Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0);
    Mat rotatedImage;
    warpAffine(image, rotatedImage, rotationMatrix, image.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(255, 255, 255));
    return rotatedImage;
}

// Функция для распознавания буквы с учётом поворотов
string recognizeCharacterWithRotation(Mat letter, tesseract::TessBaseAPI& ocr) {
    string bestResult = "n/a";
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
    return bestResult;
}

// Основной обработчик API
string processImage(const string& imagePath) {
    Mat image = imread(imagePath);
    if (image.empty()) {
        return R"({"error": "Failed to load image"})";
    }

    Mat binaryImage = preprocessImage(image);
    vector<Rect> boundingBoxes = extractContours(binaryImage);

    tesseract::TessBaseAPI ocr;
    if (ocr.Init(NULL, "ell", tesseract::OEM_LSTM_ONLY)) {
        return R"({"error": "Failed to initialize Tesseract"})";
    }
    ocr.SetVariable("tessedit_char_whitelist", "μσπλ");

    map<string, int> letterCounts;
    for (const auto& box : boundingBoxes) {
        Mat letter = binaryImage(box);
        string recognizedLetter = recognizeCharacterWithRotation(letter, ocr);
        recognizedLetter.erase(remove_if(recognizedLetter.begin(), recognizedLetter.end(), ::isspace), recognizedLetter.end());
        if (!recognizedLetter.empty()) {
            letterCounts[recognizedLetter]++;
        }
    }

    // Формируем JSON результат
    string result = "{";
    for (const auto& [letter, count] : letterCounts) {
        result += "\"" + letter + "\": " + to_string(count) + ", ";
    }
    if (!letterCounts.empty()) {
        result.pop_back();
        result.pop_back();
    }
    result += "}";
    ocr.End();
    return result;
}

int main() {
    Server svr;

    svr.Post("/predict", [](const Request& req, Response& res) {
        auto file = req.get_file_value("file");
        string tempFilePath = "temp_image.jpg";

        // Сохранение файла на диск
        ofstream ofs(tempFilePath, ios::binary);
        ofs.write(file.content.data(), file.content.size());
        ofs.close();

        // Обработка изображения
        string jsonResponse = processImage(tempFilePath);

        // Удаление временного файла
        remove(tempFilePath.c_str());

        // Отправка ответа
        res.set_content(jsonResponse, "application/json");
    });

    cout << "Сервер запущен на http://127.0.0.1:8000" << endl;
    svr.listen("0.0.0.0", 8000);

    return 0;
}
