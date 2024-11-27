#include "api_server.h"

int main() {
    Server svr;
    svr.Post("/predict", [](const Request& req, Response& res) {
        auto file = req.get_file_value("file");
        string tempFilePath = "temp_image.jpg";
        ofstream ofs(tempFilePath, ios::binary);
        ofs.write(file.content.data(), file.content.size());
        ofs.close();
        string jsonResponse = processImage(tempFilePath);
        remove(tempFilePath.c_str());
        res.set_content(jsonResponse, "application/json");
    });
    cout << "Сервер запущен на http://127.0.0.1:8000" << endl;
    svr.listen("0.0.0.0", 8000);

    return 0;
}