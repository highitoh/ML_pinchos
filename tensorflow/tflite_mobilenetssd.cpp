#include <stdio.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#define MODEL_FILENAME "../data/model/ssd_mobilenet_v2.tflite"
#define INPUT_SIZE_WIDTH 300
#define INPUT_SIZE_HEIGHT 300

#define LABEL_SIZE 32

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {

    /* Interprete Argument */
    char* image_filepath;
    float_t detection_threshold;
    if (argc == 3) {
      image_filepath = argv[1];
      detection_threshold = atof(argv[2]);
    } else {
      std::cout << "[Error] The program needs two arguments. 0:Path to Image, 1:Detection Threshold" << std::endl;
      exit(1);
    }

    /* Load Model */
    std::cout << "Loading model..." << std::endl;
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(MODEL_FILENAME);
    TFLITE_MINIMAL_CHECK(model != nullptr);
    std::cout << "Done." << std::endl;

    /* Create Interpreter */
    std::cout << "Creating interpreter..." << std::endl;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);
    std::cout << "Done." << std::endl;

    /* Allocate Tensors */
    std::cout << "Allocating tensors..." << std::endl;
    std::vector<int> dims{1, INPUT_SIZE_WIDTH, INPUT_SIZE_HEIGHT, 3};
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    // tflite::PrintInterpreterState(interpreter.get()); // for debug
    std::cout << "Done." << std::endl;

    /* Read Input Image */
    std::cout << "Reading image..." << std::endl;
    cv::Mat image = cv::imread(image_filepath);
    if (image.empty()) {
      std::cout << "[Error] Cannot Read Image File:" << image_filepath << std::endl;
      exit(1);
    }
    std::cout << "[Image] cols:" << image.cols << ", rows:" << image.rows << ", total:" << image.total() << ", elemsize:" << image.elemSize() << ", step:" << image.step << std::endl;
    std::cout << "Done." << std::endl;

    /* Preprocess the Image */
    std::cout << "Preprocessing image..." << std::endl;
    int center_crop_x = (image.cols - INPUT_SIZE_WIDTH) / 2;
    int center_crop_y = (image.rows - INPUT_SIZE_HEIGHT) / 2;
    cv::Rect roi(center_crop_x, center_crop_y, INPUT_SIZE_WIDTH, INPUT_SIZE_HEIGHT);

    // std::cout << "[Original Image] cols:" << image.cols << ", rows:" << image.rows << ", total:" << image.total() << ", elemsize:" << image.elemSize() << ", step:" << image.step << std::endl; // for debug
    image.convertTo(image, CV_32FC3);
    image = image(roi).clone();
    // cv::imwrite("cropped_image.jpg", image); // for debug
    // std::cout << "[Cropped Image] cols:" << image.cols << ", rows:" << image.rows << ", total:" << image.total() << ", elemsize:" << image.elemSize() << ", step:" << image.step << std::endl; // for debug
    std::cout << "Done." << std::endl;

    /* Prepare Input Tensor */
    std::cout << "Preparing input tensors..." << std::endl;
    float_t* input = interpreter->typed_input_tensor<float_t>(0);
    int i = 0;
    for (int y = 0; y < INPUT_SIZE_HEIGHT; y++) {
      for (int x = 0; x < INPUT_SIZE_WIDTH; x++) {
        float_t* data_ptr = (float_t*)image.data;
        input[i++] = (data_ptr[y * image.cols * image.channels() + x * image.channels() + 2] - 127.5) / 127.5f; // R
        input[i++] = (data_ptr[y * image.cols * image.channels() + x * image.channels() + 1] - 127.5) / 127.5f; // G
        input[i++] = (data_ptr[y * image.cols * image.channels() + x * image.channels() + 0] - 127.5) / 127.5f; // B
      }
    }
    std::cout << "Done." << std::endl;

    /* Run Inference */
    std::cout << "Running inference..." << std::endl;
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    std::cout << "Done." << std::endl;

    /* Get Result from Output Tensor */
    float_t* detection_boxes   = static_cast<float_t*>(interpreter->typed_output_tensor<float_t>(0));
    float_t* detection_classes = static_cast<float_t*>(interpreter->typed_output_tensor<float_t>(1));
    float_t* detection_scores  = static_cast<float_t*>(interpreter->typed_output_tensor<float_t>(2));
    float_t* num_boxes         = static_cast<float_t*>(interpreter->typed_output_tensor<float_t>(3));

    /* Output the Result */
    std::cout << "##### Result #####" << std::endl;
    for (int i = 0; i < *num_boxes; i++) {
      if (detection_scores[i] > detection_threshold) {
        int class_id   = int(detection_classes[i]);
        int confidence = int(detection_scores[i] * 100) ;
        float_t top    = int(detection_boxes[i*4 + 0] * INPUT_SIZE_HEIGHT);
        float_t left   = int(detection_boxes[i*4 + 1] * INPUT_SIZE_WIDTH);
        float_t bottom = int(detection_boxes[i*4 + 2] * INPUT_SIZE_HEIGHT);
        float_t right  = int(detection_boxes[i*4 + 3] * INPUT_SIZE_WIDTH);

        /* Output the result by Text */
        std::cout << "Class ID: " << class_id << " ";
        std::cout << "Confidence: " << confidence << "% ";
        std::cout << "top: " << top << ", left:" << left << ", bottom:" << bottom << ", right:" << right << std::endl;

        /* Add Rectangle to Output Image */
        char label[LABEL_SIZE];
        sprintf(label, "ID:%d %d%%", class_id, confidence);
        cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 255, 255));
        cv::putText(image, label, cv::Point(left, top+12), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 255, 255));
      }
    }
    cv::imwrite("result.jpg", image);

    return 0;
} 