// Copyright (c) 2026，D-Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DFEATINFER_H
#define DFEATINFER_H

#include <vector>
#include <map>
#include <unordered_map>
#include <deque>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "magic_enum/magic_enum.hpp"
#include "log_macros.h"
#include "timer_utils.h"
#include "dnn_platform.h"

using InferenceHandle = int;

struct KeyPoint {
  int x, y;
  float score;
};

struct pair_hash {
  template <class T1, class T2> size_t operator()(std::pair<T1, T2> const &pair) const {
    size_t h1 = std::hash<T1>()(pair.first);
    size_t h2 = std::hash<T2>()(pair.second);
    return h1 ^ h2;
  }
};

const float GRID_SIZE = 10.0f;

using Grid = std::unordered_map<std::pair<int, int>, std::vector<KeyPoint>, pair_hash>;

static inline std::pair<int, int> hashKeyPoint(const KeyPoint &kp) {
  int xHash = static_cast<int>(kp.x / GRID_SIZE);
  int yHash = static_cast<int>(kp.y / GRID_SIZE);
  return {xHash, yHash};
}

class DFeatInfer {
public:
  explicit DFeatInfer(const rclcpp::Logger &logger);
  ~DFeatInfer();

  /**
   * @brief Initialize the DFMatch model
   * @param model_path Path to the DFMatch model file
   * @param max_memory_count Maximum number of memory buffers to allocate
   * @return 0 on success, -1 on failure
   */
  int init(const std::string &model_path, const int &max_memory_count = 5);

  /**
   * @brief Perform forward inference using the DFMatch model asynchronously
   * @param image Input image
   * @param handle Output inference handle
   * @return 0 on success, -1 on failure
   */
  int forward(const cv::Mat &image, InferenceHandle &handle);

  /**
   * @brief Perform forward inference using the DFMatch model
   * @param image Input image
   * @param extractor_result Output keypoint and descriptor extraction result
   * @return 0 on success, -1 on failure
   */
  int postprocess(const InferenceHandle &handle,
                  std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> &extractor_result);
  
  /**
   * @brief Perform forward inference using the DFMatch model
   * @param image Input image
   * @param extractor_result Output keypoint and descriptor extraction result
   * @return 0 on success, -1 on failure
   */
  int forward(const cv::Mat &image, std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> &extractor_result);

  /**
   * @brief Get the input size required by the model
   * @param w Width of the input image
   * @param h Height of the input image
   */
  void get_model_input_size(int &w, int &h) const;

private:
  // ===================================== member functions =======================================
  /**
   * @brief Prepare input tensor for model inference
   * @param input_tensors vector to hold the prepared input tensors
   * @return 0 on success, -1 on failure
   */
  int prepare_input_tensor(std::vector<hbDNNTensor> &input_tensors);

  /**
   * @brief Prepare output tensor for model inference
   * @param output_tensors vector to hold the prepared output tensors
   * @return 0 on success, -1 on failure
   */
  int prepare_output_tensor(std::vector<hbDNNTensor> &output_tensors);

  /**
   * @brief Get an idle tensor index for processing
   * @return Index of an idle tensor, or -1 if none are available
   */
  int get_idle_tensor();

  /**
   * @brief Set a tensor as idle after processing
   * @param tensor_id Index of the tensor to set as idle
   * @return 0 on success, -1 on failure
   */
  int set_tensor_idle(const int &tensor_id);

  /**
   * @brief Fill image data into the input tensor
   * @param input_tensors Vector of input tensors to fill
   * @param image_data Pointer to the image data
   */
  int fill_img_to_input_tensor(std::vector<hbDNNTensor> &input_tensors, uint8_t *image_data);

  /**
   * @brief Apply non-maximum suppression (NMS) to keypoints
   * @param keypoints Keypoints to be suppressed
   * @param threshold Threshold for keypoint score
   * @param windowSize Size of the window used for suppression
   * @return Suppressed keypoints
   */
  std::vector<KeyPoint> applyNMS_grid_new(const std::vector<KeyPoint> &keypoints, float threshold, int windowSize);

  // ===================================== member variables =======================================
  rclcpp::Logger logger_;
  std::string model_path_;
  hbPackedDNNHandle_t packed_dnn_handle_;
  const char **model_name_list_;
  int model_count_ = 0;
  hbDNNHandle_t dnn_handle_;
  int input_count_ = 0;
  int output_count_ = 0;

  int32_t input_tensor_type_;

  int max_memory_count_ = 5;
  std::deque<std::atomic_bool> idle_tensor_;
  std::vector<std::vector<hbDNNTensor>> batch_output_tensors_;
  std::vector<std::vector<hbDNNTensor>> batch_input_tensors_;

  int model_input_w_;
  int model_input_h_;

  float point_th_high_ = 0.012;
  float point_th_low_ = 0.007;
};

#endif // DFEATINFER_H