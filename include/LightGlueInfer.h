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

#ifndef LIGHTGLUEINFER_H
#define LIGHTGLUEINFER_H

#include <vector>
#include <map>
#include <unordered_map>
#include <deque>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "magic_enum/magic_enum.hpp"
#include "log_macros.h"
#include "timer_utils.h"
#include "dnn_platform.h"

using InferenceHandle = int;

class LightGlueInfer {
public:
  explicit LightGlueInfer(const rclcpp::Logger &logger);
  ~LightGlueInfer();

  /**
   * @brief Initialize the LightGlue model
   * @param model_path Path to the LightGlue model file
   * @param max_memory_count Maximum number of memory buffers to allocate
   * @return 0 on success, -1 on failure
   */
  int init(const std::string &model_path, const int &max_memory_count = 5);

  /**
   * @brief Perform forward inference using the LightGlue model asynchronously
   * @param dfeat_result_1 Output keypoint and descriptor extraction result
   * @param dfeat_result_2 Output keypoint and descriptor extraction result
   * @param handle Output inference handle
   * @return 0 on success, -1 on failure
   */
  int forward(std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> &dfeat_result_1,
              std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> &dfeat_result_2, int &img_height, int &img_width,
              InferenceHandle &handle);

  /**
   * @brief Perform postprocess using the LightGlue model
   * @param handle Output inference handle
   * @param keypoint_1 Input keypoint 1
   * @param keypoint_2 Input keypoint 2
   * @param match_kp_1 Output keypoint 1
   * @param match_kp_2 Output keypoint 2
   * @return 0 on success, -1 on failure
   */
  int postprocess(const InferenceHandle &handle, std::vector<cv::Point2f> &keypoint_1,
                  std::vector<cv::Point2f> &keypoint_2, std::vector<cv::Point2f> &match_kp_1,
                  std::vector<cv::Point2f> &match_kp_2);
  int postprocess_v1(const InferenceHandle &handle, std::vector<cv::Point2f> &keypoint_1,
                     std::vector<cv::Point2f> &keypoint_2, std::vector<cv::Point2f> &match_kp_1,
                     std::vector<cv::Point2f> &match_kp_2);
  int postprocess_v2(const InferenceHandle &handle, std::vector<cv::Point2f> &keypoint_1,
                     std::vector<cv::Point2f> &keypoint_2, std::vector<cv::Point2f> &match_kp_1,
                     std::vector<cv::Point2f> &match_kp_2);

  /**
   * @brief Perform forward inference using the LightGlue model
   * @param dfeat_result_1 Output keypoint and descriptor extraction result
   * @param dfeat_result_2 Output keypoint and descriptor extraction result
   * @param match_kp_1 Output keypoint 1
   * @param match_kp_2 Output keypoint 2
   * @return 0 on success, -1 on failure
   */
  int forward(std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> &dfeat_result_1,
              std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> &dfeat_result_2, int &img_height, int &img_width,
              std::vector<cv::Point2f> &match_kp_1, std::vector<cv::Point2f> &match_kp_2);

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
   * @brief Normalize keypoints
   * @param kpts Keypoints to be normalized
   * @param h Height of the input image
   * @param w Width of the input image
   * @return Normalized keypoints
   */
  std::vector<cv::Point2f> NormalizeKeypoints(std::vector<cv::Point2f> &kpts, int h, int w);

  /**
   * @brief Fill image data into the input tensor
   * @param input_tensors Vector of input tensors to fill
   * @param dfeat_result_1 Output keypoint and descriptor extraction result
   * @param dfeat_result_2 Output keypoint and descriptor extraction result
   * @return 0 on success, -1 on failure
   */
  int fill_input_tensor(std::vector<hbDNNTensor> &input_tensors,
                        std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> &dfeat_result_1,
                        std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> &dfeat_result_2, int &img_height,
                        int &img_width);

  /**
   * @brief log_softmax over rows (axis=1)
   * @param x Input matrix
   * @return Log softmax over rows
   */
  Eigen::MatrixXd log_softmax_rows(Eigen::MatrixXd &x);

  /**
   * @brief log_softmax over columns (axis=0)
   * @param x Input matrix
   * @return Log softmax over columns
   */
  Eigen::MatrixXd log_softmax_cols(Eigen::MatrixXd &x);

  /**
   * @brief Sigmoid function
   * @param x Input value
   * @return Sigmoid value
   */
  inline double logsigmoid(double x);

  /**
   * @brief Log softmax
   * @param x Input vector
   * @return Log softmax vector
   */
  Eigen::VectorXd log_softmax(Eigen::VectorXd &x);

  /**
   * @brief Sigmoid log double softmax
   * @param sim Input matrix
   * @param z0 Input vector
   * @param z1 Input vector
   * @return Sigmoid log double softmax matrix
   */
  Eigen::MatrixXd sigmoid_log_double_softmax(Eigen::MatrixXd &sim, Eigen::VectorXd &z0, Eigen::VectorXd &z1);

  /**
   * @brief Filter matches
   * @param scores Log assignment matrix
   * @param matches Output matched pairs
   * @param mscores Output matching scores
   */
  void filter_matches(Eigen::MatrixXd &scores,                   // log assignment matrix, shape [M, N]
                      std::vector<std::pair<int, int>> &matches, // output: matched pairs (i, j)
                      std::vector<double> &mscores               // output: matching scores (exp of log score)
  );

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
};

#endif // LIGHTGLUEINFER_H