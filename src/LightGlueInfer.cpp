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

#include "LightGlueInfer.h"

LightGlueInfer::LightGlueInfer(const rclcpp::Logger &logger) : logger_(logger) {
}

LightGlueInfer::~LightGlueInfer() {
  int ret_code = 0;
  // Free input memory
  for (int i = 0; i < max_memory_count_; i++) {
    for (size_t j = 0; j < batch_input_tensors_[i].size(); j++) {
      ret_code = hbSysFreeMem(&TENSOR_SYSMEM(batch_input_tensors_[i][j], 0));
      HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFreeMem failed");
    }
  }
  // Free output memory
  for (int i = 0; i < max_memory_count_; i++) {
    for (size_t j = 0; j < batch_output_tensors_[i].size(); j++) {
      ret_code = hbSysFreeMem(&TENSOR_SYSMEM(batch_output_tensors_[i][j], 0));
      HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFreeMem failed");
    }
  }
  // Release dnn handle
  ret_code = hbDNNRelease(packed_dnn_handle_);
  HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNInfer failed");
  LOG_WARN(logger_, "=> release LightGlueInfer");
}

int LightGlueInfer::init(const std::string &model_path, const int &max_memory_count) {
  int ret_code = 0;
  // load model
  model_path_ = model_path;
  const char *model_path_cstr = model_path_.c_str();
  ret_code = hbDNNInitializeFromFiles(&packed_dnn_handle_, &model_path_cstr, 1);
  HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNInitializeFromFiles failed");

  // get model name
  ret_code = hbDNNGetModelNameList(&model_name_list_, &model_count_, packed_dnn_handle_);
  HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNGetModelNameList failed");

  // get model handle
  ret_code = hbDNNGetModelHandle(&dnn_handle_, packed_dnn_handle_, model_name_list_[0]);
  HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNGetModelHandle failed");

  // get input count and output count
  ret_code = hbDNNGetInputCount(&input_count_, dnn_handle_);
  HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNGetInputCount failed");
  ret_code = hbDNNGetOutputCount(&output_count_, dnn_handle_);
  HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNGetOutputCount failed");
  LOG_WARN(logger_, "=> ============ init model start ============");
  LOG_WARN(logger_, "=> model name: " << model_name_list_[0]);
  LOG_WARN(logger_, "=> input_count: " << input_count_);
  LOG_WARN(logger_, "=> output_count: " << output_count_);

  // prepare input tensor and output tensor
  max_memory_count_ = max_memory_count;
  for (int i = 0; i < max_memory_count_; i++) {
    idle_tensor_.emplace_back(true);
  }

  batch_input_tensors_.resize(max_memory_count_);
  for (int i = 0; i < max_memory_count_; i++) {
    ret_code = prepare_input_tensor(batch_input_tensors_[i]);
  }

  batch_output_tensors_.resize(max_memory_count_);
  for (int i = 0; i < max_memory_count_; ++i) {
    ret_code = prepare_output_tensor(batch_output_tensors_[i]);
  }
  LOG_WARN_ONCE(logger_, "=> ============ init model end ============");

  return ret_code;
}

int LightGlueInfer::prepare_input_tensor(std::vector<hbDNNTensor> &input_tensors) {
  int ret_code = 0;
  LOG_WARN_ONCE(logger_, "=> ----- prepare_input_tensor -----");

  // allocate memory for input tensor
  input_tensors.resize(input_count_);
  for (int i = 0; i < input_count_; i++) {
    auto &tensor = input_tensors[i];
    // get input tensor properties
    hbDNNTensorProperties properties;
    ret_code = hbDNNGetInputTensorProperties(&properties, dnn_handle_, i);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNGetInputTensorProperties failed");
    LOG_WARN_ONCE(logger_, "=> input tensor type is "
                               << magic_enum::enum_name(static_cast<hbDNNDataType>(properties.tensorType)));
    input_tensor_type_ = properties.tensorType;

#ifdef PLATFORM_X5
    if (properties.tensorType != HB_DNN_TENSOR_TYPE_F32) {
      LOG_ERROR(logger_, "=> input tensor type is not in [HB_DNN_TENSOR_TYPE_F32]");
      return -1;
    }
#endif

#if defined(PLATFORM_S100) || defined(PLATFORM_S600)

#endif

#if defined(PLATFORM_S100) || defined(PLATFORM_S600)
    // properties.quantizeAxis = 3;
    properties.alignedByteSize = properties.validShape.dimensionSize[0] * properties.validShape.dimensionSize[1] *
                                 properties.validShape.dimensionSize[2] * properties.validShape.dimensionSize[3];
    auto dim_len = properties.validShape.numDimensions;
    for (int32_t dim_i = dim_len - 1; dim_i >= 0; --dim_i) {
      if (properties.stride[dim_i] == -1) {
        auto cur_stride = properties.stride[dim_i + 1] * properties.validShape.dimensionSize[dim_i + 1];
        properties.stride[dim_i] = ALIGN_32(cur_stride);
      }
    }
#endif

    tensor.properties = properties;
    tensor.properties.tensorType = properties.tensorType;

#ifdef PLATFORM_X5
    if (properties.tensorType == HB_DNN_TENSOR_TYPE_F32) {
      int input_memSize = tensor.properties.alignedByteSize;
      ret_code = hbSysAllocCachedMem(&tensor.sysMem[0], input_memSize);
      HB_CHECK_SUCCESS(logger_, ret_code, "hbSysAllocCachedMem failed");
      LOG_WARN_ONCE(logger_, "=> input[" << i << "].memsize: " << tensor.sysMem[0].memSize);
    } else {
      return -1;
    }
#endif

#if defined(PLATFORM_S100) || defined(PLATFORM_S600)

#endif
  }
  return ret_code;
}

int LightGlueInfer::prepare_output_tensor(std::vector<hbDNNTensor> &output_tensors) {
  int ret_code = 0;
  LOG_WARN_ONCE(logger_, "=> ----- prepare_output_tensor -----");
  output_tensors.resize(output_count_);
  for (int i = 0; i < output_count_; ++i) {
    ret_code = hbDNNGetOutputTensorProperties(&output_tensors[i].properties, dnn_handle_, i);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNGetOutputTensorProperties failed");
    LOG_WARN_ONCE(logger_, "=> output tensor type is " << magic_enum::enum_name(
                               static_cast<hbDNNDataType>(output_tensors[i].properties.tensorType)));
    ret_code = hbSysAllocCachedMem(&TENSOR_SYSMEM(output_tensors[i], 0), output_tensors[i].properties.alignedByteSize);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysAllocCachedMem failed");
    LOG_WARN_ONCE(logger_, "=> output[" << i << "].memsize: " << output_tensors[i].properties.alignedByteSize);
  }
  return ret_code;
}

int LightGlueInfer::get_idle_tensor() {
  for (int i = 0; i < max_memory_count_; ++i) {
    if (idle_tensor_[i]) {
      idle_tensor_[i] = false;
      return i;
    }
  }
  return -1;
}

int LightGlueInfer::set_tensor_idle(const int &tensor_id) {
  if (tensor_id >= 0 || tensor_id < max_memory_count_) {
    idle_tensor_[tensor_id] = true;
    return 0;
  }
  return -1;
}

int LightGlueInfer::forward(std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> &dfeat_result_1,
                            std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> &dfeat_result_2, int &img_height,
                            int &img_width, InferenceHandle &handle) {
  int ret_code = 0;
  // forward
  int idle_tensor_id = get_idle_tensor();
  {
    ScopeProcessTime t(logger_, "fill_img_to_input_tensor");
    if (idle_tensor_id == -1) {
      LOG_ERROR(logger_, "=> no idle tensor");
      return -1;
    }
    ret_code =
        fill_input_tensor(batch_input_tensors_[idle_tensor_id], dfeat_result_1, dfeat_result_2, img_height, img_width);
  }
  {
    ScopeProcessTime t(logger_, "infer");
    hbDNNTensor *output = batch_output_tensors_[idle_tensor_id].data();
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
    hbDNNTaskHandle_t task_handle = nullptr;
    ret_code =
        hbDNNInfer(&task_handle, &output, batch_input_tensors_[idle_tensor_id].data(), dnn_handle_, &infer_ctrl_param);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNInfer failed");
    // wait task done
    ret_code = hbDNNWaitTaskDone(task_handle, 0);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNWaitTaskDone failed");
    ret_code = hbDNNReleaseTask(task_handle);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNReleaseTask failed");
    // make sure CPU read data from DDR before using output tensor data
    for (size_t i = 0; i < batch_output_tensors_[idle_tensor_id].size(); i++) {
      ret_code =
          hbSysFlushMem(&TENSOR_SYSMEM(batch_output_tensors_[idle_tensor_id][i], 0), HB_SYS_MEM_CACHE_INVALIDATE);
      HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFlushMem failed");
    }
  }

  handle = idle_tensor_id;

  return ret_code;
}

int LightGlueInfer::postprocess(const InferenceHandle &handle, std::vector<cv::Point2f> &keypoint_1,
                                std::vector<cv::Point2f> &keypoint_2, std::vector<cv::Point2f> &match_kp_1,
                                std::vector<cv::Point2f> &match_kp_2) {
  int ret_code = 0;
  // get shape info
  auto &outputs = batch_output_tensors_[handle];
  if (outputs.size() != 3) {
    LOG_ERROR(logger_, "=> output tensor size is not equal to 3, size=" << outputs.size());
    set_tensor_idle(handle);
    return -1;
  }
  auto sim = outputs[0];
  auto z0 = outputs[1];
  auto z1 = outputs[2];
  int *sim_shape = sim.properties.validShape.dimensionSize;
  int sim_tensor_len = sim_shape[0] * sim_shape[1] * sim_shape[2] * sim_shape[3];
  int *z0_shape = z0.properties.validShape.dimensionSize;
  int z0_tensor_len = z0_shape[0] * z0_shape[1] * z0_shape[2] * z0_shape[3];
  int *z1_shape = z1.properties.validShape.dimensionSize;
  int z1_tensor_len = z1_shape[0] * z1_shape[1] * z1_shape[2] * z1_shape[3];
  LOG_INFO(logger_, "=> sim_shape [" << sim_shape[0] << ", " << sim_shape[1] << ", " << sim_shape[2] << ", "
                                     << sim_shape[3] << "]");
  LOG_INFO(logger_,
           "=> z0_shape [" << z0_shape[0] << ", " << z0_shape[1] << ", " << z0_shape[2] << ", " << z0_shape[3] << "]");
  LOG_INFO(logger_,
           "=> z1_shape [" << z1_shape[0] << ", " << z1_shape[1] << ", " << z1_shape[2] << ", " << z1_shape[3] << "]");

  if (sim.properties.tensorType == HB_DNN_TENSOR_TYPE_F32 && z0.properties.tensorType == HB_DNN_TENSOR_TYPE_F32 &&
      z1.properties.tensorType == HB_DNN_TENSOR_TYPE_F32) {
    auto sim_data = reinterpret_cast<float *>(TENSOR_SYSMEM(sim, 0).virAddr);
    auto z0_data = reinterpret_cast<float *>(TENSOR_SYSMEM(z0, 0).virAddr);
    auto z1_data = reinterpret_cast<float *>(TENSOR_SYSMEM(z1, 0).virAddr);

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> simEigenMatrix(
        sim_data, sim_shape[1], sim_shape[2]);
    Eigen::MatrixXd simEigenMatrixXd(sim_shape[1], sim_shape[2]);
    simEigenMatrixXd = simEigenMatrix.cast<double>();

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> z0EigenMatrix(
        z0_data, z0_shape[1], z0_shape[2]);
    Eigen::MatrixXd z0EigenMatrixXd(z0_shape[1], z0_shape[2]);
    z0EigenMatrixXd = z0EigenMatrix.cast<double>();
    Eigen::VectorXd z0_vec = Eigen::Map<const Eigen::VectorXd>(z0EigenMatrixXd.data(), z0_tensor_len);

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> z1EigenMatrix(
        z1_data, z1_shape[1], z1_shape[2]);
    Eigen::MatrixXd z1EigenMatrixXd(z1_shape[1], z1_shape[2]);
    z1EigenMatrixXd = z1EigenMatrix.cast<double>();
    Eigen::VectorXd z1_vec = Eigen::Map<const Eigen::VectorXd>(z1EigenMatrixXd.data(), z1_tensor_len);

    Eigen::MatrixXd scores_tmp = sigmoid_log_double_softmax(simEigenMatrixXd, z0_vec, z1_vec);
    std::vector<std::pair<int, int>> matches_vec;
    std::vector<double> mscores_vec;
    filter_matches(scores_tmp, matches_vec, mscores_vec);
    int nums_match = 0;
    for (int i = 0; i < mscores_vec.size(); i++) {
      if (mscores_vec[i] > 0.08) {
        nums_match++;
        match_kp_1.push_back(keypoint_1[matches_vec[i].first]);
        match_kp_2.push_back(keypoint_2[matches_vec[i].second]);
      }
    }
    LOG_INFO(logger_, "=> nums match: " << nums_match);
  }

  // reset idle tensor
  set_tensor_idle(handle);
  return ret_code;
}

int LightGlueInfer::forward(std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> &dfeat_result_1,
                            std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> &dfeat_result_2, int &img_height,
                            int &img_width, std::vector<cv::Point2f> &match_kp_1,
                            std::vector<cv::Point2f> &match_kp_2) {
  int ret_code = 0;
  // forward
  InferenceHandle handle = 0;
  ret_code = forward(dfeat_result_1, dfeat_result_2, img_height, img_width, handle);
  if (ret_code != 0) return ret_code;
  // postprocess
  ret_code = postprocess(handle, dfeat_result_1.first, dfeat_result_2.first, match_kp_1, match_kp_2);
  return ret_code;
}

std::vector<cv::Point2f> LightGlueInfer::NormalizeKeypoints(std::vector<cv::Point2f> &kpts, int h, int w) {
  cv::Size size(w, h);
  cv::Point2f shift(static_cast<float>(w) / 2, static_cast<float>(h) / 2);
  float scale = static_cast<float>((std::max)(w, h)) / 2;

  std::vector<cv::Point2f> normalizedKpts;
  for (const cv::Point2f &kpt : kpts) {
    cv::Point2f normalizedKpt = (kpt - shift) / scale;
    normalizedKpts.push_back(normalizedKpt);
  }

  return normalizedKpts;
}

int LightGlueInfer::fill_input_tensor(std::vector<hbDNNTensor> &input_tensors,
                                      std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> &dfeat_result_1,
                                      std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> &dfeat_result_2,
                                      int &img_height, int &img_width) {
  int ret_code = 0;
  auto keypoint_1 = dfeat_result_1.first;
  auto desc_1 = dfeat_result_1.second;
  auto keypoint_2 = dfeat_result_2.first;
  auto desc_2 = dfeat_result_2.second;
#ifdef PLATFORM_X5
  //   hbDNNTensor &input_tensor = input_tensors[0];

  if (input_tensor_type_ == HB_DNN_TENSOR_TYPE_F32) {
    // fill image data into memory
    auto kpts1 = NormalizeKeypoints(keypoint_1, img_height, img_width);
    auto kpts2 = NormalizeKeypoints(keypoint_2, img_height, img_width);
    float *kpts1_data = new float[kpts1.size() * 2];
    float *kpts2_data = new float[kpts2.size() * 2];
    for (size_t i = 0; i < kpts1.size(); ++i) {
      kpts1_data[i * 2] = kpts1[i].x;
      kpts1_data[i * 2 + 1] = kpts1[i].y;
    }

    for (size_t i = 0; i < kpts2.size(); ++i) {
      kpts2_data[i * 2] = kpts2[i].x;
      kpts2_data[i * 2 + 1] = kpts2[i].y;
    }

    Eigen::MatrixXf desc_1_float = desc_1.cast<float>();
    Eigen::MatrixXf desc_2_float = desc_2.cast<float>();
    Eigen::MatrixXf desc_1_float_trans = desc_1_float.transpose();
    Eigen::MatrixXf desc_2_float_trans = desc_2_float.transpose();
    float *desc1 = desc_1_float_trans.data();
    float *desc2 = desc_2_float_trans.data();

    hbDNNTensor &kp1_tensor = input_tensors[0];
    hbDNNTensor &kp2_tensor = input_tensors[1];
    hbDNNTensor &desc1_tensor = input_tensors[2];
    hbDNNTensor &desc2_tensor = input_tensors[3];

    int *kp1_shape = input_tensors[0].properties.validShape.dimensionSize;
    int kp1_tensor_len = kp1_shape[0] * kp1_shape[1] * kp1_shape[2] * kp1_shape[3];
    int *kp2_shape = input_tensors[1].properties.validShape.dimensionSize;
    int kp2_tensor_len = kp2_shape[0] * kp2_shape[1] * kp2_shape[2] * kp2_shape[3];
    int *desc1_shape = input_tensors[2].properties.validShape.dimensionSize;
    int desc1_tensor_len = desc1_shape[0] * desc1_shape[1] * desc1_shape[2] * desc1_shape[3];
    int *desc2_shape = input_tensors[3].properties.validShape.dimensionSize;
    int desc2_tensor_len = desc2_shape[0] * desc2_shape[1] * desc2_shape[2] * desc2_shape[3];

    ret_code = hbSysWriteMem(&kp1_tensor.sysMem[0], (char *)kpts1_data, kp1_tensor_len * 4);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysWriteMem failed");
    ret_code = hbSysWriteMem(&kp2_tensor.sysMem[0], (char *)kpts2_data, kp2_tensor_len * 4);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysWriteMem failed");
    ret_code = hbSysWriteMem(&desc1_tensor.sysMem[0], (char *)desc1, desc1_tensor_len * 4);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysWriteMem failed");
    ret_code = hbSysWriteMem(&desc2_tensor.sysMem[0], (char *)desc2, desc2_tensor_len * 4);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysWriteMem failed");
    delete[] kpts1_data;
    delete[] kpts2_data;

    // make sure memory data is flushed to DDR before inference
    ret_code = hbSysFlushMem(&kp1_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFlushMem failed");
    ret_code = hbSysFlushMem(&kp2_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFlushMem failed");
    ret_code = hbSysFlushMem(&desc1_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFlushMem failed");
    ret_code = hbSysFlushMem(&desc2_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFlushMem failed");
  } else {
    LOG_ERROR(logger_, "=> input_tensor_type is not in [HB_DNN_TENSOR_TYPE_F32]");
    return -1;
  }
#endif

#if defined(PLATFORM_S100) || defined(PLATFORM_S600)

#endif

  return ret_code;
}

// log_softmax over rows (axis=1)
Eigen::MatrixXd LightGlueInfer::log_softmax_rows(Eigen::MatrixXd &x) {
  Eigen::MatrixXd result(x.rows(), x.cols());
  for (int i = 0; i < x.rows(); ++i) {
    double maxVal = x.row(i).maxCoeff();
    Eigen::RowVectorXd shifted = x.row(i).array() - maxVal;
    double logsum = std::log((shifted.array().exp()).sum());
    result.row(i) = shifted.array() - logsum;
  }
  return result;
}

// log_softmax over columns (axis=0)
Eigen::MatrixXd LightGlueInfer::log_softmax_cols(Eigen::MatrixXd &x) {
  Eigen::MatrixXd result(x.rows(), x.cols());
  for (int j = 0; j < x.cols(); ++j) {
    double maxVal = x.col(j).maxCoeff();
    Eigen::VectorXd shifted = x.col(j).array() - maxVal;
    double logsum = std::log((shifted.array().exp()).sum());
    result.col(j) = shifted.array() - logsum;
  }
  return result;
}

inline double LightGlueInfer::logsigmoid(double x) {
  return -std::log1p(std::exp(-x)); // log(sigmoid(x))
}

Eigen::VectorXd log_softmax(Eigen::VectorXd &x) {
  double max_val = x.maxCoeff();
  Eigen::VectorXd shifted = x.array() - max_val;
  double log_sum_exp = std::log(shifted.array().exp().sum());
  return shifted.array() - log_sum_exp;
}

Eigen::MatrixXd LightGlueInfer::sigmoid_log_double_softmax(Eigen::MatrixXd &sim, Eigen::VectorXd &z0,
                                                           Eigen::VectorXd &z1) {
  int M = sim.rows();
  int N = sim.cols();

  Eigen::VectorXd log_z0(z0.size());
  for (int i = 0; i < z0.size(); ++i) {
    log_z0[i] = -std::log1p(std::exp(-z0[i]));
  }

  Eigen::VectorXd log_z1(z1.size());
  for (int i = 0; i < z1.size(); ++i) {
    log_z1[i] = -std::log1p(std::exp(-z1[i]));
  }

  Eigen::MatrixXd certainties = log_z0.replicate(1, N) + log_z1.transpose().replicate(M, 1);

  Eigen::MatrixXd scores0 = log_softmax_rows(sim);
  Eigen::MatrixXd scores1 = log_softmax_cols(sim);

  Eigen::MatrixXd scores = scores0 + scores1 + certainties;
  return scores;
}

void LightGlueInfer::filter_matches(Eigen::MatrixXd &scores,                   // log assignment matrix, shape [M, N]
                                    std::vector<std::pair<int, int>> &matches, // output: matched pairs (i, j)
                                    std::vector<double> &mscores // output: matching scores (exp of log score)
) {
  int M = scores.rows();
  int N = scores.cols();

  Eigen::VectorXi m0(M);
  for (int i = 0; i < M; ++i) {
    scores.row(i).maxCoeff(&m0(i));
  }

  Eigen::VectorXi m1(N);
  for (int j = 0; j < N; ++j) {
    scores.col(j).maxCoeff(&m1(j));
  }

  for (int i = 0; i < M; ++i) {
    int j = m0(i);
    if (m1(j) == i) { // mutual match
      matches.emplace_back(i, j);
      double score = std::exp(scores(i, j));
      mscores.push_back(score);
    }
  }
}