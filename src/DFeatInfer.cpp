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

#include "DFeatInfer.h"

DFeatInfer::DFeatInfer(const rclcpp::Logger &logger) : logger_(logger) {
}

DFeatInfer::~DFeatInfer() {
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
  LOG_WARN(logger_, "=> release DFMatchInfer");
}

int DFeatInfer::init(const std::string &model_path, const int &max_memory_count) {
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

  // get model input size from input tensor[0]
  hbDNNTensorProperties properties;
  ret_code = hbDNNGetInputTensorProperties(&properties, dnn_handle_, 0);
#if defined(PLATFORM_S100) || defined(PLATFORM_S600)
  properties.quantizeAxis = 3;
#endif
  hbGetInputTensorHW(properties, model_input_h_, model_input_w_);
  LOG_WARN(logger_, "=> model_input_h: " << model_input_h_ << ", model_input_w: " << model_input_w_);

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

int DFeatInfer::prepare_input_tensor(std::vector<hbDNNTensor> &input_tensors) {
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
    if (properties.tensorType != HB_DNN_IMG_TYPE_Y) {
      LOG_ERROR(logger_, "=> input tensor type is not in [HB_DNN_IMG_TYPE_Y]");
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
    if (properties.tensorType == HB_DNN_IMG_TYPE_Y) {
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

int DFeatInfer::prepare_output_tensor(std::vector<hbDNNTensor> &output_tensors) {
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

int DFeatInfer::get_idle_tensor() {
  for (int i = 0; i < max_memory_count_; ++i) {
    if (idle_tensor_[i]) {
      idle_tensor_[i] = false;
      return i;
    }
  }
  return -1;
}

int DFeatInfer::set_tensor_idle(const int &tensor_id) {
  if (tensor_id >= 0 || tensor_id < max_memory_count_) {
    idle_tensor_[tensor_id] = true;
    return 0;
  }
  return -1;
}

int DFeatInfer::fill_img_to_input_tensor(std::vector<hbDNNTensor> &input_tensors, uint8_t *image_data) {
  int ret_code = 0;
#ifdef PLATFORM_X5
  hbDNNTensor &input_tensor = input_tensors[0];

  if (input_tensor_type_ == HB_DNN_IMG_TYPE_Y) {
    // fill image data into memory
    ret_code = hbSysWriteMem(&input_tensor.sysMem[0], (char *)image_data, input_tensor.sysMem[0].memSize);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysWriteMem failed");

    // make sure memory data is flushed to DDR before inference
    ret_code = hbSysFlushMem(&input_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFlushMem failed");
  } else {
    LOG_ERROR(logger_, "=> input_tensor_type is not in [HB_DNN_IMG_TYPE_Y]");
    return -1;
  }
#endif

#if defined(PLATFORM_S100) || defined(PLATFORM_S600)

#endif

  return ret_code;
}

int DFeatInfer::forward(const cv::Mat &image, InferenceHandle &handle) {
  int ret_code = 0;
  // forward
  int idle_tensor_id = get_idle_tensor();
  {
    ScopeProcessTime t(logger_, "fill_img_to_input_tensor");
    if (idle_tensor_id == -1) {
      LOG_ERROR(logger_, "=> no idle tensor");
      return -1;
    }
    // check img is gray or not
    if (image.channels() != 1) {
      LOG_ERROR(logger_, "=> input image is not gray");
      return -1;
    }
    ret_code = fill_img_to_input_tensor(batch_input_tensors_[idle_tensor_id], image.data);
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

int DFeatInfer::postprocess(const InferenceHandle &handle,
                            std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> &extractor_result) {
  ScopeProcessTime t(logger_, "postprocess");
  int ret_code = 0;
  int idle_tensor_id = handle;

  // get shape info
  auto &outputs = batch_output_tensors_[idle_tensor_id];
  if (outputs.size() != 2) {
    LOG_ERROR(logger_, "=> output tensor size is not equal to 2, size=" << outputs.size());
    set_tensor_idle(idle_tensor_id);
    return -1;
  }

  auto semi = outputs[0];
  auto desc = outputs[1];
  int *semi_shape = semi.properties.validShape.dimensionSize;
  int tensor_len = semi_shape[0] * semi_shape[1] * semi_shape[2] * semi_shape[3];
  // int H = shape[2];
  int W = semi_shape[3];
  int *desc_shape = desc.properties.validShape.dimensionSize;
  LOG_INFO(logger_, "=> semi_shape [" << semi_shape[0] << ", " << semi_shape[1] << ", " << semi_shape[2] << ", "
                                      << semi_shape[3] << "]");
  LOG_INFO(logger_, "=> desc_shape [" << desc_shape[0] << ", " << desc_shape[1] << ", " << desc_shape[2] << ", "
                                      << desc_shape[3] << "]");

  if (semi.properties.tensorType == HB_DNN_TENSOR_TYPE_S8 && desc.properties.tensorType == HB_DNN_TENSOR_TYPE_S8) {
    auto semi_data = reinterpret_cast<int8_t *>(TENSOR_SYSMEM(semi, 0).virAddr);
    auto desc_data = reinterpret_cast<int8_t *>(TENSOR_SYSMEM(desc, 0).virAddr);

    // ===================================== keypoint postprocess =======================================
    int res_count_high = 0;
    int res_count_low = 0;
    std::vector<KeyPoint> keypoints_all;
    std::vector<KeyPoint> keypoints_all_high;
    std::vector<KeyPoint> keypoints_all_low;
    float *semi_scale = semi.properties.scale.scaleData;
    for (int i = 0; i < tensor_len; i++) {
      if (semi.properties.quantiType != SCALE) {
        LOG_ERROR(logger_, "=> semi quantiType is not SCALE");
        set_tensor_idle(idle_tensor_id);
        return -1;
      }
      float semi_dequant = static_cast<float>(semi_data[i]) * semi_scale[0];

      if (semi_dequant >= point_th_high_) {
        res_count_high++;
        int kp_x = i % W;
        int kp_y = i / W;

        KeyPoint cur_kp;
        cur_kp.x = kp_x;
        cur_kp.y = kp_y;
        cur_kp.score = semi_dequant;
        keypoints_all_high.emplace_back(cur_kp);
      }

      if (semi_dequant >= point_th_low_) {
        res_count_low++;
        int kp_x = i % W;
        int kp_y = i / W;

        KeyPoint cur_kp;
        cur_kp.x = kp_x;
        cur_kp.y = kp_y;
        cur_kp.score = semi_dequant;
        keypoints_all_low.emplace_back(cur_kp);
      }
    }
    if (keypoints_all_high.size() >= 1600) {
      keypoints_all = keypoints_all_high;
    } else {
      keypoints_all = keypoints_all_low;
    }

    LOG_INFO(logger_, "=> before nms find keypoints: " << keypoints_all.size());
    float threshold = 0.0;
    int windowSize = 5;
    std::vector<KeyPoint> nmsKeypoints_kp = applyNMS_grid_new(keypoints_all, threshold, windowSize);
    std::vector<KeyPoint> sortedKeypoints = nmsKeypoints_kp;
    std::sort(sortedKeypoints.begin(), sortedKeypoints.end(),
              [](const KeyPoint &a, const KeyPoint &b) { return a.score > b.score; });
    std::vector<cv::Point2f> nmsKeypoints;
    for (int i = 0; i < sortedKeypoints.size(); i++) {
      KeyPoint kp = sortedKeypoints[i];
      nmsKeypoints.push_back(cv::Point2f(kp.x * 1.0, kp.y * 1.0));
    }
    LOG_INFO(logger_, "=> after nms nms find keypoints: " << nmsKeypoints.size());

    // ===================================== desc postprocess =======================================
    if (desc.properties.quantiType != SCALE) {
      LOG_ERROR(logger_, "=> desc quantiType is not SCALE");
      set_tensor_idle(idle_tensor_id);
      return -1;
    }
    float *desc_scale = desc.properties.scale.scaleData;

    int desc_dim = desc_shape[3];
    int kp_nums = nmsKeypoints.size();
    if (kp_nums < 256) kp_nums = 256;
    Eigen::MatrixXf matMatrixXd(kp_nums, desc_dim);

    for (int i = 0; i < nmsKeypoints.size(); i++) {
      int cur_x = nmsKeypoints[i].x;
      int cur_y = nmsKeypoints[i].y;

      Eigen::VectorXf cur_desc = Eigen::VectorXf::Zero(desc_dim);
      for (int j = 0; j < desc_dim; j++) {
        cur_desc[j] = static_cast<float>(desc_data[(cur_y * W + cur_x) * desc_dim + j]) * desc_scale[0];
      }

      float norm_sqrt = cur_desc.norm();
      Eigen::VectorXf cur_desc_normalized = cur_desc / norm_sqrt;
      matMatrixXd.row(i) = cur_desc_normalized;
    }

    if (nmsKeypoints.size() < 256) {
      for (int k = nmsKeypoints.size(); k < 256; k++) {
        nmsKeypoints.push_back(nmsKeypoints[0]);
        Eigen::VectorXf tmp_desc = Eigen::VectorXf::Zero(desc_dim);
        matMatrixXd.row(k) = matMatrixXd.row(0);
      }
    }
    Eigen::MatrixXd desc_double(matMatrixXd.cast<double>());

    extractor_result.first = nmsKeypoints;
    extractor_result.second = desc_double;
  }

  // reset idle tensor
  set_tensor_idle(idle_tensor_id);
  return ret_code;
}

int DFeatInfer::forward(const cv::Mat &image, std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> &extractor_result) {
  int ret_code = 0;
  // forward
  InferenceHandle handle = 0;
  ret_code = forward(image, handle);
  if (ret_code != 0) return ret_code;
  // postprocess
  ret_code = postprocess(handle, extractor_result);
  return ret_code;
}

std::vector<KeyPoint> DFeatInfer::applyNMS_grid_new(const std::vector<KeyPoint> &keypoints, float threshold,
                                                    int windowSize) {
  std::vector<KeyPoint> nmsKeypoints;
  Grid grid;

  for (const auto &kp : keypoints) {
    auto key = hashKeyPoint(kp);
    grid[key].push_back(kp);
  }

  std::vector<bool> suppressed(keypoints.size(), false);

  for (size_t i = 0; i < keypoints.size(); ++i) {
    if (suppressed[i]) continue;

    KeyPoint kp = keypoints[i];
    bool isLocalMaximum = true;

    auto key = hashKeyPoint(kp);
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        auto neighborKey = std::make_pair(key.first + dx, key.second + dy);
        if (grid.find(neighborKey) != grid.end()) {
          for (const auto &neighbor : grid[neighborKey]) {
            if (neighbor.x == kp.x && neighbor.y == kp.y) continue;

            float dist = std::sqrt(std::pow(kp.x - neighbor.x, 2) + std::pow(kp.y - neighbor.y, 2));
            if (dist < windowSize && neighbor.score > kp.score - threshold) {
              isLocalMaximum = false;
              break;
            }
          }
        }
        if (!isLocalMaximum) break;
      }
      if (!isLocalMaximum) break;
    }

    if (isLocalMaximum) {
      nmsKeypoints.push_back(kp);
    } else {
      suppressed[i] = true;
    }
  }

  return nmsKeypoints;
}

void DFeatInfer::get_model_input_size(int &w, int &h) const {
  w = model_input_w_;
  h = model_input_h_;
}