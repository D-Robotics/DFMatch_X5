#include <iostream>
#include <vector>
#include <chrono>
#include <map>
#include <fstream>

#include <cmath>
#include <unordered_map>

#include "dnn/hb_dnn.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace std;

#define HB_CHECK_SUCCESS(value, errmsg)                                                                                                                                                                \
  do {                                                                                                                                                                                                 \
    /*value can be call of function*/                                                                                                                                                                  \
    auto ret_code = value;                                                                                                                                                                             \
    if (ret_code != 0) {                                                                                                                                                                               \
      std::cout << errmsg << ", error code:" << ret_code << std::endl;                                                                                                                                 \
      return ret_code;                                                                                                                                                                                 \
    }                                                                                                                                                                                                  \
  } while (0);

hbPackedDNNHandle_t packed_dnn_handle_sp;
hbDNNHandle_t dnn_handle_sp;
const char **model_name_list_sp;
bool flag_init_sp = false;

float point_th_high = 0.012;

float point_th_low = 0.007;

// float point_th_low = 0.012;

int windowSize = 5;

hbPackedDNNHandle_t packed_dnn_handle_lg;
hbDNNHandle_t dnn_handle_lg;
const char **model_name_list_lg;
bool flag_init_lg = false;

struct KeyPoint {
  int x, y;
  float score;
};

struct pair_hash {
  template <class T1, class T2> size_t operator()(pair<T1, T2> const &pair) const {
    size_t h1 = hash<T1>()(pair.first);
    size_t h2 = hash<T2>()(pair.second);
    return h1 ^ h2;
  }
};

const float GRID_SIZE = 10.0f;

using Grid = std::unordered_map<std::pair<int, int>, std::vector<KeyPoint>, pair_hash>;

std::pair<int, int> hashKeyPoint(const KeyPoint &kp) {
  int xHash = static_cast<int>(kp.x / GRID_SIZE);
  int yHash = static_cast<int>(kp.y / GRID_SIZE);
  return {xHash, yHash};
}

std::vector<KeyPoint> applyNMS_grid_new(const std::vector<KeyPoint> &keypoints, float threshold, int windowSize) {
  std::vector<KeyPoint> nmsKeypoints;
  Grid grid;

  for (const auto &kp : keypoints) {
    auto key = hashKeyPoint(kp);
    grid[key].push_back(kp);
  }

  std::vector<bool> suppressed(keypoints.size(), false);

  for (size_t i = 0; i < keypoints.size(); ++i) {
    if (suppressed[i])
      continue;

    KeyPoint kp = keypoints[i];
    bool isLocalMaximum = true;

    auto key = hashKeyPoint(kp);
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        auto neighborKey = std::make_pair(key.first + dx, key.second + dy);
        if (grid.find(neighborKey) != grid.end()) {
          for (const auto &neighbor : grid[neighborKey]) {
            if (neighbor.x == kp.x && neighbor.y == kp.y)
              continue;

            float dist = std::sqrt(std::pow(kp.x - neighbor.x, 2) + std::pow(kp.y - neighbor.y, 2));
            if (dist < windowSize && neighbor.score > kp.score - threshold) {
              isLocalMaximum = false;
              break;
            }
          }
        }
        if (!isLocalMaximum)
          break;
      }
      if (!isLocalMaximum)
        break;
    }

    if (isLocalMaximum) {
      nmsKeypoints.push_back(kp);
    } else {
      suppressed[i] = true;
    }
  }

  return nmsKeypoints;
}

int prepare_tensor(hbDNNTensor *input_tensor, hbDNNTensor *output_tensor, hbDNNHandle_t dnn_handle) {
  int input_count = 0;
  int output_count = 0;
  hbDNNGetInputCount(&input_count, dnn_handle);
  hbDNNGetOutputCount(&output_count, dnn_handle);

  /** Tips:
   * For input memory size:
   * *   input_memSize = input[i].properties.alignedByteSize
   * For output memory size:
   * *   output_memSize = output[i].properties.alignedByteSize
   */
  hbDNNTensor *input = input_tensor;
  for (int i = 0; i < input_count; i++) {
    HB_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input[i].properties, dnn_handle, i), "hbDNNGetInputTensorProperties failed");
    int input_memSize = input[i].properties.alignedByteSize;
    HB_CHECK_SUCCESS(hbSysAllocCachedMem(&input[i].sysMem[0], input_memSize), "hbSysAllocCachedMem failed");
    /** Tips:
     * For input tensor, aligned shape should always be equal to the real
     * shape of the user's data. If you are going to set your input data with
     * padding, this step is not necessary.
     * */
    input[i].properties.alignedShape = input[i].properties.validShape;

    // Show how to get input name
    const char *input_name;
    HB_CHECK_SUCCESS(hbDNNGetInputName(&input_name, dnn_handle, i), "hbDNNGetInputName failed");
    std::cout << "input[" << i << "] name is " << input_name << std::endl;
  }

  hbDNNTensor *output = output_tensor;
  for (int i = 0; i < output_count; i++) {
    HB_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i), "hbDNNGetOutputTensorProperties failed");
    int output_memSize = output[i].properties.alignedByteSize;
    HB_CHECK_SUCCESS(hbSysAllocCachedMem(&output[i].sysMem[0], output_memSize), "hbSysAllocCachedMem failed");

    // Show how to get output name
    const char *output_name;
    HB_CHECK_SUCCESS(hbDNNGetOutputName(&output_name, dnn_handle, i), "hbDNNGetOutputName failed");
    std::cout << "output[" << i << "] name is " << output_name << std::endl;
  }
  return 0;
}

/** You can define read_image_2_tensor_as_other_type to prepare your data **/
int32_t read_image_2_tensor_as_gray_dfeat(cv::Mat &bgr_mat, hbDNNTensor *input_tensor) {
  hbDNNTensor *input = input_tensor;
  hbDNNTensorProperties Properties = input->properties;
  int tensor_id = 0;

  int input_h = Properties.validShape.dimensionSize[2];
  int input_w = Properties.validShape.dimensionSize[3];

  cv::Mat normalized_image_gray;
  cv::cvtColor(bgr_mat, normalized_image_gray, cv::COLOR_BGR2GRAY);

  if (bgr_mat.empty()) {
    std::cout << "image file not exist!" << std::endl;
    ;
    return -1;
  }

  if (input_h % 2 || input_w % 2) {
    std::cout << "input img height and width must aligned by 2!" << std::endl;
    return -1;
  }

  auto data = input->sysMem[0].virAddr;
  int32_t data_size = input_h * input_w;

  memcpy(data, normalized_image_gray.data, data_size);

  return 0;
}

int infer_x5_dfeat(cv::Mat &bgr_mat, std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> &extractor_result) {

  if (flag_init_sp == false) {
    flag_init_sp = true;

    std::string model_name = "./dfeat_640_640.bin";

    auto modelFileName = model_name.c_str();
    int model_count = 0;
    // Step1: get model handle
    {
      HB_CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle_sp, &modelFileName, 1), "hbDNNInitializeFromFiles failed");
      HB_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list_sp, &model_count, packed_dnn_handle_sp), "hbDNNGetModelNameList failed");
      HB_CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle_sp, packed_dnn_handle_sp, model_name_list_sp[0]), "hbDNNGetModelHandle failed");
    }
    // Show how to get dnn version
    std::cout << "DNN runtime version: " << hbDNNGetVersion() << std::endl;
  }

  std::vector<hbDNNTensor> input_tensors;
  std::vector<hbDNNTensor> output_tensors; //
  int input_count = 0;
  int output_count = 0;

  // Step2: prepare input and output tensor
  {
    HB_CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle_sp), "hbDNNGetInputCount failed");
    HB_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, dnn_handle_sp), "hbDNNGetOutputCount failed");
    input_tensors.resize(input_count);
    output_tensors.resize(output_count);
    prepare_tensor(input_tensors.data(), output_tensors.data(), dnn_handle_sp);
  }

  // Step3: set input data to input tensor
  {

    // read a single picture for input_tensor[0], for multi_input model, you
    // should set other input data according to model input properties.
    HB_CHECK_SUCCESS(read_image_2_tensor_as_gray_dfeat(bgr_mat, input_tensors.data()), "read_image_2_tensor_as_nv12 failed");
  }

  hbDNNTaskHandle_t task_handle = nullptr;
  hbDNNTensor *output = output_tensors.data();

  auto t1 = std::chrono::steady_clock::now();

  // Step4: run inference
  {
    // make sure memory data is flushed to DDR before inference
    for (int i = 0; i < input_count; i++) {
      hbSysFlushMem(&input_tensors[i].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    }

    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
    HB_CHECK_SUCCESS(hbDNNInfer(&task_handle, &output, input_tensors.data(), dnn_handle_sp, &infer_ctrl_param), "hbDNNInfer failed");
    // wait task done
    HB_CHECK_SUCCESS(hbDNNWaitTaskDone(task_handle, 0), "hbDNNWaitTaskDone failed");
  }

  auto t2 = std::chrono::steady_clock::now();
  auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;
  std::cout << "\033[31m" << "dfeat infer Time:" << time_used << "\033[0m" << std::endl;

  // Step5: do postprocess with output data
  // std::vector<Classification> top_k_cls;
  {
    // make sure CPU read data from DDR before using output tensor data
    for (int i = 0; i < output_count; i++) {
      hbSysFlushMem(&output_tensors[i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    }
    std::cout << "----------output_count-----------" << output_count << "\n";

    int *shape = output->properties.validShape.dimensionSize;
    int tensor_len = shape[0] * shape[1] * shape[2] * shape[3];
    std::cout << "---tensor_len---0---  " << tensor_len << " " << shape[0] << " * " << shape[1] << " *  " << shape[2] << " *  " << shape[3] << "\n";

    int *shape_1 = output_tensors[1].properties.validShape.dimensionSize;
    int tensor_len_1 = shape_1[0] * shape_1[1] * shape_1[2] * shape_1[3];
    std::cout << "---tensor_len---1---  " << tensor_len_1 << " " << shape_1[0] << " * " << shape_1[1] << " *  " << shape_1[2] << " *  " << shape_1[3] << "\n";

    auto semi = reinterpret_cast<int8_t *>(output_tensors[0].sysMem[0].virAddr);
    auto desc = reinterpret_cast<int8_t *>(output_tensors[1].sysMem[0].virAddr);

    auto t3_0 = std::chrono::steady_clock::now();

    int res_count_high = 0;
    int res_count_low = 0;
    std::vector<KeyPoint> keypoints_all;
    std::vector<KeyPoint> keypoints_all_high;
    std::vector<KeyPoint> keypoints_all_low;

    for (int i = 0; i < tensor_len; i++) {

      float scale = 0.00025672250194475055;
      float tmp_dequant = static_cast<float>(semi[i]) * scale;

      if (tmp_dequant >= point_th_high) {
        res_count_high++;
        int kp_x = i % 640;
        int kp_y = i / 640;

        KeyPoint cur_kp;
        cur_kp.x = kp_x;
        cur_kp.y = kp_y;
        cur_kp.score = tmp_dequant;
        keypoints_all_high.emplace_back(cur_kp);
      }

      if (tmp_dequant >= point_th_low) {
        res_count_low++;
        int kp_x = i % 640;
        int kp_y = i / 640;

        KeyPoint cur_kp;
        cur_kp.x = kp_x;
        cur_kp.y = kp_y;
        cur_kp.score = tmp_dequant;
        keypoints_all_low.emplace_back(cur_kp);
      }
    }

    // 1500
    if (keypoints_all_high.size() >= 1600) {
      keypoints_all = keypoints_all_high;
      std::cout << "------using point_th_high------------\n";
    } else {
      keypoints_all = keypoints_all_low;
      std::cout << "------using point_th_low------------\n";
    }

    std::cout << "---------------------------count------before-------nms----------" << keypoints_all.size();
    std::cout << std::endl;

    float threshold = 0.0;

    std::vector<KeyPoint> nmsKeypoints_kp = applyNMS_grid_new(keypoints_all, threshold, windowSize);

    std::vector<KeyPoint> sortedKeypoints = nmsKeypoints_kp;
    std::sort(sortedKeypoints.begin(), sortedKeypoints.end(), [](const KeyPoint &a, const KeyPoint &b) { return a.score > b.score; });

    std::cout << "---------------------------count------after-------nms----------" << sortedKeypoints.size();
    std::cout << std::endl;

    auto t3 = std::chrono::steady_clock::now();
    auto time_used_2 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t3_0).count() * 1000;
    std::cout << "\033[31m" << "dfeat postprocess nms Time:" << time_used_2 << "\033[0m" << std::endl;

    std::vector<cv::Point2f> nmsKeypoints;
    for (int i = 0; i < sortedKeypoints.size(); i++) {
      KeyPoint kp = sortedKeypoints[i];
      nmsKeypoints.push_back(cv::Point2f(kp.x * 1.0, kp.y * 1.0));
    }

    float scale_desc = 0.04756002873182297;

    int desc_dim = 256;
    int kp_nums = nmsKeypoints.size();
    if (kp_nums < 256)
      kp_nums = 256;
    Eigen::MatrixXf matMatrixXd(kp_nums, desc_dim);
    // Eigen::MatrixXf matMatrixXd(256, desc_dim);

    for (int i = 0; i < nmsKeypoints.size(); i++) {
      int cur_x = nmsKeypoints[i].x;
      int cur_y = nmsKeypoints[i].y;

      Eigen::VectorXf cur_desc = Eigen::VectorXf::Zero(desc_dim);

      for (int j = 0; j < desc_dim; j++) {
        cur_desc[j] = static_cast<float>(desc[(cur_y * 640 + cur_x) * desc_dim + j]) * scale_desc;
      }

      float norm_sqrt = cur_desc.norm();

      Eigen::VectorXf cur_desc_normalized = cur_desc / norm_sqrt;

      matMatrixXd.row(i) = cur_desc_normalized;
    }

    if (nmsKeypoints.size() < 256) {
      for (int k = nmsKeypoints.size(); k < 256; k++) {
        nmsKeypoints.push_back(nmsKeypoints[0]);
        // matMatrixXd.row(k) = matMatrixXd.row(0);
        Eigen::VectorXf tmp_desc = Eigen::VectorXf::Zero(desc_dim);
        // matMatrixXd.row(k) = tmp_desc;
        matMatrixXd.row(k) = matMatrixXd.row(0);
      }
    }

    auto t4 = std::chrono::steady_clock::now();
    auto time_used_3 = std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3).count() * 1000;
    std::cout << "\033[31m" << "dfeat postprocess desc Time:" << time_used_3 << "\033[0m" << std::endl;

    Eigen::MatrixXd desc_double(matMatrixXd.cast<double>());

    extractor_result.first = nmsKeypoints;
    extractor_result.second = desc_double;
  }

  // Step6: release resources
  {
    // release task handle
    HB_CHECK_SUCCESS(hbDNNReleaseTask(task_handle), "hbDNNReleaseTask failed");
    // free input mem
    for (int i = 0; i < input_count; i++) {
      HB_CHECK_SUCCESS(hbSysFreeMem(&(input_tensors[i].sysMem[0])), "hbSysFreeMem failed");
    }
    // free output mem
    for (int i = 0; i < output_count; i++) {
      HB_CHECK_SUCCESS(hbSysFreeMem(&(output_tensors[i].sysMem[0])), "hbSysFreeMem failed");
    }
    // release model
    // HB_CHECK_SUCCESS(hbDNNRelease(packed_dnn_handle_sp), "hbDNNRelease failed");
  }

  // return results;
  return 0;
}

std::vector<cv::Point2f> NormalizeKeypoints(std::vector<cv::Point2f> kpts, int h, int w) {
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

void saveFloatPtrToBin(const std::string &filename, float *data, size_t length) {
  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    return;
  }
  out.write(reinterpret_cast<const char *>(data), length * sizeof(float));
  out.close();
}

// log_softmax over rows (axis=1)
Eigen::MatrixXd log_softmax_rows(Eigen::MatrixXd &x) {
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
Eigen::MatrixXd log_softmax_cols(Eigen::MatrixXd &x) {
  Eigen::MatrixXd result(x.rows(), x.cols());
  for (int j = 0; j < x.cols(); ++j) {
    double maxVal = x.col(j).maxCoeff();
    Eigen::VectorXd shifted = x.col(j).array() - maxVal;
    double logsum = std::log((shifted.array().exp()).sum());
    result.col(j) = shifted.array() - logsum;
  }
  return result;
}

inline double logsigmoid(double x) {
  return -std::log1p(std::exp(-x)); // log(sigmoid(x))
}

Eigen::VectorXd log_softmax(Eigen::VectorXd &x) {
  double max_val = x.maxCoeff();
  Eigen::VectorXd shifted = x.array() - max_val;
  double log_sum_exp = std::log(shifted.array().exp().sum());
  return shifted.array() - log_sum_exp;
}

Eigen::MatrixXd sigmoid_log_double_softmax(Eigen::MatrixXd &sim, Eigen::VectorXd &z0, Eigen::VectorXd &z1) {
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

void filter_matches(Eigen::MatrixXd &scores,                   // log assignment matrix, shape [M, N]
                    std::vector<std::pair<int, int>> &matches, // output: matched pairs (i, j)
                    std::vector<double> &mscores               // output: matching scores (exp of log score)
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

int infer_x5_lg(std::vector<cv::Point2f> &keypoint_1, Eigen::MatrixXd &desc_1, std::vector<cv::Point2f> &keypoint_2, Eigen::MatrixXd &desc_2, vector<cv::Point2f> &match_kp_1,
                vector<cv::Point2f> &match_kp_2) {

  std::cout << "------infer_x5----lightglue--\n";

  if (flag_init_lg == false) {

    flag_init_lg = true;

    std::string model_name = "./lg_v2.bin";

    auto modelFileName = model_name.c_str();
    int model_count = 0;
    // Step1: get model handle
    {
      HB_CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle_lg, &modelFileName, 1), "hbDNNInitializeFromFiles failed");
      HB_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list_lg, &model_count, packed_dnn_handle_lg), "hbDNNGetModelNameList failed");
      HB_CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle_lg, packed_dnn_handle_lg, model_name_list_lg[0]), "hbDNNGetModelHandle failed");
    }
    // Show how to get dnn version
    std::cout << "DNN runtime version: " << hbDNNGetVersion() << std::endl;
  }

  std::vector<hbDNNTensor> input_tensors;
  std::vector<hbDNNTensor> output_tensors; //
  int input_count = 0;
  int output_count = 0;

  // Step2: prepare input and output tensor
  {
    HB_CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle_lg), "hbDNNGetInputCount failed");
    HB_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, dnn_handle_lg), "hbDNNGetOutputCount failed");
    input_tensors.resize(input_count);
    output_tensors.resize(output_count);
    prepare_tensor(input_tensors.data(), output_tensors.data(), dnn_handle_lg);
  }

  // Step3: set input data to input tensor
  {
    int img_height = 640;
    int img_width = 640;

    auto kpts1 = NormalizeKeypoints(keypoint_1, img_height, img_width);
    auto kpts2 = NormalizeKeypoints(keypoint_2, img_height, img_width);

    Eigen::MatrixXf desc_1_float = desc_1.cast<float>();
    Eigen::MatrixXf desc_2_float = desc_2.cast<float>();

    Eigen::MatrixXf desc_1_float_trans = desc_1_float.transpose();
    Eigen::MatrixXf desc_2_float_trans = desc_2_float.transpose();

    float *desc1 = desc_1_float_trans.data();
    float *desc2 = desc_2_float_trans.data();

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

    hbDNNTensor &input = input_tensors[0];
    hbDNNTensorProperties &Properties = input.properties;

    int *kp1_shape = input_tensors[0].properties.validShape.dimensionSize;
    int kp1_tensor_len = kp1_shape[0] * kp1_shape[1] * kp1_shape[2] * kp1_shape[3];
    std::cout << "--kp1 --tensor_len------  " << kp1_tensor_len << " " << kp1_shape[0] << " * " << kp1_shape[1] << " *  " << kp1_shape[2] << " *  " << kp1_shape[3] << "\n";

    int *kp2_shape = input_tensors[1].properties.validShape.dimensionSize;
    int kp2_tensor_len = kp2_shape[0] * kp2_shape[1] * kp2_shape[2] * kp2_shape[3];
    std::cout << "--kp2 --tensor_len------  " << kp2_tensor_len << " " << kp2_shape[0] << " * " << kp2_shape[1] << " *  " << kp2_shape[2] << " *  " << kp2_shape[3] << "\n";

    int *desc1_shape = input_tensors[2].properties.validShape.dimensionSize;
    int desc1_tensor_len = desc1_shape[0] * desc1_shape[1] * desc1_shape[2] * desc1_shape[3];
    std::cout << "--desc1 --tensor_len------  " << desc1_tensor_len << " " << desc1_shape[0] << " * " << desc1_shape[1] << " *  " << desc1_shape[2] << " *  " << desc1_shape[3] << "\n";

    int *desc2_shape = input_tensors[3].properties.validShape.dimensionSize;
    int desc2_tensor_len = desc2_shape[0] * desc2_shape[1] * desc2_shape[2] * desc2_shape[3];
    std::cout << "--desc2 --tensor_len------  " << desc2_tensor_len << " " << desc2_shape[0] << " * " << desc2_shape[1] << " *  " << desc2_shape[2] << " *  " << desc2_shape[3] << "\n";

    int input_h = Properties.validShape.dimensionSize[2];
    int input_w = Properties.validShape.dimensionSize[3];

    auto data_kp1 = input_tensors[0].sysMem[0].virAddr;
    memcpy(data_kp1, kpts1_data, kp1_tensor_len * 4);

    auto data_kp2 = input_tensors[1].sysMem[0].virAddr;
    memcpy(data_kp2, kpts2_data, kp2_tensor_len * 4);

    auto data_desc1 = input_tensors[2].sysMem[0].virAddr;
    memcpy(data_desc1, desc1, desc1_tensor_len * 4);

    auto data_desc2 = input_tensors[3].sysMem[0].virAddr;
    memcpy(data_desc2, desc2, desc2_tensor_len * 4);
  }

  hbDNNTaskHandle_t task_handle = nullptr;
  hbDNNTensor *output = output_tensors.data();

  auto t1 = std::chrono::steady_clock::now();

  // Step4: run inference
  {
    // make sure memory data is flushed to DDR before inference
    for (int i = 0; i < input_count; i++) {
      hbSysFlushMem(&input_tensors[i].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    }

    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
    HB_CHECK_SUCCESS(hbDNNInfer(&task_handle, &output, input_tensors.data(), dnn_handle_lg, &infer_ctrl_param), "hbDNNInfer failed");
    // wait task done
    HB_CHECK_SUCCESS(hbDNNWaitTaskDone(task_handle, 0), "hbDNNWaitTaskDone failed");
  }

  auto t2 = std::chrono::steady_clock::now();
  auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;
  std::cout << "\033[31m" << "lightglue infer Time:" << time_used << "\033[0m" << std::endl;

  // Step5: do postprocess with output data
  // std::vector<Classification> top_k_cls;
  {
    // make sure CPU read data from DDR before using output tensor data
    for (int i = 0; i < output_count; i++) {
      hbSysFlushMem(&output_tensors[i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    }

    int *sim_Shape = output->properties.validShape.dimensionSize;
    int tensor_len = sim_Shape[0] * sim_Shape[1] * sim_Shape[2] * sim_Shape[3];
    std::cout << "--lg-tensor_len---0---  " << tensor_len << " " << sim_Shape[0] << " * " << sim_Shape[1] << " *  " << sim_Shape[2] << " *  " << sim_Shape[3] << "\n";

    int *z0_Shape = output_tensors[1].properties.validShape.dimensionSize;
    int tensor_len_1 = z0_Shape[0] * z0_Shape[1] * z0_Shape[2] * z0_Shape[3];
    std::cout << "--lg-tensor_len---1---  " << tensor_len_1 << " " << z0_Shape[0] << " * " << z0_Shape[1] << " *  " << z0_Shape[2] << " *  " << z0_Shape[3] << "\n";

    int *z1_Shape = output_tensors[2].properties.validShape.dimensionSize;
    int tensor_len_2 = z1_Shape[0] * z1_Shape[1] * z1_Shape[2] * z1_Shape[3];
    std::cout << "--lg-tensor_len---1---  " << tensor_len_2 << " " << z1_Shape[0] << " * " << z1_Shape[1] << " *  " << z1_Shape[2] << " *  " << z1_Shape[3] << "\n";

    auto sim = reinterpret_cast<float *>(output_tensors[0].sysMem[0].virAddr);
    auto z0 = reinterpret_cast<float *>(output_tensors[1].sysMem[0].virAddr);
    auto z1 = reinterpret_cast<float *>(output_tensors[2].sysMem[0].virAddr);

    auto t3_0 = std::chrono::steady_clock::now();

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> simEigenMatrix(sim, sim_Shape[1], sim_Shape[2]);

    Eigen::MatrixXd simEigenMatrixXd(sim_Shape[1], sim_Shape[2]);

    simEigenMatrixXd = simEigenMatrix.cast<double>();

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> z0EigenMatrix(z0, z0_Shape[1], z0_Shape[2]);

    Eigen::MatrixXd z0EigenMatrixXd(z0_Shape[1], z0_Shape[2]);

    z0EigenMatrixXd = z0EigenMatrix.cast<double>();

    Eigen::VectorXd z0_vec = Eigen::Map<const Eigen::VectorXd>(z0EigenMatrixXd.data(), 256);

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> z1EigenMatrix(z1, z1_Shape[1], z1_Shape[2]);

    Eigen::MatrixXd z1EigenMatrixXd(z1_Shape[1], z1_Shape[2]);

    z1EigenMatrixXd = z1EigenMatrix.cast<double>();

    Eigen::VectorXd z1_vec = Eigen::Map<const Eigen::VectorXd>(z1EigenMatrixXd.data(), 256);

    Eigen::MatrixXd scores_tmp = sigmoid_log_double_softmax(simEigenMatrixXd, z0_vec, z1_vec);

    std::vector<std::pair<int, int>> matches_vec;
    std::vector<double> mscores_vec;

    filter_matches(scores_tmp, matches_vec, mscores_vec);

    int nums_match = 0;

    for (int i = 0; i < mscores_vec.size(); i++) {
      std::cout << mscores_vec[i] << " ";
      if (mscores_vec[i] > 0.08) {
        nums_match++;
        match_kp_1.push_back(keypoint_1[matches_vec[i].first]);
        match_kp_2.push_back(keypoint_2[matches_vec[i].second]);
      }
    }

    std::cout << std::endl;
    std::cout << "--------------nums match--------------" << nums_match << std::endl;

    auto t3 = std::chrono::steady_clock::now();
    auto time_used_2 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t3_0).count() * 1000;
    std::cout << "\033[31m" << "lightglue post process Time:" << time_used_2 << "\033[0m" << std::endl;
  }

  // Step6: release resources
  {
    // release task handle
    HB_CHECK_SUCCESS(hbDNNReleaseTask(task_handle), "hbDNNReleaseTask failed");
    // free input mem
    for (int i = 0; i < input_count; i++) {
      HB_CHECK_SUCCESS(hbSysFreeMem(&(input_tensors[i].sysMem[0])), "hbSysFreeMem failed");
    }
    // free output mem
    for (int i = 0; i < output_count; i++) {
      HB_CHECK_SUCCESS(hbSysFreeMem(&(output_tensors[i].sysMem[0])), "hbSysFreeMem failed");
    }
    // release model
    // HB_CHECK_SUCCESS(hbDNNRelease(packed_dnn_handle_mix), "hbDNNRelease failed");
  }

  // return results;
  return 0;
}

int main() {

  std::string img_folder_name = "./image_test/";
  std::string img_match_vis_name = "./image_vis/";

  std::vector<Eigen::VectorXf> desc_vec;

  while (true) {
    for (int i = 0; i < 57; i++) {
      std::string img_name_1 = img_folder_name + std::to_string(i) + ".png";

      cv::Mat bgr_mat_1 = cv::imread(img_name_1, cv::IMREAD_COLOR);

      cv::Mat bgr_mat_1_resize;
      cv::resize(bgr_mat_1, bgr_mat_1_resize, cv::Size(640, 640));

      std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> dfeat_result_1;
      infer_x5_dfeat(bgr_mat_1_resize, dfeat_result_1);

      vector<cv::Point2f> match_pt_1, match_pt_2;
      infer_x5_lg(dfeat_result_1.first, dfeat_result_1.second, dfeat_result_1.first, dfeat_result_1.second, match_pt_1, match_pt_2);
    }
  }
}
