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

#include "log_macros.h"
#include "DFeatInfer.h"
#include "LightGlueInfer.h"

int main(int argc, char **argv) {
  std::string dfeat_model_path = "./model/dfeat_640_640.bin";
  std::string lightglue_model_path = "./model/lg_v2.bin";
  if (argc > 1) dfeat_model_path = argv[1];
  if (argc > 2) lightglue_model_path = argv[2];

  rclcpp::Logger logger;
  std::shared_ptr<DFeatInfer> dfeat_infer = std::make_shared<DFeatInfer>(logger);
  std::shared_ptr<LightGlueInfer> lightglue_infer = std::make_shared<LightGlueInfer>(logger);
  int ret_code = 0;
  ret_code = dfeat_infer->init(dfeat_model_path);
  if (ret_code != 0) {
    LOG_ERROR(nullptr, "=> dfeat_infer init failed");
    return -1;
  }
  ret_code = lightglue_infer->init(lightglue_model_path);
  if (ret_code != 0) {
    LOG_ERROR(nullptr, "=> lightglue_infer init failed");
    return -1;
  }

  std::string input_img_dir = "./image_test/";
  std::string output_img_dir = "./image_vis/";

  for (int i = 0; i < 57; i++) {
    LOG_INFO(nullptr, "=> ================================================================================");
    std::string img_name_1 = input_img_dir + std::to_string(i) + ".png";
    std::string img_name_2 = input_img_dir + std::to_string(i + 1) + ".png";
    LOG_INFO(nullptr, "=> img_name_1: " << img_name_1 << ", img_name_2: " << img_name_2);
    cv::Mat bgr_mat_1 = cv::imread(img_name_1, cv::IMREAD_COLOR);
    cv::Mat bgr_mat_2 = cv::imread(img_name_2, cv::IMREAD_COLOR);

    cv::Mat bgr_mat_1_resize, bgr_mat_2_resize;
    int w, h;
    dfeat_infer->get_model_input_size(w, h);
    LOG_INFO(nullptr, "=> img size: [" << bgr_mat_1.cols << ", " << bgr_mat_1.rows << "], need resize to [" << w << ", "
                                       << h << "]");
    cv::resize(bgr_mat_1, bgr_mat_1_resize, cv::Size(w, h));
    cv::resize(bgr_mat_2, bgr_mat_2_resize, cv::Size(w, h));

    LOG_INFO(nullptr, "=> -------- DFeat infer 1 --------");
    std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> dfeat_result_1;
    cv::Mat gray_mat_1;
    cv::cvtColor(bgr_mat_1_resize, gray_mat_1, cv::COLOR_BGR2GRAY);
    {
      ScopeProcessTime t(logger, "dfeat forward", "warn");
      dfeat_infer->forward(gray_mat_1, dfeat_result_1);
    }
    LOG_INFO(nullptr, "=> -------- DFeat infer 2 --------");
    std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> dfeat_result_2;
    cv::Mat gray_mat_2;
    cv::cvtColor(bgr_mat_2_resize, gray_mat_2, cv::COLOR_BGR2GRAY);
    {
      ScopeProcessTime t(logger, "dfeat forward", "warn");
      dfeat_infer->forward(gray_mat_2, dfeat_result_2);
    }

    LOG_INFO(nullptr, "=> -------- LightGlue infer --------");
    std::vector<cv::Point2f> match_kp_1, match_kp_2;
    {
      ScopeProcessTime t(logger, "lightglue forward", "warn");
      lightglue_infer->forward(dfeat_result_1, dfeat_result_2, h, w, match_kp_1, match_kp_2);
    }

    LOG_INFO(nullptr, "=> -------- Draw match --------");
    cv::Mat imgHoriz;
    cv::hconcat(bgr_mat_1_resize, bgr_mat_2_resize, imgHoriz);
    for (int j = 0; j < match_kp_1.size(); j++) {
      cv::Point point_img1;
      cv::Point point_img2;

      point_img1.x = (int)match_kp_1[j].x;
      point_img1.y = (int)match_kp_1[j].y;

      point_img2.x = (int)match_kp_2[j].x + 640;
      point_img2.y = (int)match_kp_2[j].y;

      cv::circle(imgHoriz, point_img1, 2, cv::Scalar(255, 0, 0), -1);
      cv::circle(imgHoriz, point_img2, 2, cv::Scalar(255, 0, 0), -1);
      cv::line(imgHoriz, point_img1, point_img2, cv::Scalar(0, 0, 255), 1);

      std::string match_count_text = "Matches: " + std::to_string(match_kp_1.size());
      int font_face = cv::FONT_HERSHEY_SIMPLEX;
      double font_scale = 0.7;
      int thickness = 2;

      int baseline = 0;
      cv::Size text_size = cv::getTextSize(match_count_text, font_face, font_scale, thickness, &baseline);

      cv::Point text_org(imgHoriz.cols - text_size.width - 10, text_size.height + 10);

      cv::putText(imgHoriz, match_count_text, text_org, font_face, font_scale, cv::Scalar(0, 255, 0), thickness);
    }

    std::string img_match_vis_path =
        output_img_dir + "match__" + std::to_string(i) + "__" + std::to_string(i + 1) + ".png";
    LOG_INFO(nullptr, "=> save img: " << img_match_vis_path);
    cv::imwrite(img_match_vis_path, imgHoriz);
  }

  return 0;
}