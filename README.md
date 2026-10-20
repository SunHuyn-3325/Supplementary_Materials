# Tài liệu bổ sung: RFM-D và khung ICLM

Tài liệu này đi kèm bài báo "RFM-D: Extending RFM with Seasonal, Payment and Geographic Dimensions for Customer Segmentation and CLV–Churn Prediction in Vietnamese E-commerce".

## Cấu trúc

- `code/`: notebook pipeline và các script phân tích
  - `v4_iclm_pipeline.ipynb`: sinh dữ liệu mô phỏng, phân cụm, dự báo CLV và churn, chiến lược LLM
  - `h1_rfmd_vs_rfm3.py`: so sánh Silhouette giữa RFM-D và RFM ba biến (H1)
  - `h2_h5_regression.py`: hồi quy OLS (CLV) và logit (churn), kèm VIF (H2–H5)
  - `h2_h5_split.py`: kiểm tra độ ổn định của hệ số trên tập huấn luyện 80%
  - `evaluate_models.py`: đánh giá mô hình CLV (R², MAE, RMSE) và churn (AUC-ROC, F1, PR-AUC, MCC) theo Mục 3.7 và Bảng 5
- `params/`: tham số calibrate và nhật ký kiểm chứng
  - `vn_calibration_en.json`: toàn bộ tham số dùng để sinh dữ liệu mô phỏng (có chú thích STT khớp với Phụ lục A)
  - `calibration_log.json`: kết quả kiểm chứng calibrate (11/11 kiểm tra)
- `data/`
  - `model_dataset.csv`: 36.144 khách hàng có điểm dự báo, 8 biến RFM-D, clv_6m và churn
  - `customers_segmented.csv`: 50.000 khách hàng với nhóm phân khúc (`segment`) và 8 biến RFM-D, dùng cho so sánh H1. Đã bỏ các cột thông tin cá nhân (tên, email, số điện thoại).

## Môi trường

Đã kiểm tra với Python 3.12. Ghi lại phiên bản thực tế trên máy bạn bằng `pip show <tên gói>`.

- pandas
- numpy
- statsmodels
- scikit-learn
- xgboost (trên macOS cần `brew install libomp`)
- imbalanced-learn

Cài đặt:

```bash
pip install pandas numpy statsmodels scikit-learn xgboost imbalanced-learn
```

## Thứ tự chạy

1. Mở `code/v4_iclm_pipeline.ipynb` trong Jupyter và chạy từ đầu. Notebook đọc `params/vn_calibration_en.json`, nên cần đặt file này cùng thư mục với notebook hoặc sửa đường dẫn trong cell đầu tiên.
2. Chạy hồi quy với dữ liệu đầu vào:

   ```bash
   python code/h2_h5_regression.py data/model_dataset.csv
   python code/h2_h5_split.py data/model_dataset.csv
   ```

3. Chạy đánh giá mô hình dự báo (kết quả mong đợi: R² 0,326; MAE 3.390.886; RMSE 5.318.189; AUC 0,750; F1 0,647; PR-AUC 0,681; MCC 0,368):

   ```bash
   python code/evaluate_models.py data/model_dataset.csv
   ```

4. Chạy so sánh Silhouette (kết quả mong đợi: RFM-D k = 2, Silhouette 0,314; RFM ba biến k = 2, Silhouette 0,512):

   ```bash
   python code/h1_rfmd_vs_rfm3.py data/customers_segmented.csv
   ```

## Ghi chú

- Dữ liệu giao dịch là dữ liệu mô phỏng, sinh với hạt giống ngẫu nhiên 42 (`SEED` trong `vn_calibration_en.json`).
- Bộ dữ liệu tham chiếu Open E-commerce 1.0 (Berke et al., 2024) không được đính kèm. Tải tại Harvard Dataverse, DOI 10.7910/DVN/YGLYDY.
- Tần suất mua được sinh từ phân phối gamma với tham số trong khối `purchase_frequency` của `vn_calibration_en.json`. Đây là giả định, không có số liệu Việt Nam.
