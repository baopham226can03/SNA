"""
Script tạo Alpha158 factors từ dữ liệu Qlib
Alpha158 là tập 158 yếu tố kỹ thuật được tính từ OHLCV
"""
import qlib
from qlib.data import D
from qlib.config import REG_US
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
import pandas as pd
from pathlib import Path

# Khởi tạo Qlib
qlib_dir = "./qlib_data/us_data"
qlib.init(provider_uri=qlib_dir, region=REG_US)
print(f"Đã khởi tạo Qlib với dữ liệu tại: {qlib_dir}")

# Định nghĩa Alpha158 - 158 factors
# Qlib có sẵn class Alpha158 trong qlib.contrib.data.handler
try:
    from qlib.contrib.data.handler import Alpha158
    print("Import Alpha158 thành công!")
    
    # Cấu hình Alpha158
    # Tham số:
    # - instruments: danh sách ticker (hoặc 'all' cho tất cả)
    # - start_time: thời gian bắt đầu
    # - end_time: thời gian kết thúc
    # - infer_processors: các processor để xử lý dữ liệu
    
    MARKET = "all"  # Tất cả ticker trong database
    TRAIN_START = "2018-01-01"
    TRAIN_END = "2022-12-31"
    TEST_START = "2023-01-01"
    TEST_END = "2023-12-31"
    
    print("\nCấu hình:")
    print(f"  Market: {MARKET}")
    print(f"  Training: {TRAIN_START} đến {TRAIN_END}")
    print(f"  Testing: {TEST_START} đến {TEST_END}")
    
    # Tạo Alpha158 handler
    print("\nĐang tạo Alpha158 factors...")
    handler = Alpha158(
        instruments=MARKET,
        start_time=TRAIN_START,
        end_time=TEST_END,
        infer_processors=[
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        learn_processors=[
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ],
        label=["Ref($close, -1) / $close - 1"],  # Return ngày tiếp theo
    )
    
    # Lấy dữ liệu đã xử lý
    print("\nĐang fetch dữ liệu Alpha158...")
    handler.setup_data()
    
    # Lấy features và labels
    features = handler.fetch(col_set="feature")
    labels = handler.fetch(col_set="label")
    
    print(f"\nKích thước features (Alpha158): {features.shape}")
    print(f"Kích thước labels: {labels.shape}")
    print(f"\nDanh sách một số factors:")
    print(features.columns.tolist()[:20])
    
    # Lưu Alpha158 factors
    output_dir = Path("data/alpha158")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_file = output_dir / "alpha158_features.csv"
    labels_file = output_dir / "alpha158_labels.csv"
    
    features.to_csv(features_file)
    labels.to_csv(labels_file)
    
    print(f"\nĐã lưu Alpha158 factors vào:")
    print(f"  Features: {features_file}")
    print(f"  Labels: {labels_file}")
    
    # Thống kê
    print("\n" + "="*70)
    print("THỐNG KÊ:")
    print("="*70)
    print(f"Tổng số mẫu: {len(features)}")
    print(f"Số lượng factors: {len(features.columns)}")
    print(f"Khoảng thời gian: {features.index.get_level_values('datetime').min()} đến {features.index.get_level_values('datetime').max()}")
    print(f"Số lượng instruments: {features.index.get_level_values('instrument').nunique()}")
    
except ImportError as e:
    print(f"Lỗi import Alpha158: {e}")
    print("\nQlib có thể chưa cài đủ dependencies.")
    print("Thử cài đặt đầy đủ: pip install pyqlib[all]")
    print("\nHoặc dùng cách thủ công tính Alpha158:")
    print("Xem file: d_alpha158_manual.py")

except Exception as e:
    print(f"Lỗi khi tạo Alpha158: {e}")
    print("\nCó thể do:")
    print("1. Chưa có dữ liệu trong qlib_data/us_data")
    print("2. Dữ liệu chưa đúng định dạng Qlib")
    print("\nVui lòng chạy lại các bước:")
    print("  python b_convert_to_csv.py")
    print("  python c_convert_to_qlib.py")
