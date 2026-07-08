# TSC-CoLight

Đồ án xây dựng AI Agent điều khiển tín hiệu đèn giao thông liên vùng trên mô phỏng SUMO. Mục tiêu là tối ưu quyết định pha đèn tại nhiều nút giao cùng lúc, giảm ùn tắc và cải thiện luồng xe thông qua học tăng cường sâu.

Hướng tiếp cận chính của dự án là kết hợp DQN với GAT theo tinh thần CoLight:

- DQN (Deep Q-Network) học hàm giá trị Q cho từng đèn giao thông, từ đó chọn hành động điều khiển pha đèn.
- GAT (Graph Attention Network) biểu diễn mạng lưới giao thông như một đồ thị, trong đó mỗi nút là một cụm đèn giao thông và cạnh thể hiện quan hệ lân cận giữa các nút.
- Attention giúp mỗi nút giao học mức độ ảnh hưởng của các nút lân cận, phù hợp với bài toán điều khiển tín hiệu giao thông liên vùng.
- Trạng thái đầu vào gồm thông tin occupancy/hàng đợi và pha đèn hiện tại; đầu ra là hành động điều khiển đèn, ví dụ `next_or_not` để quyết định giữ pha hiện tại hoặc chuyển sang pha tiếp theo.

> Phạm vi README này tập trung vào mô hình DQN + GAT. Các thành phần Federated Learning trong repo không được trình bày như hướng chính của đồ án.

## Cấu Trúc Dự Án

```text
TSC-CoLight/
├── Env/                  # Môi trường SUMO và wrapper cho bài toán TSC
├── Models/               # Mạng GAT + DQN, script huấn luyện tập trung
├── FedDQN/               # Một số module DQN/CoLight tái sử dụng
├── Scenario/             # Các kịch bản SUMO: 1node, 3nodes, 4nodes, map, test
├── TransSimHub/          # Thư viện hỗ trợ mô phỏng và phân tích SUMO
├── infer.py              # Chạy mô phỏng bằng mô hình đã huấn luyện
├── logs/                 # Log huấn luyện và script vẽ reward
└── result*.png           # Một số hình kết quả mẫu
```

## Yêu Cầu Môi Trường

Khuyến nghị dùng Linux/Ubuntu hoặc WSL2 vì SUMO và PyTorch Geometric dễ thiết lập hơn.

Phần mềm cần có:

- Python 3.10 hoặc 3.11
- SUMO và biến môi trường `SUMO_HOME`
- PyTorch
- PyTorch Geometric, gồm `torch_geometric` và `torch_scatter`
- Các gói Python: `numpy`, `traci`, `sumolib`, `libsumo`, `scipy`, `loguru`, `matplotlib`

Cài SUMO trên Ubuntu:

```bash
sudo apt update
sudo apt install sumo sumo-tools sumo-doc
```

Thiết lập `SUMO_HOME`:

```bash
export SUMO_HOME=/usr/share/sumo
```

Có thể thêm dòng trên vào `~/.bashrc` nếu muốn dùng lâu dài.

Tạo môi trường Python:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Cài các thư viện chính:

```bash
pip install torch numpy scipy matplotlib loguru traci sumolib libsumo
```

Cài PyTorch Geometric theo phiên bản PyTorch đang dùng. Ví dụ với Torch CPU:

```bash
pip install torch_geometric
pip install torch_scatter -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cpu.html
```

Nếu dùng CUDA, thay `+cpu` bằng phiên bản CUDA phù hợp theo hướng dẫn của PyTorch Geometric.

## Dữ Liệu Và Kịch Bản Mô Phỏng

Các kịch bản SUMO nằm trong thư mục `Scenario/`, gồm:

- `Scenario/1node`: mô phỏng một nút giao.
- `Scenario/3nodes`: mô phỏng liên vùng 3 nút giao.
- `Scenario/4nodes`: mô phỏng liên vùng 4 nút giao.
- `Scenario/test`: kịch bản thử nghiệm mặc định của script huấn luyện.
- `Scenario/map`: kịch bản từ bản đồ lớn hơn.

Mỗi kịch bản thường có:

- `env/*.net.xml`: file mạng đường.
- `env/vehicle.sumocfg`: file cấu hình SUMO.
- `routes/vehicle.rou.xml`: luồng phương tiện.
- `add/*.xml`: detector và cấu hình bổ sung.

## Cách Chạy Huấn Luyện

Script huấn luyện mặc định:

```bash
python Models/train.py
```

Mặc định script sử dụng:

- Kịch bản: `Scenario/test/env/vehicle.sumocfg`
- Mạng đường: `Scenario/test/env/test.net.xml`
- Hành động đèn: `next_or_not`
- Log: `logs/test/next_or_not/`
- Checkpoint: `Models/result/test/next_or_not/`

Sau khi train, mô hình được lưu theo dạng:

```text
Models/result/<scenario>/<action_type>/q_net_ep*.pt
Models/result/<scenario>/<action_type>/target_net_ep*.pt
Models/result/<scenario>/<action_type>/q_net_final.pt
Models/result/<scenario>/<action_type>/target_net_final.pt
```

Nếu muốn đổi kịch bản, có thể chỉnh các biến `env_name` và `tls_action_type` trong `Models/train.py`, hoặc gọi hàm `train(...)` với đường dẫn SUMO tương ứng.

Ví dụ cấu hình 3 nút giao trong Python:

```python
from Models.train import train

train(
    sumo_cfg="Scenario/3nodes/env/vehicle.sumocfg",
    net_file="Scenario/3nodes/env/3nodes.net.xml",
    log_path="logs/3nodes/next_or_not",
    save_dir="Models/result/3nodes/next_or_not",
    num_episodes=500,
    max_steps_per_ep=500,
)
```

## Cách Chạy Mô Phỏng Bằng Mô Hình Đã Train

Chạy mô phỏng headless với checkpoint đã có:

```bash
python infer.py \
  --model-path Models/result/3nodes/next_or_not/q_net_final.pt \
  --sumo-cfg Scenario/3nodes/env/vehicle.sumocfg \
  --net-file Scenario/3nodes/env/3nodes.net.xml \
  --tls-action-type next_or_not \
  --num-seconds 500 \
  --log-path deploy_3nodes.log.monitor.csv \
  --trip-info deploy_3nodes.tripinfo.xml \
  --no-gui
```

Chạy với giao diện SUMO:

```bash
python infer.py \
  --model-path Models/result/3nodes/next_or_not/q_net_final.pt \
  --sumo-cfg Scenario/3nodes/env/vehicle.sumocfg \
  --net-file Scenario/3nodes/env/3nodes.net.xml \
  --gui
```

Nếu checkpoint `q_net_best.pt` không tồn tại, `infer.py` có cơ chế thử dùng checkpoint round mới nhất trong cùng thư mục đối với một số đường dẫn mặc định.

## Kết Quả Đầu Ra

Khi chạy huấn luyện hoặc mô phỏng, dự án sinh ra các nhóm kết quả chính:

- File monitor CSV, ví dụ `logs/test/next_or_not/0.monitor.csv` hoặc `deploy_3nodes.log.monitor.csv`, dùng để theo dõi reward trong quá trình mô phỏng.
- File tripinfo XML, ví dụ `deploy_3nodes.tripinfo.xml`, chứa thông tin chuyến đi của xe như thời gian di chuyển, waiting time, route length.
- Checkpoint PyTorch `.pt`, ví dụ `q_net_final.pt`, là trọng số mô hình đã học.
- Hình tổng hợp reward, ví dụ `result.png`, `result_3nodes.png`, `result_4nodes.png`.

Để vẽ reward từ log monitor:

```bash
python logs/plot_reward.py
```

Một số file kết quả mẫu đã có trong repo:

```text
result.png
result_3nodes.png
result_3nodes_large.png
result_4nodes.png
deploy_3nodes.tripinfo.xml
deploy_4nodes.tripinfo.xml
```

## Ý Nghĩa Mô Hình

Trong mỗi bước mô phỏng, môi trường SUMO trả về trạng thái của từng đèn giao thông. Mô hình đóng gói trạng thái này thành vector đặc trưng cho từng nút giao, sau đó truyền qua:

1. `ObservationEncoder`: mã hóa occupancy và pha đèn hiện tại.
2. `GAT`: trao đổi thông tin giữa các nút giao lân cận bằng attention.
3. `Q head`: sinh giá trị Q cho từng hành động khả thi.
4. DQN agent: chọn hành động theo epsilon-greedy khi train và chọn Q lớn nhất khi infer.

Cách thiết kế này cho phép agent không chỉ phản ứng với trạng thái cục bộ tại một nút giao mà còn xét ảnh hưởng từ vùng lân cận.

## Gợi Ý Bổ Sung Cho Báo Cáo/README

Ngoài ba yêu cầu chính, nên bổ sung thêm:

- Sơ đồ kiến trúc tổng quan: SUMO → Env Wrapper → DQN + GAT → hành động điều khiển đèn.
- Mô tả state, action, reward để người đọc hiểu bài toán RL.
- Bảng so sánh kết quả giữa các kịch bản hoặc giữa trước/sau khi dùng AI Agent.
- Thông tin checkpoint nào là mô hình tốt nhất dùng để demo.
- Các lỗi thường gặp khi chạy SUMO, đặc biệt là thiếu `SUMO_HOME` hoặc thiếu `torch_scatter`.

## Lỗi Thường Gặp

Thiếu `SUMO_HOME`:

```text
Please declare environment variable 'SUMO_HOME'
```

Khắc phục:

```bash
export SUMO_HOME=/usr/share/sumo
```

Thiếu `torch_scatter` hoặc sai phiên bản PyTorch:

```text
ModuleNotFoundError: No module named 'torch_scatter'
```

Khắc phục bằng cách cài lại `torch_scatter` theo đúng phiên bản Torch/CUDA.

Không mở được SUMO GUI:

```bash
python infer.py --no-gui
```

hoặc kiểm tra môi trường hiển thị nếu chạy trên WSL/server.
