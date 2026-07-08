# TSC-CoLight

Đồ án xây dựng AI Agent điều khiển tín hiệu đèn giao thông liên vùng trên mô phỏng SUMO. Mục tiêu là tối ưu quyết định pha đèn tại nhiều nút giao cùng lúc, giảm ùn tắc và cải thiện luồng xe thông qua học tăng cường sâu.

Hướng tiếp cận chính của dự án là kết hợp DQN với GAT theo tinh thần CoLight:

- DQN (Deep Q-Network) học hàm giá trị Q cho từng đèn giao thông, từ đó chọn hành động điều khiển pha đèn.
- GAT (Graph Attention Network) biểu diễn mạng lưới giao thông như một đồ thị, trong đó mỗi nút là một cụm đèn giao thông và cạnh thể hiện quan hệ lân cận giữa các nút.
- Attention giúp mỗi nút giao học mức độ ảnh hưởng của các nút lân cận, phù hợp với bài toán điều khiển tín hiệu giao thông liên vùng.
- Trạng thái đầu vào gồm thông tin occupancy/hàng đợi và pha đèn hiện tại; đầu ra là hành động điều khiển đèn, ví dụ `next_or_not` để quyết định giữ pha hiện tại hoặc chuyển sang pha tiếp theo.

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

Script huấn luyện:

```bash
python run_fed_dqn.py
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


