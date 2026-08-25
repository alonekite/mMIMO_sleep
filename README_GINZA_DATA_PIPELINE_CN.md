# Ginza normal/sleep 数据流水线交付说明

## 1. 这份交付解决什么问题

当前 `ginza_single_ue_cfr_normal_sleep.ipynb` 已经验证了单个 UE 的完整链路：

1. 在 Ginza 三维场景中加载 TX、UE 和混凝土材料。
2. 使用 Sionna RT `PathSolver` 求得传播路径。
3. 构造中心频点及 3276 个有效 OFDM 子载波上的 CFR。
4. 对全部 1024 个双极化码本波束进行中心频点和宽带扫描。
5. 在同一信道上计算 normal/sleep 接收功率和 SNR。
6. 从 normal-mode beamformed PDP 计算功率加权平均时延与 RMS 时延扩展。

本交付把上述已经验证的逻辑抽离成可重复运行的数据流水线，使多个 UE 位置可以逐一产生结构化样本，并支持 checkpoint、失败记录、断点续跑和最终主数据表合并。

**没有添加 MCS。** MCS 属于后续链路自适应或标签设计，不是当前 Sionna RT 仿真的直接输出。

## 2. 文件清单及目标位置

压缩包中的相对路径对应项目根目录：

| 文件 | 作用 |
| --- | --- |
| `src/mMIMO_sleep/data/__init__.py` | 数据包入口，不在导入时初始化 GPU。 |
| `src/mMIMO_sleep/data/schema.py` | 主数据表 schema、ML 候选输入、标签和 metadata 分组。 |
| `src/mMIMO_sleep/data/ginza_config.py` | 场景、天线、码本、OFDM、solver 和输出配置。 |
| `src/mMIMO_sleep/data/ginza_sample.py` | 单 UE 仿真、CFR、波束扫描、链路预算和 PDP 计算。 |
| `src/mMIMO_sleep/data/ginza_pipeline.py` | 批量处理、原子 checkpoint、resume 和结果合并。 |
| `scripts/generate_ginza_dataset.py` | 命令行入口。 |
| `configs/ginza_dataset_v1.yaml` | 已与当前 notebook 对齐的初始配置。 |
| `tests/test_dataset_schema.py` | Schema、ML 边界和配置 hash 测试。 |
| `tests/test_ginza_sample.py` | 路径、时延、LOS 启发式和链路预算测试。 |
| `tests/test_pipeline_resume.py` | 断点续跑、失败重试和配置隔离测试。 |
| `DEVIN_HANDOFF_PROMPT.md` | 可直接交给 Devin 的接入与验收指令。 |
| `README_GINZA_DATA_PIPELINE_CN.md` | 当前说明文档。 |

本交付不覆盖现有 `array_config.py`、`dft.py`、`muting.py`、`pmi.py`、`pmi_mask.py`、`sionna_array.py`、`beam_sweep.py` 或原 notebook；它直接复用这些项目文件提供的接口。

## 3. 已与 notebook 对齐的物理配置

| 参数 | 当前值 |
| --- | --- |
| 场景 | Ginza `ginza_1.xml` |
| 载频 | 3.5 GHz |
| 信道带宽 | 100 MHz |
| TX 位置 | `(-122.0, -108.5, 41.0)` m |
| TX orientation | `(0, 15/360*pi, 0)`，即实际 7.5° |
| TX 物理阵列 | `ArrayConfig(4, 8, 2, 2)`，共 128 ports |
| TX element pattern / 极化 | `tr38901` / `cross` |
| RX array / 极化 | 1×1 `iso` / `VH`，共 2 ports |
| 空间码本 | `Kv=8`, `Kh=32` |
| `num_i2` | 4 |
| 完整码本 | 1024 beams |
| FFT size | 4096 |
| 有效子载波 | 3276 |
| 子载波间隔 | 30 kHz |
| CP 长度 | 288 samples，仅记录配置，不引入 ISI 模型 |
| normal 总发射功率 | 51.13 dBm |
| 噪声系数 | 5 dB |
| 温度 | 290 K |
| `samples_per_src` | 1,000,000 |
| `max_num_paths_per_src` | 1,000,000 |
| 路径随机种子 | 36 |

混凝土覆盖与当前 notebook 一致：`thickness=0.5`、`scattering_coefficient=0.3`、`xpd_coefficient=0.3`。

## 4. 最重要的数据语义

### 4.1 valid PMI 只是 metadata

normal-mode 最优 PMI 必须来自完整 1024-beam codebook：

```text
wideband-selected PMI = argmax over all 1024 beams of
                        mean_subcarrier(sum_rx |H[k] @ w_beam|²)
```

`selected_pmi_valid=False` **不会拒绝样本，也不会改变 beam sweep**。该字段只用于后续数据审计、分层分析或 ML 清洗。NLOS UE 同样正常生成样本。

### 4.2 normal 和 sleep 共享同一次传播求解

每个 UE 仅调用一次 `PathSolver`：

```text
same paths -> same wideband H -> same normal-selected PMI
                              -> normal codeword
                              -> right-half-muted sleep codeword
```

sleep 码字不重新归一化：

```text
||w_normal||² = 1
||w_sleep||²  = 0.5
```

实际 sleep 发射功率下降由码字范数自然产生，不能再额外乘一次 `0.5`。

### 4.3 宽带平均必须在线性域进行

主字段：

```text
normal_rx_power_dbm = 30 + 10 log10(mean_k(P_rx_normal[k]))
sleep_rx_power_dbm  = 30 + 10 log10(mean_k(P_rx_sleep[k]))
normal_snr_db       = 10 log10(mean_k(SNR_normal[k]))
sleep_snr_db        = 10 log10(mean_k(SNR_sleep[k]))
```

`normal_mean_subcarrier_rx_power_dbm` 等字段另行保存 `mean_k(10 log10(...))`。这两种统计量不能互相替代。

当前为单 TX、单 UE、无额外干扰模型，因此这里的 SINR 与 SNR 相同；不虚构干扰项，也不把 RMS delay spread 直接伪装成 ISI 功率。

### 4.4 时延特征来自 normal-mode beamformed PDP

对每条路径 `p`：

```text
A_p: [RX ports, TX ports]
g_p = A_p @ w_normal_sionna
P_p = sum_rx |g_p|²

weighted_delay = sum_p(P_p * tau_p) / sum_p(P_p)
rms_delay_spread = sqrt(sum_p(P_p * (tau_p - weighted_delay)²) / sum_p(P_p))
```

`power_weighted_delay_s` 是 TA-related proxy，不等同于真实 3GPP Timing Advance 命令。`rms_delay_spread_s` 可用于分析时间色散，但本版本不做 OFDM 时域 ISI/ICI 仿真。

#### 4.4.1 Sionna RT 路径系数的复数重建

Sionna RT 1.2.2 把路径复数系数拆成两个 `TensorXf`：

- `paths.a[0]`：实部（real part）
- `paths.a[1]`：虚部（imaginary part）

本流水线在 `extract_path_arrays` 中先合成为完整复数系数：

```text
A_p = paths.a[0] + 1j * paths.a[1]
```

再把它用于 beamformed PDP，以保留每条路径的真实相位。`valid_pmi_mask`、
`selected_pmi_valid` 等字段只作为 metadata，不影响该计算。

> **与旧 notebook 的区分**：canonical notebook 的 beamformed PDP 相关 cell
> 目前只使用 `np.asarray(paths.a[0])`（实部），会丢弃虚部。旧数值
>（例如 `power_weighted_delay_s ≈ 1450.827 ns`、`rms_delay_spread_s ≈ 145.937 ns`）
> 不再作为本流水线生成数据表的正式时延基准；修复后的数据表将使用完整复数
> 系数得到的 PDP 值。

### 4.5 LOS 分类只是明确记录的启发式

Sionna RT 1.2.2 的当前 `Paths` 接口没有稳定公开每条路径的类型标记。因此：

```text
has_los = any(abs(tau_valid - distance_3d / c) <= tolerance)
```

对应字段：

```text
has_los
los_classification_method = "delay_matches_geometric_distance"
```

它只是 metadata。后续如果确认了稳定、可靠的路径类型 API，建议由 Devin 替换为真实 path-type 分类。

## 5. 主数据表结构

每行对应一个 UE 位置。主要字段包括：

| 分组 | 字段示例 |
| --- | --- |
| 身份与位置 | `sample_id`, `ue_index`, `ue_x_m`, `ue_y_m`, `ue_z_m` |
| 几何与环境 | `tx_ue_distance_2d_m`, `tx_ue_distance_3d_m`, `has_los` |
| 宽带最优 PMI | `pmi_i11`, `pmi_i12`, `pmi_i2`, `beam_index` |
| 中心频点 PMI | `center_pmi_i11`, `center_pmi_i12`, `center_pmi_i2` |
| PMI metadata | `selected_pmi_valid`, `center_selected_pmi_valid` |
| normal 输入 | `normal_rx_power_dbm`, `normal_snr_db` |
| sleep 标签 | `sleep_rx_power_dbm`, `sleep_snr_db` |
| normal/sleep 差值 | `rx_power_loss_db`, `snr_loss_db`, `mean_subcarrier_loss_db` |
| PDP 特征 | `power_weighted_delay_s`, `rms_delay_spread_s` |
| 路径摘要 | `first_path_delay_s`, `strongest_path_delay_s`, `last_path_delay_s` |
| 路径数量 | `total_path_count`, `valid_path_count`, `usable_path_count` |
| 功率模型 | `normal_tx_power_w`, `sleep_tx_power_w`, `per_subcarrier_noise_power_w` |
| 复现信息 | `seed`, `config_hash`, `git_commit`, `schema_version` |
| 执行状态 | `status`, `error_message`, `solver_runtime_s`, `cfr_runtime_s` |

初步 ML 输入：

```python
ML_INPUT_FIELDS = (
    "normal_rx_power_dbm",
    "normal_snr_db",
    "pmi_i11",
    "pmi_i12",
    "pmi_i2",
    "power_weighted_delay_s",
    "rms_delay_spread_s",
)
```

初步预测标签：

```python
LABEL_FIELDS = (
    "sleep_rx_power_dbm",
    "sleep_snr_db",
    "rx_power_loss_db",
    "snr_loss_db",
)
```

这些字段只是当前候选设计；后续可以增加几何、TA 相关特征、场景标签或 MCS，但应先审查数据泄漏。

## 6. 安装到现有项目

先由 Devin 检查目标目录中是否已经存在同名文件，避免覆盖未提交工作。预期项目根目录为：

```text
/workspace/mMIMO_sleep_rebuild_20260818
```

文件复制完成后，在该目录运行：

```bash
PYTHONPATH="$PWD/src" /opt/venv/bin/python -B \
  scripts/generate_ginza_dataset.py --show-schema

PYTHONPATH="$PWD/src" /opt/venv/bin/python -B \
  scripts/generate_ginza_dataset.py --print-config
```

这两条命令只检查 schema 和配置，不启动 GPU 仿真。

## 7. 先跑一个已验证 UE

当前 notebook 的参考 UE 是 `rx_1000.pkl index 1`，建议先单独运行：

```bash
PYTHONPATH="$PWD/src" /opt/venv/bin/python -B \
  scripts/generate_ginza_dataset.py \
  --config configs/ginza_dataset_v1.yaml \
  --start-index 1 \
  --stop-index 2 \
  --output-dir outputs/ginza_dataset_smoke \
  --format jsonl
```

重点与 notebook 对照：

```text
center PMI:                  (i11=24, i12=1, i2=0)
wideband PMI:                (i11=23, i12=1, i2=0)
normal wideband RX power:    approximately -107.048 dBm
sleep wideband RX power:     approximately -112.196 dBm
normal wideband SNR:         approximately 17.156 dB
sleep wideband SNR:          approximately 12.008 dB
weighted mean path delay:    approximately 1450.827 ns
RMS delay spread:            approximately 145.937 ns
```

GPU 光线求解、路径排序和浮点运算可能造成轻微波动；不要机械要求浮点末位完全一致。

## 8. 批量运行及断点续跑

先运行前十个 UE：

```bash
PYTHONPATH="$PWD/src" /opt/venv/bin/python -B \
  scripts/generate_ginza_dataset.py \
  --start-index 0 \
  --max-samples 10 \
  --output-dir outputs/ginza_dataset_v1 \
  --format jsonl
```

之后扩展至全部 UE，并跳过已有样本：

```bash
PYTHONPATH="$PWD/src" /opt/venv/bin/python -B \
  scripts/generate_ginza_dataset.py \
  --output-dir outputs/ginza_dataset_v1 \
  --format jsonl \
  --resume
```

重试之前失败的 UE：

```bash
PYTHONPATH="$PWD/src" /opt/venv/bin/python -B \
  scripts/generate_ginza_dataset.py \
  --output-dir outputs/ginza_dataset_v1 \
  --format jsonl \
  --resume \
  --retry-failed
```

输出结构：

```text
outputs/ginza_dataset_v1/
  manifest.json
  run_summary.json
  ginza_dataset.jsonl
  parts/
    ue_0000000.json
    ue_0000001.json
    ...
```

每个 UE 的 checkpoint 单独写入，并采用原子替换。进程中断不会损坏已经完成的 UE。不同物理配置不能写进同一目录，`manifest.json` 使用 `config_hash` 阻止误混合。

`--format auto` 会在已安装 `pyarrow` 时使用 Parquet，否则自动退回 JSONL。不会擅自安装依赖。即使最终输出为 CSV/Parquet，resume checkpoint 仍始终是 JSON 分片。

## 9. 测试和审计

离线测试不要求 GPU：

```bash
PYTHONPATH="$PWD/src" /opt/venv/bin/python -B -m pytest \
  -p no:cacheprovider \
  tests/test_dataset_schema.py \
  tests/test_ginza_sample.py \
  tests/test_pipeline_resume.py \
  -q
```

如果环境中没有 pytest，也可以使用标准库：

```bash
PYTHONPATH="$PWD/src" /opt/venv/bin/python -B -m unittest \
  discover -s tests -p 'test_*.py'
```

项目接入后仍应运行：

```bash
/opt/venv/bin/ruff check src/mMIMO_sleep tests scripts/generate_ginza_dataset.py

PYTHONPATH="$PWD/src" /opt/venv/bin/python -B -m pytest \
  -p no:cacheprovider tests/ -q

/opt/venv/bin/python /opt/smoke_test.py --mode gpu
```

## 10. 当前实现的边界

1. 没有 MCS 标签。
2. 没有完整 OFDM 时域 waveform，也没有显式 ISI/ICI 建模。
3. `power_weighted_delay_s` 是 TA-related proxy，不是真实 3GPP TA。
4. `has_los` 当前为几何时延启发式，不是 Sionna 官方逐路径类型。
5. 默认单进程、逐 UE 运行，优先保证 scene / solver / GPU 上下文安全。
6. 当前聊天环境无法访问你的 RunPod GPU、场景 XML 和 UE pickle；真正的端到端 RT 接入必须由 Devin 在项目环境完成并核对参考 UE。
7. 首次接入时，重点检查 Sionna `Paths.cfr` 输入类型、scene 生命周期、`scene.remove` 行为及 GPU 显存。若实际 Sionna API 与附件不一致，应以 RunPod 实际版本为准做最小修正。
