# Devin 分段任务：Ginza 单 UE 的 normal/sleep SINR 验证 Notebook

## 使用方式

请创建并逐步完善：

`notebook/validate_ginza_single_ue_normal_sleep.ipynb`

每次只读取并完成本文的一个 Section。完成后停止，汇报：

- 修改了哪个 cell；
- cell 的实际输出；
- assertion 是否通过；
- 当前问题或假设。

不要提前实现后续 Section。用户确认后再读取下一节。

可使用下面这句逐段下达任务：

> 请阅读 `devin_ginza_single_ue_notebook_prompt.md` 的 Section N，只完成该 Section，并运行该 cell。完成后停止并汇报实际输出，不要继续下一节。

## 全局规则（先读一次）

1. 先阅读 `pyproject.toml`、最新 handoff 文件，以及现有的 Ginza、array、codebook、PMI mask 和 Sionna RT 相关代码。
2. 使用当前 `.venv` 和已安装的 Sionna RT 1.2.2；不要重装依赖或修改环境。
3. 复用项目已有的 Ginza scene loader、ROI/UE 采样器和配置，不要重新手写一套场景。
4. 复用项目函数，不要在 Notebook 复制 `dft.py`、`pmi.py`、`muting.py`、`pmi_mask.py` 或 `sionna_array.py` 的核心逻辑。
5. element pattern 由 Sionna RT array/scene 配置负责；codebook 和 muting 函数只生成或修改端口激励权重。
6. `valid_pmi_mask` 的索引固定为 `valid_pmi_mask[i12, i11]`；完整 codebook 顺序固定为 `(i12, i11, i2)`，并且 `i2` 变化最快。
7. 对最终选中的 UE 只执行一次 PathSolver。normal 和 sleep 必须使用同一个 `paths`、同一个未归一化信道 `H`。
8. 不要根据 ray 直接“反推” `w_normal`。`w_normal` 必须由 valid codebook 穷举选择；ray/AoD 仅用于物理一致性检查。
9. 本 Notebook 暂不做全区域 sweep、批量 UE、MCS、训练或数据集生成。
10. 每个关键 tensor 都打印 shape、dtype、device；不要用无法解释的 `squeeze()`。

---

## Section 1 — Cell 1：导入、版本和实验配置

新增一个 code cell，完成：

- 导入项目模块、PyTorch、NumPy、Matplotlib、Mitsuba 和 Sionna RT。
- 打印 Python、PyTorch、CUDA、Sionna RT、Mitsuba、Dr.Jit 版本及 Mitsuba variant。
- 设置随机种子，例如 `seed = 42`。
- 从项目配置读取阵列、载频、发射功率、带宽、噪声系数等参数，不要在多个 cell 重复硬编码。
- 若链路预算参数尚未在项目中定义，列出缺失项并停止，不要静默猜值。

预期结果：环境导入成功，GPU 可用，配置集中显示。

---

## Section 2 — Cell 2：加载 Ginza 场景和阵列

新增一个 code cell，完成：

- 使用项目现有入口加载 Ginza scene。
- 使用项目现有 array mapping 构造 TX array；交叉极化和端口顺序必须与 codebook 一致。
- 复用现有 UE/RX array 配置。若项目没有定义，先采用最小可运行的 1×1 cross-polarized RX，并明确打印这是临时假设。
- 放置固定 TX，打印 TX 位置、朝向、阵列尺寸、极化方式和端口数。

必须检查：codebook 的物理端口数将能够与 Sionna TX 端口数完全一致。

---

## Section 3 — Cell 3：生成 normal codebook 和 valid PMI mask

新增一个 code cell，调用项目现有函数：

- 生成 RI=1 normal codebook。
- 生成或加载 `valid_pmi_mask[i12, i11]`。
- 建立显式映射：`flat_index <-> (i12, i11, i2)`，确认 `i2` 最快变化。
- 构造 valid beam flat indices：只有 mask 为真的空间 PMI，以及它们对应的全部 `i2`。

打印并断言：

- codebook shape 为 `(num_beams, num_tx_ports)`；
- 每个 normal weight 的功率范数约为 1；
- mask shape 与 `(num_i12, num_i11)` 一致；
- valid spatial PMI 数和 valid beam 总数正确；
- invalid 空间 PMI 的全部 `i2` 都被排除。

同时画出 `valid_pmi_mask` heatmap，坐标轴明确标为 `i11` 和 `i12`。

---

## Section 4 — Cell 4：随机放置一个可用 UE

新增一个 code cell，完成：

- 使用固定 seed 和项目现有 Ginza ROI/UE sampler 随机生成一个 UE 位置。
- UE 高度使用项目配置。
- 将 UE 加入 scene，并打印位置。
- 如果随机点不满足场景约束，可有限次数重采样；记录尝试次数。

此 cell 不运行最终 PathSolver。只完成可复现的位置选择和场景放置。

---

## Section 5 — Cell 5：运行一次 PathSolver 并可视化 rays

新增一个 code cell，完成：

- 对当前 TX–UE 运行一次 PathSolver，并将结果保存为后续复用的 `paths`。
- 参数应复用项目配置；明确记录最大反射/绕射/散射深度等设置。
- 打印有效路径数量、delay、AoD、AoA 和每类交互信息（以当前 API 实际提供的字段为准）。
- 使用 `scene.preview(paths=paths)` 或兼容的 render API 显示 TX、UE 和传播路径。

若没有有效路径，可以回到 Section 4 的采样逻辑有限重试；找到有效 UE 后仅保留该 UE 的这一次最终 PathSolver 结果。不要在后续 cell 重算路径。

---

## Section 6 — Cell 6：由 paths 构造未归一化信道 H

新增一个 code cell，完成：

- 使用 Sionna RT 1.2.2 的实际 API 从 `paths` 得到单频/窄带 CFR。
- 必须设置 `normalize=False`，以便 normal/sleep 接收功率具有可比的绝对尺度。
- 将输出转换/整理为明确的复数矩阵 `H`，其逻辑 shape 为 `[num_rx_ports, num_tx_ports]`。
- 用注释逐一说明原始 CFR 的每个轴，以及如何得到该矩阵；禁止盲目 `squeeze()`。

打印并断言：

- 原始 CFR shape；
- 最终 `H.shape`；
- `H` 中没有 NaN/Inf；
- `H.shape[-1] == codebook.shape[-1]`。

---

## Section 7 — Cell 7：在 valid beams 中选择 w_normal

新增一个 code cell，完成：

- 只对 Section 3 的 valid beams 计算 `g_b = H @ w_b`。
- 若 RX 有多个端口，使用项目既定 combiner；若没有定义，则暂用 ideal MRC，并把假设打印出来。
- 以合并后的接收信号功率选择最大 beam。
- 得到 `normal_pmi = (i11, i12, i2)`、flat index 和 `w_normal`。

打印并断言：

- 最优 beam 属于 valid 集合；
- `(i12, i11, i2)` 与 flat index 往返转换一致；
- `valid_pmi_mask[i12, i11]` 为真；
- `||w_normal||² ≈ 1`。

可额外计算一次不受 mask 限制的最佳 beam 作为诊断，但不得用它替代 valid-beam 结果。

---

## Section 8 — Cell 8：用 rays/AoD 检查 w_normal 的物理一致性

新增一个诊断 code cell，不重新选择 beam：

- 找出功率最大的传播路径，报告其 departure direction/AoD。
- 根据项目 array 坐标约定计算或绘制 `w_normal` 的阵列因子主瓣方向。
- 比较主瓣方向与 dominant path AoD，并明确角度坐标、符号和单位。
- 如果 API 能可靠获得 per-path channel，计算各路径经过 `w_normal` 后的贡献；否则明确说明 API 限制，不要伪造结果。

输出一张可读的方向图/AoD 对比图，并解释：多径环境下，最优 beam 不一定严格对准单条最强 ray；最终正确性仍以穷举得到的接收功率最大值为准。

---

## Section 9 — Cell 9：由 w_normal 生成 w_sleep

新增一个 code cell，调用项目现有 muting 函数：

- 根据 array/subarray 配置关闭物理阵列右半边，得到 `w_sleep`。
- 必须支持 `elements_per_subarray = 1, 2, 3, 4`，但本次运行使用项目当前配置。
- 不要重新生成 beam，也不要改变 element pattern。
- 不要对 `w_sleep` 重新归一化；否则会抹掉期望的功率损失。

打印并断言：

- normal/sleep weight shape 相同；
- 被关闭端口严格为 0；
- 未关闭端口与 `w_normal` 对应值一致；
- `||w_normal||² ≈ 1`；
- 对关闭一半等功率端口的当前配置，`||w_sleep||² ≈ 0.5`。

画出 normal/sleep 端口幅度分布或物理阵列 heatmap。

---

## Section 10 — Cell 10：在同一个 H 上计算 normal/sleep SINR

新增一个 code cell，完成：

- 计算 `g_normal = H @ w_normal` 和 `g_sleep = H @ w_sleep`。
- 使用同一个 RX combining 方法得到 normal/sleep 接收信号功率。
- 使用 Section 1 的同一套发射功率、带宽、温度和噪声系数计算噪声功率。
- 如果只有一个 TX 且未建模干扰，结果物理上是 SNR；可以保留项目字段名 SINR，但必须注明 `I = 0`。
- 输出 normal、sleep SNR/SINR，以及 `normal - sleep` 的 dB loss。

禁止通过直接减 6 dB 生成 sleep 结果。6 dB 只作为比较基线，真实 sleep 结果必须来自 `H @ w_sleep`。

---

## Section 11 — Cell 11：汇总结果和验收

新增最后一个 code cell/summary table，至少汇总：

- seed、TX/UE 位置；
- 路径数量和 `H.shape`；
- valid spatial PMI 数、valid beam 数；
- normal PMI `(i11, i12, i2)` 和 flat index；
- normal/sleep weight power；
- normal/sleep 接收功率；
- normal/sleep SNR/SINR；
- 实际 loss 与 6 dB baseline 的差异。

最终验收条件：

1. Notebook 从头到尾可重复运行；
2. 最终 UE 的 PathSolver 只调用一次；
3. normal 和 sleep 使用同一个未归一化 `H`；
4. beam 选择只使用 valid PMI；
5. sleep weight 没有重新归一化；
6. 所有关键 shape、索引顺序和单位都有明确输出；
7. rays 图、mask heatmap、beam/AoD 图及 normal/sleep weights 图都能正常显示。

若任一条件失败，请停在该 cell 修正，不要进入 MCS 或数据集阶段。

---

## Section 12 — Notebook 跑通后再做：最小重构与测试

只有 Section 1–11 全部通过后才执行：

- 将 Notebook 中可复用、非可视化的逻辑放回合适的 `src/` 模块，Notebook 只负责调用和展示。
- 为 PMI 顺序、valid mask 展开、right-half muting、weight norm 和 same-H normal/sleep 计算补充单元测试。
- 将实际环境、运行命令、关键结果和仍存在的假设写入新的 handoff 文件。
- 不要在这一阶段扩展到全区域 sweep、批量 UE、MCS 或模型训练。
