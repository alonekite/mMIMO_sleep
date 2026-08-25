# mMIMO_sleep 项目工作日志

**日期：2026-08-18**  
**项目：mMIMO Sleep / Sionna RT**  
**今日主题：RunPod 环境固化、代码事故恢复、双极化 PMI codebook 验证与主分支合并**

## 1. 今日目标

今天的工作原本包括两条主线：

1. 把 RunPod 上的 Python、PyTorch、Sionna RT、Mitsuba 和 Dr.Jit 环境固化为可重复构建的 Docker image，避免 Pod 重启或迁移后重新修环境。
2. 继续实现 mMIMO sleep 的双极化 codebook、subarray muting、PMI mask 和 Sionna RT 接口。

开发过程中发生了一次 `git reset --hard` 误覆盖未提交代码的事故，因此当天后半部分转为：

3. 保存恢复材料，在隔离 clone 中按 executable specification 重建丢失代码。
4. 完成全部测试、GPU 验证、notebook 重建、PR 审查与合并。

## 2. 今日最终成果

截至今天结束，以下目标均已完成：

- RunPod 专用 Docker image 构建成功并上传 Docker Hub。
- GitHub Actions 构建流程可手动重复运行。
- Docker image 已在真实 RTX 4090 Pod 上完成 GPU acceptance。
- Python interpreter 固定在 `/opt/venv/bin/python`。
- Jupyter kernel 固定为 `Python (mMIMO_sleep)`。
- 双极化 RI=1 codebook、subarray expansion、right-half muting、PMI indexing、PMI validity mask 全部恢复。
- 项目 codebook 到 Sionna RT PlanarArray ordering 的转换测试通过。
- `validate_pmi_muting_patterns.ipynb` 已按新 API 重建。
- 全部测试：`142 passed`。
- Ruff：`All checks passed!`
- GPU smoke test：全部通过。
- 恢复 PR #3 已合并到 `main`。
- `origin/main` 最终 commit：`118d5b63482244d9796968cf613653faf678315a`。

## 3. RunPod 环境与 Docker image

### 3.1 最终镜像

```text
Image: alonekite/mmimo-sleep-runpod:v1
Digest: sha256:1224066c79da9ffeddb76cda11d700c5ba9a6b0a4aa3ecb9a11532a47d61d323
Platform: linux/amd64
```

GitHub Actions 构建结果：

- Dependency assertions：通过
- CPU/import smoke test：通过
- Docker manifest platform：`linux/amd64`
- Docker Hub push：成功

真实 RunPod GPU acceptance：

```text
GPU: NVIDIA GeForce RTX 4090
Driver: 580.126.20
nvidia-smi CUDA capability: 13.0
PyTorch: 2.13.0+cu126
PyTorch CUDA runtime: 12.6
Sionna RT: 1.2.2
Mitsuba: 3.8.0
Dr.Jit: 1.3.1
```

验证命令：

```bash
/opt/venv/bin/python /opt/smoke_test.py --mode gpu
```

结果：

```text
RESULT: all checks passed (gpu mode)
```

### 3.2 环境架构

最终采用“不可变环境 + 持久项目数据”的分层方式：

| 内容 | 位置 | 生命周期 |
|---|---|---|
| Python 3.11 | Docker image | 随 image 固定 |
| PyTorch/Sionna RT/Mitsuba/Dr.Jit | `/opt/venv` | 随 image 固定 |
| Jupyter kernelspec | `/usr/local/share/jupyter/kernels/mmimo-sleep` | 随 image 固定 |
| 项目代码、notebook、数据 | `/workspace` Network Volume | 跨 Pod 持久化 |
| 项目目录 | `/workspace/mMIMO_sleep_rebuild_20260818` | 当前可信工作副本 |

以后 VS Code 使用：

```text
Python interpreter: /opt/venv/bin/python
Jupyter kernel: Python (mMIMO_sleep)
```

### 3.3 为什么不再使用 Network Volume 中的 `.venv`

旧 `.venv` 位于：

```text
/workspace/mMIMO_sleep/.venv
```

它的 Python 最终链接到 container 内的：

```text
/usr/bin/python3.11
```

Pod 迁移后，新 container 只有 Python 3.10，导致保存下来的 `.venv` 变成 dangling interpreter：

```text
bash: .venv/bin/python: No such file or directory
```

因此虚拟环境不能只依赖 Network Volume 保存。新方案把完整 Python runtime 和 binary packages 放入 `/opt/venv`，由 Docker image 保证。

## 4. Docker 构建过程中解决的问题

### 4.1 默认 pip 安装引入 CUDA 13

最初执行：

```bash
.venv/bin/pip install -e ".[dev,rt]"
```

默认 PyPI 解析出：

```text
torch 2.13.0+cu130
```

并安装了大量 CUDA 13 packages。RTX 4090 环境随后出现：

```text
Error 804: forward compatibility was attempted on non supported HW
```

最终方案：

```text
torch==2.13.0+cu126
```

并从官方 CUDA 12.6 wheel index 单独安装 PyTorch，再安装其余依赖。`constraints.txt` 同时阻止后续 pip 静默切换回 cu130。

### 4.2 CUDA compat library 遮挡 host driver

历史 container 的 dynamic linker 优先加载：

```text
/usr/local/cuda-13.1/compat/libcuda.so
```

而不是 RunPod host 注入的真实 driver。RTX 4090 不支持该 forward-compat 方式，因此触发 Error 804。

最终启动脚本采用证据驱动的幂等诊断：

- 先打印 `nvidia-smi`、`LD_LIBRARY_PATH` 和 `ldconfig` 解析顺序；
- 只有确认 compat library 遮挡真实 driver 时才禁用对应配置；
- 不在 build 阶段盲目删除 CUDA 文件。

### 4.3 Dr.Jit 的误导性 Python ABI 错误

Docker build 初次 import Dr.Jit 时，顶层错误声称 Python 3.11.14 与 3.11.15 不兼容。进一步检查 `__cause__` 和 `ldd` 后发现真正原因：

```text
ImportError: libatomic.so.1: cannot open shared object file
```

最终 Dockerfile 增加：

```text
libatomic1
```

并加入 build-time `ldconfig` assertion。修复后 Dr.Jit、Mitsuba、Sionna RT CPU import smoke test 全部通过。

### 4.4 Docker Hub private repository

RunPod 初次拉取镜像时出现：

```text
unauthorized: repository is private or does not exist
```

原因是 repository 为 private，而 RunPod 未提供 registry credentials。后续通过匹配 repository visibility/credentials 解决。

### 4.5 SSH public key

SSH 密码提示的原因是新 container 没有旧 Pod 的 `authorized_keys`。最终通过 RunPod environment variable `PUBLIC_KEY` 在启动时注入公钥，不把任何私钥或 token 写入 image 或 Git。

## 5. Git 误操作与恢复事故

### 5.1 事故

为了把 Docker commit 从 `main` 移到 feature branch，执行了：

```text
git reset --hard HEAD~1
```

该命令覆盖了六个 tracked 文件中尚未 commit/stage 的工作树修改：

```text
notebook/validate_pmi_muting_patterns.ipynb
src/mMIMO_sleep/codebook/dft.py
src/mMIMO_sleep/codebook/muting.py
src/mMIMO_sleep/codebook/pmi.py
src/mMIMO_sleep/codebook/pmi_mask.py
tests/test_pmi_mask.py
```

Git 无法从 reflog/fsck 恢复这些内容，因为它们从未 `git add`，因此从未成为 Git blob。

### 5.2 保存的恢复材料

恢复 archive：

```text
mMIMO_sleep_recovery_20260818T163058Z.tar.gz
```

SHA-256：

```text
98aef2c52646b9c9f7e5ae5ae5c1881731fbc7a08fe3df858e779f41f3fc014d
```

幸存的关键材料包括：

- `array_config.py`
- `sionna_array.py`
- `test_array_config.py`
- `test_dft.py`
- `test_muting.py`
- `test_pmi.py`
- `test_sionna_codebook.py`
- `pmi_mask_review_issue.md`
- handoff 文件
- recovery archive

这些 tests 成为重建实现的 executable specification。

### 5.3 隔离恢复策略

未继续在事故现场目录操作，而是创建：

```text
/workspace/mMIMO_sleep_rebuild_20260818
```

恢复分支：

```text
rebuild/pmi-codebook
```

原目录：

```text
/workspace/mMIMO_sleep
```

保持不动，只作为历史证据与 recovery source。所有重建、测试和提交均在隔离 clone 中完成。

## 6. 代码恢复阶段与 commits

恢复按依赖顺序逐阶段完成：

| Commit | 内容 |
|---|---|
| `da4c24d` | 恢复幸存的 `ArrayConfig`、Sionna adapter 和 executable specs |
| `b7ff7e9` | 恢复双极化 subarray DFT codebook |
| `0c388fe` | 恢复 physical-port right-half muting |
| `d965aa7` | 恢复带 `i2` 的 RI=1 PMI indexing |
| `1bed21d` | 重建 PMI mask executable specification |
| `65a0754` | 修复 PyTorch 2.13 下的测试兼容性问题 |
| `bfa503f` | 恢复 dual-polarized PMI validity mask |
| `756044a` | 重建 PMI muting validation notebook |
| `42d9ed4` | 修复恢复代码中的 Ruff lint 问题 |

恢复 PR：

```text
PR #3: Restore dual-polarized subarray PMI codebook and validation
```

合并 commit：

```text
118d5b6 Merge pull request #3 from alonekite/rebuild/pmi-codebook
```

验证结果：

- `42d9ed4` 是 `origin/main` 的 ancestor；
- recovery final tree 与合并后 main tree 无差异；
- 13 个目标文件全部存在；
- 远程 recovery branch 暂时保留。

## 7. 恢复和新增的 13 个文件

```text
notebook/validate_pmi_muting_patterns.ipynb
src/mMIMO_sleep/array_config.py
src/mMIMO_sleep/codebook/dft.py
src/mMIMO_sleep/codebook/muting.py
src/mMIMO_sleep/codebook/pmi.py
src/mMIMO_sleep/codebook/pmi_mask.py
src/mMIMO_sleep/simulation/sionna_array.py
tests/test_array_config.py
tests/test_dft.py
tests/test_muting.py
tests/test_pmi.py
tests/test_pmi_mask.py
tests/test_sionna_codebook.py
```

## 8. 当前阵列与 codebook 约定

### 8.1 ArrayConfig

主验证配置：

```python
ArrayConfig(
    num_subarray_rows=4,
    num_horizontal=8,
    elements_per_subarray=2,
    num_polarizations=2,
)
```

映射关系：

```text
4×8 logical subarrays
    ↓ 每个 logical subarray 垂直展开为 2 个 physical elements
8×8 physical elements
    ↓ dual polarization
8×8×2 = 128 physical ports
```

`elements_per_subarray` 支持：

```text
1, 2, 3, 4
```

### 8.2 Codebook

默认参数：

```text
num_i12 = 8
num_i11 = 32
num_i2 = 4
```

完整 codebook shape：

```text
(8 × 32 × 4, 128) = (1024, 128)
```

beam 顺序：

```text
(i12, i11, i2)
```

其中 `i2` 变化最快。

Flat index：

```text
beam_index = (i12 * num_horizontal_beams + i11) * num_i2 + i2
```

### 8.3 Polarization ordering

项目内部 physical-port ordering：

```text
polarization-major
```

即：

```text
[pol0 全部 physical elements, pol1 全部 physical elements]
```

每个 polarization 内部为 row-major，horizontal column 变化最快：

```text
port = pol * num_physical_elements + row * num_horizontal + col
```

`i2` co-phasing：

```text
pol1 = pol0 * exp(+j * 2π * i2 / num_i2)
```

每个 codeword 归一化为：

```text
sum(abs(w)**2) = 1
```

## 9. Sleep muting 约定

Sleep mode 关闭 physical array 的右半边：

- 左半边 active；
- 右半边 muted；
- 两个 polarization 同时关闭对应 spatial elements；
- `active_power_fraction = 0.5`；
- 只执行 elementwise mask；
- 不 renormalize；
- 不对 active ports 做功率补偿；
- 不修改 active weights 的相位。

因此默认情况下：

```text
normal codeword power ≈ 1.0
sleep codeword power ≈ 0.5
```

## 10. PMI validity mask 的物理定义

对于每个空间 PMI `(i11,i12)`：

1. 使用 normal codebook weights；
2. 用统一 muting API 得到 sleep weights；
3. 在 visible direction-cosine grid 上计算 normal total power；
4. 找到 normal peak direction；
5. 在同一方向读取 sleep total power；
6. 计算：

```text
loss_db = 10*log10(
    P_normal(normal_peak_direction)
    /
    P_sleep(normal_peak_direction)
)
```

严禁分别计算 normal peak 和 sleep peak 后相除。

双极化 total power：

```text
P_total = |E_pol0|² + |E_pol1|²
```

两个 polarization 不做 coherent complex-field addition。

Validity criterion：

```text
target = 10*log10(4) = 6.020599913... dB
abs(loss_db - target) <= tolerance_db
```

默认 tolerance：

```text
1.0 dB
```

Mask shape 与 indexing：

```text
valid_pmi_mask.shape = (num_i12, num_i11)
valid_pmi_mask[i12, i11]
```

Mask 只包含 spatial PMI。有效 `(i11,i12)` 在后续 beam selection 中展开为全部 `i2`。

### 10.1 当前计算结果

在默认配置、grid 和 `6.0206 ± 1.0 dB` 标准下：

```text
Valid spatial PMIs: 255 / 256
Invalid spatial PMIs: 1 / 256
Valid full beams after i2 expansion: 1020 / 1024
```

代表性结果：

```text
Valid:   i11=0,  i12=0, loss≈6.021 dB
Invalid: i11=16, i12=4, loss≈4.180 dB
```

## 11. Element pattern 的职责边界

`pmi_mask.py` 只处理：

- codebook weights；
- array factor；
- physical-port muting；
- total-power loss；
- PMI validity。

它不包含：

- TR 38.901 element pattern；
- element gain；
- Sionna RT scene antenna pattern；
- Ginza scene；
- UE、channel、SINR 或 MCS。

Element pattern 由后续 Sionna RT `PlanarArray` 和场景层加载。

## 12. Sionna RT ordering 集成

项目内部 element index：

```text
project_index = row * num_horizontal + col
```

Sionna RT PlanarArray 使用 column-first element ordering：

```text
sionna_index = col * num_physical_rows + row
```

`weights_to_sionna_precoding` 对每个 polarization block 独立做 permutation，再拆为 `(real, imag)`。

Runtime probe：

```text
num_physical_ports: 24
codeword.shape: [24]
array.num_ant: 24
precoding real shape: [24]
precoding imag shape: [24]
power before: 0.99999994
power after:  0.99999994
```

转换只做 ordering permutation 与 real/imag 拆分，不 renormalize，功率守恒。

## 13. 验证结果

### 13.1 完整 tests

```bash
PYTHONPATH="$PWD/src" /opt/venv/bin/python -B -m pytest \
    -p no:cacheprovider tests/ -q
```

结果：

```text
142 passed
```

无 failed、无 skipped、无 warnings。

### 13.2 Ruff

首次最终审计发现 10 条 import order、unused import/variable、line length 和 `zip(strict=...)` 问题。

修复 commit：

```text
42d9ed4 Fix lint issues in restored PMI codebook
```

最终结果：

```text
All checks passed!
```

### 13.3 Notebook

Notebook：

```text
notebook/validate_pmi_muting_patterns.ipynb
```

Canonical 状态：

```text
26 cells
11 Markdown cells
15 code cells
0 outputs
0 non-null execution counts
kernel: mmimo-sleep
display name: Python (mMIMO_sleep)
```

Canonical SHA-256：

```text
a0c06f0c3eab1162e7470cd06f01023eafc0409f2126a1c0c35542295451024f
```

临时执行副本：

```text
/tmp/validate_pmi_muting_patterns.executed.ipynb
```

结果：

- 15 个 code cells 全部执行；
- 0 error outputs；
- 8 条 summary assertions 全部 PASS；
- 临时执行前后 canonical SHA 不变。

### 13.4 Sionna RT

```text
tests/test_sionna_codebook.py: 9 passed
tests/test_rt_smoke.py: 2 passed
```

包括：

- array geometry mapping；
- project weights → Sionna precoding ordering；
- RadioMapSolver integration；
- 最小 PathSolver；
- GPU variant execution。

## 14. 当前 Git 状态

已合并恢复 PR：

```text
PR #3
origin/main = 118d5b63482244d9796968cf613653faf678315a
```

恢复分支：

```text
rebuild/pmi-codebook
```

远程仍保留，final commit：

```text
42d9ed4f89f254cd6983f9715cae4a1400592458
```

合并后验证：

- recovery final commit 是 `origin/main` ancestor；
- recovery final tree 与 main tree diff 为空；
- 13 个文件全部在 main；
- working tree clean。

## 15. 后续正式工作目录

以后不要继续在事故现场目录工作：

```text
/workspace/mMIMO_sleep
```

当前可信工作目录：

```text
/workspace/mMIMO_sleep_rebuild_20260818
```

下一步应从最新 main 创建：

```text
feature/ginza-single-ue
```

命令：

```bash
cd /workspace/mMIMO_sleep_rebuild_20260818
git fetch origin main
git switch -c feature/ginza-single-ue origin/main
```

创建后必须确认：

```bash
git branch --show-current
git rev-parse HEAD
git rev-parse origin/main
git status --short
```

在确认分支创建成功前，不应开始 Ginza notebook 开发。

## 16. 下一阶段计划

下一阶段目标：在独立 feature branch 中实现 Ginza 单 UE normal/sleep SINR 验证。

建议分步进行：

1. 复用 Ginza scene asset 和项目 scene configuration。
2. 放置一个合法随机 UE。
3. 配置 TX/UE 与双极化 PlanarArray。
4. 用 Sionna RT PathSolver 对该 UE 执行一次路径搜索。
5. 从 paths 生成 channel `H`。
6. 在 valid codebook beams 中选择 normal PMI `(i11,i12,i2)`。
7. 验证 selected normal weight 的端口顺序与波束方向。
8. 用 right-half mask 生成同一 PMI 的 sleep weight。
9. 在同一个 `H` 上应用 normal 和 sleep weights。
10. 计算 normal/sleep effective channel、received power、noise 和 SINR。
11. 可视化到达 UE 的 paths/rays，辅助检查空间 beam 是否合理。
12. 后续再加入 SINR→MCS、ΔMCS label 和数据生成循环。

当前阶段只做单 UE end-to-end sanity check，不立即扩展为全区域随机 UE 数据集。

## 17. 今日重要经验

1. 未提交的重要代码不能执行 `git reset --hard`。
2. 在结构性 Git 操作前，至少先 `git status`、`git diff`、`git add` 或创建 archive。
3. 恢复时应使用隔离 clone，不继续污染事故现场。
4. tests 可以作为丢失实现的 executable specification。
5. 每个恢复阶段只改一个模块并独立 commit，便于审计。
6. Docker image 是环境的可重复 artifact；Network Volume `.venv` 不是。
7. Binary-extension 顶层错误可能具有误导性，要查看 traceback `__cause__` 和 `ldd`。
8. CPU image build smoke test不能替代真实 GPU acceptance。
9. Array topology、port ordering、beam ordering 与 Sionna ordering 必须分别测试。
10. Notebook canonical 文件应清空 outputs，执行验证写到 `/tmp` 副本。
11. Algorithm code 和 Docker/CI 改动应使用不同 PR。
12. Element pattern、array factor 和 Sionna scene 各有明确职责边界，不能混在一个模块中。

## 18. 明日恢复工作快速入口

```bash
cd /workspace/mMIMO_sleep_rebuild_20260818

git fetch origin main
git branch --show-current
git status --short

/opt/venv/bin/python /opt/smoke_test.py --mode gpu

PYTHONPATH="$PWD/src" /opt/venv/bin/python -B -m pytest \
    -p no:cacheprovider tests/ -q
```

预期：

```text
GPU smoke test: PASS
pytest: 142 passed
working tree: clean
```

随后确认或创建：

```text
feature/ginza-single-ue
```

再开始新的 Ginza 单 UE notebook。

---

**今日结论：** 环境与代码恢复均已完成，`main` 已重新成为可信基线。下一阶段可以在独立 feature branch 上继续进行 Ginza 单 UE normal/sleep SINR 的端到端开发。
