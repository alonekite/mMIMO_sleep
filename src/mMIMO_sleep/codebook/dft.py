"""为均匀平面阵列（UPA）生成 DFT 码本。

码字按照下式展平为一维波束索引：

``beam_index = ((i12 * num_horizontal_beams) + i11) * num_i2 + i2``

其中，``i12`` 是垂直波束索引，``i11`` 是水平波束索引，``i2`` 是两个
极化之间的共相位（co-phasing）索引。展开后 ``i2`` 变化最快。

物理端口采用极化优先（polarization-major）顺序：先存放 polarization 0
的全部端口，再存放 polarization 1 的全部端口。每个极化块内部采用行优先
顺序，并且水平索引变化最快。
"""

from __future__ import annotations

from typing import Final

import torch

from mMIMO_sleep.array_config import ArrayConfig

_COMPLEX_DTYPES: Final = (torch.complex64, torch.complex128)


def _validate_positive(name: str, value: int) -> None:
    """检查计数类参数是否为正数。"""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def dft_vector(
    num_antennas: int,
    beam_index: int,
    num_beams: int,
    *,
    phase_sign: int = 1,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """生成一个单位范数的过采样 DFT 导向向量。

    Args:
        num_antennas: 当前轴上的物理天线阵元数。
        beam_index: DFT 波束索引，取值范围为 ``[0, num_beams)``。
        num_beams: 当前轴上采样的波束数。该值可以大于 ``num_antennas``，
            从而生成过采样码本。
        phase_sign: DFT 相位递进的符号，只能取 ``+1`` 或 ``-1``。应与
            仿真器采用的阵列坐标和角度约定保持一致。
        dtype: 输出复数张量的数据类型。
        device: 返回张量所在的 PyTorch 设备。

    Returns:
        形状为 ``[num_antennas]`` 的复数张量，且模平方之和为 1。
    """
    _validate_positive("num_antennas", num_antennas)
    _validate_positive("num_beams", num_beams)
    if not 0 <= beam_index < num_beams:
        raise ValueError(
            f"beam_index must be in [0, {num_beams}), got {beam_index}."
        )
    if phase_sign not in (-1, 1):
        raise ValueError(f"phase_sign must be +1 or -1, got {phase_sign}.")
    if dtype not in _COMPLEX_DTYPES:
        raise TypeError(f"dtype must be complex64 or complex128, got {dtype}.")

    # 相位本身是实数；其精度应与最终复数类型的实部精度一致。
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    element_index = torch.arange(
        num_antennas,
        dtype=real_dtype,
        device=device,
    )
    # 第 n 个阵元的相位为 phase_sign * 2π * n * beam_index / num_beams。
    # 当 num_beams > num_antennas 时，相邻波束的空间频率间隔更小，
    # 因而形成过采样 DFT 码本。
    phase = (
        phase_sign
        * 2.0
        * torch.pi
        * element_index
        * beam_index
        / num_beams
    )
    vector = torch.exp(1j * phase).to(dtype=dtype)
    # 除以 sqrt(num_antennas)，保证 sum(|vector|^2) = 1。
    return vector / torch.sqrt(
        torch.tensor(num_antennas, dtype=real_dtype, device=device)
    )


def generate_dft_codebook(
    config: ArrayConfig,
    num_vertical_beams: int,
    num_horizontal_beams: int,
    num_i2: int = 4,
    *,
    phase_sign: int = 1,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """生成双极化子阵 DFT 码本。

    首先在逻辑子阵网格上构造可分离的二维 DFT 权重。随后，每个逻辑子阵的
    权重沿垂直方向复制到 ``config.elements_per_subarray`` 个物理阵元，并
    扩展到两个极化。polarization 0 作为参考相位，polarization 1 相对于
    polarization 0 施加由 ``i2`` 决定的共相位旋转。

    Args:
        config: 阵列配置。目前仅支持 ``num_polarizations=2``。
        num_vertical_beams: 垂直方向的 DFT 波束数。
        num_horizontal_beams: 水平方向的 DFT 波束数。
        num_i2: 两个极化之间可选的共相位值数量。
        phase_sign: DFT 相位递进采用的符号约定。
        dtype: 输出复数张量的数据类型。
        device: 返回张量所在的 PyTorch 设备。

    Returns:
        形状为
        ``(num_vertical_beams * num_horizontal_beams * num_i2,
        config.num_physical_ports)`` 的复数张量。波束顺序为
        ``(i12, i11, i2)``，其中 ``i2`` 变化最快；端口采用极化优先顺序，
        每个极化块内部采用行优先顺序。
    """
    if not isinstance(config, ArrayConfig):
        raise TypeError(f"config must be ArrayConfig, got {type(config)}.")
    if config.num_polarizations != 2:
        raise NotImplementedError(
            "Only num_polarizations=2 is currently supported, "
            f"got {config.num_polarizations}."
        )
    if config.port_order != "polarization-major":
        raise NotImplementedError(
            "Only 'polarization-major' port_order is currently supported, "
            f"got {config.port_order!r}."
        )
    for name, value in (
        ("num_vertical_beams", num_vertical_beams),
        ("num_horizontal_beams", num_horizontal_beams),
        ("num_i2", num_i2),
    ):
        _validate_positive(name, value)
    if phase_sign not in (-1, 1):
        raise ValueError(f"phase_sign must be +1 or -1, got {phase_sign}.")
    if dtype not in _COMPLEX_DTYPES:
        raise TypeError(f"dtype must be complex64 or complex128, got {dtype}.")

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    eps = config.elements_per_subarray

    # 在逻辑子阵网格的两个轴上分别生成一维 DFT 导向向量。
    # vertical_vectors.shape = [num_vertical_beams, num_subarray_rows]
    vertical_vectors = torch.stack(
        [
            dft_vector(
                config.num_subarray_rows,
                i12,
                num_vertical_beams,
                phase_sign=phase_sign,
                dtype=dtype,
                device=device,
            )
            for i12 in range(num_vertical_beams)
        ]
    )
    # horizontal_vectors.shape = [num_horizontal_beams, num_horizontal]
    horizontal_vectors = torch.stack(
        [
            dft_vector(
                config.num_horizontal,
                i11,
                num_horizontal_beams,
                phase_sign=phase_sign,
                dtype=dtype,
                device=device,
            )
            for i11 in range(num_horizontal_beams)
        ]
    )

    # 利用二维 DFT 的可分离性，通过外积得到逻辑二维权重：
    # [num_vertical_beams, num_horizontal_beams,
    #  num_subarray_rows, num_horizontal]。
    # 这里 i/j 对应垂直/水平波束索引，v/h 对应逻辑阵列的行/列索引。
    logical_weights = torch.einsum(
        "iv,jh->ijvh",
        vertical_vectors,
        horizontal_vectors,
    )

    # 一个逻辑子阵在垂直方向包含 eps 个物理阵元。repeat_interleave
    # 让这些物理阵元共享相同的逻辑子阵权重，不额外引入相位差。
    physical_weights = logical_weights.repeat_interleave(eps, dim=2)

    # 将物理阵元按行优先顺序展平（水平列索引变化最快）。
    # 一维 DFT 向量已经按逻辑阵元数归一化，但垂直复制 eps 次、再复制到
    # num_polarizations 个极化后，总功率会相应增加。因此再乘以
    # 1/sqrt(eps * num_polarizations)，使最终双极化码字的总功率仍为 1，
    # 且每个极化承载 1/num_polarizations 的功率。
    scale = 1.0 / torch.sqrt(
        torch.tensor(
            eps * config.num_polarizations,
            dtype=real_dtype,
            device=device,
        )
    )
    physical_weights = physical_weights.reshape(
        num_vertical_beams,
        num_horizontal_beams,
        config.num_physical_elements,
    )
    physical_weights = physical_weights * scale

    # 构造 polarization 1 相对于 polarization 0 的 i2 共相位旋转：
    # exp(+j * 2π * i2 / num_i2)。正负号是本项目明确采用的约定。
    i2_index = torch.arange(
        num_i2,
        dtype=real_dtype,
        device=device,
    )
    i2_phase = torch.exp(1j * 2.0 * torch.pi * i2_index / num_i2).to(dtype=dtype)

    # 为每个空间波束扩展 num_i2 个版本。pol0 是参考极化，保持不变；
    # pol1 对每个 i2 施加对应的复相位。
    # 两者形状均为 [num_vertical_beams, num_horizontal_beams,
    # num_i2, num_physical_elements]。
    pol0 = physical_weights.unsqueeze(2).expand(
        num_vertical_beams,
        num_horizontal_beams,
        num_i2,
        config.num_physical_elements,
    )
    pol1 = pol0 * i2_phase.view(num_i2, 1)

    # 沿最后一维拼接两个极化块，形成 polarization-major 端口顺序：
    # [pol0 的全部物理阵元, pol1 的全部物理阵元]。
    codebook = torch.cat([pol0, pol1], dim=-1)

    # 将前三个维度展平为波束维度。由于 PyTorch 按行优先顺序 reshape，
    # 最后一维 i2 变化最快，因此最终顺序为 (i12, i11, i2)。
    return codebook.reshape(
        num_vertical_beams * num_horizontal_beams * num_i2,
        config.num_physical_ports,
    )
