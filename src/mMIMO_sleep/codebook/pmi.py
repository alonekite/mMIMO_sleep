"""PMI conversion utilities for the rank-one DFT codebook.

用于 rank-one（秩一）DFT 码本的 PMI 转换工具。
主要完成两件事：
1. 将 PMI(i11, i12, i2) 转换成码本中的一维 beam index；
2. 将一维 beam index 反向恢复成 PMI(i11, i12, i2)。
"""

# 让类型注解在运行时不立即求值。
# 这样可以减少一些前向引用（forward reference）带来的问题，
# 也是现代 Python 项目中比较常见的写法。
from __future__ import annotations

# dataclass 可以自动生成 __init__、__repr__、__eq__ 等常用方法，
# 很适合 PMI 这种只保存少量参数的数据结构。
from dataclasses import dataclass


# frozen=True:
#   PMI 对象创建后不能再修改 i11/i12/i2，避免索引被意外改动。
#
# slots=True:
#   Python 不再为每个对象创建普通的 __dict__，
#   可以减少内存开销，同时限制对象只能拥有这里声明的字段。
@dataclass(frozen=True, slots=True)
class PMI:
    """Rank-one PMI for the simplified DFT codebook.

    简化 DFT 码本中的 rank-one PMI。

    三个索引分别表示：
    - i11：horizontal beam index，水平方向波束索引
    - i12：vertical beam index，垂直方向波束索引
    - i2 ：cross-polarization co-phasing index，双极化之间的共相位索引

    将三维 PMI 展平成一维码本索引时，采用的顺序是：
        (i12, i11, i2)

    其中 i2 是变化最快的维度。
    也就是说：
        先改变 i2，
        i2 遍历完以后再改变 i11，
        i11 遍历完以后再改变 i12。
    """

    # 水平波束索引
    i11: int

    # 垂直波束索引
    i12: int

    # 双极化共相位索引
    i2: int


def _validate_dimensions(
    num_horizontal_beams: int,
    num_vertical_beams: int,
    num_i2: int,
) -> None:
    """检查码本三个维度是否合法。

    这里只检查两个基本条件：
    1. 三个数量必须都是 int；
    2. 三个数量都必须大于 0。

    函数名前面的下划线 "_" 表示：
    这是模块内部使用的辅助函数，不是主要对外接口。
    """

    # all(...) 要求里面所有条件都为 True。
    # 这里逐个检查三个输入是不是 Python 的 int。
    if not all(
        isinstance(x, int) for x in (num_horizontal_beams, num_vertical_beams, num_i2)
    ):
        raise TypeError("Beam counts must be integers.")

    # 水平波束数量必须为正数。
    if num_horizontal_beams <= 0:
        raise ValueError("num_horizontal_beams must be positive.")

    # 垂直波束数量必须为正数。
    if num_vertical_beams <= 0:
        raise ValueError("num_vertical_beams must be positive.")

    # i2 的候选数量必须为正数。
    if num_i2 <= 0:
        raise ValueError("num_i2 must be positive.")


def pmi_to_beam_index(
    pmi: PMI,
    *,
    num_horizontal_beams: int,
    num_vertical_beams: int,
    num_i2: int,
) -> int:
    """Convert ``PMI(i11, i12, i2)`` to the flat codebook row index.

    将 PMI(i11, i12, i2) 转换成展平后的码本行索引 beam_index。

    索引顺序为：
        (i12, i11, i2)

    并且 i2 是变化最快的维度。

    例如：
        num_horizontal_beams = 32
        num_vertical_beams   = 8
        num_i2               = 4

    对 PMI(i11=17, i12=3, i2=2)：

        beam_index
        = (i12 * num_horizontal_beams + i11) * num_i2 + i2
        = (3 * 32 + 17) * 4 + 2
        = 454

    注意：
    这里的 "*" 是 Python 的 keyword-only 参数分隔符。
    "*" 后面的参数必须显式写参数名，例如：

        pmi_to_beam_index(
            pmi,
            num_horizontal_beams=32,
            num_vertical_beams=8,
            num_i2=4,
        )
    """

    # 首先确认码本各个维度的数量合法。
    _validate_dimensions(num_horizontal_beams, num_vertical_beams, num_i2)

    # i11 的合法范围：
    # 0 <= i11 < num_horizontal_beams
    if not 0 <= pmi.i11 < num_horizontal_beams:
        raise ValueError(
            f"i11 must be in [0, {num_horizontal_beams}), got {pmi.i11}."
        )

    # i12 的合法范围：
    # 0 <= i12 < num_vertical_beams
    if not 0 <= pmi.i12 < num_vertical_beams:
        raise ValueError(
            f"i12 must be in [0, {num_vertical_beams}), got {pmi.i12}."
        )

    # i2 的合法范围：
    # 0 <= i2 < num_i2
    if not 0 <= pmi.i2 < num_i2:
        raise ValueError(f"i2 must be in [0, {num_i2}), got {pmi.i2}.")

    # 把三维索引 (i12, i11, i2) 展平为一维 beam_index。
    #
    # 可以分成两步理解：
    #
    # 第一步：
    #   pmi.i12 * num_horizontal_beams + pmi.i11
    #
    # 先把二维的 (i12, i11) 展平。
    #
    # 第二步：
    #   上一步结果 * num_i2 + pmi.i2
    #
    # 再把 i2 放到最内层。
    #
    # 因此最终顺序就是：
    #   (i12, i11, i2)
    # 并且 i2 变化最快。
    return (pmi.i12 * num_horizontal_beams + pmi.i11) * num_i2 + pmi.i2


def beam_index_to_pmi(
    beam_index: int,
    *,
    num_horizontal_beams: int,
    num_vertical_beams: int,
    num_i2: int,
) -> PMI:
    """Convert a flat codebook row index to ``PMI(i11, i12, i2)``.

    将一维 beam_index 反向恢复成 PMI(i11, i12, i2)。

    这是 pmi_to_beam_index() 的逆操作，使用完全相同的索引顺序：
        (i12, i11, i2)
    """

    # 首先检查三个维度是否合法。
    _validate_dimensions(num_horizontal_beams, num_vertical_beams, num_i2)

    # 总码本大小：
    #
    #   水平波束数 × 垂直波束数 × i2 候选数
    #
    # 例如：
    #   32 × 8 × 4 = 1024
    #
    # 那么合法 beam_index 就是 0 到 1023。
    total = num_horizontal_beams * num_vertical_beams * num_i2

    # beam_index 必须落在整个码本的合法范围内。
    if not 0 <= beam_index < total:
        raise ValueError(
            f"beam_index must be in [0, {total}), got {beam_index}."
        )

    # divmod(a, b) 同时返回：
    #   a // b   -> 商
    #   a % b    -> 余数
    #
    # 因为 i2 是变化最快的维度，
    # 所以 beam_index 除以 num_i2 后：
    #
    #   余数 = i2
    #   商   = 剩下的 (i12, i11) 展平索引
    #
    # 等价写法：
    #   tmp = beam_index // num_i2
    #   i2  = beam_index % num_i2
    tmp, i2 = divmod(beam_index, num_i2)

    # 接下来对剩余的 tmp 再做一次拆分。
    #
    # i11 是 (i12, i11) 中变化较快的维度，因此：
    #
    #   tmp // num_horizontal_beams -> i12
    #   tmp %  num_horizontal_beams -> i11
    #
    # 等价写法：
    #   i12 = tmp // num_horizontal_beams
    #   i11 = tmp % num_horizontal_beams
    i12, i11 = divmod(tmp, num_horizontal_beams)

    # 用反解得到的三个索引创建一个 PMI 对象并返回。
    return PMI(i11=i11, i12=i12, i2=i2)
