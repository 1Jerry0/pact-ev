"""利润口径的缺供价格：把原脚本中的“当日最高零售总价 × 惩罚倍数”换成“逐小时零售总价”。
HK(v) 返回一个数值上等于 max(v) 的浮点数（写入输出表时与原来相同），但与任何倍数相乘时返回逐小时向量 v，
因而原脚本里的 pen = 倍数 × tot × (1 + 1e-5 (23 − t)) 变为 v × (1 + 1e-5 (23 − t))，与倍数无关。
由 make_profit_copies.py 自动注入到本目录下各脚本的副本中。"""
import numpy as np


class HK(float):
    def __new__(cls, v):
        v = np.asarray(v, float)
        obj = super().__new__(cls, float(v.max()))
        obj.vec = v
        return obj

    def __rmul__(self, other):          # 倍数 × tot：忽略倍数，返回逐小时零售总价
        return self.vec.copy()

    def __mul__(self, other):           # tot × 数组 / 标量
        return self.vec * other

    def __reduce__(self):               # 供 joblib 跨进程传递
        return (HK, (self.vec,))
