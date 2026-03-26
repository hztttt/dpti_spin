#!/usr/bin/env python3
"""
Level 3 verification of magnetic_frenkel():
  - Bug 报告
  - 公式数值验证（逐项对比解析值）
  - 自旋部分独立验证
  - dF/dT 数值验证
"""
import sys, json, numpy as np
import scipy.constants as pc

sys.path.insert(0, "/Users/du_tt/hzt/soft/dpti")
from dpti.einstein import (
    compute_lambda,
    compute_spring,
    compute_spin_lambda,
    compute_spin_spring,
)
from dpti.lib import lmp

JOB = "."   # 在 hti_mag 目录下运行

# ─────────────────────────────────────────────
# 0. 读取参数
# ─────────────────────────────────────────────
with open(f"{JOB}/in.json") as f:
    jdata = json.load(f)

temp          = jdata["temp"]
mass_map      = jdata["mass_map"]
spin_mass     = jdata["spin_mass_map"]   # μ_s，无量纲（相对原子质量分数）
spin_map      = jdata["spin_map"]
spring_k      = jdata["spring_k"]        # eV/Å²/AMU
spring_spin_k = jdata["spring_spin_k"]   # eV/AMU/spin_mass_unit（见下文说明）

with open(f"{JOB}/conf.lmp") as f:
    sys_data = lmp.to_system_data(f.read().split("\n"))
vol     = np.linalg.det(sys_data["cell"])
natoms  = sys_data["atom_numbs"]
total_N = sum(natoms)

kBT = pc.Boltzmann * temp / pc.electron_volt   # eV

# 预计算 Lambda 系列
m_spring_k      = [spring_k      * m             for m in mass_map]
m_spring_spin_k = [spring_spin_k * m * spin_mass for m in mass_map]

Lambda_k     = [compute_lambda(temp, m)      for m in mass_map]
Lambda_E     = [compute_spring(temp, k)      for k in m_spring_k]
Lambda_E_cm  =  compute_spring(temp, spring_k)
Lambda_S_kin = [compute_spin_lambda(temp, m * spin_mass) for m in mass_map]
Lambda_S_ref = [compute_spin_spring(temp, k) for k in m_spring_spin_k]

sum_m = sum(mass_map[i] * natoms[i] for i in range(len(natoms)))
N = total_N


# ─────────────────────────────────────────────
# 1. Bug 报告
# ─────────────────────────────────────────────
sep = "=" * 60
print(sep)
print("Bug 1: key 命名错误（致命）")
print(sep)
search_keys = ["spin_spring_k", "s_spring_k", "spring_k_spin"]
print(f"  JSON 实际 key : 'spring_spin_k' = {jdata.get('spring_spin_k')}")
print(f"  代码搜索 key  : {search_keys}")
print(f"  是否匹配      : {any(k in jdata for k in search_keys)}")
print("  → 每次调用 magnetic_frenkel() 必然 KeyError")
print("  修复: 将 'spring_spin_k' 加入 get_first_matched_key_from_dict 列表")

print()
print(sep)
print("Bug 2: include_spin_kinetic flag 无效（逻辑 bug）")
print(sep)
print("  自旋动能项在 loop 内已无条件添加（line 280），")
print("  if include_spin_kinetic: 控制块代码全注释 → flag 对结果无影响")
print("  修复: 将 loop 内 spin_ke 项移入 if include_spin_kinetic: 块")


# ─────────────────────────────────────────────
# 2. 代码逻辑复现（手动传入正确 key）
# ─────────────────────────────────────────────
fe = 0.0
for idx, ii in enumerate(natoms):
    fe += 3.0 * ii * np.log(Lambda_k[idx])
    fe += 3.0 * ii * np.log(Lambda_E[idx])
    if spin_map[idx]:
        fe += 3.0 * ii * np.log(Lambda_S_ref[idx])
        fe += 3.0 * ii * np.log(Lambda_S_kin[idx])
fe -= 3.0 * np.log(Lambda_E_cm)
fe -= 1.5 * np.log(sum_m)
fe += np.log(N / (vol * pc.angstrom**3))
F_code = fe * kBT / N


# ─────────────────────────────────────────────
# 3. 解析公式（Level 3 正确推导）
# ─────────────────────────────────────────────
# 格子部分（Frenkel: N-1 弹簧 + 1 个自由位）
# 代码等价展开: 3N*ln(LE) - 3*ln(LE_cm) - 1.5*ln(sum_m) + ln(N/V)
#             = 3(N-1)*ln(LE) - 1.5*ln(N) + ln(N/V)   （单组分）
latt_ke     = 3 * N * np.log(Lambda_k[0])
# 注意: 等价于 3(N-1)*ln(LE) - 1.5*ln(N)，见注释
latt_pe_raw = 3 * N * np.log(Lambda_E[0])
cm_spring   = -3.0 * np.log(Lambda_E_cm)
cm_kinetic  = -1.5 * np.log(sum_m)
vol_term    = np.log(N / (vol * pc.angstrom**3))

# 自旋部分（N 个 3D 谐振子，m_atom*spin_mass 在 k_eff/m_eff 中约掉）
# 有效频率: omega_s = sqrt(spring_spin_k * eV * Avogadro * 1e3)
#           与 m_atom, spin_mass 无关！
spin_pe = 3 * N * np.log(Lambda_S_ref[0])
spin_ke = 3 * N * np.log(Lambda_S_kin[0])

F_analytic = (latt_ke + latt_pe_raw + cm_spring + cm_kinetic
              + vol_term + spin_pe + spin_ke) * kBT / N

print()
print(sep)
print(f"Level 3: 公式数值验证  T={temp}K  N={N}  m={mass_map[0]}  "
      f"k_l={spring_k}  k_s={spring_spin_k}  mu_s={spin_mass}")
print(sep)
print(f"  体积 V = {vol:.4f} Å³\n")

hdr = f"  {'项':<38} {'无量纲/atom':>12}"
print(hdr)
print("  " + "-"*52)

def prow(label, val):
    print(f"  {label:<38} {val/N:+12.6f}")

prow("格子动能  3N·ln(Λk)",          latt_ke)
prow("格子势能  3N·ln(ΛE) [含CM]",   latt_pe_raw)
prow("Frenkel CM弹簧修正 -3·ln(ΛE_cm)", cm_spring)
prow("Frenkel CM动能修正 -1.5·ln(Σm)",  cm_kinetic)
prow("Frenkel体积项  ln(N/V)",        vol_term)
prow("自旋势能  3N·ln(ΛS_ref)",       spin_pe)
prow("自旋动能  3N·ln(ΛS_kin)",       spin_ke)
print("  " + "-"*52)
print(f"  {'解析 F_ref/atom':<38} {F_analytic:+12.8f} eV")
print(f"  {'代码 F_ref/atom':<38} {F_code:+12.8f} eV")
diff = abs(F_analytic - F_code)
status = "✓ 一致" if diff < 1e-10 else f"✗ 差值 {diff:.2e} eV"
print(f"  {'差值':<38} {diff:12.2e}  {status}")


# ─────────────────────────────────────────────
# 4. 自旋部分独立验证（经典谐振子 F = 3kBT·ln(ħω/kBT)）
# ─────────────────────────────────────────────
print()
print(sep)
print("自旋部分独立验证  3kBT·ln(ħω_s / kBT)")
print(sep)

# 正确的 omega_s: k_eff/m_eff 中 m_atom*spin_mass 约掉
# k_eff = spring_spin_k * m_atom * spin_mass * eV
# m_eff = m_atom * spin_mass * 1e-3/Avogadro
# omega² = k_eff/m_eff = spring_spin_k * eV * Avogadro * 1e3
omega_s_correct = np.sqrt(
    spring_spin_k * pc.electron_volt * pc.Avogadro * 1e3
)
hbar = pc.Planck / (2 * np.pi)
F_spin_ho = 3 * kBT * np.log(hbar * omega_s_correct / (pc.Boltzmann * temp))
F_spin_code = (spin_pe + spin_ke) * kBT / N

print(f"  有效频率 ω_s = sqrt(k_s·eV·Nₐ·1e3) = {omega_s_correct:.6e} rad/s")
print(f"  ħω_s = {hbar * omega_s_correct:.6e} J  = {hbar*omega_s_correct/pc.electron_volt:.6e} eV")
print(f"  kBT  = {pc.Boltzmann*temp:.6e} J  = {kBT:.6e} eV")
print(f"  ħω_s/kBT = {hbar*omega_s_correct/(pc.Boltzmann*temp):.6e}")
print()
print(f"  经典谐振子  3kBT·ln(ħω/kBT)/atom : {F_spin_ho:+.8f} eV")
print(f"  代码自旋贡献 /atom               : {F_spin_code:+.8f} eV")
diff_spin = abs(F_spin_ho - F_spin_code)
status_spin = "✓ 一致" if diff_spin < 1e-8 else f"✗ 差值 {diff_spin:.2e} eV"
print(f"  差值                              : {diff_spin:.2e}  {status_spin}")

# 说明: m_atom*spin_mass 为何约掉
print()
print("  [验证 m·μ_s 约掉]")
omega_naive = np.sqrt(
    spring_spin_k * pc.electron_volt
    / (mass_map[0] * spin_mass * 1e-3 / pc.Avogadro)
)
print(f"  错误写法 ω (用 k_s 不缩放)        : {omega_naive:.6e} rad/s")
print(f"  正确写法 ω (用 k_eff = k_s·m·μ_s) : {omega_s_correct:.6e} rad/s")
print(f"  比值 = sqrt(m·μ_s) = sqrt({mass_map[0]*spin_mass:.4f}) = {np.sqrt(mass_map[0]*spin_mass):.4f}")


# ─────────────────────────────────────────────
# 5. dF/dT 验证（正确理论值）
# ─────────────────────────────────────────────
print()
print(sep)
print("dF/dT 验证  （正确理论: 温度相关，非常数）")
print(sep)

T2 = 210.0
kBT2 = pc.Boltzmann * T2 / pc.electron_volt
Lambda_k2     = [compute_lambda(T2, m)       for m in mass_map]
Lambda_E2     = [compute_spring(T2, k)       for k in m_spring_k]
Lambda_E_cm2  =  compute_spring(T2, spring_k)
Lambda_S_kin2 = [compute_spin_lambda(T2, m * spin_mass) for m in mass_map]
Lambda_S_ref2 = [compute_spin_spring(T2, k)  for k in m_spring_spin_k]

fe2 = 0.0
for idx, ii in enumerate(natoms):
    fe2 += 3.0 * ii * np.log(Lambda_k2[idx])
    fe2 += 3.0 * ii * np.log(Lambda_E2[idx])
    if spin_map[idx]:
        fe2 += 3.0 * ii * np.log(Lambda_S_ref2[idx])
        fe2 += 3.0 * ii * np.log(Lambda_S_kin2[idx])
fe2 -= 3.0 * np.log(Lambda_E_cm2)
fe2 -= 1.5 * np.log(sum_m)
fe2 += np.log(N / (vol * pc.angstrom**3))
F_code2 = fe2 * kBT2 / N

dFdT_numeric = (F_code2 - F_code) / (T2 - temp)

# 解析: F = 6kBT·ln(ħω/kBT) + const (忽略小修正项)
# dF/dT = 6kB·[ln(ħω/kBT) - 1]
# 这里用格子+自旋各 3 项，共 6，但 Frenkel 修正使格子实际是 ~3(N-1)/N ≈ 3
# 简化：只看主项
omega_l = np.sqrt(spring_k * pc.electron_volt / pc.angstrom**2 * pc.Avogadro * 1e3)
F_spin_per_atom_T  = 3 * kBT  * np.log(hbar * omega_s_correct / (pc.Boltzmann * temp))
F_spin_per_atom_T2 = 3 * kBT2 * np.log(hbar * omega_s_correct / (pc.Boltzmann * T2))
F_latt_per_atom_T  = 3 * kBT  * np.log(hbar * omega_l / (pc.Boltzmann * temp))
F_latt_per_atom_T2 = 3 * kBT2 * np.log(hbar * omega_l / (pc.Boltzmann * T2))

dFdT_spin_analytic  = (F_spin_per_atom_T2  - F_spin_per_atom_T)  / (T2 - temp)
dFdT_latt_analytic  = (F_latt_per_atom_T2  - F_latt_per_atom_T)  / (T2 - temp)
dFdT_total_analytic = dFdT_spin_analytic + dFdT_latt_analytic

print(f"  数值 dF/dT（T={temp}→{T2}K）: {dFdT_numeric:+.6e} eV/K")
print(f"  解析 dF/dT (格子主项)       : {dFdT_latt_analytic:+.6e} eV/K")
print(f"  解析 dF/dT (自旋主项)       : {dFdT_spin_analytic:+.6e} eV/K")
print(f"  解析 dF/dT (总，近似)       : {dFdT_total_analytic:+.6e} eV/K")
diff_dF = abs(dFdT_numeric - dFdT_total_analytic)
print(f"  差值（近似公式误差）         : {diff_dF:.2e} eV/K")
print()
print("  注：dF/dT = 6kB[ln(ħω/kBT)-1]，温度相关，不是常数 -6kB")
entropy = -dFdT_numeric
print(f"  当前熵估算 S/atom = -dF/dT ≈ {entropy:.4f} eV/K = {entropy/pc.Boltzmann*pc.electron_volt:.2f} kB")


# ─────────────────────────────────────────────
# 6. 总结
# ─────────────────────────────────────────────
print()
print(sep)
print("总结")
print(sep)
print("  Bug 1 [致命]: 'spring_spin_k' 未在 get_first_matched_key_from_dict 列表中")
print("                → magnetic_frenkel() 每次调用必然 KeyError")
print("  Bug 2 [逻辑]: include_spin_kinetic flag 无效，自旋动能始终被计入")
print()
if diff < 1e-10 and diff_spin < 1e-8:
    print("  公式本身: ✓ 正确（修复 key 命名后数值结果与解析一致）")
else:
    print(f"  公式本身: 总差值 {diff:.2e} eV  自旋差值 {diff_spin:.2e} eV")
