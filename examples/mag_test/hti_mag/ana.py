import sys, json, numpy as np
import scipy.constants as pc
sys.path.insert(0, '/Users/du_tt/hzt/soft/dpti')
from dpti.einstein import (compute_lambda, compute_spring,
                            compute_spin_lambda, compute_spin_spring)
from dpti.lib import lmp

print("=" * 60)
print("Bug 1: key 命名检查")
print("=" * 60)
with open('in.json') as f:
    jdata = json.load(f)
search = ["spin_spring_k", "s_spring_k", "spring_k_spin"]
print(f"JSON 实际 key : 'spring_spin_k' = {jdata.get('spring_spin_k')}")
print(f"代码搜索 key  : {search}")
print(f"是否能找到    : {any(k in jdata for k in search)}  <- BUG")

print()
print("=" * 60)
print("Bug 2: include_spin_kinetic flag")
print("=" * 60)
print("spin_ke 在 loop 内已无条件添加，if include_spin_kinetic 块全注释 -> flag 无效")

print()
print("=" * 60)
print("Level 3: 公式数值验证")
print("=" * 60)
temp          = jdata["temp"]
mass_map      = jdata["mass_map"]
spin_mass     = jdata["spin_mass_map"]
spin_map      = jdata["spin_map"]
spring_k      = jdata["spring_k"]
spring_spin_k = jdata["spring_spin_k"]

m_spring_k      = [spring_k * m for m in mass_map]
m_spring_spin_k = [spring_spin_k * m * spin_mass for m in mass_map]

with open('conf.lmp') as f:
    sys_data = lmp.to_system_data(f.read().split("\n"))
vol     = np.linalg.det(sys_data["cell"])
natoms  = sys_data["atom_numbs"]
total_N = sum(natoms)

Lambda_k     = [compute_lambda(temp, m)      for m in mass_map]
Lambda_E     = [compute_spring(temp, k)      for k in m_spring_k]
Lambda_E_cm  =  compute_spring(temp, spring_k)
Lambda_S_kin = [compute_spin_lambda(temp, m * spin_mass) for m in mass_map]
Lambda_S_ref = [compute_spin_spring(temp, k) for k in m_spring_spin_k]

print(f"T={temp}K  N={total_N}  m={mass_map[0]}  k_l={spring_k}  k_s={spring_spin_k}  mu_s={spin_mass}")
print(f"Vol={vol:.2f} A^3\n")

# 代码逻辑复现 (手动传入正确的 spring_spin_k)
fe = 0.0; sum_m = 0.0
for idx, ii in enumerate(natoms):
    fe += 3.0 * ii * np.log(Lambda_k[idx])
    fe += 3.0 * ii * np.log(Lambda_E[idx])
    sum_m += mass_map[idx] * ii
    if spin_map[idx]:
        fe += 3.0 * ii * np.log(Lambda_S_ref[idx])
        fe += 3.0 * ii * np.log(Lambda_S_kin[idx])
fe -= 3.0 * np.log(Lambda_E_cm)
fe -= 1.5 * np.log(sum_m)
fe += np.log(total_N / (vol * (pc.angstrom**3)))
fe_code = fe * pc.Boltzmann * temp / pc.electron_volt / total_N

# 解析推导 F = 3kBT*ln(hbar*omega/kBT) 逐项
kBT = pc.Boltzmann * temp / pc.electron_volt
N   = total_N
latt_ke     = 3*N * np.log(Lambda_k[0])
latt_pe     = 3*(N-1) * np.log(Lambda_E[0])
frenkel_vol = np.log(N / (vol * pc.angstrom**3))
frenkel_cm  = 1.5 * np.log(N / sum_m)
spin_pe     = 3*N * np.log(Lambda_S_ref[0])
spin_ke     = 3*N * np.log(Lambda_S_kin[0])
fe_analytic = (latt_ke+latt_pe+frenkel_vol+frenkel_cm+spin_pe+spin_ke) * kBT / N

print(f"{'项':<32} {'无量纲/atom':>12}")
print(f"{'格子动能  3N*ln(Lk)/N':<32} {latt_ke/N:+12.6f}")
print(f"{'格子势能  3(N-1)*ln(LE)/N':<32} {latt_pe/N:+12.6f}")
print(f"{'Frenkel体积 ln(N/V)/N':<32} {frenkel_vol/N:+12.6f}")
print(f"{'Frenkel质心修正/N':<32} {frenkel_cm/N:+12.6f}")
print(f"{'自旋势能  3N*ln(LS_ref)/N':<32} {spin_pe/N:+12.6f}")
print(f"{'自旋动能  3N*ln(LS_kin)/N':<32} {spin_ke/N:+12.6f}")
print("-"*46)
print(f"解析 F_ref/atom = {fe_analytic:+.8f} eV")
print(f"代码 F_ref/atom = {fe_code:+.8f} eV")
diff = abs(fe_analytic - fe_code)
print(f"差值            =  {diff:.2e} eV  {'[公式一致]' if diff<1e-10 else '[不一致!]'}")

# 自旋部分: 独立用经典谐振子公式验证
omega_s   = np.sqrt(spring_spin_k * pc.electron_volt
                    / (mass_map[0] * spin_mass * 1e-3 / pc.Avogadro))
hbar      = pc.Planck / (2*np.pi)
fe_spin_ho   = 3 * kBT * np.log(hbar * omega_s / (pc.Boltzmann * temp))
fe_spin_code = (spin_pe + spin_ke) * kBT / N
print(f"\n自旋部分独立验证 (经典谐振子 3kBT*ln(hbar*w/kBT)):")
print(f"  解析: {fe_spin_ho:.8f} eV/atom")
print(f"  代码: {fe_spin_code:.8f} eV/atom")
print(f"  差值: {abs(fe_spin_ho-fe_spin_code):.2e} eV")

# 温度依赖性: 验证 dF/dT 的 -(3+3)kB = -6kB/atom (自旋+格子各3)
T2 = 400
kBT2 = pc.Boltzmann * T2 / pc.electron_volt
Lambda_S_kin2 = [compute_spin_lambda(T2, m * spin_mass) for m in mass_map]
Lambda_S_ref2 = [compute_spin_spring(T2, k) for k in m_spring_spin_k]
fe2 = 0.0; sum_m2 = 0.0
Lambda_k2 = [compute_lambda(T2, m) for m in mass_map]
Lambda_E2 = [compute_spring(T2, k) for k in m_spring_k]
Lambda_E_cm2 = compute_spring(T2, spring_k)
for idx, ii in enumerate(natoms):
    fe2 += 3.0*ii*(np.log(Lambda_k2[idx]) + np.log(Lambda_E2[idx]))
    sum_m2 += mass_map[idx]*ii
    if spin_map[idx]:
        fe2 += 3.0*ii*(np.log(Lambda_S_ref2[idx]) + np.log(Lambda_S_kin2[idx]))
fe2 -= 3.0*np.log(Lambda_E_cm2) + 1.5*np.log(sum_m2)
fe2 += np.log(total_N/(vol*(pc.angstrom**3)))
fe2_code = fe2 * pc.Boltzmann * T2 / pc.electron_volt / total_N

dFdT_numeric = (fe2_code - fe_code) / (T2 - temp)
dFdT_expect  = -6 * pc.Boltzmann / pc.electron_volt   # -6kB (latt 3 + spin 3)
print(f"\ndF/dT 温度依赖验证 (T={temp}->400K):")
print(f"  数值 dF/dT  = {dFdT_numeric:.6e} eV/K")
print(f"  理论 -6k_B  = {dFdT_expect:.6e} eV/K")
print(f"  差值        = {abs(dFdT_numeric-dFdT_expect):.2e} eV/K")
