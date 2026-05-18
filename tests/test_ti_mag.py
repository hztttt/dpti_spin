import json

import numpy as np
import pytest
import scipy.constants as pc

from dpti import ti_mag


def _write_conf(path):
    path.write_text(
        """2 atoms

1 atom types

0.0 1.0 xlo xhi
0.0 1.0 ylo yhi
0.0 1.0 zlo zhi
""",
        encoding="utf-8",
    )


def _write_log(path, temp, total_energy, spin_kinetic=None):
    labels = [
        "Step",
        "KinEng",
        "PotEng",
        "TotEng",
        "Enthalpy",
        "Temp",
        "Press",
        "Volume",
    ]
    if spin_kinetic is not None:
        labels.append("SpinKinEng")
    labels.extend(
        [
            "c_allmsd[1]",
            "c_allmsd[2]",
            "c_allmsd[3]",
            "c_allmsd[4]",
            "c_spinmsd[1]",
            "c_spinmsd[2]",
            "c_spinmsd[3]",
            "c_spinmsd[4]",
        ]
    )

    rows = []
    for step in range(4):
        row = [step, 1.0, 2.0, total_energy, total_energy + 1.0, temp, 0.0, 20.0]
        if spin_kinetic is not None:
            row.append(spin_kinetic)
        row.extend([0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4])
        rows.append(" ".join(str(v) for v in row))

    path.write_text(
        "\n".join(["LAMMPS", " ".join(labels), *rows, "Loop time of 1 on 1 procs"]),
        encoding="utf-8",
    )


def _make_job(tmp_path, include_spin_kinetic=False, with_ske=True):
    job = tmp_path / "new_job"
    job.mkdir()
    _write_conf(job / "conf.lmp")
    settings = {
        "equi_conf": "conf.lmp",
        "mass_map": [1.0],
        "spin_mass_map": 0.01,
        "ens": "nvt",
        "path": "t",
        "stat_skip": 0,
        "stat_bsize": 2,
        "include_spin_kinetic": include_spin_kinetic,
    }
    (job / "ti_settings.json").write_text(json.dumps(settings), encoding="utf-8")
    for idx, temp in enumerate([200.0, 300.0]):
        task = job / f"task.{idx:06d}"
        task.mkdir()
        (task / "thermo.out").write_text(str(temp), encoding="utf-8")
        spin_kinetic = 4.0 + idx if with_ske else None
        _write_log(task / "log.lammps", temp, 10.0 + idx, spin_kinetic)
    return job, settings


def test_gen_lammps_input_outputs_ske_column():
    text = ti_mag._gen_lammps_input(
        conf_file="conf.lmp",
        mass_map=[1.0],
        spin_mass=0.01,
        model="graph.pb",
        nsteps=10,
        timestep=0.001,
        ens="nvt",
        temp=300.0,
    )

    assert (
        "thermo_style    custom step ke pe etotal enthalpy temp press vol "
        "ske c_allmsd[*] c_spinmsd[*]\n"
    ) in text


def test_post_tasks_subtracts_spin_kinetic_by_default(tmp_path):
    job, settings = _make_job(tmp_path, include_spin_kinetic=False, with_ske=True)

    ti_mag.post_tasks(
        str(job),
        settings,
        Eo=0.0,
        To=None,
        scheme="trapezoidal",
    )

    ti_out = np.loadtxt(job / "ti.out")
    expected = (10.0 - 4.0 + 1.5 * pc.Boltzmann * 200.0 / pc.electron_volt) / 2.0
    assert ti_out[0, 2] == pytest.approx(expected)


def test_post_tasks_can_include_spin_kinetic(tmp_path):
    job, settings = _make_job(tmp_path, include_spin_kinetic=True, with_ske=True)

    ti_mag.post_tasks(
        str(job),
        settings,
        Eo=0.0,
        To=None,
        scheme="trapezoidal",
    )

    ti_out = np.loadtxt(job / "ti.out")
    expected = (10.0 + 1.5 * pc.Boltzmann * 200.0 / pc.electron_volt) / 2.0
    assert ti_out[0, 2] == pytest.approx(expected)


def test_post_tasks_requires_ske_for_default_legacy_logs(tmp_path):
    job, settings = _make_job(tmp_path, include_spin_kinetic=False, with_ske=False)

    with pytest.raises(RuntimeError, match="SpinKinEng/ske"):
        ti_mag.post_tasks(
            str(job),
            settings,
            Eo=0.0,
            To=None,
            scheme="trapezoidal",
        )
