import json

import numpy as np
import pytest

from dpti.equi import npt_avg_ref_conf

# thermo header: cols 8-13 are lx ly lz xy xz yz (block-averaged by the code)
THERMO_HEADER = "Step KinEng PotEng TotEng Enthalpy Temp Press Volume Lx Ly Lz Xy Xz Yz"
# lx rows average to 10.0, ly to 8.0, lz to 6.0, tilts to 0
THERMO_ROWS = [
    [0, 1.0, 2.0, 3.0, 4.0, 300.0, 0.0, 480.0, 10.0, 8.0, 6.0, 0.0, 0.0, 0.0],
    [1, 1.0, 2.0, 3.0, 4.0, 300.0, 0.0, 480.0, 10.4, 8.0, 6.4, 0.0, 0.0, 0.0],
    [2, 1.0, 2.0, 3.0, 4.0, 300.0, 0.0, 480.0, 9.8, 8.0, 5.8, 0.0, 0.0, 0.0],
    [3, 1.0, 2.0, 3.0, 4.0, 300.0, 0.0, 480.0, 9.8, 8.0, 5.8, 0.0, 0.0, 0.0],
]


def _write_log(path):
    lines = ["LAMMPS log", THERMO_HEADER]
    for row in THERMO_ROWS:
        lines.append(" ".join(str(v) for v in row))
    lines.append("Loop time of 1 on 1 procs")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dump_frame(step, atom_head, atom_rows):
    return "\n".join(
        [
            "ITEM: TIMESTEP",
            str(step),
            "ITEM: NUMBER OF ATOMS",
            str(len(atom_rows)),
            "ITEM: BOX BOUNDS xy xz yz pp pp pp",
            "0.0 10.1 0.0",
            "0.0 8.1 0.0",
            "0.0 6.1 0.0",
            atom_head,
            *atom_rows,
        ]
    )


def _write_avgposi(path):
    head = "ITEM: ATOMS id type f_ap[1] f_ap[2] f_ap[3]"
    first = ["1 1 9.0 9.0 9.0", "2 1 9.0 9.0 9.0"]
    # atom 2 sits outside the averaged cell -> must wrap (10.5 -> 0.5)
    last = ["1 1 1.0 2.0 3.0", "2 1 10.5 4.0 5.0"]
    path.write_text(
        _dump_frame(0, head, first) + "\n" + _dump_frame(100, head, last) + "\n",
        encoding="utf-8",
    )


def _write_equi_dump(path, spins_last):
    head = (
        "ITEM: ATOMS id type x y z vx vy vz "
        "c_spin[1] c_spin[2] c_spin[3] c_spin[4] c_spin[5] c_spin[6] c_spin[7]"
    )
    first = [
        "1 1 1.1 2.1 3.1 0 0 0 9.9 1 0 0 0 0 0",
        "2 1 4.1 4.1 5.1 0 0 0 9.9 0 1 0 0 0 0",
    ]
    last = [
        "1 1 1.2 2.2 3.2 0 0 0 {} {} {} {} 0 0 0".format(*spins_last[0]),
        "2 1 4.2 4.2 5.2 0 0 0 {} {} {} {} 0 0 0".format(*spins_last[1]),
    ]
    path.write_text(
        _dump_frame(0, head, first) + "\n" + _dump_frame(100, head, last) + "\n",
        encoding="utf-8",
    )


def _make_npt_dir(tmp_path, name="npt"):
    d = tmp_path / name
    d.mkdir()
    (d / "equi_settings.json").write_text(
        json.dumps({"stat_skip": 0, "stat_bsize": 2}), encoding="utf-8"
    )
    _write_log(d / "log.lammps")
    _write_avgposi(d / "dump.avgposi")
    _write_equi_dump(d / "dump.equi", [(2.2, 0.0, 0.0, 1.0), (2.0, 0.6, 0.8, 0.0)])
    return d


def _parse_conf(conf):
    lines = conf.split("\n")
    cell = {}
    for ln in lines:
        if "xlo xhi" in ln:
            cell["lx"] = float(ln.split()[1])
        if "ylo yhi" in ln:
            cell["ly"] = float(ln.split()[1])
        if "zlo zhi" in ln:
            cell["lz"] = float(ln.split()[1])
    atoms = []
    in_atoms = False
    for ln in lines:
        if ln.startswith("Atoms"):
            in_atoms = True
            continue
        if in_atoms and ln.strip():
            atoms.append([float(v) for v in ln.split()])
    return cell, np.array(atoms)


def test_avg_box_avg_posi_last_spin(tmp_path):
    npt = _make_npt_dir(tmp_path)
    conf = npt_avg_ref_conf(str(npt))
    assert "Atoms # spin" in conf
    cell, atoms = _parse_conf(conf)

    # cell = block average of the thermo columns, not the last dump box
    assert cell["lx"] == pytest.approx(10.0)
    assert cell["ly"] == pytest.approx(8.0)
    assert cell["lz"] == pytest.approx(6.0)

    # positions from the LAST avgposi frame; atom 2 wrapped 10.5 -> 0.5
    np.testing.assert_allclose(atoms[0][2:5], [1.0, 2.0, 3.0], atol=1e-8)
    np.testing.assert_allclose(atoms[1][2:5], [0.5, 4.0, 5.0], atol=1e-8)

    # spins from the LAST dump.equi frame (spx spy spz sp columns)
    np.testing.assert_allclose(atoms[0][5:9], [0.0, 0.0, 1.0, 2.2], atol=1e-8)
    np.testing.assert_allclose(atoms[1][5:9], [0.6, 0.8, 0.0, 2.0], atol=1e-8)


def test_spin_dir_overrides_npt_spins(tmp_path):
    npt = _make_npt_dir(tmp_path)
    nvt = tmp_path / "nvt"
    nvt.mkdir()
    _write_equi_dump(nvt / "dump.equi", [(2.5, 1.0, 0.0, 0.0), (2.4, 0.0, 1.0, 0.0)])

    conf = npt_avg_ref_conf(str(npt), spin_dir=str(nvt))
    _, atoms = _parse_conf(conf)
    np.testing.assert_allclose(atoms[0][5:9], [1.0, 0.0, 0.0, 2.5], atol=1e-8)
    np.testing.assert_allclose(atoms[1][5:9], [0.0, 1.0, 0.0, 2.4], atol=1e-8)


def test_natoms_mismatch_raises(tmp_path):
    npt = _make_npt_dir(tmp_path)
    bad = tmp_path / "bad"
    bad.mkdir()
    head = (
        "ITEM: ATOMS id type x y z vx vy vz "
        "c_spin[1] c_spin[2] c_spin[3] c_spin[4] c_spin[5] c_spin[6] c_spin[7]"
    )
    (bad / "dump.equi").write_text(
        _dump_frame(0, head, ["1 1 0 0 0 0 0 0 2.2 0 0 1 0 0 0"]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="natoms mismatch"):
        npt_avg_ref_conf(str(npt), spin_dir=str(bad))
