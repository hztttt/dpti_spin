import glob
import json
import os
import shutil
import unittest
from types import SimpleNamespace

from context import dpti

import dpti.hti_liq_mag


class TestHtiLiqMagMakeTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.isdir("tmp_hti_liq_mag/"):
            shutil.rmtree("tmp_hti_liq_mag/")
        os.mkdir("tmp_hti_liq_mag/")

    def _base_jdata(self, spring_lambda_mode="joint"):
        jdata = {
            "equi_conf": "tests/benchmark_hti_liq/three_step/conf.lmp",
            "model": "tests/benchmark_hti_liq/three_step/graph.pb",
            "mass_map": [118.71],
            "spin_mass": 0.01,
            "spin_map": [1],
            "spin_ref": {"style": "spring", "k": 1.0, "s0": 2.5},
            "spring_lambda_mode": spring_lambda_mode,
            "soft_param": {
                "sigma_0_0": 2.527854,
                "epsilon": 0.01,
                "activation": 0.5,
                "n": 1,
                "alpha_lj": 0.5,
                "rcut": 6.0,
            },
            "lambda_soft_on": [0.2],
            "lambda_deep_on": [0.2],
            "lambda_soft_off": [0.2],
            "lambda_spin_ref_off": [0.2],
            "temp": 1200.0,
            "nsteps": 1,
            "timestep": 0.001,
            "thermo_freq": 1,
            "dump_freq": 1,
        }
        return jdata

    def test_protect_eps_is_applied_to_all_steps(self):
        test_dir = "tmp_hti_liq_mag/protect_eps"
        jdata = {
            "equi_conf": "tests/benchmark_hti_liq/three_step/conf.lmp",
            "model": "tests/benchmark_hti_liq/three_step/graph.pb",
            "mass_map": [118.71],
            "spin_mass": 0.01,
            "spin_map": [1],
            "spin_mod_k": 0.12,
            "spring_lambda_mode": "joint",
            "soft_param": {
                "sigma_0_0": 2.527854,
                "epsilon": 0.01,
                "activation": 0.5,
                "n": 1,
                "alpha_lj": 0.5,
                "rcut": 6.0,
            },
            "lambda_soft_on": [0.0, 0.5, 1.0],
            "lambda_deep_on": [0.0, 0.5, 1.0],
            "lambda_soft_off": [0.0, 0.5, 1.0],
            "protect_eps": 1e-8,
            "temp": 1200.0,
            "nsteps": 10,
            "timestep": 0.001,
            "thermo_freq": 1,
            "dump_freq": 1,
        }

        dpti.hti_liq_mag.make_tasks(iter_name=test_dir, jdata=jdata)

        for step_dir in ["00.soft_on", "01.deep_on", "02.soft_off"]:
            lambda_files = sorted(
                glob.glob(os.path.join(test_dir, step_dir, "task.*", "lambda.out"))
            )
            self.assertTrue(lambda_files, msg=step_dir)
            with open(lambda_files[0]) as fp:
                first_lambda = float(fp.read().strip())
            with open(lambda_files[-1]) as fp:
                last_lambda = float(fp.read().strip())
            self.assertAlmostEqual(first_lambda, 1e-8, places=14, msg=step_dir)
            self.assertAlmostEqual(last_lambda, 1.0 - 1e-8, places=14, msg=step_dir)

    def test_handle_gen_resolves_model_relative_to_param_file(self):
        case_dir = "tmp_hti_liq_mag/handle_gen_paths"
        assets_dir = os.path.join(case_dir, "assets")
        param_dir = os.path.join(case_dir, "params")
        os.makedirs(assets_dir)
        os.makedirs(param_dir)

        shutil.copyfile(
            "tests/benchmark_hti_liq/three_step/conf.lmp",
            os.path.join(assets_dir, "conf.lmp"),
        )
        model_name = "frozen_model.pth"
        shutil.copyfile(
            "tests/benchmark_hti_liq/three_step/graph.pb",
            os.path.join(assets_dir, model_name),
        )

        jdata = {
            "equi_conf": "../assets/conf.lmp",
            "model": f"../assets/{model_name}",
            "mass_map": [118.71],
            "spin_mass": 0.01,
            "spin_map": [1],
            "spin_mod_k": 0.12,
            "spring_lambda_mode": "joint",
            "soft_param": {
                "sigma_0_0": 2.527854,
                "epsilon": 0.01,
                "activation": 0.5,
                "n": 1,
                "alpha_lj": 0.5,
                "rcut": 6.0,
            },
            "lambda_soft_on": [0.2],
            "lambda_deep_on": [0.2],
            "lambda_soft_off": [0.2],
            "protect_eps": 1e-8,
            "temp": 1200.0,
            "nsteps": 1,
            "timestep": 0.001,
            "thermo_freq": 1,
            "dump_freq": 1,
        }
        param_path = os.path.join(param_dir, "hti.json")
        with open(param_path, "w") as fp:
            json.dump(jdata, fp)

        output_dir = os.path.join(case_dir, "new_job")
        args = SimpleNamespace(PARAM=param_path, output=output_dir)
        dpti.hti_liq_mag.handle_gen(args)

        self.assertTrue(os.path.isfile(os.path.join(output_dir, model_name)))
        in_lammps = os.path.join(output_dir, "01.deep_on", "task.000000", "in.lammps")
        self.assertTrue(os.path.isfile(in_lammps))
        with open(in_lammps) as fp:
            lammps_input = fp.read()
        self.assertIn(f"deepspin {model_name}", lammps_input)

    def test_joint_spin_ref_is_full_until_soft_off(self):
        test_dir = "tmp_hti_liq_mag/joint_spin_ref"
        dpti.hti_liq_mag.make_tasks(iter_name=test_dir, jdata=self._base_jdata("joint"))

        with open(os.path.join(test_dir, "00.soft_on", "task.000000", "in.lammps")) as fp:
            soft_on = fp.read()
        self.assertIn("spring/spin/mod 1.0000000000e+00 2.5000000000e+00", soft_on)
        self.assertIn("variable        e_diff equal c_lj_pe/v_LAMBDA", soft_on)
        self.assertNotIn("v_l_spring_spin/v_LAMBDA", soft_on)

        with open(os.path.join(test_dir, "02.soft_off", "task.000000", "in.lammps")) as fp:
            soft_off = fp.read()
        self.assertIn(
            "variable        e_diff equal -(c_lj_pe+v_l_spring_spin)/v_INV_LAMBDA",
            soft_off,
        )

    def test_split_has_separate_spin_ref_off_stage(self):
        test_dir = "tmp_hti_liq_mag/split_spin_ref"
        dpti.hti_liq_mag.make_tasks(iter_name=test_dir, jdata=self._base_jdata("split"))

        self.assertTrue(os.path.isdir(os.path.join(test_dir, "03.spin_ref_off")))
        with open(os.path.join(test_dir, "02.soft_off", "task.000000", "in.lammps")) as fp:
            soft_off = fp.read()
        self.assertIn("variable        e_diff equal -c_lj_pe/v_INV_LAMBDA", soft_off)
        self.assertNotIn("-(c_lj_pe+v_l_spring_spin)/v_INV_LAMBDA", soft_off)

        with open(os.path.join(test_dir, "03.spin_ref_off", "task.000000", "in.lammps")) as fp:
            spin_ref_off = fp.read()
        self.assertIn("pair_style      deepspin graph.pb", spin_ref_off)
        self.assertIn(
            "variable        e_diff equal -v_l_spring_spin/v_INV_LAMBDA",
            spin_ref_off,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("tmp_hti_liq_mag/")


if __name__ == "__main__":
    unittest.main()
