from __future__ import annotations

import os
import shutil
import time
import string
import random
import subprocess
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod

import h5py
import torch
import pandas as pd
import numpy as np

from .raypyng.rml import RMLFile


@dataclass
class RayOutput:
    x_loc: torch.Tensor
    y_loc: torch.Tensor
    z_loc: torch.Tensor
    x_dir: torch.Tensor
    y_dir: torch.Tensor
    z_dir: torch.Tensor
    energy: torch.Tensor


class RayBackend(metaclass=ABCMeta):
    @abstractmethod
    def run(self, raypyng_rml: RMLFile, exported_planes: list[str], run_id: str | None = None) -> dict[str, RayOutput]:
        ...


class LocalBackendBase(RayBackend):
    """
    Runs a backend directly (no docker/podman inside).
    Creates a per-run directory, writes workfile.rml there, executes backend, parses output.
    """

    def __init__(
        self,
        workdir: str = "/dev/shm/ray-workdir",
        max_retry: int = 10,
        verbose: bool = False,
        additional_mount_files: list[str] | None = None,
        device: torch.device = torch.device("cpu"),
        seed: int | None = None,
        env: dict[str, str] | None = None,
        cleanup_run_dir: bool = False,
    ) -> None:
        self.workdir = os.path.abspath(workdir)
        os.makedirs(self.workdir, exist_ok=True)

        self.max_retry = max_retry
        self.verbose = verbose
        self.additional_mount_files = additional_mount_files or []
        self.device = device
        self.seed = seed
        self.env = env or {}
        self.cleanup_run_dir = cleanup_run_dir

    def _make_run_id(self) -> str:
        return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(16))

    def _prepare_run_dir(self, run_id: str) -> str:
        run_dir = os.path.join(self.workdir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Copy any auxiliary files into run_dir (thread-safe / isolated)
        for f in self.additional_mount_files:
            shutil.copy(f, run_dir)

        return run_dir

    def _write_rml(self, run_dir: str, raypyng_rml: RMLFile) -> str:
        rml_path = os.path.join(run_dir, "workfile.rml")
        raypyng_rml.write(rml_path)
        if not os.path.isfile(rml_path):
            raise FileNotFoundError(f"RML workfile not found: {rml_path}")
        return rml_path

    def _merged_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.env)
        return env

    @abstractmethod
    def _exec_once(self, rml_path: str, exported_planes: list[str], run_dir: str) -> int:
        ...

    @abstractmethod
    def _outputs_ready(self, rml_path: str, exported_planes: list[str], run_dir: str) -> bool:
        ...

    @abstractmethod
    def _extract(self, rml_path: str, exported_planes: list[str], run_dir: str) -> dict[str, RayOutput]:
        ...

    def run(self, raypyng_rml: RMLFile, exported_planes: list[str], run_id: str | None = None) -> dict[str, RayOutput]:
        run_id = run_id or self._make_run_id()
        run_dir = self._prepare_run_dir(run_id)

        tic = time.perf_counter()
        rml_path = self._write_rml(run_dir, raypyng_rml)

        last_rc: int | None = None
        try:
            for attempt in range(self.max_retry + 1):
                last_rc = self._exec_once(rml_path, exported_planes, run_dir)

                if self._outputs_ready(rml_path, exported_planes, run_dir):
                    out = self._extract(rml_path, exported_planes, run_dir)
                    if self.verbose:
                        toc = time.perf_counter()
                        print(f"[{run_id}] success in {toc - tic:.2f}s (rc={last_rc})")
                    return out

                if self.verbose:
                    print(f"[{run_id}] attempt {attempt+1} failed (rc={last_rc}), retrying...")
                else:
                    print(f"Run {attempt+1} failed. Retry ...")

            raise RuntimeError(f"[{run_id}] backend failed after {self.max_retry+1} attempts (last_rc={last_rc})")

        finally:
            if self.cleanup_run_dir:
                shutil.rmtree(run_dir, ignore_errors=True)


class RayBackendLocalRayUI(LocalBackendBase):
    """
    Runs Ray-UI in background mode and exports CSVs.

    Equivalent to:
      rayui -b
        load <rml>
        trace noanalyze
        export "<plane>" "RawRaysBeam" <workdir>
        quit
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # For headless usage in containers (safe on desktop too):
        #self.env.setdefault("QT_QPA_PLATFORM", "offscreen")

    @staticmethod
    def _send(proc: subprocess.Popen, command: str) -> None:
        assert proc.stdin is not None
        proc.stdin.write((command + "\n").encode("utf-8"))
        proc.stdin.flush()

    def _exec_once(self, rml_path: str, exported_planes: list[str], run_dir: str) -> int:
        env = self._merged_env()
        workdir = os.path.dirname(rml_path)
        print(workdir)
        print(os.getcwd())


        # Use list form, not shell=True
        proc = subprocess.Popen(
            ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24", "rayui", "-b"],
            cwd=run_dir,
            env=env,
            stdin=subprocess.PIPE,
            stdout=None if self.verbose else subprocess.DEVNULL,
            stderr=None if self.verbose else subprocess.STDOUT,
        )

        try:
            self._send(proc, f"load {rml_path}")
            self._send(proc, "trace noanalyze")
            for plane in exported_planes:
                # export "<plane>" "RawRaysBeam" <workdir>
                self._send(proc, f'export "{plane}" "RawRaysBeam" {workdir}')
            self._send(proc, "quit")

            if proc.stdin:
                proc.stdin.close()
            return int(proc.wait())

        finally:
            # if something goes wrong, ensure the process is not left behind
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass

    def _outputs_ready(self, rml_path: str, exported_planes: list[str], run_dir: str) -> bool:
        for plane in exported_planes:
            csv_path = os.path.join(run_dir, f"{plane}-RawRaysBeam.csv")
            if not os.path.isfile(csv_path):
                return False
        return True

    def _extract(self, rml_path: str, exported_planes: list[str], run_dir: str) -> dict[str, RayOutput]:
        ray_output: dict[str, RayOutput] = {}

        for plane in exported_planes:
            csv_path = os.path.join(run_dir, f"{plane}-RawRaysBeam.csv")

            raw = pd.read_csv(
                csv_path,
                sep="\t",
                skiprows=1,
                engine="c",
                dtype="float32",
                usecols=[
                    f"{plane}_OX", f"{plane}_OY", f"{plane}_OZ",
                    f"{plane}_DX", f"{plane}_DY", f"{plane}_DZ",
                    f"{plane}_EN",
                ],
            )

            def t(col: str) -> torch.Tensor:
                return torch.tensor(raw[col].values, dtype=torch.float32, device=self.device)

            ray_output[plane] = RayOutput(
                x_loc=t(f"{plane}_OX"),
                y_loc=t(f"{plane}_OY"),
                z_loc=t(f"{plane}_OZ"),
                x_dir=t(f"{plane}_DX"),
                y_dir=t(f"{plane}_DY"),
                z_dir=t(f"{plane}_DZ"),
                energy=t(f"{plane}_EN"),
            )

        return ray_output


class RayBackendLocalRayX(LocalBackendBase):
    """
    RayX syntax:
      rayx -s <seed> <rml_file>
    Produces: <rml_file_base>.h5
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _exec_once(self, rml_path: str, exported_planes: list[str], run_dir: str) -> int:
        env = self._merged_env()

        cmd = ["rayx", "-S"]
        if self.seed is not None:
            # IMPORTANT: correct order per your syntax
            cmd += ["-s", str(self.seed)]
        cmd += [rml_path]

        if self.verbose:
            print("EXEC:", " ".join(cmd))

        proc = subprocess.run(
            cmd,
            cwd=run_dir,
            env=env,
            stdout=None if self.verbose else subprocess.DEVNULL,
            stderr=None if self.verbose else subprocess.STDOUT,
            check=False,
        )
        return int(proc.returncode)

    def _outputs_ready(self, rml_path: str, exported_planes: list[str], run_dir: str) -> bool:
        return os.path.isfile(os.path.splitext(rml_path)[0] + ".h5")

    def _extract(self, rml_path: str, exported_planes: list[str], run_dir: str) -> dict[str, RayOutput]:
        h5_path = os.path.splitext(rml_path)[0] + ".h5"
        ray_output: dict[str, RayOutput] = {}

        with h5py.File(h5_path, "r") as h5f:
            rayx = h5f["rayx"]
            object_names = [n.decode("utf-8") for n in rayx["object_names"][:]]
            name_to_index = {name: i for i, name in enumerate(object_names)}

            events = rayx["events"]
            obj_id = events["object_id"][:]  # numpy

            for plane in exported_planes:
                # Missing plane -> empty output
                if plane not in name_to_index:
                    ray_output[plane] = RayOutput(
                        x_loc=torch.empty(0, device=self.device),
                        y_loc=torch.empty(0, device=self.device),
                        z_loc=torch.empty(0, device=self.device),
                        x_dir=torch.empty(0, device=self.device),
                        y_dir=torch.empty(0, device=self.device),
                        z_dir=torch.empty(0, device=self.device),
                        energy=torch.empty(0, device=self.device),
                    )
                    continue

                idx = name_to_index[plane]
                inds = np.nonzero(obj_id == idx)[0]

                if inds.size == 0:
                    ray_output[plane] = RayOutput(
                        x_loc=torch.empty(0, device=self.device),
                        y_loc=torch.empty(0, device=self.device),
                        z_loc=torch.empty(0, device=self.device),
                        x_dir=torch.empty(0, device=self.device),
                        y_dir=torch.empty(0, device=self.device),
                        z_dir=torch.empty(0, device=self.device),
                        energy=torch.empty(0, device=self.device),
                    )
                    continue

                def to_torch(col: str) -> torch.Tensor:
                    arr = events[col][inds].astype(np.float32, copy=False)
                    arr = np.ascontiguousarray(arr)
                    t = torch.from_numpy(arr)
                    return t.to(self.device) if self.device.type != "cpu" else t

                ray_output[plane] = RayOutput(
                    x_loc=to_torch("position_x"),
                    y_loc=to_torch("position_y"),
                    z_loc=to_torch("position_z"),
                    x_dir=to_torch("direction_x"),
                    y_dir=to_torch("direction_y"),
                    z_dir=to_torch("direction_z"),
                    energy=to_torch("energy"),
                )

        return ray_output

