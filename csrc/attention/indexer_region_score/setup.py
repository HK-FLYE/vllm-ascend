# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Build the native CANN operator and its standalone queued PyTorch adapter."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup

try:
    import torch_npu
    from torch.utils.cpp_extension import BuildExtension
    from torch_npu.utils.cpp_extension import NpuExtension
except ImportError as error:
    raise RuntimeError(
        "Install matching torch/torch_npu, source CANN set_env.sh, and use pip install --no-build-isolation"
    ) from error

ROOT = Path(__file__).resolve().parent
PACKAGE = "vllm_ascend_indexer"
VENDOR = "vllm_ascend_indexer"
BUILD_JOBS = 2


def target_soc() -> str:
    soc = os.environ.get("SOC_VERSION", "ascend910b").lower()
    if soc.startswith("ascend910b"):
        return "ascend910b"
    if soc.startswith("ascend910_93"):
        return "ascend910_93"
    raise RuntimeError("The standalone BF16 wheel supports SOC_VERSION=ascend910b or ascend910_93 only")


class IndexerBuildExtension(BuildExtension):
    def run(self):
        cann = os.environ.get("ASCEND_HOME_PATH")
        if not cann or not Path(cann).is_dir():
            raise RuntimeError("Source the CANN toolkit set_env.sh before building IndexerRegionScore")
        soc = target_soc()
        native_build = Path(self.build_temp).resolve() / f"indexer_native_{soc}"
        # CANN packages copy kernel headers into generated source packages.
        # A clean native staging directory prevents an incremental stale copy.
        if native_build.exists():
            shutil.rmtree(native_build)
        subprocess.run(
            [
                "cmake",
                "-S",
                str(ROOT / "packaging"),
                "-B",
                str(native_build),
                f"-DASCEND_CANN_PACKAGE_PATH={cann}",
                f"-DASCEND_COMPUTE_UNIT={soc}",
                f"-DASCEND_PYTHON_EXECUTABLE={sys.executable}",
                "-DENABLE_BINARY_PACKAGE=ON",
                "-DENABLE_SOURCE_PACKAGE=ON",
                "-DENABLE_TEST=OFF",
            ],
            check=True,
        )
        subprocess.run(["cmake", "--build", str(native_build), "--parallel", str(BUILD_JOBS)], check=True)
        subprocess.run(
            ["cmake", "--build", str(native_build), "--target", "binary", "--parallel", str(BUILD_JOBS)], check=True
        )
        subprocess.run(["cmake", "--install", str(native_build)], check=True)
        vendor_source = native_build / "packages" / "vendors" / VENDOR
        if not (vendor_source / "op_api" / "lib" / "libcust_opapi.so").is_file():
            raise RuntimeError("CANN build did not install the IndexerRegionScore vendor package")
        package_output = Path(self.get_ext_fullpath(f"{PACKAGE}._C")).resolve().parent
        vendor_output = package_output / "_cann_ops_custom" / "vendors" / VENDOR
        if vendor_output.exists():
            shutil.rmtree(vendor_output)
        shutil.copytree(
            vendor_source, vendor_output, symlinks=False, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        (package_output / "_native_info.json").write_text(json.dumps({"soc": soc, "vendor": VENDOR}) + "\n")
        super().run()


NPU_INCLUDE = Path(torch_npu.__file__).resolve().parent / "include" / "third_party" / "acl" / "inc"
extension = NpuExtension(
    name=f"{PACKAGE}._C",
    sources=[str(path.relative_to(ROOT)) for path in sorted((ROOT / PACKAGE / "csrc").glob("*.cpp"))],
    depends=[
        "indexer_region_score_torch_adpt.h",
        "indexer_region_score_dispatch.h",
        f"{PACKAGE}/csrc/ops_common.h",
    ],
    include_dirs=[str(ROOT)],
    extra_compile_args=[f"-I{NPU_INCLUDE}", "-O2", "-g0"],
)

setup(
    name="vllm-ascend-indexer-region-score",
    version="0.1.0",
    description="Standalone BF16 IndexerRegionScore for Ascend 910B and A3",
    python_requires=">=3.10",
    packages=find_packages(include=[PACKAGE, f"{PACKAGE}.*"]),
    install_requires=["torch", "torch_npu"],
    ext_modules=[extension],
    cmdclass={"build_ext": IndexerBuildExtension.with_options(use_ninja=False)},
    include_package_data=True,
    zip_safe=False,
)
