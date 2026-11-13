#!/usr/bin/env python3
"""Run the installed SeisSol CUDA interoperability control in job-local storage."""

import os
from pathlib import Path
import subprocess
import tempfile


def main():
    prefix = Path(__file__).resolve().parents[3]
    loaded_prefix = Path(os.environ['EBROOTSEISSOL']).resolve(strict=True)
    if loaded_prefix != prefix:
        raise RuntimeError("Load the SeisSol module that provides this test script")

    temp_root = Path(os.environ['TMPDIR']).resolve(strict=True)
    if not temp_root.is_dir() or not os.access(temp_root, os.W_OK):
        raise RuntimeError("TMPDIR must name writable job-local storage")

    llvm = Path(os.environ['EBROOTLLVM'])
    acpp = Path(os.environ['EBROOTADAPTIVECPP']) / 'bin' / 'acpp'
    omp_runtime = Path(subprocess.check_output(
        [str(llvm / 'bin' / 'clang++'), '-print-file-name=libomp.so'],
        text=True,
    ).strip()).resolve(strict=True)
    if not omp_runtime.is_file():
        raise RuntimeError("The loaded LLVM module must provide libomp.so")

    source = prefix / 'share' / 'SeisSol' / 'tests' / 'sycl_cuda_interop.cpp'
    with tempfile.TemporaryDirectory(prefix='seissol-cuda-interop-', dir=temp_root) as work:
        executable = Path(work) / 'seissol-sycl-cuda-interop'
        subprocess.run([
            str(acpp), '--acpp-targets=generic', '-O2', str(source),
            '-Wl,-rpath,' + str(omp_runtime.parent), '-o', str(executable),
        ], check=True, cwd=work)
        runtime_env = os.environ.copy()
        runtime_env['ACPP_VISIBILITY_MASK'] = 'cuda'
        subprocess.run([str(executable)], check=True, cwd=work, env=runtime_env)


if __name__ == '__main__':
    main()
