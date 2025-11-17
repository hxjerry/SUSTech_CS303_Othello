from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import os
from pathlib import Path

class get_pybind_include:
    def __str__(self):
        return pybind11.get_include()

def find_xsimd_include():
    candidates = [
        Path(os.environ.get("XSIMD_INCLUDE_DIR", "")),
        Path("/usr/include"),
        Path("/usr/local/include"),
    ]
    for path in candidates:
        if path and (path / "xsimd").exists():
            return str(path)
    raise RuntimeError("Unable to locate xsimd headers; set XSIMD_INCLUDE_DIR.")

xsimd_include = find_xsimd_include()

ext_modules = [
    Extension(
        'othello._core',
        ['src/cpp/bindings.cpp'],
        include_dirs=[get_pybind_include(), 'src/cpp/include', xsimd_include],
        language='c++',
        extra_compile_args=['-std=c++17'],
    ),
]

setup(
    name='othello',
    version='0.1.0',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)