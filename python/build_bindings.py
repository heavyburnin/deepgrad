import os
import subprocess

SRC_DIR = os.path.join(os.path.dirname(__file__), "../src")
BUILD_DIR = SRC_DIR

def build():
    c_file = os.path.join(SRC_DIR, "tensor_ops.c")
    so_file = os.path.join(BUILD_DIR, "tensor_ops.so")

    compile_cmd = [
        "gcc", "-O3", "-march=native", "-mavx2", "-shared", "-fPIC",
        c_file,
        "-o", so_file
    ]

    print("Compiling C SIMD backend...")
    subprocess.check_call(compile_cmd)
    print("Build complete:", so_file)

if __name__ == "__main__":
    build()
