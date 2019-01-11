import os

def dispatch_winobench(NUM_GEMM_THREADS = 8, NUM_GEMM_TILE = 8, NUM_WINOGRAD_BLOCK = 16):
    OUT_FILE = "logs/tvmbench_winograd_gemmthread{}_gemmtile{}_winogradblock{}.tsv".format(str(NUM_GEMM_THREADS), str(NUM_GEMM_TILE), str(NUM_WINOGRAD_BLOCK))
    os.system("python3 wino_test_tvm.py {} {} {} {}".format(OUT_FILE, str(NUM_GEMM_THREADS), str(NUM_GEMM_TILE), str(NUM_WINOGRAD_BLOCK)))

if __name__ == "__main__":
    for NUM_GEMM_THREADS in [1, 2, 4, 8, 16, 32]:
        for NUM_GEMM_TILE in [1, 2, 4, 8, 16, 32]:
            for NUM_WINOGRAD_BLOCK in [1, 2, 4, 8, 16, 32]:
                  print("Dispatching experiment {}, {}, {}".format(str(NUM_GEMM_THREADS), str(NUM_GEMM_TILE), str(NUM_WINOGRAD_BLOCK)))
                  dispatch_winobench(NUM_GEMM_THREADS, NUM_GEMM_TILE, NUM_WINOGRAD_BLOCK)