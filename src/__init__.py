import os

# evita um travamento do XGBoost/OpenMP no Windows dentro do Jupyter
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
