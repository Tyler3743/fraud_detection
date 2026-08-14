import glob
import os

_ON_KAGGLE = os.path.isdir("/kaggle/working")


def _detect_raw_dir() -> str:
    env = os.environ.get("AML_RAW_DIR")
    if env:
        return env
    if _ON_KAGGLE:
        hits = sorted(glob.glob("/kaggle/input/**/HI-Small_Trans.csv", recursive=True))
        if not hits:
            raise FileNotFoundError(
                "Không thấy HI-Small_Trans.csv trong /kaggle/input.\n"
                "Vào Add Data -> thêm dataset "
                "'ealtman2019/ibm-transactions-for-anti-money-laundering-aml',\n"
                "hoặc đặt biến môi trường AML_RAW_DIR trỏ tới thư mục chứa file."
            )
        return os.path.dirname(hits[0])
    return "dataset_high"


RAW_DIR = _detect_raw_dir()
OUT_DIR = os.environ.get("AML_OUT_DIR") or (
    "/kaggle/working/dataset_high" if _ON_KAGGLE else "dataset_high"
)
WORK_DIR = os.path.dirname(os.path.abspath(OUT_DIR))

# --- input gốc (read-only trên Kaggle) ---
TRANS_CSV = os.path.join(RAW_DIR, "HI-Small_Trans.csv")

# --- file trung gian (ghi được) ---
SPLIT_CSV = os.path.join(OUT_DIR, "HI-Small_Trans_split_index.csv")
TX_FEAT = os.path.join(OUT_DIR, "transaction_features.parquet")
NODE_FEAT = os.path.join(OUT_DIR, "node_edge_features.parquet")
EDGE_ATTR = os.path.join(OUT_DIR, "edge_attr_{}.parquet")
TXN_MATRIX = os.path.join(OUT_DIR, "txn_matrix_{}.parquet")
GRAPHS_PT = os.path.join(OUT_DIR, "graphs.pt")
TXN_NODES = os.path.join(OUT_DIR, "txn_nodes.npy")

# --- output kết quả ---
RESULTS_CSV = os.path.join(WORK_DIR, "results.csv")
SCORES_DIR = os.path.join(WORK_DIR, "scores")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SCORES_DIR, exist_ok=True)


def describe() -> str:
    return (f"môi trường : {'Kaggle' if _ON_KAGGLE else 'local'}\n"
            f"RAW_DIR    : {RAW_DIR}\n"
            f"OUT_DIR    : {OUT_DIR}\n"
            f"WORK_DIR   : {WORK_DIR}")


if __name__ == "__main__":
    print(describe())
    print(f"\nTRANS_CSV tồn tại: {os.path.exists(TRANS_CSV)}")