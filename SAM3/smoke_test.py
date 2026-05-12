from pathlib import Path

from ttd_dataset import TTDDatasetBuilder


def main() -> None:
    project_root = Path("/users/7/yu001011/csci5527/CSCI5527-final")
    builder = TTDDatasetBuilder(project_root)
    train_val_df, test_df = builder.build_records("Single-TB")

    print("train_val rows:", len(train_val_df))
    print("test rows:", len(test_df))
    if not test_df.empty:
        print("sample image:", test_df.iloc[0]["image_abs_path"])
        print("sample mask:", test_df.iloc[0]["mask_path_sanitized"])


if __name__ == "__main__":
    main()
