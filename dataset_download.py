from huggingface_hub import snapshot_download
import os

local_dir = os.path.join(os.getcwd(), "TACK_Tunnel_Data")

snapshot_download(
    repo_id="TACK-project/TACK_Tunnel_Data", 
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print("Download Complete!")