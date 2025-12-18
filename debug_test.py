import sys
import os
import torch
import numpy as np

# Add the workspace root to the Python path to allow for cross-directory imports
# This enables us to import from both 'RGA' and 'lbm' directories
workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if workspace_path not in sys.path:
    sys.path.insert(0, workspace_path)

# Now we can import the functions from their original locations
try:
    from RGA.train_rel_attack_rpga_hdrlbm import base_cubemap_to_sh
    from lbm.examples.inference.bg_relight_inference_hdr import load_hdr_and_compute_sh
    from submodules.envlight.envlight.light import EnvLight as EnvLightClass
except ImportError as e:
    print(f"Failed to import necessary modules. Please ensure the script is in the 'RGA' directory and that 'lbm' and 'submodules' are in the workspace root. Error: {e}")
    sys.exit(1)

def main():
    """
    Compares two methods of calculating SH coefficients for the same HDR file
    to debug discrepancies between the training and inference pipelines.
    """
    hdr_path = "/workspace/RGA/hdri/carla_hdr/06_Foggy_Dens12.hdr"
    if not os.path.exists(hdr_path):
        print(f"Error: HDR file not found at {hdr_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"--- Debugging SH calculation for: {os.path.basename(hdr_path)} ---")

    # --- Method 1: EnvLight -> Base Cubemap -> SH (Simulates training pipeline) ---
    print("\n[Method 1] Simulating training pipeline: HDR -> EnvLight.base -> base_cubemap_to_sh")
    sh1 = None
    try:
        env_light = EnvLightClass(
            path=hdr_path, device=device, scale=1.0, min_res=16, max_res=512, trainable=False
        )
        base_cubemap = env_light.base.detach()
        
        sh1_tensor = base_cubemap_to_sh(base_cubemap, device)
        sh1 = sh1_tensor.cpu().numpy()
        
        print("  - Successfully calculated SH coefficients.")
        print(f"  - SH.shape: {sh1.shape}")
        print(f"  - SH[:5]: {sh1[:5]}")
    except Exception as e:
        print(f"  - Error in Method 1: {e}")

    # --- Method 2: Direct HDR loading -> SH (Simulates inference pipeline) ---
    print("\n[Method 2] Simulating inference pipeline: load_hdr_and_compute_sh")
    sh2 = None
    try:
        sh2_np, _ = load_hdr_and_compute_sh(hdr_path)
        sh2 = sh2_np

        print("  - Successfully calculated SH coefficients.")
        print(f"  - SH.shape: {sh2.shape}")
        print(f"  - SH[:5]: {sh2[:5]}")
    except Exception as e:
        print(f"  - Error in Method 2: {e}")

    # --- Comparison ---
    print("\n--- Comparison ---")
    if sh1 is not None and sh2 is not None:
        if sh1.shape == sh2.shape:
            abs_diff = np.abs(sh1 - sh2)
            mae = np.mean(abs_diff)
            max_diff = np.max(abs_diff)
            print(f"  - Mean Absolute Error (MAE): {mae:.6f}")
            print(f"  - Maximum Absolute Difference: {max_diff:.6f}")
            
            if np.allclose(sh1, sh2, atol=1e-4):
                print("  - Conclusion: The two methods produce NEARLY IDENTICAL SH coefficients.")
            else:
                print("  - Conclusion: The two methods produce DIFFERENT SH coefficients. This is a likely source of the bug.")
        else:
            print(f"  - Error: SH coefficient shapes do not match! Method 1: {sh1.shape}, Method 2: {sh2.shape}")
    else:
        print("  - Could not compare SH coefficients due to errors in one of the methods.")

if __name__ == '__main__':
    main()
