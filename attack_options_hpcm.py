import argparse

from arguments import ModelParams, PipelineParams, get_combined_args


def get_attack_args():
    """Defines and parses command-line arguments for HPCM-based attack (MIN only)."""
    parser = argparse.ArgumentParser("RGA physical attack (HPCM mining, MIN-only)")

    # Add model and pipeline parameters from the original arguments file
    model_params = ModelParams(parser, sentinel=False)
    pipeline_params = PipelineParams(parser)

    # =================================================================================
    # Training (step-based, no epochs)
    # =================================================================================
    parser.add_argument(
        "--total_steps",
        type=int,
        default=2000,
        help="Total number of optimization steps (batches). Training stops when reached.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for the attack.")
    parser.add_argument(
        "--global_step_start",
        type=int,
        default=60000,
        help="Renderer 'iteration' start value (controls staged rendering).",
    )
    parser.add_argument(
        "--max_cams",
        type=int,
        default=0,
        help="Limit number of cameras for quick testing (0=all).",
    )

    # =================================================================================
    # HPCM mining
    # =================================================================================
    parser.add_argument(
        "--hpcm_temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for mining distribution (lower => more greedy).",
    )
    parser.add_argument(
        "--hpcm_momentum",
        type=float,
        default=0.5,
        help="EMA momentum for difficulty table update: new = m*old + (1-m)*loss.",
    )
    parser.add_argument(
        "--hpcm_uniform_prob",
        type=float,
        default=0.05,
        help="With this probability, sample uniformly instead of mining (exploration).",
    )
    parser.add_argument(
        "--hpcm_init_score",
        type=float,
        default=10.0,
        help="Initial difficulty score for all configurations.",
    )
    parser.add_argument(
        "--hpcm_save_interval",
        type=int,
        default=200,
        help="Save HPCM table every N steps (0 disables).",
    )
    parser.add_argument(
        "--hpcm_save_history_npz",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If enabled, also save step-suffixed npz snapshots (hpcm_table_step_XXXXXX.npz) at each hpcm_save_interval.",
    )
    parser.add_argument(
        "--hpcm_export_summary",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Export human-readable HPCM summaries (text + stats csv + TopK csv) at each hpcm_save_interval.",
    )
    parser.add_argument(
        "--hpcm_export_plots",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Export HPCM plots (histograms/heatmaps/stats curves) at each hpcm_save_interval.",
    )
    parser.add_argument(
        "--hpcm_export_full_csv",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Export a full per-config CSV (can be large) at each hpcm_save_interval.",
    )
    parser.add_argument(
        "--hpcm_topk",
        type=int,
        default=20,
        help="Top-K configs/states to report in HPCM summary outputs.",
    )
    parser.add_argument(
        "--hpcm_plot_max_pitches",
        type=int,
        default=12,
        help="Max number of pitch-slice heatmaps to export per save (0 disables pitch heatmaps).",
    )
    parser.add_argument(
        "--hpcm_resample_max",
        type=int,
        default=50,
        help="Max resample attempts when a mined state has no cameras (should be rare if dataset is complete).",
    )
    parser.add_argument(
        "--hpcm_group_by_hdr",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Group sampled configs by hdr_id inside each step and run compute_batch_loss per group (enables detector batching, fewer build_mips).",
    )
    parser.add_argument(
        "--hpcm_precompute_hdr_sh",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Precompute SH coefficients for each HDR base once at startup (saves repeated base_cubemap_to_sh).",
    )

    # =================================================================================
    # Detector
    # =================================================================================
    parser.add_argument(
        "--detector",
        type=str,
        default="yolox",
        choices=["yolov3", "yolox", "faster-rcnn", "mask-rcnn", "d-detr", "pvt", "detr"],
        help="Object detector to use for the attack.",
    )
    parser.add_argument("--target_class_name", type=str, default="car", help="Target class name for the attack (COCO class).")
    parser.add_argument("--score_thresh", type=float, default=0.5, help="Score threshold for detection.")

    # =================================================================================
    # Albedo Optimization (MIN only)
    # =================================================================================
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "sgd", "adamw"], help="Optimizer to use for albedo.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for the albedo optimizer.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for the SGD optimizer.")
    parser.add_argument("--perturb_albedo", default=True, action=argparse.BooleanOptionalAction, help="Randomly perturb albedo init.")
    parser.add_argument("--perturb_budget_factor", type=float, default=1e-8, help="Factor (n) for calculating the albedo perturbation budget.")
    parser.add_argument(
        "--albedo_init_method",
        type=str,
        default="perturb",
        choices=["perturb", "random"],
        help='Albedo initialization method: "perturb" adds random noise to original albedo, "random" initializes randomly within original range.',
    )
    parser.add_argument("--reg_loss_weight", type=float, default=0.0, help="Weight for the regression loss component in the total adversarial loss.")

    # =================================================================================
    # Environment / Relighting (discrete HDR selection; NO envlight optimization)
    # =================================================================================
    parser.add_argument("--environment_texture", type=str, default="", help="Path to the environment texture (HDR). If set, overrides HDR bank.")
    parser.add_argument("--environment_scale", type=float, default=1.0, help="Scale of the environment light.")
    parser.add_argument("--hdr_rotation", action="store_true", default=False, help="Enable random rotation of the HDR environment map during training.")
    parser.add_argument("--enable_lbm_relight", default=True, action=argparse.BooleanOptionalAction, help="Enable LBM background relighting.")
    parser.add_argument("--lbm_ckpt_dir", type=str, default="/workspace/RGA/lbm_ckpt2", help="Path to LBM checkpoints directory.")
    parser.add_argument("--hdr_bank_dir", type=str, default="/workspace/RGA/hdri/carla_hdr", help="Directory of HDR/EXR files as discrete envmaps.")

    # =================================================================================
    # Model / Scene Loading
    # =================================================================================
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration to load; -1 means latest (GIR style).")
    parser.add_argument("--second_stage_step", type=int, default=30000, help="Step count for the second stage of rendering.")

    # =================================================================================
    # I/O and System
    # =================================================================================
    parser.add_argument("--save_dir", type=str, default="RGA_output", help="Directory to save outputs.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the training on.")

    # =================================================================================
    # Debug / Visualization I/O (optional)
    # =================================================================================
    parser.add_argument("--save_temp_imgs_for_det", default=False, action=argparse.BooleanOptionalAction, help="Save temp images used for detector input (temp_imgs_for_det/*).")
    parser.add_argument("--save_visualizations", default=False, action=argparse.BooleanOptionalAction, help="Save training visualization grids (visualizations/*).")
    parser.add_argument(
        "--hpcm_det_vis_interval",
        type=int,
        default=200,
        help="(HPCM) Save detector visualization (images with bboxes) every N steps, but on the NEXT step. 0 disables.",
    )

    # =================================================================================
    # Final evaluation (optional)
    # =================================================================================
    parser.add_argument(
        "--run_final_eval",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Run a final, detailed evaluation on all data splits after training completes.",
    )
    parser.add_argument("--eval_on_train", default=False, action="store_true", help="Enable evaluation on the training set in the final evaluation.")
    parser.add_argument("--save_final_full_images_mw", default=True, action=argparse.BooleanOptionalAction, help="At final stage, render multi-weather images and run OFFLINE evaluation on them.")
    parser.add_argument("--save_final_eval_vis", default=True, action=argparse.BooleanOptionalAction, help="During final offline evaluation, save detection results (images with bboxes).")
    parser.add_argument(
        "--run_stage2_eval",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Run final evaluation Stage2 (cross-detector eval on saved images). Default: disabled.",
    )

    args = get_combined_args(parser)
    return args, model_params, pipeline_params


if __name__ == "__main__":
    args, _, _ = get_attack_args()
    print("--- Parsed Arguments (HPCM) ---")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")
    print("------------------------------")


