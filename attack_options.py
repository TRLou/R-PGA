import argparse
from arguments import ModelParams, PipelineParams, get_combined_args

def get_attack_args():
    """Defines and parses command-line arguments for the adversarial attack."""
    parser = argparse.ArgumentParser("RGA physical attack baseline (GIR + MMDet)")
    
    # Add model and pipeline parameters from the original arguments file
    model_params = ModelParams(parser, sentinel=False)
    pipeline_params = PipelineParams(parser)

    # =================================================================================
    # Training / Evaluation
    # =================================================================================
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for the attack.')
    parser.add_argument('--max_cams', type=int, default=0, help='Limit number of cameras for quick testing (0=all).')
    parser.add_argument('--eval_on_train', default=False, action='store_true', help="Enable evaluation on the training set at the end of each epoch.")
    parser.add_argument('--run_final_eval', default=True, action=argparse.BooleanOptionalAction, help="Run a final, detailed evaluation on all data splits after training completes.")
    parser.add_argument('--eval_vis_interval', type=int, default=5, help="Save eval-test visualization every N epochs (0 disables). Default: 5.")

    # =================================================================================
    # Model / Scene Loading
    # =================================================================================
    parser.add_argument('--iteration', type=int, default=-1, help='Iteration to load; -1 means latest (GIR style).')
    parser.add_argument('--second_stage_step', type=int, default=30000, help="Step count for the second stage of rendering.")
    
    # =================================================================================
    # Detector
    # =================================================================================
    parser.add_argument('--detector', type=str, default='yolox', 
                        choices=['yolov3', 'yolox', 'faster-rcnn', 'mask-rcnn', 'd-detr', 'pvt', 'detr'],
                        help="Object detector to use for the attack.")
    parser.add_argument('--target_class_name', type=str, default='car', help="Target class name for the attack (COCO class).")
    parser.add_argument('--score_thresh', type=float, default=0.5, help="Score threshold for detection.")


 
    # =================================================================================
    # Albedo Optimization (Min Phase)
    # =================================================================================
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'], help='Optimizer to use for albedo.')
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for the albedo optimizer.")
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the SGD optimizer.') # 默认0.9
    parser.add_argument('--perturb_albedo', default=True, action=argparse.BooleanOptionalAction, help='随机初始化albedo')
    parser.add_argument('--perturb_budget_factor', type=float, default=1e-8, help='Factor (n) for calculating the albedo perturbation budget.')
    parser.add_argument('--albedo_init_method', type=str, default='perturb', choices=['perturb', 'random'], help='Albedo initialization method: "perturb" adds random noise to original albedo, "random" initializes randomly within original range.')
    parser.add_argument('--reg_loss_weight', type=float, default=0.0, help='Weight for the regression loss component in the total adversarial loss.')
    parser.add_argument(
        '--enable_loss_var_reg',
        default=True,
        action=argparse.BooleanOptionalAction,
        help='If enabled, add a regularization term in MIN phase: variance of per-sample losses within the current batch.'
    )
    parser.add_argument(
        '--loss_var_reg_weight',
        type=float,
        default=0.001,
        help='Weight (lambda) for the MIN-phase per-batch loss variance regularizer.'
    )

    # =================================================================================
    # Min-Max Adversarial Training (EnvLight)
    # =================================================================================
    parser.add_argument('--enable_min_max', default=True, action=argparse.BooleanOptionalAction, help='Enable Min-Max adversarial training for envlight.')
    parser.add_argument('--min_steps', type=int, default=5, help='Number of steps to optimize albedo (min phase).')
    parser.add_argument('--max_steps', type=int, default=1, help='Number of steps to optimize envlight (max phase).')
    parser.add_argument('--env_lr', type=float, default=1e-4, help='Learning rate for optimizing envlight in the max phase.')
    parser.add_argument('--diversity_lambda', type=float, default=0.1, help='Weight for diversity loss in the max phase.')
    parser.add_argument('--reset_envlight_each_epoch', default=False, action=argparse.BooleanOptionalAction, help='Reset envlight to its initial checkpoint state at the start of each epoch.')
    parser.add_argument('--shuffle_each_epoch', default=True, action=argparse.BooleanOptionalAction, help='Shuffle camera order at the start of each epoch.')
    parser.add_argument('--env_delta_max', type=float, default=1e-6, help='Hard clamp range for envlight.base_train parameter: [-env_delta_max, env_delta_max].')
    parser.add_argument('--env_init_bound', type=float, default=1e-6, help='Hard clamp range for envlight.init_base latent tensor: [-env_init_bound, env_init_bound].')
    
    # =================================================================================
    # SGLD Settings for Max Phase (optional)
    # =================================================================================
    parser.add_argument('--use_sgld', default=True, action='store_true', help='Use SGLD for envlight updates in the max phase instead of Adam.')
    parser.add_argument('--sgld_lr', type=float, default=1e-2, help='SGLD learning rate for envlight updates.')
    parser.add_argument('--sgld_noise_std', type=float, default=1e-4, help='Std of Gaussian noise injected in SGLD updates.')


    # =================================================================================
    # Environment / Relighting
    # =================================================================================
    parser.add_argument('--environment_texture', type=str, default="", help="Path to the environment texture (HDR).")
    parser.add_argument('--environment_scale', type=float, default=1.0, help="Scale of the environment light.")
    parser.add_argument('--hdr_rotation', action='store_true', default=False, help="Enable random rotation of the HDR environment map during training.")
    parser.add_argument('--enable_lbm_relight', default=True, action=argparse.BooleanOptionalAction, help='Enable LBM background relighting.')
    parser.add_argument('--lbm_ckpt_dir', type=str, default='/workspace/RGA/lbm_ckpt2', help='Path to LBM checkpoints directory.') # /workspace/lbm/checkpoints
    parser.add_argument('--hdr_bank_dir', type=str, default='/workspace/RGA/hdri/carla_hdr', help='Directory of HDR/EXR files to seed EnvLight replay buffer.')
    # HDR 可视化（训练前/训练后）
    parser.add_argument('--enable_hdr_bank_vis_pre', default=False, action=argparse.BooleanOptionalAction, help='Enable pre-training visualization: iterate HDR bank and render random views per HDR.')
    parser.add_argument('--enable_hdr_bank_vis_post', default=False, action=argparse.BooleanOptionalAction, help='Enable post-training visualization: render random views using updated envlight bases (ReplayBuffer/current).')
    parser.add_argument('--hdr_vis_num_views', type=int, default=5, help='Number of random camera views per HDR/base for visualization.')
    parser.add_argument('--hdr_vis_seed', type=int, default=0, help='Random seed for selecting views in HDR visualization.')

    # =================================================================================
    # Replay Buffer for EnvLight
    # =================================================================================
    parser.add_argument('--use_replay_buffer', default=True, action=argparse.BooleanOptionalAction, help='Enable replay buffer for envlight states in max phase.')
    parser.add_argument('--buffer_size', type=int, default=50, help='Max number of envlight states to keep in the replay buffer.')
    parser.add_argument(
        '--buffer_replace_strategy',
        type=str,
        default='replace_self',
        choices=['fifo', 'replace_self'],
        help="Replay buffer replacement policy when full: "
             "'fifo' pops the oldest; 'replace_self' replaces the entry that was sampled for the current max-phase."
    )


    # =================================================================================
    # LR Scheduler Settings (for SGD)
    # =================================================================================
    parser.add_argument('--use_lr_scheduler', default=True, action=argparse.BooleanOptionalAction, help="Enable learning rate scheduler for SGD optimizer.")
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine', choices=['cosine', 'step'], help="Type of learning rate scheduler to use.")
    parser.add_argument('--lr_step_size', type=int, default=10, help="Step size for StepLR scheduler.")
    parser.add_argument('--lr_gamma', type=float, default=0.5, help="Gamma for StepLR scheduler.")

    # =================================================================================
    # I/O and System
    # =================================================================================
    parser.add_argument('--save_dir', type=str, default='RGA_output', help="Directory to save outputs.")
    parser.add_argument('--device', type=str, default='cuda', help="Device to run the training on.")

    # =================================================================================
    # Debug / Visualization I/O (optional)
    # =================================================================================
    parser.add_argument('--save_temp_imgs_for_det', default=False, action=argparse.BooleanOptionalAction, help="Save temp images used for detector input (temp_imgs_for_det/*).")
    parser.add_argument('--save_visualizations', default=False, action=argparse.BooleanOptionalAction, help="Save training visualization grids (visualizations/*).")
    
    # =================================================================================
    # Final Visualization (optional)
    # =================================================================================
    parser.add_argument('--save_final_full_images_mw', default=True, action=argparse.BooleanOptionalAction, help="At final stage, render multi-weather images and run OFFLINE evaluation on them.")
    parser.add_argument('--save_final_eval_vis', default=True, action=argparse.BooleanOptionalAction, help="During final offline evaluation, save detection results (images with bboxes).")


    args = get_combined_args(parser)
    return args, model_params, pipeline_params

if __name__ == '__main__':
    args, _, _ = get_attack_args()
    print("--- Parsed Arguments ---")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")
    print("------------------------")
