import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torchvision.transforms as transforms
import cv2
from sugar_scene.gs_model import GaussianSplattingWrapper
from sugar_utils.general_utils import str2bool
import open3d as o3d
from sugar_scene.sugar_model import SuGaR, extract_texture_image_and_uv_from_gaussians, \
    convert_refined_sugar_into_gaussians
from main_utils import NPSCalculator
from sugar_utils.spherical_harmonics import (
    eval_sh, RGB2SH, SH2RGB,
)
import main_utils
import json
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    ssd300_vgg16,
    maskrcnn_resnet50_fpn
)
from PIL import Image
from mmdet.apis import init_detector, inference_detector, inference_detector_custom

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Script to train a full macarons model in large 3D scenes.')
    parser.add_argument('-s', '--scene_path', type=str, help='(Required) path to the scene data to use.')
    parser.add_argument('-i', '--iteration_to_load', type=int, default=7000, help='iteration to load.')
    parser.add_argument('-c', '--checkpoint_path', type=str, help='(Required) path to the vanilla 3D Gaussian Splatting Checkpoint to load.')
    parser.add_argument('-m', '--refined_model_path', type=str, help='(Required) Path to the refine model checkpoint.')
    parser.add_argument('-o', '--mesh_output_dir', type=str, default=None, help='path to the output directory.')
    parser.add_argument('-n', '--n_gaussians_per_surface_triangle', default=None, type=int, help='Number of gaussians per surface triangle.')
    parser.add_argument('--square_size', default=None, type=int, help='Size of the square to use for the texture.')
    parser.add_argument('--eval', type=str2bool, default=True, help='Use eval split.')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='Index of GPU to use.')
    parser.add_argument('--postprocess_mesh', type=str2bool, default=False, help='If True, postprocess the mesh.')
    parser.add_argument('--postprocess_density_threshold', type=float, default=0.1, help='Threshold to use for postprocessing the mesh.')
    parser.add_argument('--postprocess_iterations', type=int, default=5, help='Number of iterations for mesh postprocessing.')
    parser.add_argument('--use_white_background', type=bool, default=False, help='Use white background.')
    parser.add_argument("--data", type=str, default='./yolov5/data/028.yaml', help='datasets yaml')
    parser.add_argument("--printfile", type=str, default='general_utils/30values.txt', help='NPS file')

    parser.add_argument('--one_batch_iters', type=int, default=5, help='')
    parser.add_argument('--attack_iters', type=int, default=10, help='')
    parser.add_argument('--batch_size', type=int, default=8, help='')

    parser.add_argument('--nps_weight', type=float, default=0., help='Weight of NPS.')
    parser.add_argument('--clr_weight', type=float, default=0.3, help='Weight of Color loss.')
    parser.add_argument('--reg_weight', type=float, default=0., help='Weight of regularization.')
    parser.add_argument('--use_augment', type=bool, default=False, help='')

    parser.add_argument('--save_ori_view', type=bool, default=False, help='')
    parser.add_argument('--synth_views', type=str, default='synth_views_gm.pth',
                        help='Path to the .pth file containing distilled views (keys: c2ws, centers).')

    # parser.add_argument('--inter_traj', type=bool, default=True, help='')
    # parser.add_argument('--exp_dir', type=str, default='exp_vd_cloudy2_output')
    # parser.add_argument('--anno_path', type=str, default=f'vis_res/vd_cloudy2_anno', help='')
    # parser.add_argument('--mask_path', type=str, default=f'seg_car_mask/red_mask_vd_cloudy2', help='')
    # parser.add_argument('--ref_path', type=str, default=f'vis_res/exp_vd_cloudy2', help='')
    # parser.add_argument('--sugar_mesh_path', type=str, default=f'output_vd_cloudy2', help='')

    parser.add_argument('--inter_traj', type=bool, default=True, help='')
    parser.add_argument('--exp_dir', type=str, default='exp_vd_cloudy2_output_gm')
    parser.add_argument('--anno_path', type=str, default=f'vis_res/vd_cloudy2_anno_gm', help='')
    parser.add_argument('--mask_path', type=str, default=f'seg_car_mask/red_mask_vd_cloudy2_gm', help='')
    parser.add_argument('--ref_path', type=str, default=f'vis_res/exp_vd_cloudy2_gm', help='')
    parser.add_argument('--sugar_mesh_path', type=str, default=f'output_vd_cloudy2', help='')

    args = parser.parse_args()
    return args

def attack():
    #############
    args = parse_args()
    LEARNING_RATE = 0.05

    # 创建结果目录
    os.makedirs(f'./vis_res/{args.exp_dir}', exist_ok=True)
    os.makedirs(f'./adv_gs_res/{args.exp_dir}', exist_ok=True)
    os.makedirs(f'./vis_detect_res/{args.exp_dir}', exist_ok=True)

    ##########################
    # 初始化检测模型
    ##### yolox
    config_file = '/workspace/mmdet/configs/yolox/yolox_l_8xb8-300e_coco.py'
    ckp_file = '/workspace/mmdet/checkpoints/yolox_l_8x8_300e_coco.pth'
    ##### yolov3
    # config_file = '/workspace/mmdet/configs/yolo/yolov3_d53_8xb8-amp-ms-608-273e_coco.py'
    # ckp_file = '/workspace/mmdet/checkpoints/yolov3_d53_fp16_mstrain-608_273e_coco.pth'
    ##### fr
    # config_file = '/workspace/mmdet/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
    # ckp_file = '/workspace/mmdet/checkpoints/faster_rcnn_r50_fpn_1x_coco.pth'

    tar_model = init_detector(config_file, ckp_file, device=device)
    for p in tar_model.parameters():
        p.requires_grad = False
    tar_model.eval()
    if not hasattr(tar_model, 'CLASSES'):
        tar_model.CLASSES = main_utils.coco_classes

    ##########################
    # 加载 Gaussian Splatting
    source_path = args.scene_path
    iteration_to_load = args.iteration_to_load
    gs_checkpoint_path = args.checkpoint_path
    use_train_test_split = args.eval
    refined_model_path = args.refined_model_path
    n_skip_images_for_eval_split = 8

    nerfmodel = GaussianSplattingWrapper(
        source_path=source_path,
        output_path=gs_checkpoint_path,
        iteration_to_load=iteration_to_load,
        load_gt_images=False,
        eval_split=use_train_test_split,
        eval_split_interval=n_skip_images_for_eval_split,
    )

    scene_name = os.path.basename(source_path.rstrip('/'))
    sugar_mesh_path = os.path.join(
        args.sugar_mesh_path, 'coarse_mesh', scene_name,
        refined_model_path.split('/')[-2]
            .split('_normalconsistency')[0]
            .replace('sugarfine', 'sugarmesh') + '.ply'
    )
    o3d_mesh = o3d.io.read_triangle_mesh(sugar_mesh_path)
    checkpoint = torch.load(refined_model_path, map_location=nerfmodel.device)

    if args.n_gaussians_per_surface_triangle is None:
        n_gaussians_per_surface_triangle = int(
            refined_model_path.split('/')[-2].split('_gaussperface')[-1]
        )
    else:
        n_gaussians_per_surface_triangle = args.n_gaussians_per_surface_triangle

    refined_sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=checkpoint['state_dict']['_points'],
        colors=SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'][:, 0, :]),
        initialize=False,
        sh_levels=nerfmodel.gaussians.active_sh_degree + 1,
        keep_track_of_knn=False,
        knn_to_track=0,
        beta_mode='average',
        surface_mesh_to_bind=o3d_mesh,
        n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle,
    )
    refined_sugar.load_state_dict(checkpoint['state_dict'])
    refined_sugar.train()

    ##########################
    # 渲染 & 优化参数
    if args.use_white_background:
        bg_tensor = torch.ones(3, dtype=torch.float, device=nerfmodel.device)
    else:
        bg_tensor = None
    current_sh_levels = 4
    compute_color_in_rasterizer = False
    use_densifier = False
    regularize = False
    use_same_scale_in_all_directions = False
    enforce_entropy_regularization = False

    gs_sh_dc = refined_sugar._sh_coordinates_dc
    gs_sh_dc.requires_grad_()
    ori_gs_sh_dc = gs_sh_dc.clone().detach()
    # optimizer = optim.Adam([gs_sh_dc], lr=LEARNING_RATE)
    optimizer = optim.AdamW([gs_sh_dc],
                            lr=LEARNING_RATE,
                            betas=(0.9, 0.999),
                            eps=1e-8,
                            weight_decay=1e-4)
    transform = transforms.ToTensor()
    nps_calculator = NPSCalculator(
        args.printfile, refined_sugar.image_height, refined_sugar.image_width
    ).cuda()

    attack_iters = args.attack_iters
    p3d_cameras = refined_sugar.nerfmodel.training_cameras.p3d_cameras
    c2ws_all = refined_sugar.nerfmodel.training_cameras.camera_to_worlds
    z_near = p3d_cameras[0].znear.item()
    z_far = p3d_cameras[0].zfar.item()
    K = p3d_cameras[0].K
    # 相机 & 轨迹插值
    if args.inter_traj:
        new_views = 2000
        new_c2ws, new_centers = main_utils.interpolate_trajectory(c2ws_all, num_new=new_views)
        torch.save({'c2ws': new_c2ws, 'centers': new_centers}, 'ori_views_1000.pth')
    else:
        new_c2ws, new_centers = torch.load(args.synth_views)['c2ws'], torch.load(args.synth_views)['centers']
        new_views = new_centers.shape[0]

    batch_size = args.batch_size
    image_h = refined_sugar.image_height
    image_w = refined_sugar.image_width

    # ======== 筛选有效视角 ========
    valid_views = []
    if args.save_ori_view:
        for v in range(new_views):
            valid_views.append(v)
    else:
        for v in range(new_views):
            anno_path = args.anno_path + f'/results_view{v}_0.json'
            if os.path.exists(anno_path):
                gt_bbox, _ = main_utils.load_labelme_annotation(anno_path)
                if gt_bbox.shape[0] == 1:
                    valid_views.append(v)
    print(f'Found {len(valid_views)} valid views to attack')

    # ======== 批次渲染 + 攻击函数 ========
    def attack_batch(idx_batch, iteration):
        iter_conf_batch = 0.0
        valid_count = 0

        # 渲染
        tmp_list, mask_list, ref_list = [], [], []
        for v in idx_batch:
            outputs = refined_sugar.render_image_gaussian_rasterizer_custom(
                z_near=z_near,
                z_far=z_far,
                camera_center=new_centers[v],
                K=K,
                c2w=new_c2ws[v],
                bg_color=bg_tensor,
                sh_deg=current_sh_levels - 1,
                sh_rotations=None,
                compute_color_in_rasterizer=compute_color_in_rasterizer,
                compute_covariance_in_rasterizer=True,
                return_2d_radii=use_densifier or regularize,
                quaternions=None,
                use_same_scale_in_all_directions=use_same_scale_in_all_directions,
                return_opacities=enforce_entropy_regularization,
            )
            pred_rgb = outputs.view(1, image_h, image_w, 3)
            tmp_rgb = pred_rgb.permute(0, 3, 1, 2).squeeze(0)
            tmp_list.append(tmp_rgb)
            if not args.save_ori_view:
                mask_v = transform(
                    Image.open(args.mask_path + f"/results_view{v}_0_mask.png")
                ).cuda().unsqueeze(0).expand(-1, 3, -1, -1)
                mask_v = F.interpolate(mask_v, size=(image_h, image_w), mode='nearest')
                mask_list.append(mask_v.squeeze(0))

                ref_v = transform(
                    Image.open(args.ref_path + f"/results_view{v}_0.png")
                ).cuda()
                ref_list.append(ref_v)

        tmp_rgbs_bg = torch.stack(tmp_list, dim=0)
        if not args.save_ori_view:
            masks = torch.stack(mask_list, dim=0)
            refs = torch.stack(ref_list, dim=0)
            res_imgs = torch.where(masks.bool(), tmp_rgbs_bg, refs)
            nps_losses = [
                nps_calculator(masks[n] * tmp_rgbs_bg[n])
                for n in range(tmp_rgbs_bg.shape[0])
            ]
            nps_loss = args.nps_weight * torch.stack(nps_losses).sum()
            reg_loss = args.reg_weight * torch.norm(ori_gs_sh_dc - gs_sh_dc, p=2)
        else:
            res_imgs = tmp_rgbs_bg
            nps_loss = torch.tensor(0.0).cuda()
            reg_loss = torch.tensor(0.0).cuda()

        res_imgs = torch.clamp(res_imgs, 0.0, 1.0)
        inputs = (
            main_utils.augment_image(res_imgs)
            if args.use_augment
            else res_imgs
        )
        loss = torch.tensor(0.0, requires_grad=True).cuda()
        if not args.save_ori_view:
            # 检测
            resized_inputs = inputs.permute(0, 2, 3, 1)
            preds = inference_detector_custom(tar_model, resized_inputs * 255)   # [b,h,w,3]
            # feats_adv = tar_model.extract_feat(inputs)
            # feats_clean = tar_model.extract_feat(refs)

            # 计算对抗 loss
            for i_in_batch, view_idx in enumerate(idx_batch):
                # print(f'MINING view {view_idx}...')
                pred = preds[i_in_batch]
                th_mask = pred.pred_instances.scores >= 0.5
                # tmp_adv_loss, _ = main_utils.compute_adv_loss(
                #     pred_bboxes=pred.pred_instances.bboxes[th_mask],
                #     pred_scores=pred.pred_instances.scores[th_mask],
                #     pred_classes=pred.pred_instances.labels[th_mask],
                #     gt_bbox=main_utils.load_labelme_annotation(
                #         args.anno_path + f'/results_view{view_idx}_0.json'
                #     )[0],
                #     target_class_idx=3 - 1
                # )

                tmp_adv_loss, _ = main_utils.compute_adv_total_loss(
                    pred_bboxes=pred.pred_instances.bboxes[th_mask],
                    pred_scores=pred.pred_instances.scores[th_mask],
                    pred_classes=pred.pred_instances.labels[th_mask],
                    gt_bbox=main_utils.load_labelme_annotation(
                        args.anno_path + f'/results_view{view_idx}_0.json'
                    )[0].cuda(),
                    target_class_idx=2,
                )

                loss = loss + tmp_adv_loss
                valid_count += 1

                loss = loss + nps_loss + reg_loss

            print(f'Batch({idx_batch[0]}~{idx_batch[-1]}) loss: {loss.item():.4f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 保存中间视图
        if iteration % 5 == 0:
            for i_in_batch, view_idx in enumerate(idx_batch):
                res_img_np = (
                    res_imgs[i_in_batch].permute(1, 2, 0)
                    .detach().cpu().numpy() * 255
                ).astype(np.uint8)
                cv2.imwrite(
                    f'./vis_res/{args.exp_dir}/results_view{view_idx}_{iteration}.png',
                    cv2.cvtColor(res_img_np, cv2.COLOR_RGB2BGR)
                )

        return loss.item(), valid_count

    ##########################
    # 主循环
    iter_conf = 0.0
    iter_num = 0
    for i in range(attack_iters + 1):
        print(f'attack iter {i}')
        for start in range(0, len(valid_views), batch_size):
            for j in range(args.one_batch_iters):
                print(f'one_batch_iter:{j}')
                idx_batch = valid_views[start:start + batch_size]
                batch_conf, batch_valid = attack_batch(idx_batch, i)
                iter_conf += batch_conf
                iter_num += batch_valid

        if iter_num > 0:
            print(f"Avg conf of iter {i}: {iter_conf / iter_num:.4f}")

        # 定期保存模型和点云
        if i % 10 == 0:
            refined_sugar.save_model(
                path=f"./adv_gs_res/{args.exp_dir}/sugar_{i}.pt",
                train_losses=None, epoch=None, iteration=None,
                optimizer_state_dict=optimizer.state_dict(),
            )
            gaussians = refined_sugar.nerfmodel.gaussians
            gaussians.save_ply(f"./adv_gs_res/{args.exp_dir}/gs_point_cloud_{i}.ply")
            refined_gaussians = convert_refined_sugar_into_gaussians(refined_sugar)
            refined_gaussians.save_ply(f"./adv_gs_res/{args.exp_dir}/gs_point_cloud_{i}.ply")

        if args.save_ori_view:
            print("Already Saved All Ori Views.")
            break

if __name__ == '__main__':
    args = parse_args()
    attack()
