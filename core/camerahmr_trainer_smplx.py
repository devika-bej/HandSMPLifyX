import torch
import pickle
import smplx 
import pytorch_lightning as pl
from typing import Dict
from omegaconf import DictConfig
from loguru import logger
import numpy as np

from .backbones import create_backbone
from .losses_mean import (
    Keypoint3DLoss, Keypoint2DLoss, Keypoint2DLossScaled,
    ParameterLoss, VerticesLoss, TranslationLoss, HandVerticesLoss
)
from .cam_model.fl_net import FLNet
from .smplx_wrapper import SMPLXLayer_
from .heads.smplx_head_cliff_with_hands import build_smplx_head
from .utils.train_utils import (
    trans_points2d_parallel, load_valid, perspective_projection,
    convert_to_full_img_cam
)
from .utils.eval_utils import pck_accuracy, reconstruction_error
from .utils.geometry import aa_to_rotmat
from .utils.pylogger import get_pylogger
from .utils.renderer_cam import render_image_group
from .constants import (
    NUM_JOINTS_SMPLX, H36M_TO_J14, CAM_MODEL_CKPT, DOWNSAMPLE_MAT, NUM_BETAS_SMPLX,
    REGRESSOR_H36M, VITPOSE_BACKBONE, SMPL_MODEL_DIR, SMPLX_MODEL_DIR, SMPLX2SMPL
)

log = get_pylogger(__name__)

def scale_and_translation_transform_batch(P, T):
    P = P.detach().cpu().numpy()
    T = T.detach().cpu().numpy()
    P_mean = np.mean(P, axis=1, keepdims=True)
    P_trans = P - P_mean
    P_scale = np.sqrt(np.sum(P_trans ** 2, axis=(1, 2), keepdims=True) / P.shape[1])
    P_normalised = P_trans / P_scale

    T_mean = np.mean(T, axis=1, keepdims=True)
    T_scale = np.sqrt(np.sum((T - T_mean) ** 2, axis=(1, 2), keepdims=True) / T.shape[1])

    P_transformed = P_normalised * T_scale + T_mean

    return P_transformed

class CameraHMR(pl.LightningModule):
    """
    Pytorch Lightning Module for Camera Human Mesh Recovery (CameraHMR) with SMPLX.
    This module integrates backbone feature extraction, camera modeling, SMPLX fitting,
    and loss functions for training a 3D human mesh recovery pipeline.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['init_renderer'])
        self.cfg = cfg

        # Backbone feature extractor
        self.backbone = create_backbone()
        self.backbone.load_state_dict(torch.load(VITPOSE_BACKBONE, map_location='cpu')['state_dict'])

        # Camera model
        self.cam_model = FLNet()
        load_valid(self.cam_model, CAM_MODEL_CKPT)

        # SMPLX Head
        self.smplx_head = build_smplx_head()

        # Loss functions
        loss_type = cfg.TRAIN.LOSS_TYPE
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type=loss_type)
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type=loss_type)
        self.keypoint_2d_loss_scaled = Keypoint2DLossScaled(loss_type=loss_type)
        self.trans_loss = TranslationLoss(loss_type=loss_type)
        self.vertices_loss = VerticesLoss(loss_type=loss_type)
        self.hand_vertices_loss = HandVerticesLoss(loss_type=loss_type)
        self.smplx_parameter_loss = ParameterLoss()

        # SMPLX model
        self.smplx_layer = SMPLXLayer_(model_path=SMPLX_MODEL_DIR, use_pca=False, num_betas=NUM_BETAS_SMPLX, flat_hand_mean=True)

        # For validation on SMPL datasets
        self.smpl = smplx.SMPL(model_path=SMPL_MODEL_DIR)
        smplx2smpl = pickle.load(open(SMPLX2SMPL, 'rb'))
        self.register_buffer('smplx2smpl_matreg', torch.tensor(smplx2smpl['matrix'][None], dtype=torch.float32))

        # Initialize ActNorm layers flag
        self.register_buffer('initialized', torch.tensor(False))

        # Disable automatic optimization
        self.automatic_optimization = False

        # Additional configurations
        self.J_regressor = torch.from_numpy(np.load(REGRESSOR_H36M))
        # Store validation outputs
        self.validation_step_output = []

    def get_parameters(self):
        """Aggregate model parameters for optimization."""
        return list(self.smplx_head.parameters()) + list(self.backbone.parameters())

    def configure_optimizers(self):
        """Configure optimizers for training."""
        optimizer = torch.optim.AdamW(
            params=[
                {
                    'params': filter(lambda p: p.requires_grad, self.get_parameters()),
                    'lr': self.cfg.TRAIN.LR
                }
            ],
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY
        )
        return optimizer

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        # Use RGB image as input
        x = batch['img']
        batch_size = x.shape[0]

        # Compute conditioning features using the backbone
        conditioning_feats = self.backbone(x[:,:,:,32:-32])

        cx, cy = batch['box_center'][:, 0], batch['box_center'][:, 1]

        b = batch['box_size']
        img_h = batch['img_size'][:,0]
        img_w = batch['img_size'][:,1]
        if train:
            cam_intrinsics = batch['cam_int']
            fl_h = cam_intrinsics[:,0,0]
        else:
           cam_intrinsics = batch['cam_int']
           fl_h = cam_intrinsics[:,0,0]
           cam, features = self.cam_model(batch['img_full_resized'])
           vfov = cam[:, 1]
           fl_h = (img_h / (2 * torch.tan(vfov / 2)))
           cam_intrinsics[:,0,0]=fl_h
           cam_intrinsics[:,1,1]=fl_h

        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b],
                                dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / cam_intrinsics[:, 0, 0].unsqueeze(-1)
        bbox_info[:, 2] = (bbox_info[:, 2] / cam_intrinsics[:, 0, 0])

        bbox_info = bbox_info.cuda().float()
        pred_smpl_params, pred_cam, _ = self.smplx_head(conditioning_feats, bbox_info=bbox_info)

        # Compute model vertices, joints and the projected joints
        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size, -1)
        smpl_output = self.smplx_layer(**{k: v.float() for k,v in pred_smpl_params.items()})


        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices
        pred_hand_joints = smpl_output.hand_joints
        pred_face_joints =smpl_output.face_joints
        pred_feet_joints =smpl_output.feet_joints
        pred_lhand_verts = smpl_output.lhand_verts
        pred_rhand_verts = smpl_output.rhand_verts
        pred_body_verts = smpl_output.body_verts
        # Initialize the output dictionary
        output = {
            'pred_keypoints_3d': pred_keypoints_3d.view(batch_size, -1, 3),
            'pred_vertices': pred_vertices.view(batch_size, -1, 3),
            'pred_cam': pred_cam,
            'pred_hand_joints':pred_hand_joints,
            'pred_face_joints':pred_face_joints,
            'pred_feet_joints':pred_feet_joints,
            'pred_lhand_verts':pred_lhand_verts,
            'pred_rhand_verts':pred_rhand_verts,
            'pred_body_verts':pred_body_verts,
            'pred_smpl_params': {k: v.clone() for k, v in pred_smpl_params.items()}
        }

        # Compute camera translation
        device = pred_smpl_params['body_pose'].device
        cam_t = convert_to_full_img_cam(
            pare_cam=output['pred_cam'],
            bbox_height=batch['box_size'],
            bbox_center=batch['box_center'],
            img_w=img_w,
            img_h=img_h,
            focal_length=batch['cam_int'][:, 0, 0]
        )
        output['pred_cam_t'] = cam_t

        # Project 3D joints to 2D using the perspective projection
        joints2d = perspective_projection(
            output['pred_keypoints_3d'],
            rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
            translation=cam_t,
            cam_intrinsics=batch['cam_int']
        )
        output['pred_keypoints_2d'] = joints2d.view(batch_size, -1, 2)
        # if train:
        # self.perspective_projection_vis(batch, output)
        return output, fl_h
    
    def perspective_projection_vis(self, input_batch, output, max_save_img=1):
        import os
        import cv2

        translation = output['pred_cam_t'].detach()
        vertices = output['pred_vertices'].detach()

        # translation = input_batch['translation'].detach()[:,:3]
        # vertices = input_batch['gt_vertices'].detach()
        for i in range(len(input_batch['imgname'])):
            cy, cx = input_batch['img_size'][i] // 2
            img_h, img_w = cy*2, cx*2
            imgname = input_batch['imgname'][i]
            save_filename = os.path.join('.', f'{self.global_step:08d}_{i:02d}_{os.path.basename(imgname)}')
            # focal_length_ = (img_w * img_w + img_h * img_h) ** 0.5  # Assumed fl
      
            focal_length_ = input_batch['cam_int'][i, 0, 0]
            focal_length = (focal_length_, focal_length_)

            rendered_img = render_image_group(
                image=cv2.imread(imgname),
                camera_translation=translation[i],
                vertices=vertices[i],
                focal_length=focal_length,
                camera_center=(cx, cy),
                camera_rotation=None,
                save_filename=save_filename,
                faces=self.smplx.faces,
            )
            # if i >= (max_save_img - 1):
            #     break

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        # Extract predictions
        pred_smpl_params = output['pred_smpl_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']
        
        # Get batch size, device, and data type for consistency
        batch_size = pred_smpl_params['body_pose'].shape[0]
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype

        # Get ground truth annotations
        gt_keypoints_3d = batch['keypoints_3d']
        gt_smpl_params = batch['smpl_params']
        
        # Image size for projection normalization
        img_size = batch['img_size'].rot90().T.unsqueeze(1)

        # Calculate 3D keypoint loss
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d[:, :NUM_JOINTS_SMPLX], gt_keypoints_3d[:, :NUM_JOINTS_SMPLX], pelvis_id=25 + 14)
        # Calculate vertices loss
        loss_body_vertices = self.vertices_loss(output['pred_body_verts'], batch['gt_body_verts'])
        loss_lhand_vertices = self.hand_vertices_loss(output['pred_lhand_verts'], batch['gt_lhand_verts'], output['pred_hand_joints'][:,:21,:], batch['gt_hand_joints'][:,:21,:-1])
        loss_rhand_vertices = self.hand_vertices_loss(output['pred_rhand_verts'], batch['gt_rhand_verts'], output['pred_hand_joints'][:,21:,:], batch['gt_hand_joints'][:,21:,:-1])
        loss_hand_joints= self.keypoint_3d_loss(output['pred_hand_joints'], batch['gt_hand_joints'], pelvis_id=0)
        # Compute loss on SMPL parameters
        loss_smpl_params = {}
        for param_name, pred_param in pred_smpl_params.items():
            gt_param = gt_smpl_params[param_name].view(batch_size, -1)
            if 'betas' not in param_name:
                gt_param = aa_to_rotmat(gt_param.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            # if 'hand' in param_name or 'jaw' in param_name or 'eye' in param_name:
            #     continue
            loss_smpl_params[param_name] = self.smplx_parameter_loss(pred_param.view(batch_size, -1), gt_param.view(batch_size, -1))

        # Total loss calculation
        loss = (
            self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d +
            sum([loss_smpl_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_smpl_params]) +
            self.cfg.LOSS_WEIGHTS['HAND_VERTICES'] * (loss_body_vertices + loss_lhand_vertices + loss_rhand_vertices) + 
            self.cfg.LOSS_WEIGHTS['HAND_KEYPOINTS_3D'] * loss_hand_joints 
        )

        # If configured, compute 2D cropped vertices loss
        if self.cfg.LOSS_WEIGHTS.get('VERTS2D_CROP'):
            gt_verts2d = batch['proj_verts']
            pred_verts2d = output['pred_verts2d']
            pred_verts2d_cropped = trans_points2d_parallel(pred_verts2d, batch['_trans']) / self.cfg.MODEL.IMAGE_SIZE - 0.5
            gt_verts_2d_cropped = trans_points2d_parallel(gt_verts2d[:, :, :2], batch['_trans']) / self.cfg.MODEL.IMAGE_SIZE - 0.5
            loss_proj_vertices_cropped = self.keypoint_2d_loss(pred_verts2d_cropped, gt_verts_2d_cropped)
            loss += self.cfg.LOSS_WEIGHTS['VERTS2D_CROP'] * loss_proj_vertices_cropped

        if self.cfg.LOSS_WEIGHTS.get('HAND_KEYPOINTS_2D_CROP'):
            lh_gt_keypoints_2d_cropped = trans_points2d_parallel(batch['orig_keypoints_2d'][:,69:84,:2], batch['_trans_lh']) / self.cfg.MODEL.IMAGE_SIZE - 0.5
            lh_pred_keypoints_2d_cropped = trans_points2d_parallel( output['pred_keypoints_2d'][:,69:84], batch['_trans_lh']) / self.cfg.MODEL.IMAGE_SIZE - 0.5
            rh_gt_keypoints_2d_cropped = trans_points2d_parallel(batch['orig_keypoints_2d'][:,84:99,:2], batch['_trans_rh']) / self.cfg.MODEL.IMAGE_SIZE - 0.5
            rh_pred_keypoints_2d_cropped = trans_points2d_parallel( output['pred_keypoints_2d'][:,84:99], batch['_trans_rh']) / self.cfg.MODEL.IMAGE_SIZE - 0.5
            lh_gt_keypoints_2d_cropped_ = torch.cat([lh_gt_keypoints_2d_cropped,batch['orig_keypoints_2d'][:,69:84,2].unsqueeze(-1)], dim=-1)
            rh_gt_keypoints_2d_cropped_ = torch.cat([rh_gt_keypoints_2d_cropped, batch['orig_keypoints_2d'][:,84:99,2].unsqueeze(-1)], dim=-1)
            loss_hand_keypoints_2d_cropped = self.keypoint_2d_loss(lh_pred_keypoints_2d_cropped, lh_gt_keypoints_2d_cropped_)+ self.keypoint_2d_loss(rh_pred_keypoints_2d_cropped, rh_gt_keypoints_2d_cropped_)
            loss += self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D_CROP'] * loss_hand_keypoints_2d_cropped

        # If configured, compute 2D cropped keypoints loss
        if self.cfg.LOSS_WEIGHTS.get('KEYPOINTS_2D_CROP'):
            gt_keypoints_2d = batch['keypoints_2d'].clone()
            pred_keypoints_2d_cropped = trans_points2d_parallel(pred_keypoints_2d, batch['_trans']) / self.cfg.MODEL.IMAGE_SIZE - 0.5
            # print(pred_keypoints_2d_cropped[:, :NUM_JOINTS_SMPLX].shape, gt_keypoints_2d[:, :NUM_JOINTS_SMPLX].shape)
            loss_keypoints_2d_cropped = self.keypoint_2d_loss(pred_keypoints_2d_cropped[:, :NUM_JOINTS_SMPLX], gt_keypoints_2d[:, :NUM_JOINTS_SMPLX])
            loss += self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D_CROP'] * loss_keypoints_2d_cropped

        # If configured, compute translation loss
        if self.cfg.LOSS_WEIGHTS.get('TRANS_LOSS'):
            gt_trans = batch['translation'][:, :3]
            pred_trans = output['pred_cam_t']
            loss_trans = self.trans_loss(pred_trans, gt_trans)
            loss += self.cfg.LOSS_WEIGHTS['TRANS_LOSS'] * loss_trans

        #Add all hand losses

        # Collect individual losses for reporting
        losses = {
            'loss': loss.detach(),
            'loss_keypoints_3d': loss_keypoints_3d.detach(),
            'loss_vertices': loss_body_vertices.detach(),
            'loss_lhand_vertices': loss_lhand_vertices.detach(),
            'loss_rhand_vertices': loss_rhand_vertices.detach(),
            'loss_hand_joints': loss_hand_joints.detach(),
            'loss_kp2d_cropped': loss_keypoints_2d_cropped.detach() if 'loss_keypoints_2d_cropped' in locals() else None,
            'loss_hand_kp2d_cropped': loss_hand_keypoints_2d_cropped.detach() if 'loss_hand_keypoints_2d_cropped' in locals() else None,

        }
        
        # Add SMPL parameter losses
        for k, v in loss_smpl_params.items():
            losses[f'loss_{k}'] = v.detach()
    
        output['losses'] = losses
        return loss

    def forward(self, batch: Dict) -> Dict:
        return self.forward_step(batch, train=False)

    def training_step(self, joint_batch: Dict, batch_idx: int) -> Dict:
        batch = joint_batch['img']
        optimizer = self.optimizers(use_pl_optimizer=True)

        output,_ = self.forward_step(batch, train=True) 
        loss = self.compute_loss(batch, output, train=True)
        if torch.isnan(loss):
            log.error(f"NaN loss detected at batch_idx: {batch_idx}, imgname: {batch.get('imgname', 'N/A')}")
            for k,v in output['losses'].items():
                log.error(f"Loss component {k}: {v}")

        optimizer.zero_grad()
        self.manual_backward(loss)
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL, error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        optimizer.step()

        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        for k, v in output['losses'].items():
            if k != 'loss' and v is not None:
                self.log(f'train/{k}', v, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return output

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        batch_size = batch['img'].shape[0]
        output,_ = self.forward_step(batch, train=False)
        dataset_names = batch['dataset']
        joint_mapper_h36m = H36M_TO_J14
        J_regressor_batch_smpl = self.J_regressor[None, :].expand(batch['img'].shape[0], -1, -1).float().cuda()

        if '3dpw' in dataset_names[0]: # 14 joints evaluation
            neutral_hand_pose = torch.eye(3).repeat(batch_size, 15, 1, 1).cuda()
            pred_smpl_params = output['pred_smpl_params']
            smplx_neutral_hand_output = self.smplx_layer(global_orient=pred_smpl_params['global_orient'], 
                                            body_pose=pred_smpl_params['body_pose'], 
                                            left_hand_pose=neutral_hand_pose, 
                                            right_hand_pose=neutral_hand_pose, 
                                            betas=pred_smpl_params['betas'])
            # For 3dpw vertices are generated in dataset.py because gender is needed
            gt_cam_vertices = batch['vertices']
            # Get 14 predicted joints from the mesh
            gt_keypoints_3d = torch.matmul(J_regressor_batch_smpl, gt_cam_vertices)
            gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
            gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            gt_cam_vertices = gt_cam_vertices - gt_pelvis
            # Convert predicted vertices to SMPL Fromat
            # Get 14 predicted joints from the mesh
            pred_cam_vertices = smplx_neutral_hand_output.vertices 
            pred_cam_vertices = torch.matmul(self.smplx2smpl_matreg.to(pred_cam_vertices.device), pred_cam_vertices)

            pred_keypoints_3d = torch.matmul(J_regressor_batch_smpl, pred_cam_vertices)
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_cam_vertices = pred_cam_vertices - pred_pelvis
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1))
            error_verts = torch.sqrt(((pred_cam_vertices - gt_cam_vertices) ** 2).sum(dim=-1))
            error_lhand_joints = torch.zeros((batch_size, 21))
            error_rhand_joints = torch.zeros((batch_size, 21))
            r_error, _ = reconstruction_error(
                pred_keypoints_3d.float().cpu().numpy(),
                gt_keypoints_3d.float().cpu().numpy(),
                reduction=None
            )
            
        else:
            if 'shape-eval' in dataset_names[0]:
                neutral_hand_pose = torch.eye(3).repeat(batch_size, 15, 1, 1).cuda()
                neutral_body_pose = torch.eye(3).repeat(batch_size, 21, 1, 1).cuda()
                neutral_orient = torch.eye(3).repeat(batch_size, 1, 1, 1).cuda()
                pred_smpl_params = output['pred_smpl_params']
                smplx_neutral_hand_output = self.smplx_layer(global_orient=neutral_orient,
                                                body_pose=neutral_body_pose,
                                                left_hand_pose=neutral_hand_pose, 
                                                right_hand_pose=neutral_hand_pose, 
                                                betas=pred_smpl_params['betas'])
                gt_cam_vertices = batch['vertices']
                pred_vertices = smplx_neutral_hand_output.vertices 

                pred_vertices_aligned = torch.tensor(
                    scale_and_translation_transform_batch(pred_vertices, gt_cam_vertices)
                )
                gt_np = gt_cam_vertices.detach().cpu().numpy()
                pred_np = pred_vertices_aligned.detach().cpu().numpy()
                r_error = np.linalg.norm(gt_np - pred_np, axis=-1)  # shape: (B, N)
                error = torch.zeros((batch_size, 1))
                error_verts = torch.zeros((batch_size, 1))
                error_lhand_joints = torch.zeros((batch_size, 21))
                error_rhand_joints = torch.zeros((batch_size, 21))
                import trimesh
                import os

                for i in range(batch_size):
                    imgname = batch['imgname'][i]
                    save_filename = os.path.join('.', f'{self.global_step:08d}_{i:02d}_{os.path.basename(imgname)}')

                    gt_mesh = trimesh.Trimesh(
                        vertices=gt_cam_vertices[i].detach().cpu().numpy(),
                        faces=self.smpl.faces,
                        process=False
                    )

                    pred_mesh = trimesh.Trimesh(
                        vertices=pred_cam_vertices[i].detach().cpu().numpy(),
                        faces=self.smpl.faces,
                        process=False
                    )

                    # Combine meshes
                    combined_mesh = trimesh.util.concatenate([gt_mesh, pred_mesh])

                    # Export combined OBJ
                    combined_mesh.export(save_filename + "_combined.obj")
            else:
                # RICH is evaluated in SMPL format to be compatbile with other benchmarks
                neutral_hand_pose = torch.eye(3).repeat(batch_size, 15, 1, 1).cuda()
                pred_smpl_params = output['pred_smpl_params']
                smplx_neutral_hand_output = self.smplx_layer(global_orient=pred_smpl_params['global_orient'], 
                                                body_pose=pred_smpl_params['body_pose'], 
                                                left_hand_pose=neutral_hand_pose, 
                                                right_hand_pose=neutral_hand_pose, 
                                                betas=pred_smpl_params['betas'])
                gt_cam_vertices_smpl = batch['vertices']
                # pred_vertices = output['pred_vertices']
                pred_vertices = smplx_neutral_hand_output.vertices 
                pred_cam_vertices_smpl = torch.matmul(self.smplx2smpl_matreg.to(pred_vertices.device), pred_vertices)
                gt_keypoints3d_smpl = batch['keypoints_3d']
                gt_pelvis = (gt_keypoints3d_smpl[:, [1], :] + gt_keypoints3d_smpl[:, [2], :]) / 2.0
                gt_keypoints_3d = gt_keypoints3d_smpl - gt_pelvis

                pred_keypoints_3d = torch.matmul(self.smpl.J_regressor, pred_cam_vertices_smpl)
                pred_pelvis = (pred_keypoints_3d[:, [1], :] + pred_keypoints_3d[:, [2], :]) / 2.0
                pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
                
                pred_cam_vertices_smpl = pred_cam_vertices_smpl - pred_pelvis
                gt_cam_vertices_smpl = gt_cam_vertices_smpl - gt_pelvis

                error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1))
                error_verts = torch.sqrt(((pred_cam_vertices_smpl - gt_cam_vertices_smpl) ** 2).sum(dim=-1))
                error_lhand_joints = torch.zeros((batch_size, 21))
                error_rhand_joints = torch.zeros((batch_size, 21))

                r_error, _ = reconstruction_error(
                    pred_keypoints_3d.float().cpu().numpy(),
                    gt_keypoints_3d.float().cpu().numpy(),
                    reduction=None
                )

        val_mpjpe = error.mean(-1)
        val_pve = error_verts.mean(-1)
        val_pampjpe = torch.tensor(r_error.mean(-1))
        val_lhand_mpjpe = error_lhand_joints.mean(-1)
        val_rhand_mpjpe = error_rhand_joints.mean(-1)

        if dataloader_idx==0:
            self.log('val_loss',val_pve.mean(), logger=True, sync_dist=True)
        self.log('val_pve',val_pve.mean(), logger=True, sync_dist=True)
        self.log('val_mpjpe',val_mpjpe.mean(), logger=True, sync_dist=True)
        self.log('val_pampjpe',val_pampjpe.mean(), logger=True, sync_dist=True)

        self.validation_step_output.append({'val_loss': val_pve , 'val_loss_mpjpe': val_mpjpe, 'val_loss_pampjpe':val_pampjpe, 'dataloader_idx': dataloader_idx})


    def on_validation_epoch_end(self, dataloader_idx=0):
        # Flatten outputs if it's a list of lists
        outputs = self.validation_step_output
        if outputs and isinstance(outputs[0], list):
            outputs = [item for sublist in outputs for item in sublist]
        val_dataset = self.cfg.DATASETS.VAL_DATASETS.split('_')
        # Proceed with the assumption outputs is a list of dictionaries
        for dataloader_idx in range(len(val_dataset)):
            dataloader_outputs = [x for x in outputs if x.get('dataloader_idx') == dataloader_idx]
            if dataloader_outputs:  # Ensure there are outputs for this dataloader
                avg_val_loss = torch.stack([x['val_loss'] for x in dataloader_outputs]).mean()*1000
                avg_mpjpe_loss = torch.stack([x['val_loss_mpjpe'] for x in dataloader_outputs]).mean()*1000
                avg_pampjpe_loss = torch.stack([x['val_loss_pampjpe'] for x in dataloader_outputs]).mean()*1000

                logger.info('PA-MPJPE: '+str(dataloader_idx)+str(avg_pampjpe_loss))
                logger.info('MPJPE: '+str(dataloader_idx)+str(avg_mpjpe_loss))
                logger.info('PVE: '+str(dataloader_idx)+ str(avg_val_loss))

            if dataloader_idx==0:
                self.log('val_loss',avg_val_loss, logger=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()
