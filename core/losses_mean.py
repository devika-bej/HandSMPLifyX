import torch
import torch.nn as nn


    
class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        batch_size = conf.shape[0]

        loss = (conf * self.loss_fn(pred_keypoints_2d.float(), gt_keypoints_2d[:, :, :-1])).sum(dim=(1,2))
        return loss.mean()

class Keypoint2DHandLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        super(Keypoint2DHandLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        batch_size = conf.shape[0]

        loss = (conf * self.loss_fn(pred_keypoints_2d.float(), gt_keypoints_2d[:, :, :-1])).sum(dim=(1,2))
        return loss.mean()

class Keypoint2DLossScaled(nn.Module):

    def __init__(self, loss_type: str = 'l1'):

        super(Keypoint2DLossScaled, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor, box_size, img_size) -> torch.Tensor:
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        batch_size = conf.shape[0]

        loss = (conf * self.loss_fn(pred_keypoints_2d.float(), gt_keypoints_2d[:, :, :-1]))

        loss_scale = (img_size.squeeze(1)/box_size.unsqueeze(-1)).mean(1)
        loss = (loss*loss_scale.unsqueeze(-1).unsqueeze(-1)).sum(dim=(1,2))

        return loss.mean()

class VerticesLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        super(VerticesLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_vertices: torch.Tensor, gt_vertices: torch.Tensor):
        batch_size = pred_vertices.shape[0]
        gt_vertices = gt_vertices.clone()
        loss = self.loss_fn(pred_vertices, gt_vertices).sum(dim=(1,2))
        return loss.mean()

class HandVerticesLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        super(HandVerticesLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_vertices: torch.Tensor, gt_vertices: torch.Tensor, pred_hand_keypoints, gt_hand_keypoints):
        gt_vertices_ = gt_vertices.clone()
        pred_vertices_ = pred_vertices.clone()
        conf = gt_vertices_[:, :, -1].unsqueeze(-1)
        gt_vertices_ = gt_vertices_[:, :, :-1] - gt_hand_keypoints[:, 0].unsqueeze(dim=1)
        pred_vertices_ = pred_vertices_ - pred_hand_keypoints[:, 0].unsqueeze(dim=1)
        loss = (conf*self.loss_fn(pred_vertices_, gt_vertices_)).sum(dim=(1,2))
        return loss.mean()
    
class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor, pelvis_id: int = 39):

        batch_size = pred_keypoints_3d.shape[0]
        gt_keypoints_3d_ = gt_keypoints_3d.clone()
        pred_keypoints_3d_ = pred_keypoints_3d.clone()

        pred_keypoints_3d_ = pred_keypoints_3d_ - pred_keypoints_3d_[:, pelvis_id, :].unsqueeze(dim=1)
        gt_keypoints_3d_[:, :, :-1] = gt_keypoints_3d_[:, :, :-1] - gt_keypoints_3d_[:, pelvis_id, :-1].unsqueeze(dim=1)
        conf = gt_keypoints_3d_[:, :, -1].unsqueeze(-1)
        gt_keypoints_3d_ = gt_keypoints_3d_[:, :, :-1]
        loss = (conf * self.loss_fn(pred_keypoints_3d_.float(), gt_keypoints_3d_)).sum(dim=(1,2))
        return loss.mean()

class ParameterLoss(nn.Module):

    def __init__(self):
        """
        SMPL parameter loss module.
        """
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor):

        batch_size = pred_param.shape[0]
        num_dims = len(pred_param.shape)
        mask_dimension = [batch_size] + [1] * (num_dims-1)
        loss_param = self.loss_fn(pred_param.float(), gt_param)
        return loss_param.mean()

class TranslationLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):

        super(TranslationLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')


    def forward(self, pred_trans: torch.Tensor, gt_trans: torch.Tensor):

        loss = self.loss_fn(pred_trans, gt_trans)
        return loss.mean()