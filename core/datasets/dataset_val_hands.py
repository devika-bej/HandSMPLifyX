from torch.utils.data import Dataset, ConcatDataset
import numpy as np
import os
import cv2
import torch
import copy
from ..configs import DATASET_FOLDERS, DATASET_FILES
from .utils import expand_to_aspect_ratio, get_example
from torchvision.transforms import Normalize
from ..utils.pylogger import get_pylogger
from ..smplx_wrapper import SMPLX_
log = get_pylogger(__name__)
from ..constants import FLIP_KEYPOINT_PERMUTATION, NUM_JOINTS, NUM_BETAS,NUM_JOINTS_SMPLX, NUM_PARAMS_SMPL, NUM_PARAMS_SMPLX, SMPLX2SMPL, SMPLX_MODEL_DIR, SMPL_MODEL_DIR
import smplx
import pickle
body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
FLIP_KEYPOINT_PERMUTATION = body_permutation + [25 + i for i in extra_permutation]
SMPLX_MODEL_DIR='/ps/archive/alignment/models/smplx'


def resize_image(img, target_size):
    height, width = img.shape[:2]
    aspect_ratio = width / height

    # Calculate the new size while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Resize the image using OpenCV
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new blank image with the target size
    final_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    # Paste the resized image onto the blank image, centering it
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    final_img[start_y:start_y + new_height, start_x:start_x + new_width] = resized_img

    return aspect_ratio, final_img

class DatasetVal(Dataset):
    def __init__(self, cfg, dataset, is_train=False):
        super(DatasetVal, self).__init__()

        self.dataset = dataset
        self.is_train = is_train
        self.cfg = cfg
        self.IMG_SIZE = cfg.MODEL.IMAGE_SIZE
        self.BBOX_SHAPE = cfg.MODEL.get('BBOX_SHAPE', None)
        self.MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.STD = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.normalize_img = Normalize(mean=cfg.MODEL.IMAGE_MEAN,
                                    std=cfg.MODEL.IMAGE_STD)
        self.use_skimage_antialias = cfg.DATASETS.get('USE_SKIMAGE_ANTIALIAS', False)
        self.border_mode = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
        }[cfg.DATASETS.get('BORDER_MODE', 'constant')]

        self.img_dir = DATASET_FOLDERS[dataset] 
        self.data = np.load(DATASET_FILES[is_train][dataset], allow_pickle=True)
        self.imgname = self.data['imgname']
        self.scale = self.data['scale']
        self.center = self.data['center']

        if 'pose_cam' in self.data:
            if 'smplx' in self.dataset:
                self.pose = self.data['pose_cam'][:, :NUM_PARAMS_SMPLX*3].astype(np.float)
            else:
                self.pose = self.data['pose_cam'][:, :NUM_PARAMS_SMPL*3].astype(np.float)
        else:
            self.pose = np.zeros((len(self.imgname), 24*3), dtype=np.float32)

        if 'part' in self.data:
            self.keypoints = self.data['part']
        elif 'gtkps' in self.data:
            self.keypoints = self.data['gtkps'][:,:NUM_JOINTS_SMPLX]
        elif 'body_keypoints_2d' in self.data:
            self.keypoints = self.data['body_keypoints_2d']
        else:
            self.keypoints = np.zeros((len(self.imgname), NUM_JOINTS_SMPLX, 3))
        if self.keypoints.shape[2]<3:
            ones_array = np.ones((self.keypoints.shape[0],self.keypoints.shape[1],1))
            self.keypoints = np.concatenate((self.keypoints, ones_array), axis=2)
        
        if 'shape' in self.data:
            self.betas = self.data['shape'].astype(np.float)[:,:NUM_BETAS] 
        else:
            self.betas = np.zeros((len(self.imgname), 10), dtype=np.float32)

        if 'cam_int' in self.data:
            self.cam_int = self.data['cam_int']
        else:
            self.cam_int = np.zeros((len(self.imgname),3,3), dtype=np.float32)
        try:
            gender = self.data['gender']
            self.gender = np.array([
                0 if str(g).strip().lower() in ['m', 'male'] else
                1 if str(g).strip().lower() in ['f', 'female'] else
                2 if str(g).strip().lower() in ['n', 'neutral'] else
                -1
                for g in gender
            ], dtype=np.int32)
        except KeyError:
            self.gender = -1 * np.ones(len(self.imgname), dtype=np.int32)


        self.smpl_gt_male = smplx.SMPL(SMPL_MODEL_DIR,
                                gender='male')
        self.smpl_gt_female = smplx.SMPL(SMPL_MODEL_DIR,
                                    gender='female')
        self.smpl_gt_neutral = smplx.SMPL(SMPL_MODEL_DIR,
                                    gender='neutral')
        
        self.smplx_gt_male = SMPLX_(gender='male',model_path=SMPLX_MODEL_DIR, use_pca=False, num_betas=10, flat_hand_mean=True) 
        self.smplx_gt_female = SMPLX_(gender='female',model_path=SMPLX_MODEL_DIR, use_pca=False, num_betas=10, flat_hand_mean=True) 
        self.smplx_gt_neutral = SMPLX_(gender='neutral',model_path=SMPLX_MODEL_DIR, use_pca=False, num_betas=10, flat_hand_mean=True) 
        self.smplx2smpl = pickle.load(open(SMPLX2SMPL, 'rb'))
        self.smplx2smpl = torch.tensor(self.smplx2smpl['matrix'][None],
                                        dtype=torch.float32)


        self.length = self.scale.shape[0]
        log.info(f'Loaded {self.dataset} dataset, num samples {self.length}')

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        keypoints_2d = self.keypoints[index].copy()
        orig_keypoints_2d = self.keypoints[index].copy()
        center_x = center[0]
        center_y = center[1]
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=self.BBOX_SHAPE).max()
        if bbox_size < 1:
            #Todo raise proper error
            breakpoint()

        augm_config = copy.deepcopy(self.cfg.DATASETS.CONFIG)
        imgname = os.path.join(self.img_dir, self.imgname[index])
        cv_img = cv2.imread(imgname, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        cv_img = cv_img[:, :, ::-1]
        aspect_ratio, img_full_resized = resize_image(cv_img, 256)
        img_full_resized = np.transpose(img_full_resized.astype('float32'),
                        (2, 0, 1))/255.0
        item['img_full_resized'] = self.normalize_img(torch.from_numpy(img_full_resized).float())
        if 'smplx' in self.dataset:
            smpl_params = {'global_orient': self.pose[index][:3].astype(np.float32),
                        'body_pose': self.pose[index][3:66].astype(np.float32),
                        'left_hand_pose':self.pose[index][75:120].astype(np.float32),
                        'right_hand_pose':self.pose[index][120:165].astype(np.float32),
                        'jaw_pose':self.pose[index][66:69].astype(np.float32),
                        'leye_pose':self.pose[index][69:72].astype(np.float32),
                        'reye_pose':self.pose[index][72:75].astype(np.float32),
                        'expression': np.zeros(10).astype(np.float32),
                        'betas': self.betas[index].astype(np.float32)
                        }
            item['smpl_params'] = smpl_params
        else:
            smpl_params = {'global_orient': self.pose[index][:3].astype(np.float32),
                        'body_pose': self.pose[index][3:].astype(np.float32),
                        'betas': self.betas[index].astype(np.float32)
                        }
            item['smpl_params'] = smpl_params

        img_patch_rgba = None
        img_patch_cv = None
        img_patch_rgba, \
        img_patch_cv,\
        keypoints_2d, \
        img_size, cx, cy, bbox_w, bbox_h, trans, scale_aug = get_example(imgname,
                                      center_x, center_y,
                                      bbox_size, bbox_size,
                                      keypoints_2d,
                                      FLIP_KEYPOINT_PERMUTATION,
                                      self.IMG_SIZE, self.IMG_SIZE,
                                      self.MEAN, self.STD, self.is_train, augm_config,
                                      is_bgr=True, return_trans=True,
                                      use_skimage_antialias=self.use_skimage_antialias,
                                      border_mode=self.border_mode,
                                      dataset=self.dataset
                                      )
        new_center = np.array([cx, cy])
        img_patch = img_patch_rgba[:3,:,:]
        item['cam_int'] = np.array(self.cam_int[index]).astype(np.float32)
        item['img'] = img_patch
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['orig_keypoints_2d'] = orig_keypoints_2d
        item['box_center'] = new_center
        item['box_size'] = bbox_w * scale_aug
        item['img_size'] = 1.0 * img_size.copy()
        item['_scale'] = scale
        item['_trans'] = trans
        item['imgname'] = imgname
        item['dataset'] = self.dataset
        item['gender'] = self.gender[index]
        if 'smplx' in self.dataset:
            if self.gender[index] == 1:
                model = self.smplx_gt_female
                smpl_model = self.smpl_gt_female
            elif self.gender[index] == 0:
                model = self.smplx_gt_male
                smpl_model = self.smpl_gt_male
            elif self.gender[index] == 2:
                model = self.smplx_gt_neutral
                smpl_model = self.smpl_gt_neutral
            else:
                raise Exception("Gender undefined")
            if 'shape-eval' in self.dataset:
                # T-pose
                gt_smpl_out = model(
                    global_orient=torch.zeros(3).unsqueeze(0),
                    body_pose=torch.zeros(63).unsqueeze(0),
                    betas=torch.from_numpy(item['smpl_params']['betas']).unsqueeze(0),
                    left_hand_pose=torch.zeros(45).unsqueeze(0),
                    right_hand_pose=torch.zeros(45).unsqueeze(0),
                    jaw_pose=torch.zeros(3).unsqueeze(0),
                    leye_pose=torch.zeros(3).unsqueeze(0),
                    reye_pose=torch.zeros(3).unsqueeze(0),
                    expression=torch.zeros(10).unsqueeze(0))
                gt_vertices = gt_smpl_out.vertices.detach()
                item['vertices'] = gt_vertices[0].float()
            else:
                gt_smpl_out = model(
                    global_orient=torch.from_numpy(item['smpl_params']['global_orient']).unsqueeze(0),
                    body_pose=torch.from_numpy(item['smpl_params']['body_pose']).unsqueeze(0),
                    betas=torch.from_numpy(item['smpl_params']['betas']).unsqueeze(0),
                    left_hand_pose=torch.zeros(45).unsqueeze(0),
                    right_hand_pose=torch.zeros(45).unsqueeze(0),
                    jaw_pose=torch.zeros(3).unsqueeze(0),
                    leye_pose=torch.zeros(3).unsqueeze(0),
                    reye_pose=torch.zeros(3).unsqueeze(0),
                    expression=torch.zeros(10).unsqueeze(0))
        
                gt_vertices = gt_smpl_out.vertices.detach()
                gt_vertices = torch.matmul(self.smplx2smpl, gt_vertices)
                item['keypoints_3d'] = torch.matmul(smpl_model.J_regressor, gt_vertices[0])
                item['vertices'] = gt_vertices[0].float()
                item['gt_hand_joints'] = gt_smpl_out.hand_joints[0].detach()
        else:
            if self.gender[index] == 1:
                model = self.smpl_gt_female
            elif self.gender[index] == 0:
                model = self.smpl_gt_male
            elif self.gender[index] == 2:
                model = self.smpl_gt_neutral
            else:
                raise Exception("Gender undefined")
            gt_smpl_out = model(
                        global_orient=torch.from_numpy(item['smpl_params']['global_orient']).unsqueeze(0),
                        body_pose=torch.from_numpy(item['smpl_params']['body_pose']).unsqueeze(0),
                        betas=torch.from_numpy(item['smpl_params']['betas']).unsqueeze(0))
            
            gt_vertices = gt_smpl_out.vertices.detach()  
            item['keypoints_3d'] = torch.matmul(model.J_regressor, gt_vertices[0])
            item['vertices'] = gt_vertices[0].float()

        return item
    def __len__(self):
        return int(len(self.imgname))
        
       