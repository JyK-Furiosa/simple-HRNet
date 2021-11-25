import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from models.hrnet import HRNet
from models.poseresnet import PoseResNet
from models.detectors.YOLOv3 import YOLOv3

from misc.utils import find_person_id_associations


def predict_next_bbox_from_keypoints(IMAGE_SHAPE, pts, margin_ratio = 0.3):
    """

    Args:
        keypoints: [Identity, 17, 3] [X,Y]

    Returns:

    """
    raw_top_left = np.min(pts, axis=1)
    raw_down_right = np.max(pts, axis=1)

    center  = (raw_top_left + raw_down_right) /2
    margin = np.abs(center - raw_top_left) * margin_ratio


    top_left = np.clip(raw_top_left - margin, 0, None)
    down_right = np.clip(raw_down_right + margin, None, np.array(IMAGE_SHAPE)[::-1])

    bbox = np.concatenate([top_left,down_right,np.ones((pts.shape[0],1)),np.ones((pts.shape[0],1)),np.zeros((pts.shape[0],1))], axis=1).astype(np.int32)

    return bbox

class SimpleHRNet:
    """
    SimpleHRNet class.

    The class provides a simple and customizable method to load the HRNet network, load the official pre-trained
    weights, and predict the human pose on single images.
    Multi-person support with the YOLOv3 detector is also included (and enabled by default).
    """

    def __init__(self,
                 c,
                 nof_joints,
                 checkpoint_path,
                 model_name='HRNet',
                 resolution=(384, 288),
                 interpolation=cv2.INTER_CUBIC,
                 multiperson=True,
                 return_heatmaps=False,
                 return_bounding_boxes=False,
                 max_batch_size=32,
                 yolo_model_def="./models/detectors/yolo/config/yolov3.cfg",
                 yolo_class_path="./models/detectors/yolo/data/coco.names",
                 yolo_weights_path="./models/detectors/yolo/weights/yolov3.weights",
                 device=torch.device("cpu")):
        """
        Initializes a new SimpleHRNet object.
        HRNet (and YOLOv3) are initialized on the torch.device("device") and
        its (their) pre-trained weights will be loaded from disk.

        Args:
            c (int): number of channels (when using HRNet model) or resnet size (when using PoseResNet model).
            nof_joints (int): number of joints.
            checkpoint_path (str): path to an official hrnet checkpoint or a checkpoint obtained with `train_coco.py`.
            model_name (str): model name (HRNet or PoseResNet).
                Valid names for HRNet are: `HRNet`, `hrnet`
                Valid names for PoseResNet are: `PoseResNet`, `poseresnet`, `ResNet`, `resnet`
                Default: "HRNet"
            resolution (tuple): hrnet input resolution - format: (height, width).
                Default: (384, 288)
            interpolation (int): opencv interpolation algorithm.
                Default: cv2.INTER_CUBIC
            multiperson (bool): if True, multiperson detection will be enabled.
                This requires the use of a people detector (like YOLOv3).
                Default: True
            return_heatmaps (bool): if True, heatmaps will be returned along with poses by self.predict.
                Default: False
            return_bounding_boxes (bool): if True, bounding boxes will be returned along with poses by self.predict.
                Default: False
            max_batch_size (int): maximum batch size used in hrnet inference.
                Useless without multiperson=True.
                Default: 16
            yolo_model_def (str): path to yolo model definition file.
                Default: "./models/detectors/yolo/config/yolov3.cfg"
            yolo_class_path (str): path to yolo class definition file.
                Default: "./models/detectors/yolo/data/coco.names"
            yolo_weights_path (str): path to yolo pretrained weights file.
                Default: "./models/detectors/yolo/weights/yolov3.weights.cfg"
            device (:class:`torch.device`): the hrnet (and yolo) inference will be run on this device.
                Default: torch.device("cpu")
        """

        self.c = c
        self.nof_joints = nof_joints
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.resolution = resolution  # in the form (height, width) as in the original implementation
        self.interpolation = interpolation
        self.multiperson = multiperson
        self.return_heatmaps = return_heatmaps
        self.return_bounding_boxes = return_bounding_boxes
        self.max_batch_size = max_batch_size
        self.yolo_model_def = yolo_model_def
        self.yolo_class_path = yolo_class_path
        self.yolo_weights_path = yolo_weights_path
        self.device = device

        if model_name in ('HRNet', 'hrnet'):
            self.model = HRNet(c=c, nof_joints=nof_joints)
        elif model_name in ('PoseResNet', 'poseresnet', 'ResNet', 'resnet'):
            self.model = PoseResNet(resnet_size=c, nof_joints=nof_joints)
        else:
            raise ValueError('Wrong model name.')

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

        if 'cuda' in str(self.device):
            print("device: 'cuda' - ", end="")

            if 'cuda' == str(self.device):
                # if device is set to 'cuda', all available GPUs will be used
                print("%d GPU(s) will be used" % torch.cuda.device_count())
                device_ids = None
            else:
                # if device is set to 'cuda:IDS', only that/those device(s) will be used
                print("GPU(s) '%s' will be used" % str(self.device))
                device_ids = [int(x) for x in str(self.device)[5:].split(',')]

            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        elif 'cpu' == str(self.device):
            print("device: 'cpu'")
        else:
            raise ValueError('Wrong device name.')

        self.model = self.model.to(device)
        self.model.eval()

        if not self.multiperson:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        else:
            self.detector = YOLOv3(model_def=yolo_model_def,
                                   class_path=yolo_class_path,
                                   weights_path=yolo_weights_path,
                                   classes=('person',),
                                   max_batch_size=self.max_batch_size,
                                   device=device)
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resolution[0], self.resolution[1])),  # (height, width)
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


        #######
        self.renew = True
        self.prev_boxes = None
        self.prev_pts = None
        self.prev_person_ids = None
        self.next_person_id = 0
        self.d = {}
        self.predicted_bbox = None
        self.bbox = None
        self.pts = None
        self.person_ids = None

        self.db = {}
        self.dp = {}

    def predict_custom(self, input_frame):
        IMAGE_SHAPE = [input_frame.shape[0], input_frame.shape[1]]
        if self.renew or self.predicted_bbox == None:
            self.bbox, self.pts = self.predict(input_frame)
        else:
            self.bbox, self.pts = self.predict(input_frame, self.predicted_bbox)

        if len(self.pts) == 0:
            self.renew = True
            self.person_ids = np.array((), dtype=np.int32)
        else:
            self.predicted_bbox = predict_next_bbox_from_keypoints(IMAGE_SHAPE, self.pts[..., [1, 0]])

            if self.prev_pts is None and self.prev_person_ids is None:
                self.person_ids = np.arange(self.next_person_id, len(self.pts) + self.next_person_id,
                                             dtype=np.int32)
                self.next_person_id = len(self.pts) + 1
            else:
                self.bbox, self.pts, self.person_ids = find_person_id_associations(
                    boxes=self.bbox, pts=self.pts, prev_boxes=self.prev_boxes, prev_pts=self.prev_pts,
                    prev_person_ids=self.prev_person_ids,
                    next_person_id=self.next_person_id, pose_alpha=0.5, similarity_threshold=0.4, smoothing_alpha=0,
                )
                self.next_person_id = max(self.next_person_id, np.max(self.person_ids) + 1)

            self.prev_boxes = self.bbox.copy()
            self.prev_pts = self.pts.copy()
            self.prev_person_ids = self.person_ids

            for identity in range(len(self.pts)):

                self.db[self.person_ids[identity]] = self.prev_boxes[identity, :]####
                self.dp[self.person_ids[identity]] = self.prev_pts[identity, :, :]

                if self.person_ids[identity] in self.d:
                    self.d[self.person_ids[identity]][0:242, :, :] = self.d[self.person_ids[identity]][1:243, :, :]
                    self.d[self.person_ids[identity]][242, :, :] = self.pts[identity, :, 1::-1]
                else:
                    self.d[self.person_ids[identity]] = np.zeros(shape=(243, 17, 2))
                    self.d[self.person_ids[identity]][242, :, :] = self.pts[identity, :, 1::-1]

            self.prev_boxes = np.array(list(self.db.values()))####
            self.prev_pts = np.array(list(self.dp.values()))
            self.prev_person_ids = np.array(list(self.db.keys()))


    def predict(self, image, predict_bbox=None):
        """
        Predicts the human pose on a single image or a stack of n images.

        Args:
            image (:class:`np.ndarray`):
                the image(s) on which the human pose will be estimated.

                image is expected to be in the opencv format.
                image can be:
                    - a single image with shape=(height, width, BGR color channel)

        Returns:
            :class:`np.ndarray`:
                a numpy array containing human joints for each (detected) person.

                Format:
                        shape=(# of people, # of joints (nof_joints), 3);  dtype=(np.float32).

                Each joint has 3 values: (y position, x position, joint confidence).

                If self.return_heatmaps, the class returns a list with (heatmaps, human joints)
                If self.return_bounding_boxes, the class returns a list with (bounding boxes, human joints)
                If self.return_heatmaps and self.return_bounding_boxes, the class returns a list with
                    (heatmaps, bounding boxes, human joints)
        """
        if len(image.shape) == 3:
            return self._predict_single(image, predict_bbox)
        else:
            raise ValueError('Wrong image format.')


    def _predict_single(self, image, predict_bbox):
        if not self.multiperson:
            old_res = image.shape
            if self.resolution is not None:
                image = cv2.resize(
                    image,
                    (self.resolution[1], self.resolution[0]),  # (width, height)
                    interpolation=self.interpolation
                )
            
            images = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(dim=0)
            boxes = np.asarray([[0, 0, old_res[1], old_res[0]]], dtype=np.float32)  # [x1, y1, x2, y2]
            heatmaps = np.zeros((1, self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                                dtype=np.float32)

        else:
            if predict_bbox is not None:
                detections_ = predict_bbox
            else:
                detections_ = self.detector.predict_single(image)

            if detections_ is None:
                return [],[]
            
            detections = []
            for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections_):
                area = (x2 - x1) * (y2 - y1)

                if area < 5000:
                    continue
                else:
                    detections.append([x1, y1, x2, y2, conf, cls_conf, cls_pred]) 
                    



            nof_people = len(detections) if detections is not None else 0
            boxes = np.empty((nof_people, 4), dtype=np.int32)
            images = torch.empty((nof_people, 3, self.resolution[0], self.resolution[1]))  # (height, width)
            heatmaps = np.zeros((nof_people, self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                                dtype=np.float32)

            if detections is not None:
                for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
                    x1 = int(round(x1.item()))
                    x2 = int(round(x2.item()))
                    y1 = int(round(y1.item()))
                    y2 = int(round(y2.item()))

                    # Adapt detections to match HRNet input aspect ratio (as suggested by xtyDoge in issue #14)
                    correction_factor = self.resolution[0] / self.resolution[1] * (x2 - x1) / (y2 - y1)
                    if correction_factor > 1:
                        # increase y side
                        center = y1 + (y2 - y1) // 2
                        length = int(round((y2 - y1) * correction_factor))
                        y1 = max(0, center - length // 2)
                        y2 = min(image.shape[0], center + length // 2)
                    elif correction_factor < 1:
                        # increase x side
                        center = x1 + (x2 - x1) // 2
                        length = int(round((x2 - x1) * 1 / correction_factor))
                        x1 = max(0, center - length // 2)
                        x2 = min(image.shape[1], center + length // 2)

                    boxes[i] = [x1, y1, x2, y2]

                    images[i] = self.transform(image[y1:y2, x1:x2, ::-1])

 



        if images.shape[0] > 0:
            images = images.to(self.device)
            # print(images.shape)
            
            with torch.no_grad():

                if len(images) <= self.max_batch_size:
                    out = self.model(images)

                else:
                    out = torch.empty(
                        (images.shape[0], self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                        device=self.device
                    )
                    for i in range(0, len(images), self.max_batch_size):
                        out[i:i + self.max_batch_size] = self.model(images[i:i + self.max_batch_size])

            out = out.detach().cpu().numpy()
            pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
            # For each human, for each joint: y, x, confidence
            for i, human in enumerate(out):
                heatmaps[i] = human
                for j, joint in enumerate(human):
                    pt = np.unravel_index(np.argmax(joint), (self.resolution[0] // 4, self.resolution[1] // 4))
                    # 0: pt_y / (height // 4) * (bb_y2 - bb_y1) + bb_y1
                    # 1: pt_x / (width // 4) * (bb_x2 - bb_x1) + bb_x1
                    # 2: confidences
                    pts[i, j, 0] = pt[0] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                    pts[i, j, 1] = pt[1] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                    pts[i, j, 2] = joint[pt]

        else:
            pts = np.empty((0, 0, 3), dtype=np.float32)

        res = list()
        if self.return_heatmaps:
            res.append(heatmaps)
        if self.return_bounding_boxes:
            res.append(boxes)
        res.append(pts)

        if len(res) > 1:
            return res
        else:
            return res[0]
