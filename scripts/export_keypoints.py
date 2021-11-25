import os
import sys
import argparse
import ast
import csv
import cv2
import json
import time
import torch
import pathlib
sys.path.insert(1, os.getcwd())
from SimpleHRNet_custom import SimpleHRNet
from misc.visualization import check_video_rotation
import numpy as np

def main(format, filename, hrnet_m, hrnet_c, hrnet_j, hrnet_weights, image_resolution, single_person, use_tiny_yolo,
         max_batch_size, csv_output_filename, csv_delimiter, json_output_filename, device):
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    # print(device)

    image_resolution = ast.literal_eval(image_resolution)


    assert format in ('csv', 'json')
    if format == 'csv':
        assert csv_output_filename.endswith('.csv')
        fd = open(csv_output_filename, 'wt', newline='')
        csv_output = csv.writer(fd, delimiter=csv_delimiter)
    elif format == 'json':
        assert json_output_filename.endswith('.json')
        fd = open(json_output_filename, 'wt')
        json_data = {}

    if use_tiny_yolo:
        yolo_model_def = "./models/detectors/yolo/config/yolov3-tiny.cfg"
        yolo_class_path = "./models/detectors/yolo/data/coco.names"
        yolo_weights_path = "./models/detectors/yolo/weights/yolov3-tiny.weights"
    else:
        yolo_model_def = "./models/detectors/yolo/config/yolov3.cfg"
        yolo_class_path = "./models/detectors/yolo/data/coco.names"
        yolo_weights_path = "./models/detectors/yolo/weights/yolov3.weights"

    model = SimpleHRNet(48, 17, "./weights/officialhrnet_w48_384x288.pth",
                    model_name='HRNet',
                    resolution=ast.literal_eval('(384, 288)'),
                    multiperson=True,
                    return_bounding_boxes=True,
                    max_batch_size=16,
                    yolo_model_def=yolo_model_def,
                    yolo_class_path=yolo_class_path,
                    yolo_weights_path=yolo_weights_path,
                    device=device)
    index=0
    ext='jpg'
    pathes = [str(path) for path in pathlib.Path('../../seg/coco/val2017').glob(f'*.{ext}')]
    pathes = pathes[:100]
    nof_frames=len(pathes)

    pth_=''
    for path in pathes:
        valid_d = {}

        t = time.time()

        frame = cv2.imread(path)
        # if rotation_code is not None:
        #     frame = cv2.rotate(frame, rotation_code)

        model.predict_custom(frame)

        if len(model.pts) == 0:
            model.renew = True
            model.person_ids = np.array((), dtype=np.int32)
            prediction = []
        else:
            for ids in sorted(model.person_ids):
                valid_d[ids] = model.d[ids]
            if len(model.pts) != 0:
                print(np.array(list(valid_d.values())).shape)
                print(np.amax(np.array(list(valid_d.values()))), np.amin(np.array(list(valid_d.values()))))
                pts = np.array(list(valid_d.values()))
        # csv format is:
        #   frame_index,detection_index,<point 0>,<point 1>,...,<point hrnet_j>
        # where each <point N> corresponds to three elements:
        #   y_coordinate,x_coordinate,confidence

        # json format is:
        #   {frame_index: [[<point 0>,<point 1>,...,<point hrnet_j>], ...], ...}
        # where each <point N> corresponds to three elements:
        #   [y_coordinate,x_coordinate,confidence]
        if pth_=='':
            pth_ = pts
        else:
            pth_ = np.concatenate((pth_,pts),axis=0)
        print(pth_.shape)
        fps = 1. / (time.time() - t)
        print('\rframe: % 4d / %d - framerate: %f fps ' % (index, nof_frames - 1, fps), end='')

        index += 1
    np.save('output.npy',pth_)
    if format == 'json':
        json.dump(json_data, fd)

    fd.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract and save keypoints in csv or json format.\n'
                    'csv format is:\n'
                    '  frame_index,detection_index,<point 0>,<point 1>,...,<point hrnet_j>\n'
                    'where each <point N> corresponds to three elements:\n'
                    '  y_coordinate,x_coordinate,confidence\n'
                    'json format is:\n'
                    '  {frame_index: [[<point 0>,<point 1>,...,<point hrnet_j>], ...], ...}\n'
                    'where each <point N> corresponds to three elements:\n'
                    '[y_coordinate,x_coordinate,confidence]')
    parser.add_argument("--format", help="output file format. CSV or JSON.",
                        type=str, default=None)
    parser.add_argument("--filename", "-f", help="open the specified video",
                        type=str, default=None)
    parser.add_argument("--hrnet_m", "-m", help="network model - HRNet or PoseResNet", type=str, default='HRNet')
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels (if model is HRNet), "
                                                "resnet size (if model is PoseResNet)", type=int, default=48)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument("--hrnet_weights", "-w", help="hrnet parameters - path to the pretrained weights",
                        type=str, default="./weights/pose_hrnet_w48_384x288.pth")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(384, 288)')
    parser.add_argument("--single_person",
                        help="disable the multiperson detection (YOLOv3 or an equivalen detector is required for"
                             "multiperson detection)",
                        action="store_true")
    parser.add_argument("--use_tiny_yolo",
                        help="Use YOLOv3-tiny in place of YOLOv3 (faster person detection). Ignored if --single_person",
                        action="store_true")
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--csv_output_filename", help="filename of the csv that will be written.",
                        type=str, default='output.csv')
    parser.add_argument("--csv_delimiter", help="csv delimiter", type=str, default=',')
    parser.add_argument("--json_output_filename", help="filename of the json file that will be written.",
                        type=str, default='output.json')
    parser.add_argument("--device", help="device to be used (default: cuda, if available)."
                                         "Set to `cuda` to use all available GPUs (default); "
                                         "set to `cuda:IDS` to use one or more specific GPUs "
                                         "(e.g. `cuda:0` `cuda:1,2`); "
                                         "set to `cpu` to run on cpu.", type=str, default=None)
    args = parser.parse_args()
    main(**args.__dict__)
