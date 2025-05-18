import copy
import pickle
import os

import numpy as np
import json

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate


class CpddDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]  # train or val

        split_dir = os.path.join(self.root_path, 'detection', (self.split + '.txt'))
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None # [Sport_Complex_01_Day/000129, ...]

        self.cpdd_infos = []
        self.include_data(self.mode)  # 加载cpdd_infos_train.pkl或cpdd_infos_val.pkl
        self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI
        self.map_class_to_waymo = self.dataset_cfg.MAP_CLASS_TO_WAYMO

    def include_data(self, mode):
        self.logger.info('Loading Cpdd dataset.')
        cpdd_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / "infos" / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                cpdd_infos.extend(infos)

        self.cpdd_infos.extend(cpdd_infos)
        self.logger.info('Total samples for CUSTOM dataset: %d' % (len(cpdd_infos)))

    def get_label(self, idx):
        file_path = idx.split('-')
        label_file = self.root_path / 'detection' / (file_path[0] + '/label/' + file_path[1] + '.json')
        lidar_file = self.root_path / 'scenes' / (file_path[0] + '/lidar/' + file_path[1] + '.bin')
        assert label_file.exists()
        assert lidar_file.exists()
        points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 8)
        with open(label_file, 'r') as f:
            label_data = json.load(f)

        # [N, 8]: (x y z dx dy dz heading_angle category_id)
        gt_boxes = []
        gt_names = []
        num_points_in_gt = []
        for obj in label_data:
            x, y, z = obj['psr']['position']['x'], obj['psr']['position']['y'], obj['psr']['position']['z']
            dx, dy, dz = obj['psr']['scale']['x'], obj['psr']['scale']['y'], obj['psr']['scale']['z']
            heading_angle = obj['psr']['rotation']['z']
            obj_type = obj['obj_type']
            # if obj_type not in self.map_class_to_kitti:  # 过滤掉不需要的类别
            #     continue
            gt_boxes.append([x, y, z, dx, dy, dz, heading_angle])
            # gt_names.append(self.map_class_to_kitti[obj_type])  # 转换为kitti的类别
            gt_names.append(obj_type)
        
        gt_boxes = np.array(gt_boxes, dtype=np.float32).reshape(-1, 7)

        import torch
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(  # 选取点云中在gt_box内的点
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)
        
        for i in range(len(gt_boxes)):
            gt_points = points[point_indices[i] > 0]
            num_points_in_gt.append(gt_points.shape[0])

        return gt_boxes, np.array(gt_names), np.array(num_points_in_gt)

    def get_lidar(self, idx):
        file_path = idx.split('-')
        idx = file_path[0] + '/lidar/' + file_path[1] + '.bin'
        lidar_file = self.root_path / 'scenes' / idx
        assert lidar_file.exists()
        point_features = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 8)
        return point_features

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split

        split_dir = self.root_path / 'detection' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs

        return len(self.cpdd_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.cpdd_infos)

        info = copy.deepcopy(self.cpdd_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        input_dict = {
            'frame_id': self.sample_id_list[index],
            'points': points
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.cpdd_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos, map_name_to_waymo):
            from ..waymo.waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            for anno in eval_det_annos:
                anno['name'] = anno['name'].astype('U20')
                for k in range(anno['name'].shape[0]):
                    anno['name'][k] = map_name_to_waymo[anno['name'][k]]
            for anno in eval_gt_annos:
                anno['difficulty'] = np.zeros(len(anno['name']))
                anno['name'] = anno['name'].astype('U20')
                for k in range(anno['name'].shape[0]):
                    anno['name'][k] = map_name_to_waymo[anno['name'][k]]
            
            waymo_class_names = [map_name_to_waymo[x] for x in class_names]

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=waymo_class_names,
                distance_thresh=96, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.cpdd_infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos ,self.map_class_to_waymo)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=8):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                annotations = {}
                gt_boxes_lidar, name, num_points_in_gt = self.get_label(sample_idx)
                annotations['name'] = name
                annotations['gt_boxes_lidar'] = gt_boxes_lidar[:, :7]  # (x y z dx dy dz heading_angle)
                annotations['num_points_in_gt'] = num_points_in_gt
                info['annos'] = annotations

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        # create a thread pool to improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / 'infos' / ('cpdd_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(  # 选取点云中在gt_box内的点
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]  # 将点云坐标移动到坐标系原点
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        # Output the num of all classes in database
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def create_label_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
        with open(save_label_path, 'w') as f:
            for idx in range(gt_boxes.shape[0]):
                boxes = gt_boxes[idx]
                name = gt_names[idx]
                if name not in class_names:
                    continue
                line = "{x} {y} {z} {l} {w} {h} {angle} {name}\n".format(
                    x=boxes[0], y=boxes[1], z=(boxes[2]), l=boxes[3],
                    w=boxes[4], h=boxes[5], angle=boxes[6], name=name
                )
                f.write(line)


def create_cpdd_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = CpddDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / dataset_cfg.INFO_PATH['train'][0]
    val_filename = save_path / dataset_cfg.INFO_PATH['test'][0]

    save_path.mkdir(parents=True, exist_ok=True)

    print('------------------------Start to generate data infos------------------------')

    dataset.set_split(train_split)
    cpdd_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(cpdd_infos_train, f)
    print('Custom info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    cpdd_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(cpdd_infos_val, f)
    print('Custom info train file is saved to %s' % val_filename)

    print('------------------------Start create groundtruth database for data augmentation------------------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    print('------------------------Data preparation done------------------------')


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_cpdd_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_cpdd_infos(
            dataset_cfg=dataset_cfg,
            class_names=list(dataset_cfg.MAP_CLASS_TO_KITTI.keys()),
            data_path=Path(dataset_cfg.DATA_PATH),
            save_path=Path(dataset_cfg.DATA_PATH) / 'infos',
        )
