# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.


import os
import pickle
import numpy as np
from ...utils import common_utils
import tensorflow as tf
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2

from ...utils.box_utils import boxes_to_corners_3d
from scipy.spatial import ConvexHull, Delaunay
import copy

try:
    tf.enable_eager_execution()
except:
    pass

WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist', 'Truck']


def generate_labels(frame, pose):
    obj_name, difficulty, dimensions, locations, heading_angles = [], [], [], [], []
    tracking_difficulty, speeds, accelerations, obj_ids = [], [], [], []
    num_points_in_gt = []
    laser_labels = frame.laser_labels

    for i in range(len(laser_labels)):
        box = laser_labels[i].box
        class_ind = laser_labels[i].type
        loc = [box.center_x, box.center_y, box.center_z]
        heading_angles.append(box.heading)
        obj_name.append(WAYMO_CLASSES[class_ind])
        difficulty.append(laser_labels[i].detection_difficulty_level)
        tracking_difficulty.append(laser_labels[i].tracking_difficulty_level)
        dimensions.append([box.length, box.width, box.height])  # lwh in unified coordinate of OpenPCDet
        locations.append(loc)
        obj_ids.append(laser_labels[i].id)
        num_points_in_gt.append(laser_labels[i].num_lidar_points_in_box)
        speeds.append([laser_labels[i].metadata.speed_x, laser_labels[i].metadata.speed_y])
        accelerations.append([laser_labels[i].metadata.accel_x, laser_labels[i].metadata.accel_y])

    annotations = {}
    annotations['name'] = np.array(obj_name)
    annotations['difficulty'] = np.array(difficulty)
    annotations['dimensions'] = np.array(dimensions)
    annotations['location'] = np.array(locations)
    annotations['heading_angles'] = np.array(heading_angles)

    annotations['obj_ids'] = np.array(obj_ids)
    annotations['tracking_difficulty'] = np.array(tracking_difficulty)
    annotations['num_points_in_gt'] = np.array(num_points_in_gt)
    annotations['speed_global'] = np.array(speeds)
    annotations['accel_global'] = np.array(accelerations)

    annotations = common_utils.drop_info_with_name(annotations, name='unknown')
    if annotations['name'].__len__() > 0:
        global_speed = np.pad(annotations['speed_global'], ((0, 0), (0, 1)), mode='constant', constant_values=0)  # (N, 3)
        speed = np.dot(global_speed, np.linalg.inv(pose[:3, :3].T))
        speed = speed[:, :2]
        
        gt_boxes_lidar = np.concatenate([
            annotations['location'], annotations['dimensions'], annotations['heading_angles'][..., np.newaxis], speed],
            axis=1
        )
    else:
        gt_boxes_lidar = np.zeros((0, 9))
    annotations['gt_boxes_lidar'] = gt_boxes_lidar
    return annotations

def modify_labels(points, sem_label, ori_anno):
    def in_hull(p, hull):
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        return hull.find_simplex(p) >= 0

    def extract_pc_in_box3d(pc, box3d):
        """pc: (N,3), box3d: (8,3)"""
        box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
        return pc[box3d_roi_inds, :], box3d_roi_inds
    def count_label(location, dimension, angle, points_with_label):
        box_info = np.expand_dims(np.concatenate([location, dimension, angle.reshape(-1)], axis=-1), 0)
        # box_info = lidar_boxes[anno_idx]
        corners = boxes_to_corners_3d(box_info)

        selected_points, selected_idx = extract_pc_in_box3d(points_with_label, corners[0])
        selected_labels = selected_points[:, -1]
        label_cnt = np.bincount(selected_points[:, -1].astype(np.int32), minlength=40)
        return label_cnt, selected_points
    points_with_label = np.concatenate([
        points, sem_label
    ], axis=-1).astype(np.float32) # 6(x y z int elong nlz) + 1

    names = ori_anno['name']
    ret_names = copy.deepcopy(ori_anno['name'])
    locations = ori_anno['location']
    dimensions = ori_anno['dimensions']
    angles = ori_anno['heading_angles']
    point_cnts = ori_anno['num_points_in_gt']
    lidar_boxes = ori_anno['gt_boxes_lidar']

    for anno_idx in range(names.shape[0]):
        name = names[anno_idx]
        if name not in ['Vehicle']:
            continue
        location = locations[anno_idx]
        dimension = dimensions[anno_idx]
        angle = angles[anno_idx]
        point_cnt = point_cnts[anno_idx]
        if point_cnt < 1:
            continue
        # location - dimension * angle

        label_cnt, _ = count_label(location, dimension + 0.1, angle, points_with_label)
        non_vehicle_cnt = label_cnt[2] + label_cnt[3] + label_cnt[4] # Truck, Bus, Other_vehicle
        vehicle_cnt = label_cnt[1]
        if point_cnt > 20 and (non_vehicle_cnt + vehicle_cnt) < 1:
            print(f"strange...{point_cnt} {non_vehicle_cnt + vehicle_cnt}\n")
            label_cnt, _ = count_label(location, dimension + 2.0, angle, points_with_label)
            non_vehicle_cnt = label_cnt[2] + label_cnt[3] + label_cnt[4] # Truck, Bus, Other_vehicle
            vehicle_cnt = label_cnt[1]
        if non_vehicle_cnt > vehicle_cnt:
            ret_names[anno_idx] = 'Truck'
        
    return ret_names

def convert_range_image_to_point_cloud_labels(frame,
                                              range_images,
                                              segmentation_labels,
                                              ri_index=(0, 1)):
    """Convert segmentation labels from range images to point clouds.
    Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
    range_image_second_return]}.
    segmentation_labels: A dict of {laser_name, [range_image_first_return,
    range_image_second_return]}.
    ri_index: 0 for the first return, 1 for the second return.
    Returns:
    point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
    points that are not labeled.
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        point_labels_single = []
        for cur_ri_index in ri_index:
            range_image = range_images[c.name][cur_ri_index]
            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            range_image_mask = range_image_tensor[..., 0] > 0

            if c.name in segmentation_labels:
                sl = segmentation_labels[c.name][cur_ri_index]
                sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
                sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
            else:
                num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
                sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

            point_labels_single.append(sl_points_tensor.numpy())
        point_labels.append(np.concatenate(point_labels_single, axis=0))

    return point_labels

def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1)):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)

    for c in calibrations:
        points_single, cp_points_single, points_NLZ_single, points_intensity_single, points_elongation_single \
            = [], [], [], [], []
        for cur_ri_index in ri_index:
            range_image = range_images[c.name][cur_ri_index]
            if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0
            range_image_NLZ = range_image_tensor[..., 3]
            range_image_intensity = range_image_tensor[..., 1]
            range_image_elongation = range_image_tensor[..., 2]
            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.where(range_image_mask))
            points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
            points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
            points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))
            cp = camera_projections[c.name][0]
            cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))

            points_single.append(points_tensor.numpy())
            cp_points_single.append(cp_points_tensor.numpy())
            points_NLZ_single.append(points_NLZ_tensor.numpy())
            points_intensity_single.append(points_intensity_tensor.numpy())
            points_elongation_single.append(points_elongation_tensor.numpy())

        # points.append(np.concatenate(points_single, axis=0))
        # cp_points.append(np.concatenate(cp_points_single, axis=0))
        # points_NLZ.append(np.concatenate(points_NLZ_single, axis=0))
        # points_intensity.append(np.concatenate(points_intensity_single, axis=0))
        # points_elongation.append(np.concatenate(points_elongation_single, axis=0))
        points.append(tf.concat(points_single, axis=0))
        cp_points.append(tf.concat(cp_points_single, axis=0))
        points_NLZ.append(tf.concat(points_NLZ_single, axis=0))
        points_intensity.append(tf.concat(points_intensity_single, axis=0))
        points_elongation.append(tf.concat(points_elongation_single, axis=0))

    return points, cp_points, points_NLZ, points_intensity, points_elongation


def save_lidar_points(frame, cur_save_path, use_two_returns=True, do_semantic_label=False, ori_annotation=None):
    ret_outputs = frame_utils.parse_range_image_and_camera_projection(frame)
    if len(ret_outputs) == 4:
        range_images, camera_projections, seg_labels, range_image_top_pose = ret_outputs
    else:
        assert len(ret_outputs) == 3
        range_images, camera_projections, range_image_top_pose = ret_outputs

    points, cp_points, points_in_NLZ_flag, points_intensity, points_elongation = convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1) if use_two_returns else (0,)
    )

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_in_NLZ_flag = np.concatenate(points_in_NLZ_flag, axis=0).reshape(-1, 1)
    points_intensity = np.concatenate(points_intensity, axis=0).reshape(-1, 1)
    points_elongation = np.concatenate(points_elongation, axis=0).reshape(-1, 1)

    num_points_of_each_lidar = [point.shape[0] for point in points]
    save_points = np.concatenate([
        points_all, points_intensity, points_elongation, points_in_NLZ_flag
    ], axis=-1).astype(np.float32)

    if do_semantic_label:
        if not cur_save_path.exists():
            np.save(cur_save_path, save_points)
    else:
        np.save(cur_save_path, save_points)
    # print('saving to ', cur_save_path)

    if do_semantic_label:
        if frame.lasers[0].ri_return1.segmentation_label_compressed:
            sl_points = convert_range_image_to_point_cloud_labels(
                frame, range_images, seg_labels, ri_index=(0, 1) if use_two_returns else (0,)
            )
            points_label = np.concatenate(sl_points, axis=0)
            label_filename = os.path.basename(cur_save_path)
            label_save_path = os.path.join(os.path.dirname(cur_save_path), 'sem_label', label_filename)
            np.save(label_save_path, points_label)

            modified_names = modify_labels(save_points, points_label, ori_annotation)
            ori_annotation['name'] = modified_names

    return num_points_of_each_lidar


def process_single_sequence(sequence_file, save_path, sampled_interval, has_label=True, use_two_returns=True, update_info_only=False, do_semantic_label=False):
    sequence_name = os.path.splitext(os.path.basename(sequence_file))[0]

    # print('Load record (sampled_interval=%d): %s' % (sampled_interval, sequence_name))
    if not sequence_file.exists():
        print('NotFoundError: %s' % sequence_file)
        return []

    dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
    cur_save_dir = save_path / sequence_name
    cur_save_dir.mkdir(parents=True, exist_ok=True)
    if do_semantic_label:
        sem_label_dir = os.path.join(cur_save_dir, 'sem_label')
        os.makedirs(sem_label_dir, exist_ok=True)
    pkl_file = cur_save_dir / ('%s.pkl' % sequence_name)

    sequence_infos = []
    if pkl_file.exists() and not do_semantic_label: # temporary do everything
        sequence_infos = pickle.load(open(pkl_file, 'rb'))
        sequence_infos_old = None
        if not update_info_only:
            print('Skip sequence since it has been processed before: %s' % pkl_file)
            return sequence_infos
        else:
            sequence_infos_old = sequence_infos
            sequence_infos = []

    for cnt, data in enumerate(dataset):
        if cnt % sampled_interval != 0:
            continue
        # print(sequence_name, cnt)
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        info = {}
        pc_info = {'num_features': 5, 'lidar_sequence': sequence_name, 'sample_idx': cnt}
        info['point_cloud'] = pc_info

        info['frame_id'] = sequence_name + ('_%03d' % cnt)
        info['metadata'] = {
            'context_name': frame.context.name,
            'timestamp_micros': frame.timestamp_micros
        }
        image_info = {}
        for j in range(5):
            width = frame.context.camera_calibrations[j].width
            height = frame.context.camera_calibrations[j].height
            image_info.update({'image_shape_%d' % j: (height, width)})
        info['image'] = image_info

        pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)
        info['pose'] = pose

        if has_label:
            annotations = generate_labels(frame, pose=pose)
            info['annos'] = annotations

        if update_info_only and sequence_infos_old is not None:
            assert info['frame_id'] == sequence_infos_old[cnt]['frame_id']
            num_points_of_each_lidar = sequence_infos_old[cnt]['num_points_of_each_lidar']
        else:
            num_points_of_each_lidar = save_lidar_points(
                frame, cur_save_dir / ('%04d.npy' % cnt), use_two_returns=use_two_returns, do_semantic_label=do_semantic_label and has_label, ori_annotation=annotations,
            )
        info['num_points_of_each_lidar'] = num_points_of_each_lidar

        sequence_infos.append(info)

    with open(pkl_file, 'wb') as f:
        pickle.dump(sequence_infos, f)

    print('Infos are saved to (sampled_interval=%d): %s' % (sampled_interval, pkl_file))
    return sequence_infos


