# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from calendar import c
from cycler import K
from estimater import *
from datareader import *
import argparse
import sys
import pyzed.sl as sl


def get_mask(image_size, file):
    mask = cv2.imread(file, -1)
    if len(mask.shape) == 3:
      for c in range(3):
        if mask[..., c].sum() > 0:
          mask = mask[..., c]
          break
    mask = cv2.resize(mask, (image_size.width, image_size.height), interpolation=cv2.INTER_NEAREST).astype(
        bool).astype(np.uint8)
    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str,
                        default=f'{code_dir}/demo_data/realtime/mesh/finger.ply')
    parser.add_argument('--mask_file', type=str,
                        default=f'{code_dir}/demo_data/realtime/mask_2.png')
    parser.add_argument('--test_scene_dir', type=str,
                        default=f'{code_dir}/demo_data/realtime')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    init = sl.InitParameters()
    # Set configuration parameters for the ZED
    init.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 60  # The framerate is lowered to avoid any USB3 bandwidth issues
    init.depth_maximum_distance = 8000
    init.depth_minimum_distance = 200
    init.coordinate_units = sl.UNIT.MILLIMETER

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
        
    fx = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.fx
    fy = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.fy
    cx = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.cx
    cy = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.cy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    image_size = zed.get_camera_information().camera_configuration.resolution

    # Declare your sl.Mat matrices
    image_zed = sl.Mat()
    depth_image_zed = sl.Mat()

    mesh = trimesh.load(args.mesh_file)
    print(mesh)
    debug = args.debug
    debug_dir = args.debug_dir
    os.system(
        f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    print(to_origin, extents)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                         scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    counter = 0
    while True:
        err = zed.grab()
        if err != sl.ERROR_CODE.SUCCESS:
            continue
        logging.info(f'i:{counter}')
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_image_zed, sl.MEASURE.DEPTH)

        color = image_zed.get_data()[...,:3]
        color = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
        depth = depth_image_zed.get_data().astype(np.uint16)/1e3
        depth = cv2.resize(depth, (image_size.width, image_size.height),interpolation=cv2.INTER_NEAREST)
        if counter == 0:
            mask = get_mask(image_size, args.mask_file).astype(bool)
            pose = est.register(K=K, rgb=color, depth=depth,
                                ob_mask=mask, iteration=args.est_refine_iter)

            if debug >= 3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, K)
                valid = depth >= 0.1
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(
                    f'{debug_dir}/scene_complete.ply', pcd)
        else:
            pose = est.track_one(rgb=color, depth=depth,
                                 K=K, iteration=args.track_refine_iter)

        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(
            f'{debug_dir}/ob_in_cam/{counter}.txt', pose.reshape(4, 4))

        if debug >= 1:
            center_pose = pose@np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(
                K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1,
                                K=K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('1', vis[..., ::-1])
            cv2.waitKey(1)

        if debug >= 2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(
                f'{debug_dir}/track_vis/{counter}.png', vis)
        counter += 1