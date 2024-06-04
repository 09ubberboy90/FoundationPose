# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from foundationpose.estimater import *
from foundationpose.datareader import *
import sys
import pyzed.sl as sl
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose

class Tracker(Node):
    def __init__(self):
        super().__init__('tracker')
        self.code_dir = "/home/ubb/Documents/FoundationPose"
        self.mesh_file =f'{self.code_dir}/demo_data/realtime/mesh/finger.ply'
        self.mask_file =f'{self.code_dir}/demo_data/realtime/mask_2.png'
        self.test_scene_dir =f'{self.code_dir}/demo_data/realtime'
        self.est_refine_iter =5
        self.track_refine_iter =2
        self.debug =1
        self.debug_dir =f'{self.code_dir}/debug'
        
        self.publisher_ = self.create_publisher(Pose, 'pose', 10)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
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

        self.zed = sl.Camera()
        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()
            
        fx = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.fx
        fy = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.fy
        cx = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.cx
        cy = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        self.image_size = self.zed.get_camera_information().camera_configuration.resolution

        # Declare your sl.Mat matrices
        self.image_zed = sl.Mat()
        self.depth_image_zed = sl.Mat()

        self.mesh = trimesh.load(self.mesh_file)
        self.debug = self.debug
        self.debug_dir = self.debug_dir
        os.system(
            f'rm -rf {self.debug_dir}/* && mkdir -p {self.debug_dir}/track_vis {self.debug_dir}/ob_in_cam')

        self.to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals, mesh=self.mesh,
                            scorer=self.scorer, refiner=self.refiner, debug_dir=self.debug_dir, debug=self.debug, glctx=self.glctx)
        logging.info("estimator initialization done")
        self.counter = 0


    def timer_callback(self):
        err = self.zed.grab()
        if err != sl.ERROR_CODE.SUCCESS:
            return
        logging.info(f'i:{self.counter}')
        self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
        self.zed.retrieve_measure(self.depth_image_zed, sl.MEASURE.DEPTH)

        color = self.image_zed.get_data()[...,:3]
        color = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
        depth = self.depth_image_zed.get_data().astype(np.uint16)/1e3
        depth = cv2.resize(depth, (self.image_size.width, self.image_size.height),interpolation=cv2.INTER_NEAREST)
        if self.counter == 0:
            mask = get_mask(self.image_size, self.mask_file).astype(bool)
            pose = self.est.register(K=self.K, rgb=color, depth=depth,
                                ob_mask=mask, iteration=self.est_refine_iter)

            if self.debug >= 3:
                m = self.mesh.copy()
                m.apply_transform(pose)
                m.export(f'{self.debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, self.K)
                valid = depth >= 0.1
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(
                    f'{self.debug_dir}/scene_complete.ply', pcd)
        else:
            pose = self.est.track_one(rgb=color, depth=depth,
                                 K=self.K, iteration=self.track_refine_iter)

        os.makedirs(f'{self.debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(
            f'{self.debug_dir}/ob_in_cam/{self.counter}.txt', pose.reshape(4, 4))

        if self.debug >= 1:
            center_pose = pose@np.linalg.inv(self.to_origin)
            vis = draw_posed_3d_box(
                self.K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1,
                                K=self.K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('1', vis[..., ::-1])
            cv2.waitKey(1)

        if self.debug >= 2:
            os.makedirs(f'{self.debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(
                f'{self.debug_dir}/track_vis/{self.counter}.png', vis)
        self.counter += 1

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


def main(args=None):
    rclpy.init()
    tracker = Tracker()
    rclpy.spin(tracker)
    tracker.destroy_node()
    rclpy.shutdown()
    sys.exit(0)

if __name__ == '__main__':
    main()