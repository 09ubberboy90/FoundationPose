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
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge

class Tracker(Node):
    def __init__(self):
        super().__init__('tracker')
        self.code_dir = "/home/ubb/Documents/FoundationPose"
        self.mesh_file =f'{self.code_dir}/demo_data/realtime/mesh/untitled.ply'
        self.mask_file =f'{self.code_dir}/demo_data/realtime/mask.png'
        self.test_scene_dir =f'{self.code_dir}/demo_data/realtime'
        self.est_refine_iter =5
        self.track_refine_iter =2
        self.debug = 1
        self.debug_dir =f'{self.code_dir}/debug'
        
        self.publisher = self.create_publisher(Pose, 'pose', 10)
        self.publisher_stamped = self.create_publisher(PoseStamped, 'pose_stamp', 10)
        self.image_pub = self.create_publisher(Image, '/detected_object', 10)
        
        self.image_sub = self.create_subscription(Image, '/zed/zed_node/left/image_rect_color', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/zed/zed_node/depth/depth_registered', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/zed/zed_node/left/camera_info', self.camera_info_callback, 10)
        self.color = None
        self.depth = None
        self.camera_info = None
        self.bridge = CvBridge()
        
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        set_logging_format(logging.WARN)
        set_seed(0)

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
        if self.color is None or self.depth is None or self.camera_info is None:
            return
        timing = self.get_clock().now()
        if self.counter == 0:
            mask = get_mask(self.image_size, self.mask_file).astype(bool)
            pose = self.est.register(K=self.K, rgb=self.color, depth=self.depth,
                                ob_mask=mask, iteration=self.est_refine_iter)

            if self.debug >= 3:
                m = self.mesh.copy()
                m.apply_transform(pose)
                m.export(f'{self.debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(self.depth, self.K)
                valid = self.depth >= 0.1
                pcd = toOpen3dCloud(xyz_map[valid], self.color[valid])
                o3d.io.write_point_cloud(
                    f'{self.debug_dir}/scene_complete.ply', pcd)
        else:
            pose = self.est.track_one(rgb=self.color, depth=self.depth,
                                 K=self.K, iteration=self.track_refine_iter)

        os.makedirs(f'{self.debug_dir}/ob_in_cam', exist_ok=True)
        pose = pose.reshape(4, 4)
        np.savetxt(
            f'{self.debug_dir}/ob_in_cam/{self.counter}.txt', pose)
        self.publish_pose(pose)

        if self.debug >= 1:
            center_pose = pose@np.linalg.inv(self.to_origin)
            vis = draw_posed_3d_box(
                self.K, img=self.color, ob_in_cam=center_pose, bbox=self.bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1,
                                K=self.K, thickness=3, transparency=0, is_input_rgb=True)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(vis, 'rgb8'))
        

        if self.debug >= 2:
            os.makedirs(f'{self.debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(
                f'{self.debug_dir}/track_vis/{self.counter}.png', vis)
        self.counter += 1
        self.get_logger().info('Publishing: "%s"' % str(self.get_clock().now() - timing))
    
    def publish_pose(self, pose):
        stamped = PoseStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = "zed_camera_link" 
        msg = Pose()
        msg.position.x = float(pose[0,3])
        msg.position.y = float(pose[1,3])
        msg.position.z = float(pose[2,3])
        quat = R.from_matrix(pose[:3, :3]).as_quat()
        msg.orientation.x = quat[0]
        msg.orientation.y = quat[1]
        msg.orientation.z = quat[2]
        msg.orientation.w = quat[3]
        self.publisher.publish(msg)
        stamped.pose = msg
        self.publisher_stamped.publish(stamped)
        
    def image_callback(self, msg):
        self.color = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
    
    def depth_callback(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, 'mono16')/1e3
    
    def camera_info_callback(self, msg:CameraInfo):
        self.camera_info = msg
        self.fx = self.camera_info.k[0]
        self.fy = self.camera_info.k[4]
        self.cx = self.camera_info.k[2]
        self.cy = self.camera_info.k[5]
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        self.image_size = self.camera_info.width, self.camera_info.height
    
def get_mask(image_size, file):
    mask = cv2.imread(file, -1)
    if len(mask.shape) == 3:
      for c in range(3):
        if mask[..., c].sum() > 0:
          mask = mask[..., c]
          break
    mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST).astype(
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