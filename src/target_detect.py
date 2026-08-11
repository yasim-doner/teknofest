#!/usr/bin/env python3

import math
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Vector3
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32


def calculate_angles_from_camera_pixel(u, v, width, height, fx=525.0, fy=525.0):
    """
    Sadece kamera görüntüsündeki hedef piksel (u, v) ile resim merkezi (u_c, v_c)
    arasındaki piksel farkları (du, dv) ve kameranın odak uzaklıklarını (fx, fy)
    kullanarak mesafe hesabı yapmadan açısal sapmayı (yaw_offset, pitch_offset) hesaplar.

    yaw_offset = atan2(du, fx)
    pitch_offset = atan2(-dv, fy)
    """
    u_c = width / 2.0
    v_c = height / 2.0

    du = float(u - u_c)
    dv = float(v - v_c)

    yaw_rad = math.atan2(du, fx)
    pitch_rad = math.atan2(-dv, fy)

    return math.degrees(yaw_rad), math.degrees(pitch_rad), du, dv


class TargetDetector(Node):
    """
    Kamera görüntüsü işleme ile hedef (hedef tahtası / nişan noktası) tespiti ve
    turret / lazer atış açılarını (/laser_angle) kamera piksel sapmaları üzerinden hesaplayan ROS 2 Düğümü.
    """

    def __init__(self):
        super().__init__("target_detect")

        # Parametreler
        self.declare_parameter("image_topic", "/rover/camera/image_raw")
        self.declare_parameter("fx", 525.0)
        self.declare_parameter("fy", 525.0)
        self.declare_parameter("active_stage", 8)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.fx = float(self.get_parameter("fx").value)
        self.fy = float(self.get_parameter("fy").value)
        self.active_stage = int(self.get_parameter("active_stage").value)

        self.bridge = CvBridge()

        # State & Data Variables
        self.current_stage = 0
        self.laser_active = False

        # QoS Ayarları
        self.camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Publishers
        self.laser_angle_pub = self.create_publisher(
            Vector3,
            "/laser_angle",
            10,
        )
        self.target_point_pub = self.create_publisher(
            PointStamped,
            "/teknofest/target_point",
            10,
        )
        self.debug_pub = self.create_publisher(
            Image,
            "/teknofest/target_debug_image",
            10,
        )

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            self.camera_qos,
        )
        self.stage_sub = self.create_subscription(
            Int32,
            "/teknofest/stage_id",
            self.stage_callback,
            10,
        )
        self.laser_on_sub = self.create_subscription(
            Bool,
            "/teknofest/laser_on",
            self.laser_on_callback,
            10,
        )

        self.get_logger().info(
            f"=== Target Detect Node Başlatıldı (Topic={self.image_topic}, fx={self.fx}, fy={self.fy}) ==="
        )

    def stage_callback(self, msg):
        self.current_stage = int(msg.data)

    def laser_on_callback(self, msg):
        self.laser_active = bool(msg.data)

    def detect_target_in_image(self, frame):
        """
        Siyah-beyaz hedef tahtasını uzaktan (küçük boyutlarda) ortak merkezli (konsantrik)
        çemberler arayarak tespit eder. Gürültülere ve renk değişimlerine karşı dayanıklıdır.
        Dönen değer: (target_found, u_center, v_center, radius, contour)
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Gürültü giderme ve kenar algılama (Canny)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is None or len(contours) == 0:
            return False, 0, 0, 0, None

        candidates = []

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 10 or area > (w * h * 0.2):
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < 3 or radius > 120:
                continue

            # Dairesellik kontrolü (4 * pi * area / perimeter^2)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            # Bounding box en-boy oranı (Aspect ratio)
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = float(bw) / float(bh) if bh > 0 else 0.0

            if circularity > 0.4 and 0.55 <= aspect <= 1.45:
                candidates.append({
                    'idx': i,
                    'center': (cx, cy),
                    'radius': radius,
                    'area': area,
                    'cnt': cnt
                })

        # Yakın merkezli (ortak merkezli / konsantrik) çember gruplarını bul
        groups = []
        for i, c1 in enumerate(candidates):
            group = [c1]
            for j, c2 in enumerate(candidates):
                if i == j:
                    continue
                dist = np.hypot(c1['center'][0] - c2['center'][0], c1['center'][1] - c2['center'][1])
                if dist < max(4.0, c1['radius'] * 0.25):
                    group.append(c2)

            if len(group) >= 2:  # En az 2 konsantrik hiyerarşik çember
                groups.append(group)

        groups.sort(key=lambda g: len(g), reverse=True)

        if len(groups) > 0:
            best_group = groups[0]
            avg_cx = int(round(np.mean([c['center'][0] for c in best_group])))
            avg_cy = int(round(np.mean([c['center'][1] for c in best_group])))
            max_r = int(round(np.max([c['radius'] for c in best_group])))
            best_cnt = best_group[0]['cnt']

            return True, avg_cx, avg_cy, max_r, best_cnt

        return False, 0, 0, 0, None

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Görüntü dönüştürme hatası: {e}")
            return

        h, w = frame.shape[:2]
        found, u, v, radius, cnt = self.detect_target_in_image(frame)

        debug_img = frame.copy()
        center_u, center_v = w // 2, h // 2

        # Resim merkezini (artı göstergesi) çiz
        cv2.drawMarker(
            debug_img,
            (center_u, center_v),
            (255, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2,
        )

        if found:
            # Kamera piksel verisinden açı hesabı (Uzaklık / Karekök olmadan)
            yaw_deg, pitch_deg, du, dv = calculate_angles_from_camera_pixel(
                u, v, w, h, self.fx, self.fy
            )

            # /laser_angle yayınla
            angle_msg = Vector3()
            angle_msg.x = float(yaw_deg)
            angle_msg.y = float(pitch_deg)
            angle_msg.z = 0.0
            self.laser_angle_pub.publish(angle_msg)

            # /teknofest/target_point yayınla
            pt_msg = PointStamped()
            pt_msg.header.stamp = self.get_clock().now().to_msg()
            pt_msg.header.frame_id = "camera_link"
            pt_msg.point.x = float(du)
            pt_msg.point.y = float(dv)
            pt_msg.point.z = 0.0
            self.target_point_pub.publish(pt_msg)

            # Visual Debug Drawing
            cv2.circle(debug_img, (u, v), radius, (0, 255, 0), 2)
            cv2.circle(debug_img, (u, v), 4, (0, 0, 255), -1)
            cv2.line(debug_img, (center_u, center_v), (u, v), (0, 255, 255), 2)

            info_str = f"Yaw: {yaw_deg:.2f} deg, Pitch: {pitch_deg:.2f} deg"
            pix_str = f"du: {du:.1f}px, dv: {dv:.1f}px"

            cv2.putText(
                debug_img,
                info_str,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                debug_img,
                pix_str,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        else:
            cv2.putText(
                debug_img,
                "Hedef Aranıyor...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Debug görüntüsünü yayınla
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Debug görüntü yayınlama hatası: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TargetDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
