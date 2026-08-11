#!/usr/bin/env python3

import glob
import math
import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int32, String
from visualization_msgs.msg import Marker, MarkerArray


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class SignDetector(Node):
    # sim.launch.py içindeki stage başlangıç konumları.
    STAGE_POSITIONS = {
        1: (16.0, 0.0),
        2: (11.0, 0.0),
        3: (6.0, 0.0),
        4: (2.0, 10.0),
        5: (7.0, 10.0),
        6: (14.0, 10.0),
        7: (18.0, 20.0),
        8: (11.5, 20.0),
        9: (7.8, 20.0),
        10: (4.8, 20.0),
        11: (20.0, -8.0),
    }

    """
    Teknofest parkurundaki kırmızı çerçeveli fiziksel levhaları sırasıyla sayar.

    Bu sürüm:
    - worker thread kullanmaz,
    - template matching kullanmaz,
    - HoughCircles kullanmaz,
    - her kamera karesini hızlı connected-components yöntemiyle işler,
    - ilk fiziksel levhayı stage_1, sonrakileri stage_2... olarak yayınlar,
    - Stage 8 sonrasında görülen sonraki yakın levhayı STOP olarak yayınlar.
    """

    def __init__(self):
        super().__init__("sign_detect")

        self.declare_parameter(
            "image_topic",
            "/rover/camera/image_raw",
        )
        self.declare_parameter("initial_stage", 0)

        self.declare_parameter("min_component_area", 30)
        self.declare_parameter("max_component_area_ratio", 0.15)
        self.declare_parameter("min_candidate_size", 8)
        self.declare_parameter("max_candidate_size", 220)

        self.declare_parameter("min_aspect", 0.45)
        self.declare_parameter("max_aspect", 1.75)
        self.declare_parameter("min_white_ratio", 0.08)
        self.declare_parameter("min_red_fill_ratio", 0.02)

        self.declare_parameter("enter_radius", 14)
        self.declare_parameter("exit_radius", 9)
        self.declare_parameter("enter_frames", 2)
        self.declare_parameter("exit_frames", 3)

        self.declare_parameter("stop_guard_seconds", 3.0)
        self.declare_parameter("diagnostic_every_frames", 5)
        self.declare_parameter("min_stage_travel_distance", 1.5)
        self.declare_parameter("min_stage_interval_seconds", 2.0)
        self.declare_parameter("use_stage_geofence", True)
        self.declare_parameter("stage_geofence_radius", 3.0)
        self.declare_parameter("use_odom_fallback", True)
        self.declare_parameter("odom_fallback_radius", 2.5)
        self.declare_parameter("final_stage", 10)
        self.declare_parameter("stop_after_final_stage", True)

        # Görüntünün alt kısmındaki koniler stage levhası değildir.
        self.declare_parameter("max_stage_candidate_y_ratio", 0.78)
        self.declare_parameter("upper_enter_radius", 10)

        # Şablon eşleme (Template matching) parametreleri
        self.declare_parameter(
            "textures_dir",
            "/home/myazou/rover_ws/src/teknofest/textures",
        )
        self.declare_parameter("use_template_matching", True)
        self.declare_parameter("template_match_threshold", 0.40)
        self.declare_parameter("show_unmatched_candidates", False)
        self.declare_parameter("real_sign_diameter", 0.60)
        self.declare_parameter("enable_physical_size_filter", True)
        self.declare_parameter("min_sign_diameter", 0.35)
        self.declare_parameter("max_sign_diameter", 0.85)

        self.image_topic = str(
            self.get_parameter("image_topic").value
        )
        self.initial_stage = max(
            0,
            min(
                11,
                int(
                    self.get_parameter(
                        "initial_stage"
                    ).value
                ),
            ),
        )

        self.min_component_area = max(
            1,
            int(
                self.get_parameter(
                    "min_component_area"
                ).value
            ),
        )
        self.max_component_area_ratio = float(
            self.get_parameter(
                "max_component_area_ratio"
            ).value
        )
        self.min_candidate_size = max(
            2,
            int(
                self.get_parameter(
                    "min_candidate_size"
                ).value
            ),
        )
        self.max_candidate_size = max(
            self.min_candidate_size + 1,
            int(
                self.get_parameter(
                    "max_candidate_size"
                ).value
            ),
        )

        self.min_aspect = float(
            self.get_parameter("min_aspect").value
        )
        self.max_aspect = float(
            self.get_parameter("max_aspect").value
        )
        self.min_white_ratio = float(
            self.get_parameter(
                "min_white_ratio"
            ).value
        )
        self.min_red_fill_ratio = float(
            self.get_parameter(
                "min_red_fill_ratio"
            ).value
        )

        self.enter_radius = max(
            1,
            int(
                self.get_parameter(
                    "enter_radius"
                ).value
            ),
        )
        self.exit_radius = max(
            1,
            int(
                self.get_parameter(
                    "exit_radius"
                ).value
            ),
        )
        self.enter_frames = max(
            1,
            int(
                self.get_parameter(
                    "enter_frames"
                ).value
            ),
        )
        self.exit_frames = max(
            1,
            int(
                self.get_parameter(
                    "exit_frames"
                ).value
            ),
        )

        self.stop_guard_seconds = max(
            0.0,
            float(
                self.get_parameter(
                    "stop_guard_seconds"
                ).value
            ),
        )
        self.diagnostic_every_frames = max(
            1,
            int(
                self.get_parameter(
                    "diagnostic_every_frames"
                ).value
            ),
        )

        self.min_stage_travel_distance = max(
            0.0,
            float(
                self.get_parameter(
                    "min_stage_travel_distance"
                ).value
            ),
        )
        self.min_stage_interval_seconds = max(
            0.0,
            float(
                self.get_parameter(
                    "min_stage_interval_seconds"
                ).value
            ),
        )

        self.use_stage_geofence = bool(
            self.get_parameter(
                "use_stage_geofence"
            ).value
        )
        self.stage_geofence_radius = max(
            0.5,
            float(
                self.get_parameter(
                    "stage_geofence_radius"
                ).value
            ),
        )

        self.use_odom_fallback = bool(
            self.get_parameter(
                "use_odom_fallback"
            ).value
        )
        self.odom_fallback_radius = max(
            0.5,
            float(
                self.get_parameter(
                    "odom_fallback_radius"
                ).value
            ),
        )

        self.final_stage = max(
            1,
            min(
                11,
                int(
                    self.get_parameter(
                        "final_stage"
                    ).value
                ),
            ),
        )
        self.stop_after_final_stage = bool(
            self.get_parameter(
                "stop_after_final_stage"
            ).value
        )

        self.max_stage_candidate_y_ratio = float(
            self.get_parameter(
                "max_stage_candidate_y_ratio"
            ).value
        )
        self.upper_enter_radius = max(
            1,
            int(
                self.get_parameter(
                    "upper_enter_radius"
                ).value
            ),
        )

        self.textures_dir = str(
            self.get_parameter("textures_dir").value
        )
        self.use_template_matching = bool(
            self.get_parameter("use_template_matching").value
        )
        self.template_match_threshold = float(
            self.get_parameter("template_match_threshold").value
        )
        self.show_unmatched_candidates = bool(
            self.get_parameter("show_unmatched_candidates").value
        )
        self.real_sign_diameter = float(
            self.get_parameter("real_sign_diameter").value
        )
        self.real_sign_radius = self.real_sign_diameter / 2.0
        self.enable_physical_size_filter = bool(
            self.get_parameter("enable_physical_size_filter").value
        )
        self.min_sign_diameter = float(
            self.get_parameter("min_sign_diameter").value
        )
        self.max_sign_diameter = float(
            self.get_parameter("max_sign_diameter").value
        )

        self.templates = {}
        if self.use_template_matching:
            self.load_templates()

        self.bridge = CvBridge()

        # Kamera publisher'ı RELIABLE olduğu için aynı QoS kullanılır.
        self.camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.debug_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Marker'lar için TRANSIENT_LOCAL — RViz bağlanma anında da görür
        self.marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            self.camera_qos,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/rover/odom",
            self.odom_callback,
            10,
        )

        self.stage_sub = self.create_subscription(
            Int32,
            "/teknofest/stage_id",
            self.stage_callback,
            10,
        )

        self.stage_pub = self.create_publisher(
            Int32,
            "/teknofest/stage_id",
            10,
        )
        self.stage_order_pub = self.create_publisher(
            Int32,
            "/teknofest/stage_order",
            10,
        )

        self.label_pub = self.create_publisher(
            String,
            "/teknofest/sign_label",
            10,
        )
        self.confidence_pub = self.create_publisher(
            Float32,
            "/teknofest/sign_confidence",
            10,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/teknofest/sign_markers",
            self.marker_qos,
        )
        self.debug_pub = self.create_publisher(
            Image,
            "/teknofest/sign_debug_image",
            self.debug_qos,
        )
        self.final_stop_pub = self.create_publisher(
            Bool,
            "/teknofest/final_stop",
            10,
        )

        self.frame_count = 0
        self.stage_id = self.initial_stage
        self.last_stage_change_time = None
        self.last_stage_change_x = None
        self.last_stage_change_y = None

        self.final_stop_active = False

        self.last_radius = 0
        self.last_center = None
        self.last_candidate_count = 0
        self.last_red_pixels = 0
        self.last_process_seconds = 0.0

        self.current_matched_label = "none"
        self.current_match_score = 0.0

        self.first_callback_seen = False

        self.odom_x = None
        self.odom_y = None
        self.odom_yaw = 0.0

        self.detected_markers_history = []
        self.detected_signs = {}

        # 1Hz'de kayıtlı tabelaları RViz'e yeniden yayınla
        self.create_timer(1.0, self._republish_saved_markers)

        self.publish_state()

        self.get_logger().info(
            "=== Basit ve hızlı tabela algılayıcı başladı ==="
        )
        self.get_logger().info(
            f"Kamera={self.image_topic}, "
            f"başlangıç_stage={self.stage_id}, "
            f"enter={self.enter_radius}px, "
            f"exit={self.exit_radius}px, "
            f"üst_enter={self.upper_enter_radius}px, "
            f"üst_y_oranı={self.max_stage_candidate_y_ratio:.2f}, "
            f"min_stage_mesafe={self.min_stage_travel_distance:.1f}m, "
            f"min_stage_süre={self.min_stage_interval_seconds:.1f}s, "
            f"stage_geofence={self.stage_geofence_radius:.1f}m, "
            f"odom_fallback={self.use_odom_fallback} ({self.odom_fallback_radius:.1f}m), "
            f"final_stage={self.final_stage}, "
            f"final_duruş={self.stop_after_final_stage}, "
            "kamera_QoS=RELIABLE"
        )

    def now_seconds(self):
        return (
            self.get_clock()
            .now()
            .nanoseconds
            / 1_000_000_000.0
        )

    def _republish_saved_markers(self):
        """1Hz timer callback: Kayıtlı tüm tabelaları (stage + stop) RViz'e periyodik olarak yayınla."""
        if not self.detected_signs:
            return

        marker_array = MarkerArray()
        now_time = self.get_clock().now().to_msg()
        frame_id = "odom" if (self.odom_x is not None) else "base_link"

        for sign_key, sdata in self.detected_signs.items():
            if isinstance(sign_key, int):
                marker_id_base = sign_key * 10
                label_text = f"{sdata['label']} (Stage {sign_key})"
            else:
                marker_id_base = 900
                label_text = f"{sdata['label']} ({sign_key})"

            p_text = Marker()
            p_text.header.frame_id = frame_id
            p_text.header.stamp = now_time
            p_text.ns = "sign_map_text"
            p_text.id = marker_id_base
            p_text.type = Marker.TEXT_VIEW_FACING
            p_text.action = Marker.ADD
            p_text.pose.position.x = sdata["x"]
            p_text.pose.position.y = sdata["y"]
            p_text.pose.position.z = 0.80
            p_text.pose.orientation.w = 1.0
            p_text.scale.z = 0.45
            p_text.color.r = 0.0
            p_text.color.g = 0.9
            p_text.color.b = 1.0
            p_text.color.a = 1.0
            p_text.text = label_text

            marker_array.markers.append(p_text)

            p_shape = Marker()
            p_shape.header.frame_id = frame_id
            p_shape.header.stamp = now_time
            p_shape.ns = "sign_map_shape"
            p_shape.id = marker_id_base + 1
            p_shape.type = Marker.CYLINDER
            p_shape.action = Marker.ADD
            p_shape.pose.position.x = sdata["x"]
            p_shape.pose.position.y = sdata["y"]
            p_shape.pose.position.z = 0.30
            p_shape.pose.orientation.w = 1.0
            p_shape.scale.x = 0.50
            p_shape.scale.y = 0.50
            p_shape.scale.z = 0.60
            p_shape.color.r = 0.0
            p_shape.color.g = 0.6
            p_shape.color.b = 1.0
            p_shape.color.a = 0.85

            marker_array.markers.append(p_shape)

        if marker_array.markers:
            self.marker_pub.publish(marker_array)

    @staticmethod
    def make_red_mask(hsv):
        # Kırmızı renk HSV aralığı (Doygunluk S >= 55 ile kahverengi taşlar elenir)
        lower_red_1 = np.array([0, 55, 45], dtype=np.uint8)
        upper_red_1 = np.array([16, 255, 255], dtype=np.uint8)

        lower_red_2 = np.array([160, 55, 45], dtype=np.uint8)
        upper_red_2 = np.array([179, 255, 255], dtype=np.uint8)

        mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

        red_mask = cv2.bitwise_or(mask_1, mask_2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return red_mask

    def load_templates(self):
        self.templates = {}
        if not os.path.exists(self.textures_dir):
            self.get_logger().warning(
                f"Şablon dizini bulunamadı: {self.textures_dir}"
            )
            return

        for filepath in glob.glob(os.path.join(self.textures_dir, "*.png")):
            key = os.path.basename(filepath).replace(".png", "")
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            h, w = gray.shape[:2]
            inner_gray = gray_norm[int(h * 0.12) : int(h * 0.88), int(w * 0.12) : int(w * 0.88)]
            inner_gray_norm = cv2.normalize(inner_gray, None, 0, 255, cv2.NORM_MINMAX)
            inner_edges = cv2.Canny(inner_gray_norm, 30, 120)

            _, inner_thresh = cv2.threshold(
                inner_gray_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            stage_num = None
            is_stop = (key == "sign_stop")
            is_final = ("exit" in key or key in ["sign_10_exit", "sign_11_exit"])

            if key.startswith("sign_") and not is_stop:
                num_part = key.replace("sign_", "").replace("_exit", "")
                if num_part.isdigit():
                    stage_num = int(num_part)

            self.templates[key] = {
                "key": key,
                "gray": gray_norm,
                "inner_gray": inner_gray_norm,
                "inner_edges": inner_edges,
                "thresh": inner_thresh,
                "stage_num": stage_num,
                "is_stop": is_stop,
                "is_final": is_final,
            }

        self.get_logger().info(
            f"{len(self.templates)} adet tabela şablonu yüklendi: {list(self.templates.keys())}"
        )

    def match_candidate_template(self, frame, box):
        if not self.use_template_matching or not self.templates:
            return None, 0.0, None

        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]
        if rh < 6 or rw < 6:
            return None, 0.0, None

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_gray_norm = cv2.normalize(roi_gray, None, 0, 255, cv2.NORM_MINMAX)

        inner_roi = roi_gray_norm[int(rh * 0.12) : int(rh * 0.88), int(rw * 0.12) : int(rw * 0.88)]
        if inner_roi.size == 0 or inner_roi.shape[0] < 4 or inner_roi.shape[1] < 4:
            return None, 0.0, None

        inner_roi_norm = cv2.normalize(inner_roi, None, 0, 255, cv2.NORM_MINMAX)
        inner_roi_edges = cv2.Canny(inner_roi_norm, 30, 120)

        roi_std = float(np.std(inner_roi_norm))
        if roi_std > 5.0:
            _, roi_thresh = cv2.threshold(
                inner_roi_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        else:
            roi_thresh = None

        best_key = None
        best_score = -1.0
        best_tmpl = None

        for key, tmpl in self.templates.items():
            t_gray = cv2.resize(tmpl["gray"], (rw, rh))
            res_gray = max(0.0, float(np.max(cv2.matchTemplate(roi_gray_norm, t_gray, cv2.TM_CCOEFF_NORMED))))

            ih, iw = inner_roi_norm.shape[:2]
            t_inner = cv2.resize(tmpl["inner_gray"], (iw, ih))
            res_inner = max(0.0, float(np.max(cv2.matchTemplate(inner_roi_norm, t_inner, cv2.TM_CCOEFF_NORMED))))

            t_edges = cv2.resize(tmpl["inner_edges"], (iw, ih))
            res_edge = max(0.0, float(np.max(cv2.matchTemplate(inner_roi_edges, t_edges, cv2.TM_CCOEFF_NORMED))))

            if roi_thresh is not None:
                t_thresh = cv2.resize(tmpl["thresh"], (iw, ih))
                if float(np.std(roi_thresh)) > 0:
                    res_thresh = max(0.0, float(np.max(cv2.matchTemplate(roi_thresh, t_thresh, cv2.TM_CCOEFF_NORMED))))
                else:
                    res_thresh = res_inner
            else:
                res_thresh = res_inner

            combined_score = float(
                0.30 * res_gray + 0.40 * res_inner + 0.30 * max(res_edge, res_thresh)
            )

            if combined_score > best_score:
                best_score = combined_score
                best_key = key
                best_tmpl = tmpl

        return best_key, best_score, best_tmpl

    def find_candidates(
        self,
        frame,
        hsv,
        red_mask,
    ):
        height, width = frame.shape[:2]
        raw_candidates = []

        # STRATEJİ 1: İç Beyaz Daire → Kırmızı duvar üzerindeki tabelaları ayırır
        # Tabela içindeki SAF BEYAZ disk (S<50, V>160) aranır — gök/zemin renkleri elenir
        white_mask = cv2.inRange(hsv, np.array([0, 0, 160], dtype=np.uint8), np.array([179, 50, 255], dtype=np.uint8))
        kernel_w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        white_clean = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_w, iterations=2)
        white_contours, _ = cv2.findContours(white_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for wc in white_contours:
            wx, wy, ww, wh = cv2.boundingRect(wc)
            if ww < 8 or wh < 8 or ww > width * 0.30 or wh > height * 0.30:
                continue

            aspect = float(ww) / float(wh)
            if aspect < 0.40 or aspect > 2.5:
                continue

            # Şablon eşleme için: beyaz iç disk + küçük margin
            small_margin = max(2, int(max(ww, wh) * 0.20))
            sx1 = max(0, wx - small_margin)
            sy1 = max(0, wy - small_margin)
            sx2 = min(width, wx + ww + small_margin)
            sy2 = min(height, wy + wh + small_margin)

            # Kırmızı halka varlığı için daha geniş kontrol alanı
            ring_margin = max(4, int(max(ww, wh) * 0.55))
            rx1 = max(0, wx - ring_margin)
            ry1 = max(0, wy - ring_margin)
            rx2 = min(width, wx + ww + ring_margin)
            ry2 = min(height, wy + wh + ring_margin)

            if np.count_nonzero(red_mask[ry1:ry2, rx1:rx2]) < 12:
                continue

            cx = wx + ww // 2
            cy = wy + wh // 2
            radius = max(rx2 - rx1, ry2 - ry1) // 2

            best_key, match_score, tmpl_meta = self.match_candidate_template(frame, (sx1, sy1, sx2, sy2))

            if match_score >= 0.10:
                raw_candidates.append(
                    {
                        "cx": cx,
                        "cy": cy,
                        "radius": radius,
                        "box": (rx1, ry1, rx2, ry2),
                        "white_ratio": 0.5,
                        "red_ratio": 0.3,
                        "score": 60.0 + match_score * 60.0,
                        "frame_height": height,
                        "matched_key": best_key,
                        "match_score": match_score,
                        "tmpl_meta": tmpl_meta,
                    }
                )

        # STRATEJİ 2: Dış Kırmızı Kontur Tespiti (Normal gri/beyaz zeminlerdeki tabelalar)
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_connect, iterations=2)
        red_contours, _ = cv2.findContours(red_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in red_contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)

            if box_width < 10 or box_height < 10 or box_width > width * 0.60 or box_height > height * 0.60:
                continue

            aspect = float(box_width) / float(box_height)
            if aspect < 0.40 or aspect > 2.5:
                continue

            cx = x + box_width // 2
            cy = y + box_height // 2

            if cy > int(height * self.max_stage_candidate_y_ratio):
                continue

            radius = max(box_width, box_height) // 2
            expand = max(2, int(radius * 0.10))

            x1 = max(0, x - expand)
            y1 = max(0, y - expand)
            x2 = min(width, x + box_width + expand)
            y2 = min(height, y + box_height + expand)

            if x2 <= x1 or y2 <= y1:
                continue

            hsv_roi = hsv[y1:y2, x1:x2]
            red_roi = red_mask[y1:y2, x1:x2]

            w_mask = (hsv_roi[:, :, 1] < 70) & (hsv_roi[:, :, 2] > 110)
            white_ratio = float(np.count_nonzero(w_mask) / w_mask.size)
            red_fill_ratio = float(np.count_nonzero(red_roi) / red_roi.size)

            if white_ratio < 0.04 or red_fill_ratio < 0.02:
                continue

            best_key, match_score, tmpl_meta = self.match_candidate_template(frame, (x1, y1, x2, y2))

            score = radius + 20.0 * white_ratio + 10.0 * red_fill_ratio + 30.0 * max(0.0, match_score)

            raw_candidates.append(
                {
                    "cx": cx,
                    "cy": cy,
                    "radius": radius,
                    "box": (x1, y1, x2, y2),
                    "white_ratio": white_ratio,
                    "red_ratio": red_fill_ratio,
                    "score": score,
                    "frame_height": height,
                    "matched_key": best_key,
                    "match_score": match_score,
                    "tmpl_meta": tmpl_meta,
                }
            )

        # Non-Maximum Suppression (Çakışan küçük kutuları tek tam tabelaya indirge)
        raw_candidates.sort(key=lambda item: item["score"], reverse=True)
        candidates = []

        for cand in raw_candidates:
            keep = True
            cx1, cy1, r1 = cand["cx"], cand["cy"], cand["radius"]
            for existing in candidates:
                cx2, cy2, r2 = existing["cx"], existing["cy"], existing["radius"]
                dist = math.hypot(cx1 - cx2, cy1 - cy2)
                if dist < max(r1, r2) * 0.85:
                    keep = False
                    break
            if keep:
                candidates.append(cand)

        self.last_candidate_count = len(candidates)
        return candidates

    def stage_travel_distance(self):
        if (
            self.odom_x is None
            or self.odom_y is None
            or self.last_stage_change_x is None
            or self.last_stage_change_y is None
        ):
            return None

        dx = self.odom_x - self.last_stage_change_x
        dy = self.odom_y - self.last_stage_change_y

        return math.hypot(dx, dy)

    def distance_to_expected_stage(
        self,
        stage_id,
    ):
        target = self.STAGE_POSITIONS.get(
            int(stage_id)
        )

        if (
            target is None
            or self.odom_x is None
            or self.odom_y is None
        ):
            return None

        target_x, target_y = target

        return math.hypot(
            self.odom_x - target_x,
            self.odom_y - target_y,
        )

    def stage_interval_seconds(self):
        if self.last_stage_change_time is None:
            return None

        return (
            self.now_seconds()
            - self.last_stage_change_time
        )

    def stage_change_allowed(self, target_stage=None):
        if target_stage is None:
            target_stage = min(
                self.stage_id + 1,
                11,
            )

        distance = self.stage_travel_distance()
        interval = self.stage_interval_seconds()
        target_distance = (
            self.distance_to_expected_stage(
                target_stage
            )
        )

        distance_ok = (
            self.stage_id == 0
            or distance is None
            or distance
            >= self.min_stage_travel_distance
        )

        interval_ok = (
            self.stage_id == 0
            or interval is None
            or interval
            >= self.min_stage_interval_seconds
        )

        geofence_ok = (
            not self.use_stage_geofence
            or target_distance is None
            or target_distance
            <= self.stage_geofence_radius
        )

        target_text = (
            "bilinmiyor"
            if target_distance is None
            else f"{target_distance:.2f}m"
        )

        if (
            distance_ok
            and interval_ok
            and geofence_ok
        ):
            return True, (
                f"son_stage_mesafe="
                f"{distance if distance is not None else -1.0:.2f}m, "
                f"süre="
                f"{interval if interval is not None else -1.0:.2f}s, "
                f"stage_{target_stage}_uzaklık="
                f"{target_text}"
            )

        return False, (
            f"son_stage_mesafe="
            f"{distance if distance is not None else -1.0:.2f}m/"
            f"{self.min_stage_travel_distance:.2f}m, "
            f"süre="
            f"{interval if interval is not None else -1.0:.2f}s/"
            f"{self.min_stage_interval_seconds:.2f}s, "
            f"stage_{target_stage}_uzaklık="
            f"{target_text}/"
            f"{self.stage_geofence_radius:.2f}m"
        )

    def record_stage_change(self):
        self.last_stage_change_time = (
            self.now_seconds()
        )

        if (
            self.odom_x is not None
            and self.odom_y is not None
        ):
            self.last_stage_change_x = self.odom_x
            self.last_stage_change_y = self.odom_y

    def stage_callback(self, msg):
        self.stage_id = int(msg.data)

    def force_stage(self, new_stage, reason):
        new_stage = int(new_stage)
        if new_stage <= self.stage_id:
            return

        old_stage = self.stage_id
        self.stage_id = min(self.final_stage, new_stage)
        self.record_stage_change()

        self.get_logger().warning(
            f"Odometri Yedeği (Odom Fallback): "
            f"stage_{old_stage} -> stage_{self.stage_id}. Neden: {reason}"
        )
        self.publish_stage()

    def check_stage_progression(self):
        if self.odom_x is None or self.odom_y is None:
            return

        if self.stage_id >= self.final_stage:
            return

        interval = self.stage_interval_seconds()
        if interval is not None and self.stage_id > 0 and interval < self.min_stage_interval_seconds:
            return

        travel_dist = self.stage_travel_distance()
        if travel_dist is not None and self.stage_id > 0 and travel_dist < self.min_stage_travel_distance:
            return

        next_stage = self.stage_id + 1

        target_x, target_y = None, None
        source_desc = ""

        if next_stage in self.detected_signs:
            sign_info = self.detected_signs[next_stage]
            target_x = sign_info["x"]
            target_y = sign_info["y"]
            source_desc = f"Algılanan Tabela ({sign_info['label']})"
        elif next_stage in self.STAGE_POSITIONS:
            target_x, target_y = self.STAGE_POSITIONS[next_stage]
            source_desc = "Varsayılan Stage Konumu"

        if target_x is not None and target_y is not None:
            dist = math.hypot(self.odom_x - target_x, self.odom_y - target_y)
            threshold = self.odom_fallback_radius

            if dist <= threshold:
                old_stage = self.stage_id
                self.stage_id = min(self.final_stage, next_stage)
                self.record_stage_change()
                self.get_logger().info(
                    f"[STAGE GEÇİŞİ] Araç Stage {self.stage_id} konumuna ulaştı/geçti ({source_desc}): "
                    f"Stage {old_stage} -> {self.stage_id} (Konum: ({target_x:.2f}, {target_y:.2f}), Mesafe: {dist:.2f}m <= {threshold:.2f}m)"
                )
                self.publish_stage()

                if self.stage_id >= self.final_stage:
                    self.final_stop_active = True
                    self.publish_final_stop()

    def publish_stage(self):
        stage_msg = Int32()
        stage_msg.data = int(self.stage_id)
        self.stage_pub.publish(stage_msg)

        order_msg = Int32()
        order_msg.data = int(self.stage_id)
        self.stage_order_pub.publish(order_msg)

    def odom_callback(self, msg):
        self.odom_x = float(msg.pose.pose.position.x)
        self.odom_y = float(msg.pose.pose.position.y)
        self.odom_yaw = quaternion_to_yaw(msg.pose.pose.orientation)

        if self.last_stage_change_x is None:
            self.last_stage_change_x = self.odom_x
            self.last_stage_change_y = self.odom_y

        self.check_stage_progression()

    def publish_final_stop(self):
        msg = Bool()
        msg.data = True
        self.final_stop_pub.publish(msg)

    def estimate_candidate_position(self, candidate):
        if candidate is None:
            return None, None, None

        cx = candidate["cx"]
        cy = candidate["cy"]
        radius = candidate["radius"]

        if radius <= 0:
            return None, None, None

        focal_length = 554.0
        depth = (focal_length * self.real_sign_radius) / max(1.0, float(radius))
        x_cam = ((cx - 320.0) * depth) / focal_length
        y_cam = ((cy - 240.0) * depth) / focal_length

        x_rel = depth
        y_rel = -x_cam
        z_rel = 0.40 - y_cam

        if self.odom_x is not None and self.odom_y is not None:
            yaw = self.odom_yaw
            x_world = self.odom_x + x_rel * math.cos(yaw) - y_rel * math.sin(yaw)
            y_world = self.odom_y + x_rel * math.sin(yaw) + y_rel * math.cos(yaw)
            z_world = z_rel
            return x_world, y_world, z_world

        return None, None, None

    def process_sign_candidates(self, candidates):
        if not candidates:
            self.last_radius = 0
            self.last_center = None
            self.current_matched_label = "none"
            self.current_match_score = 0.0
            return

        best_candidate = candidates[0]
        self.last_radius = int(best_candidate["radius"])
        self.last_center = (int(best_candidate["cx"]), int(best_candidate["cy"]))

        top_matched = False

        for candidate in candidates:
            matched_key = candidate.get("matched_key", None)
            match_score = candidate.get("match_score", 0.0)
            tmpl_meta = candidate.get("tmpl_meta", None)

            x_world, y_world, _ = self.estimate_candidate_position(candidate)

            if not (matched_key and match_score >= self.template_match_threshold):
                continue

            # En yüksek skorluyu mevcut label olarak işaretle
            if not top_matched or match_score > self.current_match_score:
                self.current_matched_label = matched_key
                self.current_match_score = match_score
                top_matched = True

            if tmpl_meta:
                stage_num = tmpl_meta.get("stage_num", None)
                is_stop = tmpl_meta.get("is_stop", False) or (matched_key == "sign_stop")

                sign_key = stage_num if stage_num is not None else ("STOP" if is_stop else None)

                if sign_key is not None and x_world is not None and y_world is not None:
                    if sign_key not in self.detected_signs:
                        # İlk tespit → konumu kaydet + marker ekle
                        self.detected_signs[sign_key] = {
                            "x": x_world,
                            "y": y_world,
                            "label": matched_key,
                            "score": match_score,
                            "is_stop": is_stop,
                        }
                        tag_desc = f"Stage {stage_num}" if stage_num is not None else "STOP Tabelası"
                        self.get_logger().info(
                            f"[TABELA KONUMU KAYDEDİLDİ] {tag_desc} ({matched_key}) harita konumu kaydedildi: "
                            f"({x_world:.2f}, {y_world:.2f})."
                        )
                        self.publish_sign_marker(candidate)
                    else:
                        # Tekrar tespit → sadece konumu güncelle, yeni marker koyma
                        old = self.detected_signs[sign_key]
                        old["x"] = 0.70 * old["x"] + 0.30 * x_world
                        old["y"] = 0.70 * old["y"] + 0.30 * y_world
                        old["score"] = max(old["score"], match_score)

        if not top_matched:
            self.current_matched_label = "candidate"
            self.current_match_score = 0.0

    def publish_sign_marker(self, candidate):
        marker_array = MarkerArray()
        now_time = self.get_clock().now().to_msg()

        # Frame ID ve Odometri kontrolü
        use_odom = (self.odom_x is not None and self.odom_y is not None)
        frame_id = "odom" if use_odom else "base_link"

        if candidate is not None:
            cx = candidate["cx"]
            cy = candidate["cy"]
            radius = candidate["radius"]
            matched_key = candidate.get("matched_key", None)
            match_score = candidate.get("match_score", 0.0)

            if radius > 0:
                focal_length = 554.0
                depth = (focal_length * self.real_sign_radius) / max(1.0, float(radius))
                x_cam = ((cx - 320.0) * depth) / focal_length
                y_cam = ((cy - 240.0) * depth) / focal_length

                x_rel = depth
                y_rel = -x_cam
                z_rel = 0.40 - y_cam

                if use_odom:
                    yaw = self.odom_yaw
                    x_world = self.odom_x + x_rel * math.cos(yaw) - y_rel * math.sin(yaw)
                    y_world = self.odom_y + x_rel * math.sin(yaw) + y_rel * math.cos(yaw)
                    z_world = z_rel
                else:
                    x_world, y_world, z_world = x_rel, y_rel, z_rel

                is_matched = bool(matched_key and match_score >= self.template_match_threshold)
                if is_matched or self.show_unmatched_candidates:
                    label = matched_key if is_matched else "Detected Sign"

                    # Real-time active tracking text marker
                    track_text = Marker()
                    track_text.header.frame_id = frame_id
                    track_text.header.stamp = now_time
                    track_text.ns = "sign_realtime_text"
                    track_text.id = 9999
                    track_text.type = Marker.TEXT_VIEW_FACING
                    track_text.action = Marker.ADD
                    track_text.pose.position.x = x_world
                    track_text.pose.position.y = y_world
                    track_text.pose.position.z = z_world + 0.50
                    track_text.pose.orientation.x = 0.0
                    track_text.pose.orientation.y = 0.0
                    track_text.pose.orientation.z = 0.0
                    track_text.pose.orientation.w = 1.0
                    track_text.scale.z = 0.40
                    track_text.color.r = 0.0
                    track_text.color.g = 1.0
                    track_text.color.b = 0.3
                    track_text.color.a = 1.0
                    track_text.text = f"{label} ({match_score:.2f})" if matched_key else f"Candidate (r={radius}px)"
                    marker_array.markers.append(track_text)

                    # Real-time active tracking shape marker
                    track_shape = Marker()
                    track_shape.header.frame_id = frame_id
                    track_shape.header.stamp = now_time
                    track_shape.ns = "sign_realtime_shape"
                    track_shape.id = 9998
                    track_shape.type = Marker.CYLINDER
                    track_shape.action = Marker.ADD
                    track_shape.pose.position.x = x_world
                    track_shape.pose.position.y = y_world
                    track_shape.pose.position.z = z_world
                    track_shape.pose.orientation.x = 0.0
                    track_shape.pose.orientation.y = 0.0
                    track_shape.pose.orientation.z = 0.0
                    track_shape.pose.orientation.w = 1.0
                    track_shape.scale.x = 0.45
                    track_shape.scale.y = 0.45
                    track_shape.scale.z = 0.50
                    track_shape.color.r = 1.0
                    track_shape.color.g = 0.2
                    track_shape.color.b = 0.2
                    track_shape.color.a = 0.85
                    marker_array.markers.append(track_shape)

        # Permanent markers for all saved detected signs
        for stage_num, sdata in self.detected_signs.items():
            p_text = Marker()
            p_text.header.frame_id = frame_id
            p_text.header.stamp = now_time
            p_text.ns = "sign_map_text"
            p_text.id = int(stage_num) * 10
            p_text.type = Marker.TEXT_VIEW_FACING
            p_text.action = Marker.ADD
            p_text.pose.position.x = sdata["x"]
            p_text.pose.position.y = sdata["y"]
            p_text.pose.position.z = 0.80
            p_text.pose.orientation.x = 0.0
            p_text.pose.orientation.y = 0.0
            p_text.pose.orientation.z = 0.0
            p_text.pose.orientation.w = 1.0
            p_text.scale.z = 0.45
            p_text.color.r = 0.0
            p_text.color.g = 0.9
            p_text.color.b = 1.0
            p_text.color.a = 1.0
            p_text.text = f"{sdata['label']} (Stage {stage_num})"
            marker_array.markers.append(p_text)

            p_shape = Marker()
            p_shape.header.frame_id = frame_id
            p_shape.header.stamp = now_time
            p_shape.ns = "sign_map_shape"
            p_shape.id = int(stage_num) * 10 + 1
            p_shape.type = Marker.CYLINDER
            p_shape.action = Marker.ADD
            p_shape.pose.position.x = sdata["x"]
            p_shape.pose.position.y = sdata["y"]
            p_shape.pose.position.z = 0.30
            p_shape.pose.orientation.x = 0.0
            p_shape.pose.orientation.y = 0.0
            p_shape.pose.orientation.z = 0.0
            p_shape.pose.orientation.w = 1.0
            p_shape.scale.x = 0.50
            p_shape.scale.y = 0.50
            p_shape.scale.z = 0.60
            p_shape.color.r = 0.0
            p_shape.color.g = 0.6
            p_shape.color.b = 1.0
            p_shape.color.a = 0.80
            marker_array.markers.append(p_shape)

        if marker_array.markers:
            self.marker_pub.publish(marker_array)

    def publish_state(self):
        label_msg = String()
        label_msg.data = str(self.current_matched_label)
        self.label_pub.publish(label_msg)

        confidence_msg = Float32()
        confidence_msg.data = float(self.current_match_score)
        self.confidence_pub.publish(confidence_msg)

    def draw_debug(
        self,
        frame,
        candidates,
    ):
        debug = frame.copy()

        if candidates:
            for candidate in candidates:
                match_score = candidate.get("match_score", 0.0)
                matched_label = candidate.get("matched_key", "none")
                is_matched = match_score >= self.template_match_threshold

                if is_matched or self.show_unmatched_candidates:
                    x1, y1, x2, y2 = candidate["box"]
                    text_color = (0, 255, 0) if is_matched else (0, 255, 255)

                    cv2.rectangle(
                        debug,
                        (x1, y1),
                        (x2, y2),
                        text_color,
                        2,
                    )
                    cv2.circle(
                        debug,
                        (
                            candidate["cx"],
                            candidate["cy"],
                        ),
                        candidate["radius"],
                        text_color,
                        2,
                    )

                    cv2.putText(
                        debug,
                        f"{matched_label} ({match_score:.2f})",
                        (
                            x1,
                            max(22, y1 - 22),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        text_color,
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        debug,
                        (
                            f"r={candidate['radius']} "
                            f"white={candidate['white_ratio']:.2f} "
                            f"red={candidate['red_ratio']:.2f}"
                        ),
                        (
                            x1,
                            max(38, y1 - 6),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        text_color,
                        1,
                        cv2.LINE_AA,
                    )

        matched_key = self.current_matched_label
        match_score = self.current_match_score

        status = (
            f"stage={self.stage_id} "
            f"candidates={len(candidates) if candidates else 0} "
            f"match={matched_key} ({match_score:.2f}) "
            f"r={self.last_radius} "
            f"proc={self.last_process_seconds:.3f}s"
        )

        cv2.putText(
            debug,
            status,
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return debug

    def image_callback(self, msg):
        started_at = time.perf_counter()

        if not self.first_callback_seen:
            self.first_callback_seen = True
            self.get_logger().info(
                "İlk kamera callback'i alındı."
            )

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
        except Exception as exc:
            self.get_logger().error(
                "Kamera görüntüsü bgr8'e "
                f"çevrilemedi: {exc}"
            )
            return

        self.frame_count += 1

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV,
        )
        red_mask = self.make_red_mask(
            hsv
        )

        self.last_red_pixels = int(
            np.count_nonzero(red_mask)
        )

        candidates = self.find_candidates(
            frame,
            hsv,
            red_mask,
        )

        self.process_sign_candidates(candidates)
        self.publish_state()

        self.last_process_seconds = (
            time.perf_counter()
            - started_at
        )

        debug = self.draw_debug(
            frame,
            candidates,
        )

        try:
            debug_msg = (
                self.bridge.cv2_to_imgmsg(
                    debug,
                    encoding="bgr8",
                )
            )
            debug_msg.header = msg.header
            self.debug_pub.publish(
                debug_msg
            )
        except Exception as exc:
            self.get_logger().error(
                "Debug görüntüsü yayınlanamadı: "
                f"{exc}"
            )

        if (
            self.frame_count
            % self.diagnostic_every_frames
            == 0
        ):
            is_matched = (
                self.current_matched_label not in ("none", "candidate")
                and self.current_match_score >= self.template_match_threshold
            )

            if is_matched or self.show_unmatched_candidates:
                center_text = (
                    "none"
                    if self.last_center is None
                    else (
                        f"{self.last_center[0]},"
                        f"{self.last_center[1]}"
                    )
                )

                self.get_logger().info(
                    "[VISION] "
                    f"frame={self.frame_count} "
                    f"red_px={self.last_red_pixels} "
                    f"candidates={self.last_candidate_count} "
                    f"center={center_text} "
                    f"r={self.last_radius} "
                    f"match={self.current_matched_label} ({self.current_match_score:.2f}) "
                    f"stage={self.stage_id} "
                    f"odom=({self.odom_x if self.odom_x is not None else 'none'},"
                    f"{self.odom_y if self.odom_y is not None else 'none'}) "
                    f"proc={self.last_process_seconds:.3f}s"
                )


def main(args=None):
    rclpy.init(args=args)
    node = SignDetector()

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
