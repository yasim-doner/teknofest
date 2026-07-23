#!/usr/bin/env python3

import math
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
        self.declare_parameter("final_stage", 10)
        self.declare_parameter("stop_after_final_stage", True)

        # Görüntünün alt kısmındaki koniler stage levhası değildir.
        self.declare_parameter("max_stage_candidate_y_ratio", 0.78)
        self.declare_parameter("upper_enter_radius", 10)

        # Simülasyon yedeği: bir levha kaçarsa stage sırası kaymasın.
        self.declare_parameter("use_odom_fallback", True)
        self.declare_parameter("stage5_to6_x", 12.5)
        self.declare_parameter("stage5_lane_y", 10.0)
        self.declare_parameter("stage5_lane_tolerance", 3.0)
        self.declare_parameter("stage6_to7_min_x", 15.0)
        self.declare_parameter("stage6_to7_y", 17.0)

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

        self.use_odom_fallback = bool(
            self.get_parameter(
                "use_odom_fallback"
            ).value
        )
        self.stage5_to6_x = float(
            self.get_parameter(
                "stage5_to6_x"
            ).value
        )
        self.stage5_lane_y = float(
            self.get_parameter(
                "stage5_lane_y"
            ).value
        )
        self.stage5_lane_tolerance = max(
            0.1,
            float(
                self.get_parameter(
                    "stage5_lane_tolerance"
                ).value
            ),
        )
        self.stage6_to7_min_x = float(
            self.get_parameter(
                "stage6_to7_min_x"
            ).value
        )
        self.stage6_to7_y = float(
            self.get_parameter(
                "stage6_to7_y"
            ).value
        )

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
        self.stop_pub = self.create_publisher(
            Bool,
            "/teknofest/stop_detected",
            10,
        )

        self.final_stop_pub = self.create_publisher(
            Bool,
            "/teknofest/final_stop",
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
        self.debug_pub = self.create_publisher(
            Image,
            "/teknofest/sign_debug_image",
            self.debug_qos,
        )

        self.frame_count = 0
        self.stage_id = self.initial_stage

        # Normal başlangıçta ilk levhaya hazırız.
        # stage:=8 doğrudan testinde mevcut stage tekrar artırılmasın.
        self.encounter_active = self.initial_stage > 0
        self.enter_count = 0
        self.exit_count = 0

        self.stop_active = False
        self.stop_completed = False

        # Stage 10 tabelası geride kaldığında kalıcı olarak True olur.
        self.final_stop_active = False
        self.stage8_started_at = (
            self.now_seconds()
            if self.stage_id == 8
            else None
        )

        self.last_radius = 0
        self.last_center = None
        self.last_candidate_count = 0
        self.last_red_pixels = 0
        self.last_process_seconds = 0.0

        # İlk kamera callback'inin gelip gelmediğini açıkça gösterir.
        self.first_callback_seen = False

        self.odom_x = None
        self.odom_y = None
        self.last_odom_fallback_stage = None

        self.last_stage_change_x = None
        self.last_stage_change_y = None
        self.last_stage_change_time = None

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
            f"odom_yedek={self.use_odom_fallback}, "
            f"min_stage_mesafe={self.min_stage_travel_distance:.1f}m, "
            f"min_stage_süre={self.min_stage_interval_seconds:.1f}s, "
            f"stage_geofence={self.stage_geofence_radius:.1f}m, "
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

    @staticmethod
    def make_red_mask(hsv):
        # Gazebo materyallerindeki kırmızı tonlar için iki HSV aralığı.
        lower_red_1 = np.array(
            [0, 55, 45],
            dtype=np.uint8,
        )
        upper_red_1 = np.array(
            [18, 255, 255],
            dtype=np.uint8,
        )

        lower_red_2 = np.array(
            [160, 55, 45],
            dtype=np.uint8,
        )
        upper_red_2 = np.array(
            [179, 255, 255],
            dtype=np.uint8,
        )

        mask_1 = cv2.inRange(
            hsv,
            lower_red_1,
            upper_red_1,
        )
        mask_2 = cv2.inRange(
            hsv,
            lower_red_2,
            upper_red_2,
        )

        red_mask = cv2.bitwise_or(
            mask_1,
            mask_2,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )

        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )
        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        return red_mask

    def find_best_candidate(
        self,
        frame,
        hsv,
        red_mask,
    ):
        height, width = frame.shape[:2]
        frame_area = height * width
        max_area = (
            frame_area
            * self.max_component_area_ratio
        )

        count, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                red_mask,
                connectivity=8,
            )
        )

        candidates = []

        # 0 arka plan bileşenidir.
        for component_id in range(1, count):
            x = int(
                stats[
                    component_id,
                    cv2.CC_STAT_LEFT,
                ]
            )
            y = int(
                stats[
                    component_id,
                    cv2.CC_STAT_TOP,
                ]
            )
            box_width = int(
                stats[
                    component_id,
                    cv2.CC_STAT_WIDTH,
                ]
            )
            box_height = int(
                stats[
                    component_id,
                    cv2.CC_STAT_HEIGHT,
                ]
            )
            area = int(
                stats[
                    component_id,
                    cv2.CC_STAT_AREA,
                ]
            )

            if area < self.min_component_area:
                continue

            if area > max_area:
                continue

            if box_width < self.min_candidate_size:
                continue

            if box_height < self.min_candidate_size:
                continue

            if box_width > self.max_candidate_size:
                continue

            if box_height > self.max_candidate_size:
                continue

            aspect = (
                box_width
                / float(box_height)
            )

            if not (
                self.min_aspect
                <= aspect
                <= self.max_aspect
            ):
                continue

            cx = int(
                round(
                    centroids[
                        component_id,
                        0,
                    ]
                )
            )
            cy = int(
                round(
                    centroids[
                        component_id,
                        1,
                    ]
                )
            )

            # Yol ve koniler çoğunlukla görüntünün alt bölümündedir.
            if cy > int(height * 0.88):
                continue

            radius = int(
                math.ceil(
                    0.5
                    * max(
                        box_width,
                        box_height,
                    )
                )
            )

            expand = max(
                2,
                int(radius * 0.30),
            )

            x1 = max(0, x - expand)
            y1 = max(0, y - expand)
            x2 = min(
                width,
                x + box_width + expand,
            )
            y2 = min(
                height,
                y + box_height + expand,
            )

            if x2 <= x1 or y2 <= y1:
                continue

            hsv_roi = hsv[y1:y2, x1:x2]
            red_roi = red_mask[y1:y2, x1:x2]

            white_mask = (
                (hsv_roi[:, :, 1] < 125)
                & (hsv_roi[:, :, 2] > 90)
            )

            white_ratio = float(
                np.count_nonzero(white_mask)
                / white_mask.size
            )

            red_fill_ratio = float(
                np.count_nonzero(red_roi)
                / red_roi.size
            )

            if white_ratio < self.min_white_ratio:
                continue

            if (
                red_fill_ratio
                < self.min_red_fill_ratio
            ):
                continue

            # Yakın/büyük levha öncelikli, renk oranları destekleyicidir.
            score = (
                radius
                + 20.0 * white_ratio
                + 10.0 * red_fill_ratio
            )

            candidates.append(
                {
                    "cx": cx,
                    "cy": cy,
                    "radius": radius,
                    "box": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    "white_ratio": white_ratio,
                    "red_ratio": red_fill_ratio,
                    "score": score,
                    "frame_height": height,
                }
            )

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        self.last_candidate_count = len(
            candidates
        )

        if not candidates:
            return None

        upper_candidates = [
            item
            for item in candidates
            if item["cy"]
            <= int(
                height
                * self.max_stage_candidate_y_ratio
            )
        ]

        # Stage levhası için üst bölgedeki adayı tercih et.
        # Alt bölgedeki kırmızı koniler debug görüntüsünde kalır,
        # fakat stage sayacını ilerletmez.
        if upper_candidates:
            return upper_candidates[0]

        return candidates[0]

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

    def stage_change_allowed(self):
        next_stage_id = min(
            self.stage_id + 1,
            11,
        )

        distance = self.stage_travel_distance()
        interval = self.stage_interval_seconds()
        target_distance = (
            self.distance_to_expected_stage(
                next_stage_id
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
                f"stage_{next_stage_id}_uzaklık="
                f"{target_text}"
            )

        return False, (
            f"son_stage_mesafe="
            f"{distance if distance is not None else -1.0:.2f}m/"
            f"{self.min_stage_travel_distance:.2f}m, "
            f"süre="
            f"{interval if interval is not None else -1.0:.2f}s/"
            f"{self.min_stage_interval_seconds:.2f}s, "
            f"stage_{next_stage_id}_uzaklık="
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

    def update_stage(self, candidate):
        if self.final_stop_active:
            self.last_radius = 0
            self.last_center = None
            return

        if candidate is None:
            radius = 0
        else:
            radius = int(
                candidate["radius"]
            )

        self.last_radius = radius
        self.last_center = (
            None
            if candidate is None
            else (
                int(candidate["cx"]),
                int(candidate["cy"]),
            )
        )

        if candidate is None:
            candidate_is_upper = False
        else:
            candidate_is_upper = (
                candidate["cy"]
                <= int(
                    candidate["frame_height"]
                    * self.max_stage_candidate_y_ratio
                )
            )

        effective_enter_radius = (
            self.upper_enter_radius
            if candidate_is_upper
            else self.enter_radius
        )

        close_sign = (
            candidate is not None
            and candidate_is_upper
            and radius >= effective_enter_radius
        )

        sign_gone = (
            candidate is None
            or not candidate_is_upper
            or radius <= self.exit_radius
        )

        # Stage 8'den sonra bir sonraki yakın levha STOP olarak kullanılır.
        stop_guard_passed = (
            self.stage_id == 8
            and self.stage8_started_at is not None
            and (
                self.now_seconds()
                - self.stage8_started_at
            )
            >= self.stop_guard_seconds
        )

        if (
            self.stage_id == 8
            and not self.stop_completed
            and not self.encounter_active
            and stop_guard_passed
        ):
            if close_sign:
                self.enter_count += 1
            else:
                self.enter_count = 0

            if self.enter_count >= self.enter_frames:
                self.enter_count = 0
                self.stop_active = True
                self.encounter_active = True

                self.get_logger().warning(
                    f"STOP levhası algılandı "
                    f"(r={radius}px)."
                )

            return

        if self.stop_active:
            if sign_gone:
                self.exit_count += 1
            else:
                self.exit_count = 0

            if self.exit_count >= self.exit_frames:
                self.exit_count = 0
                self.stop_active = False
                self.stop_completed = True
                self.encounter_active = False

                self.get_logger().info(
                    "STOP levhası geride kaldı."
                )

            return

        if not self.encounter_active:
            if close_sign:
                self.enter_count += 1
            else:
                self.enter_count = 0

            if self.enter_count < self.enter_frames:
                return

            allowed, gate_reason = (
                self.stage_change_allowed()
            )

            if not allowed:
                # Aynı yanlış nesne her iki karede bir tekrar denemesin.
                self.enter_count = 0

                self.get_logger().warning(
                    "Yeni tabela adayı reddedildi: "
                    f"stage_{self.stage_id} korunuyor, "
                    f"r={radius}px, {gate_reason}."
                )
                return

            self.enter_count = 0
            self.encounter_active = True

            if self.stage_id < self.final_stage:
                old_stage = self.stage_id
                self.stage_id += 1
                self.record_stage_change()

                if self.stage_id == 8:
                    self.stage8_started_at = (
                        self.now_seconds()
                    )
                    self.stop_completed = False

                self.get_logger().info(
                    f"Yeni fiziksel levha: "
                    f"stage_{old_stage} -> "
                    f"stage_{self.stage_id} "
                    f"(r={radius}px, {gate_reason})."
                )

            return

        # Mevcut levhanın görüntüden çıkmasını bekle.
        if sign_gone:
            self.exit_count += 1
        else:
            self.exit_count = 0

        if self.exit_count >= self.exit_frames:
            self.exit_count = 0

            if (
                self.stop_after_final_stage
                and self.stage_id == self.final_stage
            ):
                self.encounter_active = True
                self.final_stop_active = True

                self.get_logger().warning(
                    f"FINAL: stage_{self.final_stage} levhası "
                    "geride kaldı. Parkur tamamlandı; "
                    "kalıcı duruş komutu gönderiliyor."
                )

                self.publish_state()
                return

            self.encounter_active = False

            self.get_logger().info(
                f"stage_{self.stage_id} levhası "
                "geride kaldı; sonraki levha için hazır."
            )

    def force_stage(
        self,
        new_stage,
        reason,
    ):
        new_stage = int(new_stage)

        if new_stage <= self.stage_id:
            return

        old_stage = self.stage_id
        self.stage_id = min(
            self.final_stage,
            new_stage,
        )
        self.record_stage_change()

        self.enter_count = 0
        self.exit_count = 0

        # Geofence geçişinden sonra görünür durumdaki levhanın aynı anda
        # bir stage daha artırmasını engelle.
        self.encounter_active = True

        if self.stage_id == 8:
            self.stage8_started_at = (
                self.now_seconds()
            )
            self.stop_completed = False

        self.get_logger().warning(
            f"Odometri yedeği: "
            f"stage_{old_stage} -> "
            f"stage_{self.stage_id}. "
            f"Neden: {reason}"
        )

        self.publish_state()

    def odom_callback(self, msg):
        self.odom_x = float(
            msg.pose.pose.position.x
        )
        self.odom_y = float(
            msg.pose.pose.position.y
        )

        if not self.use_odom_fallback:
            return

        # Stage 5 parkuru yaklaşık y=10 doğrultusunda +x yönündedir.
        # Stage 6 levhası kaçsa bile kayan engelden önce Stage 6 açılır.
        if (
            self.stage_id == 5
            and self.odom_x
            >= self.stage5_to6_x
            and abs(
                self.odom_y
                - self.stage5_lane_y
            )
            <= self.stage5_lane_tolerance
        ):
            self.force_stage(
                6,
                (
                    f"x={self.odom_x:.2f}, "
                    f"y={self.odom_y:.2f}"
                ),
            )
            return

        # Stage 6'dan sonra parkur kuzeye dönüp Stage 7 alanına çıkar.
        # Stage 7 levhası gelmeden önce sıra Stage 6'da kalır; Stage 7
        # bölgesine ulaşınca kesin olarak Stage 7'ye geçilir.
        if (
            self.stage_id == 6
            and self.odom_x
            >= self.stage6_to7_min_x
            and self.odom_y
            >= self.stage6_to7_y
        ):
            self.force_stage(
                7,
                (
                    f"x={self.odom_x:.2f}, "
                    f"y={self.odom_y:.2f}"
                ),
            )

    def publish_state(self):
        stage_msg = Int32()
        stage_msg.data = int(
            self.stage_id
        )
        self.stage_pub.publish(stage_msg)

        order_msg = Int32()
        order_msg.data = int(
            self.stage_id
        )
        self.stage_order_pub.publish(
            order_msg
        )

        stop_msg = Bool()
        stop_msg.data = bool(
            self.stop_active
        )
        self.stop_pub.publish(stop_msg)

        final_stop_msg = Bool()
        final_stop_msg.data = bool(
            self.final_stop_active
        )
        self.final_stop_pub.publish(
            final_stop_msg
        )

        label_msg = String()

        if self.final_stop_active:
            label_msg.data = "final_stop"
        elif self.stop_active:
            label_msg.data = "stop"
        elif self.stage_id > 0:
            label_msg.data = (
                f"stage_{self.stage_id}"
            )
        else:
            label_msg.data = "none"

        self.label_pub.publish(label_msg)

        confidence_msg = Float32()

        if self.last_radius <= 0:
            confidence_msg.data = 0.0
        else:
            confidence_msg.data = float(
                min(
                    1.0,
                    self.last_radius
                    / max(
                        1.0,
                        float(
                            self.enter_radius
                        ),
                    ),
                )
            )

        self.confidence_pub.publish(
            confidence_msg
        )

    def draw_debug(
        self,
        frame,
        candidate,
    ):
        debug = frame.copy()

        if candidate is not None:
            x1, y1, x2, y2 = (
                candidate["box"]
            )

            cv2.rectangle(
                debug,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2,
            )
            cv2.circle(
                debug,
                (
                    candidate["cx"],
                    candidate["cy"],
                ),
                candidate["radius"],
                (0, 255, 255),
                2,
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
                    max(
                        22,
                        y1 - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        status = (
            f"stage={self.stage_id} "
            f"stop={int(self.stop_active)} "
            f"final={int(self.final_stop_active)} "
            f"candidate={int(candidate is not None)} "
            f"r={self.last_radius} "
            f"armed={int(not self.encounter_active)} "
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

        candidate = self.find_best_candidate(
            frame,
            hsv,
            red_mask,
        )

        self.update_stage(candidate)
        self.publish_state()

        self.last_process_seconds = (
            time.perf_counter()
            - started_at
        )

        debug = self.draw_debug(
            frame,
            candidate,
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
            center_text = (
                "none"
                if self.last_center is None
                else (
                    f"{self.last_center[0]},"
                    f"{self.last_center[1]}"
                )
            )

            if self.stage_id >= self.final_stage:
                next_stage_id = self.final_stage
                next_distance_text = "FINAL"
            else:
                next_stage_id = min(
                    self.stage_id + 1,
                    self.final_stage,
                )
                next_stage_distance = (
                    self.distance_to_expected_stage(
                        next_stage_id
                    )
                )
                next_distance_text = (
                    "none"
                    if next_stage_distance is None
                    else f"{next_stage_distance:.2f}"
                )

            self.get_logger().info(
                "[VISION] "
                f"frame={self.frame_count} "
                f"red_px={self.last_red_pixels} "
                f"candidates={self.last_candidate_count} "
                f"center={center_text} "
                f"r={self.last_radius} "
                f"stage={self.stage_id} "
                f"next_stage={next_stage_id} "
                f"next_dist={next_distance_text} "
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
