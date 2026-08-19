#!/usr/bin/env python3

import sys
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
    Hem ROS 2 kamera topic'leri hem de bilgisayarın dahili/harici web kamerası ile çalışabilir.
    Bölgesel Arama (ROI Tracking) ve Hız Tahmini (Motion Prediction) ile hareketli hedeflerde yüksek başarım sağlar.
    """

    def __init__(self, use_webcam_cli=False, device_id_cli=0, show_window_cli=False):
        super().__init__("target_detect")

        # Parametreler
        self.declare_parameter("image_topic", "/rover/camera/image_raw")
        self.declare_parameter("fx", 525.0)
        self.declare_parameter("fy", 525.0)
        self.declare_parameter("active_stage", 9)
        self.declare_parameter("laser_duration", 1.2)
        self.declare_parameter("use_webcam", use_webcam_cli)
        self.declare_parameter("device_id", device_id_cli)
        self.declare_parameter("show_window", show_window_cli or use_webcam_cli)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.fx = float(self.get_parameter("fx").value)
        self.fy = float(self.get_parameter("fy").value)
        self.active_stage = int(self.get_parameter("active_stage").value)
        self.laser_duration = float(self.get_parameter("laser_duration").value)
        self.use_webcam = bool(self.get_parameter("use_webcam").value)
        self.device_id = int(self.get_parameter("device_id").value)
        self.show_window = bool(self.get_parameter("show_window").value)

        self.bridge = CvBridge()
        self.cap = None
        self.timer = None

        # State & Data Variables
        self.current_stage = 0
        self.laser_active = False
        self.parking_brake_active = False
        self.result_saved = False
        self.shot_started = False
        self.shot_start_time = None
        self.shot_completed = False

        # ROI & Motion Tracking State
        self.tracking = False
        self.last_target_center = None  # (u, v)
        self.last_target_radius = None  # float
        self.velocity = [0.0, 0.0]      # [vx, vy] in px/frame
        self.tracking_confidence = 0
        self.roi_missed_count = 0
        self.max_roi_missed_frames = 3

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
        self.laser_pub = self.create_publisher(
            Bool,
            "/teknofest/laser_on",
            10,
        )
        self.release_pub = self.create_publisher(
            Int32,
            "/teknofest/release",
            10,
        )

        # Subscriptions
        self.stage_sub = self.create_subscription(
            Int32,
            "/teknofest/stage_id",
            self.stage_callback,
            10,
        )
        self.parking_brake_sub = self.create_subscription(
            Bool,
            "/rover/parking_brake",
            self.parking_brake_callback,
            10,
        )

        if self.use_webcam:
            self.get_logger().info(f"Kamera Modu: Bilgisayar Kamerası (Device Index={self.device_id})")
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                self.get_logger().error(f"HATA: Cihaz indeks {self.device_id} olan bilgisayar kamerası açılamadı!")
            else:
                self.get_logger().info("Bilgisayar kamerası başarıyla açıldı. Görüntü işleme başlatılıyor...")
                # 30 FPS timer
                self.timer = self.create_timer(1.0 / 30.0, self.webcam_timer_callback)
        else:
            self.get_logger().info(f"Kamera Modu: ROS Topic ({self.image_topic})")
            self.image_sub = self.create_subscription(
                Image,
                self.image_topic,
                self.image_callback,
                self.camera_qos,
            )

        self.get_logger().info(
            f"=== Target Detect Node Başlatıldı (fx={self.fx}, fy={self.fy}, show_window={self.show_window}) ==="
        )

    def stage_callback(self, msg):
        self.current_stage = int(msg.data)

    def parking_brake_callback(self, msg):
        self.parking_brake_active = bool(msg.data)

    def is_target_detection_active(self):
        """
        Hedef tespiti sadece araç rampada tepedeyken ve park freni aktif durumdayken (Stage 9 + Park Freni) çalışır.
        """
        return self.current_stage in (9, self.active_stage) and self.parking_brake_active

    def publish_laser(self, enable: bool):
        msg = Bool()
        msg.data = bool(enable)
        self.laser_active = bool(enable)
        self.laser_pub.publish(msg)

    def publish_release(self, stage_num: int):
        msg = Int32()
        msg.data = int(stage_num)
        self.release_pub.publish(msg)
        self.get_logger().info(f"Stage {stage_num} serbest bırakıldı (/teknofest/release={stage_num}).")

    def save_target_result(self, debug_img, yaw_deg, pitch_deg, du, dv, score):
        try:
            import json, os
            out_dir = "/home/myazou/rover_ws/src/teknofest"
            img_path = os.path.join(out_dir, "target_result.jpg")
            json_path = os.path.join(out_dir, "target_result.json")

            cv2.imwrite(img_path, debug_img)

            res_data = {
                "timestamp": float(self.get_clock().now().nanoseconds / 1e9),
                "yaw_deg": float(yaw_deg),
                "pitch_deg": float(pitch_deg),
                "du_px": float(du),
                "dv_px": float(dv),
                "score": float(score),
                "image_saved": img_path,
            }
            with open(json_path, "w") as f:
                json.dump(res_data, f, indent=2)

            self.get_logger().warning(
                f"[HEDEF TESPİT EDİLDİ & KAYDEDİLDİ] Yaw: {yaw_deg:.2f}°, Pitch: {pitch_deg:.2f}°. "
                f"Görsel '{img_path}', veriler '{json_path}' dosyasına kaydedildi."
            )
        except Exception as e:
            self.get_logger().error(f"Hedef sonucu kaydetme hatası: {e}")

    def detect_target_in_image(self, frame, is_roi=False):
        """
        Siyah-beyaz hedef tahtasını uzaktan/yakından ortak merkezli (konsantrik)
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
            return False, 0, 0, 0, None, 0.0

        candidates = []

        max_area_ratio = 0.85 if is_roi else 0.25
        max_radius = min(w, h) * 0.45 if is_roi else 150.0

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 10 or area > (w * h * max_area_ratio):
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < 3 or radius > max_radius:
                continue

            # Dairesellik kontrolü (4 * pi * area / perimeter^2)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            # Bounding box en-boy oranı (Aspect ratio)
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = float(bw) / float(bh) if bh > 0 else 0.0

            if circularity > 0.50 and 0.60 <= aspect <= 1.40:
                candidates.append({
                    'idx': i,
                    'center': (cx, cy),
                    'radius': radius,
                    'area': area,
                    'cnt': cnt,
                    'circ': circularity
                })

        # Yakın merkezli (ortak merkezli / konsantrik) çember gruplarını bul ve 1:2:3 oran filtrelemesi uygula
        evaluated_groups = []
        
        for i, c1 in enumerate(candidates):
            raw_group = [c1]
            for j, c2 in enumerate(candidates):
                if i == j:
                    continue
                dist = np.hypot(c1['center'][0] - c2['center'][0], c1['center'][1] - c2['center'][1])
                if dist < max(4.0, c1['radius'] * 0.30):
                    raw_group.append(c2)

            if len(raw_group) < 2:
                continue

            # Benzer yarıçapa sahip çemberleri (iç/dış kenar çiftleri) ayıkla/birleştir
            raw_group.sort(key=lambda c: c['radius'])
            unique_rings = []
            for c in raw_group:
                if not unique_rings:
                    unique_rings.append(c)
                else:
                    prev_r = unique_rings[-1]['radius']
                    # Yarıçap farkı en az %15 olmalı ki farklı bir halka sayılsın
                    if (c['radius'] - prev_r) / prev_r > 0.15:
                        unique_rings.append(c)

            k = len(unique_rings)
            if k < 2:
                continue

            # Hedef halkaları yarıçap oranları: 3 cm, 6 cm, 9 cm (Oranlar: 1 : 2 : 3)
            radii = [c['radius'] for c in unique_rings]
            avg_circ = float(np.mean([c['circ'] for c in unique_rings]))
            score = -1.0
            best_sub_cnts = unique_rings

            if k >= 3:
                # 3 halka tespiti: r1/r3 ~ 1/3 (0.333) ve r2/r3 ~ 2/3 (0.667)
                r1, r2, r3 = radii[0], radii[1], radii[-1]
                ratio1 = r1 / r3
                ratio2 = r2 / r3
                err = abs(ratio1 - (3.0 / 9.0)) + abs(ratio2 - (6.0 / 9.0))

                # Perspektif ve açı sapması toleransı (maks %25 sapma)
                if err < 0.25:
                    score = (100.0 - (err * 100.0)) * avg_circ
            elif k == 2:
                # 2 halka esneklik takibi (r1/r2 için beklenen oranlar: 3/9=0.333, 6/9=0.667, 3/6=0.500)
                r1, r2 = radii[0], radii[1]
                ratio = r1 / r2
                target_ratios = [3.0 / 9.0, 6.0 / 9.0, 3.0 / 6.0]
                min_err = min(abs(ratio - tr) for tr in target_ratios)

                if min_err < 0.18:
                    score = (50.0 - (min_err * 100.0)) * avg_circ

            if score >= 35.0:
                evaluated_groups.append({
                    'score': score,
                    'rings': best_sub_cnts,
                    'num_rings': k
                })

        # En yüksek oran uyum skoruna sahip grubu seç
        evaluated_groups.sort(key=lambda g: g['score'], reverse=True)

        if len(evaluated_groups) > 0:
            best_group = evaluated_groups[0]['rings']
            best_score = evaluated_groups[0]['score']
            avg_cx = int(round(np.mean([c['center'][0] for c in best_group])))
            avg_cy = int(round(np.mean([c['center'][1] for c in best_group])))
            max_r = int(round(np.max([c['radius'] for c in best_group])))
            best_cnt = best_group[0]['cnt']

            return True, avg_cx, avg_cy, max_r, best_cnt, best_score

        return False, 0, 0, 0, None, 0.0

    def webcam_timer_callback(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.get_logger().warning("Bilgisayar kamerasından kare okunamadı!")
                return
            self.process_frame(frame)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Görüntü dönüştürme hatası: {e}")
            return

        self.process_frame(frame)

    def process_frame(self, frame):
        if not self.is_target_detection_active():
            if self.shot_started and not self.shot_completed:
                self.publish_laser(False)
            self.tracking = False
            self.result_saved = False
            self.shot_started = False
            self.shot_start_time = None
            self.shot_completed = False
            return

        h, w = frame.shape[:2]
        debug_img = frame.copy()
        center_u, center_v = w // 2, h // 2

        found = False
        u, v, radius = 0, 0, 0
        target_score = 0.0
        roi_box = None  # (x1, y1, x2, y2)
        search_mode = "GLOBAL"

        # 1. ROI (Bölgesel Arama) ile Takip Modu
        if self.tracking and self.last_target_center is not None:
            last_u, last_v = self.last_target_center
            last_r = self.last_target_radius or 30.0

            # Tahmini yeni merkez (Hız kompanzasyonu ile)
            pred_u = last_u + self.velocity[0]
            pred_v = last_v + self.velocity[1]

            margin_x = max(int(last_r * 3.5), 90)
            margin_y = max(int(last_r * 3.5), 90)

            x1 = max(0, int(pred_u - margin_x))
            y1 = max(0, int(pred_v - margin_y))
            x2 = min(w, int(pred_u + margin_x))
            y2 = min(h, int(pred_v + margin_y))

            roi_box = (x1, y1, x2, y2)

            if (x2 - x1) > 20 and (y2 - y1) > 20:
                roi_frame = frame[y1:y2, x1:x2]
                found_roi, local_u, local_v, r_roi, _, score_roi = self.detect_target_in_image(roi_frame, is_roi=True)

                if found_roi:
                    found = True
                    u = x1 + local_u
                    v = y1 + local_v
                    radius = r_roi
                    target_score = score_roi
                    search_mode = "ROI_TRACKING"

                    # Hız tahmini güncelleme (Exponential Moving Average)
                    alpha = 0.4
                    new_vx = u - last_u
                    new_vy = v - last_v
                    self.velocity[0] = alpha * new_vx + (1.0 - alpha) * self.velocity[0]
                    self.velocity[1] = alpha * new_vy + (1.0 - alpha) * self.velocity[1]

                    self.last_target_center = (u, v)
                    self.last_target_radius = radius
                    self.tracking_confidence += 1
                    self.roi_missed_count = 0
                else:
                    self.roi_missed_count += 1
                    if self.roi_missed_count >= self.max_roi_missed_frames:
                        # ROI'de bulamazsa takibi sıfırla, Global aramaya geç
                        self.tracking = False
                        self.last_target_center = None
                        self.velocity = [0.0, 0.0]

        # 2. Global (Tüm Kare) Arama Modu (Takipte değilse veya ROI başarısız olduysa)
        if not found:
            found_g, u_g, v_g, r_g, _, score_g = self.detect_target_in_image(frame, is_roi=False)
            if found_g:
                found = True
                u, v, radius = u_g, v_g, r_g
                target_score = score_g
                search_mode = "GLOBAL_SEARCH"

                self.tracking = True
                self.last_target_center = (u, v)
                self.last_target_radius = radius
                self.velocity = [0.0, 0.0]
                self.tracking_confidence = 1
                self.roi_missed_count = 0
            else:
                self.tracking = False
                self.tracking_confidence = 0

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

            # ROI kutusunu görselleştir (Takip modundaysa)
            if roi_box is not None and search_mode == "ROI_TRACKING":
                rx1, ry1, rx2, ry2 = roi_box
                cv2.rectangle(debug_img, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)
                cv2.putText(
                    debug_img,
                    f"ROI Track (Conf: {self.tracking_confidence})",
                    (rx1, max(15, ry1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                )

            info_str = f"Yaw: {yaw_deg:.2f} deg, Pitch: {pitch_deg:.2f} deg"
            pix_str = f"du: {du:.1f}px, dv: {dv:.1f}px [{search_mode}] Score: {target_score:.1f}"

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

            # Lazer Atış Mantığı (Sadece target_detect kontrol eder)
            if not self.shot_started and not self.shot_completed:
                self.shot_started = True
                self.shot_start_time = self.get_clock().now()
                self.publish_laser(True)
                self.get_logger().warning("[TARGET DETECT] Hedef kilitlendi! Lazer açılıyor (/teknofest/laser_on = True).")
                if not self.result_saved:
                    self.save_target_result(debug_img, yaw_deg, pitch_deg, du, dv, target_score)
                    self.result_saved = True

        # Atış süresinin dolup dolmadığını kontrol et ve Stage 9'u serbest bırak
        if self.shot_started and not self.shot_completed:
            elapsed = (self.get_clock().now() - self.shot_start_time).nanoseconds / 1e9
            if elapsed < self.laser_duration:
                self.publish_laser(True)
            else:
                self.publish_laser(False)
                self.shot_completed = True
                self.get_logger().warning(
                    f"[TARGET DETECT] Lazer atışı ({self.laser_duration}s) tamamlandı. Stage 9 serbest bırakılıyor."
                )
                self.publish_release(9)

        else:
            cv2.putText(
                debug_img,
                "Hedef Aranıyor (Global)...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Ekranda pencere göster (isteğe bağlı veya webcam modunda varsayılan)
        if self.show_window:
            cv2.imshow("Target Detection (Bilgisayar Kamerasi)", debug_img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC veya 'q' tuşuna basıldığında çık
                self.get_logger().info("Kullanıcı çıkış yaptı (q/ESC).")
                rclpy.shutdown()

        # Debug görüntüsünü yayınla
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Debug görüntü yayınlama hatası: {e}")


    def destroy_node(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    use_webcam_cli = False
    device_id_cli = 0
    show_window_cli = False

    # CLI argümanlarını kontrol et (--webcam, --device X, --show)
    for arg in sys.argv[1:]:
        if arg in ("--webcam", "-w"):
            use_webcam_cli = True
            show_window_cli = True
        elif arg.startswith("--device=") or arg.startswith("-d="):
            try:
                device_id_cli = int(arg.split("=")[1])
            except ValueError:
                pass
        elif arg in ("--show", "-s"):
            show_window_cli = True

    rclpy.init(args=args)
    node = TargetDetector(
        use_webcam_cli=use_webcam_cli,
        device_id_cli=device_id_cli,
        show_window_cli=show_window_cli,
    )

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
