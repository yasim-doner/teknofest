#!/usr/bin/env python3

import os
import signal
import subprocess

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int32, String


class SignDetector(Node):
    """
    Gazebo TEKNOFEST etap tabelalarını algılar.

    Çalışma sırası:
    1. Beyaz içli, kırmızı çerçeveli, yaklaşık dairesel tabela adaylarını bulur.
    2. Birden fazla aday varsa her birini sınıflandırır.
    3. Dış çerçeve yerine tabelanın ortasındaki rakam/yazıyı karşılaştırır.
    4. En iyi ve ikinci en iyi sınıf arasındaki fark yeterliyse sonucu kabul eder.
    5. Aynı etiket art arda birkaç kare görülmeden kararlı sonuç yayınlamaz.

    Bu node yalnızca algılama yapar; /rover/cmd_vel yayınlamaz.
    Parkur sırası denetimi ayrı bir mission_manager node'unda yapılmalıdır.
    """

    def __init__(self):
        super().__init__("sign_detector")

        self.declare_parameter(
            "image_topic",
            "/rover/camera/image_raw"
        )
        self.declare_parameter("min_confidence", 0.18)
        self.declare_parameter("min_margin", 0.015)
        self.declare_parameter("stable_frames", 3)
        self.declare_parameter("lost_frames", 5)
        self.declare_parameter("min_candidate_size", 12)
        self.declare_parameter("max_candidate_size", 60)
        self.declare_parameter("max_candidates", 5)

        self.image_topic = str(
            self.get_parameter("image_topic").value
        )

        self.min_confidence = float(
            self.get_parameter("min_confidence").value
        )

        self.min_margin = float(
            self.get_parameter("min_margin").value
        )

        self.stable_frames = max(
            1,
            int(self.get_parameter("stable_frames").value)
        )

        self.lost_frames_limit = max(
            1,
            int(self.get_parameter("lost_frames").value)
        )

        self.min_candidate_size = max(
            6,
            int(self.get_parameter("min_candidate_size").value)
        )

        self.max_candidate_size = max(
            self.min_candidate_size + 1,
            int(self.get_parameter("max_candidate_size").value),
        )

        self.max_candidates = max(
            1,
            int(self.get_parameter("max_candidates").value)
        )

        self.bridge = CvBridge()
        self.templates = self.load_templates()

        if not self.templates:
            raise RuntimeError(
                "Hiç tabela şablonu yüklenemedi."
            )

        self.pending_label = None
        self.pending_count = 0

        self.stable_label = None
        self.stable_confidence = 0.0

        self.lost_frames = 0

        self.stage_pub = self.create_publisher(
            Int32,
            "/teknofest/stage_id",
            10
        )

        self.stop_pub = self.create_publisher(
            Bool,
            "/teknofest/stop_detected",
            10
        )

        self.label_pub = self.create_publisher(
            String,
            "/teknofest/sign_label",
            10
        )

        self.conf_pub = self.create_publisher(
            Float32,
            "/teknofest/sign_confidence",
            10
        )

        self.debug_pub = self.create_publisher(
            Image,
            "/teknofest/sign_debug_image",
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        # Sign detector başladıktan sonra rqt_image_view
        # penceresini otomatik açmak için kullanılır.
        self.image_viewer_process = None

        # Gazebo ve kamera topic'lerinin oluşması için
        # rqt_image_view iki saniye sonra açılır.
        self.image_viewer_timer = self.create_timer(
            2.0,
            self.start_image_viewer,
        )

        self.get_logger().info(
            f"Sign detector başladı. "
            f"Kamera={self.image_topic}, "
            f"şablon={len(self.templates)}, "
            f"eşik={self.min_confidence:.3f}, "
            f"fark_eşiği={self.min_margin:.3f}, "
            f"kararlı_kare={self.stable_frames}"
        )

    @staticmethod
    def composite_alpha_on_white(image):
        if image.ndim != 3:
            return image

        if image.shape[2] != 4:
            return image

        bgr = image[:, :, :3].astype(np.float32)

        alpha = (
            image[:, :, 3:4].astype(np.float32)
            / 255.0
        )

        white = np.full_like(
            bgr,
            255.0
        )

        result = (
            bgr * alpha
            + white * (1.0 - alpha)
        )

        return result.astype(np.uint8)

    @staticmethod
    def trim_nonwhite(image):
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        mask = gray < 248

        ys, xs = np.where(mask)

        if len(xs) == 0 or len(ys) == 0:
            return image

        margin = 3

        x1 = max(
            0,
            int(xs.min()) - margin
        )

        x2 = min(
            image.shape[1],
            int(xs.max()) + margin + 1
        )

        y1 = max(
            0,
            int(ys.min()) - margin
        )

        y2 = min(
            image.shape[0],
            int(ys.max()) + margin + 1
        )

        return image[y1:y2, x1:x2]

    @staticmethod
    def square_pad(image, value=0):
        height, width = image.shape[:2]

        size = max(
            height,
            width
        )

        if image.ndim == 2:
            canvas = np.full(
                (size, size),
                value,
                dtype=image.dtype,
            )

        else:
            canvas = np.full(
                (
                    size,
                    size,
                    image.shape[2]
                ),
                value,
                dtype=image.dtype,
            )

        y0 = (size - height) // 2
        x0 = (size - width) // 2

        canvas[
            y0:y0 + height,
            x0:x0 + width
        ] = image

        return canvas

    @staticmethod
    def rotate_image(image, angle):
        height, width = image.shape[:2]

        matrix = cv2.getRotationMatrix2D(
            (
                width / 2.0,
                height / 2.0
            ),
            angle,
            1.0,
        )

        return cv2.warpAffine(
            image,
            matrix,
            (
                width,
                height
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(
                255,
                255,
                255
            ),
        )

    @staticmethod
    def extract_glyph(image):
        """
        Ortak kırmızı çerçeveyi dışarıda bırakır ve yalnızca
        merkezdeki rakam/STOP yazısını normalize eder.
        """

        image = SignDetector.square_pad(
            image,
            value=255
        )

        image = cv2.resize(
            image,
            (
                160,
                160
            ),
            interpolation=cv2.INTER_CUBIC,
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (
                3,
                3
            ),
            0
        )

        inner_mask = np.zeros(
            (
                160,
                160
            ),
            dtype=np.uint8
        )

        cv2.circle(
            inner_mask,
            (
                80,
                80
            ),
            52,
            255,
            -1
        )

        dark = np.zeros(
            (
                160,
                160
            ),
            dtype=np.uint8
        )

        dark[
            (gray < 155)
            & (inner_mask == 255)
        ] = 255

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                3,
                3
            ),
        )

        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                dark,
                connectivity=8,
            )
        )

        cleaned = np.zeros_like(dark)

        for index in range(
            1,
            count
        ):
            area = int(
                stats[
                    index,
                    cv2.CC_STAT_AREA
                ]
            )

            if area >= 8:
                cleaned[
                    labels == index
                ] = 255

        ys, xs = np.where(
            cleaned > 0
        )

        if len(xs) == 0 or len(ys) == 0:
            return np.zeros(
                (
                    96,
                    96
                ),
                dtype=np.uint8
            )

        margin = 4

        x1 = max(
            0,
            int(xs.min()) - margin
        )

        x2 = min(
            cleaned.shape[1],
            int(xs.max()) + margin + 1
        )

        y1 = max(
            0,
            int(ys.min()) - margin
        )

        y2 = min(
            cleaned.shape[0],
            int(ys.max()) + margin + 1
        )

        glyph = cleaned[
            y1:y2,
            x1:x2
        ]

        glyph = SignDetector.square_pad(
            glyph,
            value=0
        )

        glyph = cv2.resize(
            glyph,
            (
                96,
                96
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        return glyph

    def load_templates(self):
        share_dir = get_package_share_directory(
            "rover_sim"
        )

        texture_dir = os.path.join(
            share_dir,
            "models",
            "signs",
            "materials",
            "textures",
        )

        specs = [
            (
                f"stage_{stage_id}",
                stage_id,
                f"sign_{stage_id}.png"
            )
            for stage_id in range(
                1,
                12
            )
        ]

        specs.append(
            (
                "stop",
                0,
                "sign_stop.png"
            )
        )

        templates = []

        for label, stage_id, filename in specs:
            path = os.path.join(
                texture_dir,
                filename
            )

            image = cv2.imread(
                path,
                cv2.IMREAD_UNCHANGED
            )

            if image is None:
                self.get_logger().warning(
                    f"Şablon okunamadı: {path}"
                )
                continue

            image = self.composite_alpha_on_white(
                image
            )

            if image.ndim == 2:
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_GRAY2BGR
                )

            image = self.trim_nonwhite(
                image
            )

            glyph = self.extract_glyph(
                image
            )

            templates.append(
                {
                    "label": label,
                    "stage_id": stage_id,
                    "glyph": glyph,
                }
            )

        return templates

    @staticmethod
    def red_mask(hsv):
        low_red = cv2.inRange(
            hsv,
            (
                0,
                65,
                45
            ),
            (
                15,
                255,
                255
            )
        )

        high_red = cv2.inRange(
            hsv,
            (
                165,
                65,
                45
            ),
            (
                180,
                255,
                255
            )
        )

        return cv2.bitwise_or(
            low_red,
            high_red
        )

    def validate_circle(
        self,
        frame,
        cx,
        cy,
        radius
    ):
        height, width = frame.shape[:2]

        if radius < self.min_candidate_size / 2.0:
            return None

        if radius > self.max_candidate_size / 2.0:
            return None

        if cy > int(height * 0.82):
            return None

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        red = self.red_mask(hsv)

        yy, xx = np.ogrid[
            :height,
            :width
        ]

        distance = np.sqrt(
            (xx - cx) ** 2
            + (yy - cy) ** 2
        )

        inner = (
            distance
            <= radius * 0.72
        )

        ring = (
            (distance >= radius * 0.72)
            & (distance <= radius * 1.18)
        )

        inner_count = int(
            np.count_nonzero(inner)
        )

        ring_count = int(
            np.count_nonzero(ring)
        )

        if inner_count == 0:
            return None

        if ring_count == 0:
            return None

        white_pixels = (
            (hsv[:, :, 1] < 90)
            & (hsv[:, :, 2] > 120)
            & inner
        )

        dark_pixels = (
            (gray < 130)
            & inner
        )

        red_pixels = (
            (red > 0)
            & ring
        )

        white_ratio = (
            np.count_nonzero(
                white_pixels
            )
            / inner_count
        )

        dark_ratio = (
            np.count_nonzero(
                dark_pixels
            )
            / inner_count
        )

        red_ratio = (
            np.count_nonzero(
                red_pixels
            )
            / ring_count
        )

        if white_ratio < 0.26:
            return None

        if dark_ratio < 0.008:
            return None

        if red_ratio < 0.012:
            return None

        quality = (
            0.50 * white_ratio
            + 0.30 * min(
                1.0,
                red_ratio * 8.0
            )
            + 0.20 * min(
                1.0,
                dark_ratio * 7.0
            )
        )

        return {
            "cx": int(cx),
            "cy": int(cy),
            "radius": int(radius),
            "quality": float(quality),
        }

    def candidates_from_white_contours(
        self,
        frame
    ):
        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        white_mask = cv2.inRange(
            hsv,
            (
                0,
                0,
                125
            ),
            (
                180,
                95,
                255
            ),
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                3,
                3
            ),
        )

        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        contours, _ = cv2.findContours(
            white_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        frame_area = (
            frame.shape[0]
            * frame.shape[1]
        )

        candidates = []

        for contour in contours:
            area = cv2.contourArea(
                contour
            )

            if area < 45:
                continue

            if area > frame_area * 0.035:
                continue

            perimeter = cv2.arcLength(
                contour,
                True
            )

            if perimeter <= 0:
                continue

            circularity = (
                4.0
                * np.pi
                * area
                / (
                    perimeter
                    * perimeter
                )
            )

            _, _, width, height = cv2.boundingRect(
                contour
            )

            if width < self.min_candidate_size:
                continue

            if height < self.min_candidate_size:
                continue

            if width > self.max_candidate_size:
                continue

            if height > self.max_candidate_size:
                continue

            aspect = width / float(height)

            if not (
                0.72
                <= aspect
                <= 1.38
            ):
                continue

            if circularity < 0.42:
                continue

            (cx, cy), radius = (
                cv2.minEnclosingCircle(
                    contour
                )
            )

            validated = self.validate_circle(
                frame,
                cx,
                cy,
                radius * 1.10
            )

            if validated is not None:
                candidates.append(
                    validated
                )

        return candidates

    def candidates_from_hough(
        self,
        frame
    ):
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (
                7,
                7
            ),
            1.5
        )

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=120,
            param2=23,
            minRadius=max(
                5,
                self.min_candidate_size // 2
            ),
            maxRadius=max(
                8,
                self.max_candidate_size // 2
            ),
        )

        candidates = []

        if circles is None:
            return candidates

        rounded_circles = np.round(
            circles[0]
        ).astype(int)

        for cx, cy, radius in rounded_circles:
            validated = self.validate_circle(
                frame,
                int(cx),
                int(cy),
                int(radius),
            )

            if validated is not None:
                candidates.append(
                    validated
                )

        return candidates

    @staticmethod
    def candidate_iou(
        candidate_a,
        candidate_b
    ):
        ax1 = (
            candidate_a["cx"]
            - candidate_a["radius"]
        )

        ay1 = (
            candidate_a["cy"]
            - candidate_a["radius"]
        )

        ax2 = (
            candidate_a["cx"]
            + candidate_a["radius"]
        )

        ay2 = (
            candidate_a["cy"]
            + candidate_a["radius"]
        )

        bx1 = (
            candidate_b["cx"]
            - candidate_b["radius"]
        )

        by1 = (
            candidate_b["cy"]
            - candidate_b["radius"]
        )

        bx2 = (
            candidate_b["cx"]
            + candidate_b["radius"]
        )

        by2 = (
            candidate_b["cy"]
            + candidate_b["radius"]
        )

        x1 = max(
            ax1,
            bx1
        )

        y1 = max(
            ay1,
            by1
        )

        x2 = min(
            ax2,
            bx2
        )

        y2 = min(
            ay2,
            by2
        )

        intersection = (
            max(
                0,
                x2 - x1
            )
            * max(
                0,
                y2 - y1
            )
        )

        area_a = (
            max(
                0,
                ax2 - ax1
            )
            * max(
                0,
                ay2 - ay1
            )
        )

        area_b = (
            max(
                0,
                bx2 - bx1
            )
            * max(
                0,
                by2 - by1
            )
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    def find_candidates(self, frame):
        candidates = (
            self.candidates_from_white_contours(
                frame
            )
        )

        candidates.extend(
            self.candidates_from_hough(
                frame
            )
        )

        candidates = sorted(
            candidates,
            key=lambda item: (
                item["radius"],
                item["quality"],
            ),
            reverse=True,
        )

        kept = []

        for candidate in candidates:
            is_separate = all(
                self.candidate_iou(
                    candidate,
                    old
                ) < 0.45
                for old in kept
            )

            if is_separate:
                kept.append(
                    candidate
                )

        return kept[
            :self.max_candidates
        ]

    @staticmethod
    def crop_candidate(
        frame,
        candidate
    ):
        cx = candidate["cx"]
        cy = candidate["cy"]

        radius = int(
            candidate["radius"]
            * 1.18
        )

        height, width = frame.shape[:2]

        x1 = max(
            0,
            cx - radius
        )

        y1 = max(
            0,
            cy - radius
        )

        x2 = min(
            width,
            cx + radius
        )

        y2 = min(
            height,
            cy + radius
        )

        crop = frame[
            y1:y2,
            x1:x2
        ].copy()

        if crop.size == 0:
            return None, None

        return crop, (
            x1,
            y1,
            x2,
            y2
        )

    @staticmethod
    def binary_dice(
        mask_a,
        mask_b
    ):
        a = mask_a > 0
        b = mask_b > 0

        denominator = (
            np.count_nonzero(a)
            + np.count_nonzero(b)
        )

        if denominator == 0:
            return 0.0

        intersection = np.count_nonzero(
            a & b
        )

        return (
            2.0
            * intersection
            / denominator
        )

    @staticmethod
    def binary_iou(
        mask_a,
        mask_b
    ):
        a = mask_a > 0
        b = mask_b > 0

        union = np.count_nonzero(
            a | b
        )

        if union == 0:
            return 0.0

        intersection = np.count_nonzero(
            a & b
        )

        return intersection / union

    @staticmethod
    def correlation(
        mask_a,
        mask_b
    ):
        value = cv2.matchTemplate(
            mask_a,
            mask_b,
            cv2.TM_CCOEFF_NORMED,
        )[0, 0]

        if np.isnan(value):
            return 0.0

        return max(
            0.0,
            float(value)
        )

    def classify_candidate(
        self,
        crop
    ):
        best_by_label = {}

        for angle in (
            -12,
            -8,
            -4,
            0,
            4,
            8,
            12
        ):
            rotated = self.rotate_image(
                crop,
                angle
            )

            glyph = self.extract_glyph(
                rotated
            )

            for template in self.templates:
                template_glyph = template[
                    "glyph"
                ]

                dice = self.binary_dice(
                    glyph,
                    template_glyph
                )

                iou = self.binary_iou(
                    glyph,
                    template_glyph
                )

                corr = self.correlation(
                    glyph,
                    template_glyph
                )

                score = (
                    0.45 * dice
                    + 0.35 * iou
                    + 0.20 * corr
                )

                label = template[
                    "label"
                ]

                previous = best_by_label.get(
                    label
                )

                if (
                    previous is None
                    or score > previous["score"]
                ):
                    best_by_label[label] = {
                        "label": label,
                        "stage_id": template[
                            "stage_id"
                        ],
                        "score": float(score),
                    }

        if not best_by_label:
            return None

        ranked = sorted(
            best_by_label.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        best = dict(
            ranked[0]
        )

        if len(ranked) > 1:
            second = ranked[1]

        else:
            second = {
                "label": "none",
                "score": 0.0,
            }

        best["second_label"] = second[
            "label"
        ]

        best["second_score"] = float(
            second["score"]
        )

        best["margin"] = float(
            best["score"]
            - second["score"]
        )

        best["accepted"] = (
            best["score"]
            >= self.min_confidence
            and best["margin"]
            >= self.min_margin
        )

        return best

    def choose_best_detection(
        self,
        frame
    ):
        candidates = self.find_candidates(
            frame
        )

        detections = []

        for candidate in candidates:
            crop, box = self.crop_candidate(
                frame,
                candidate
            )

            if crop is None:
                continue

            classification = self.classify_candidate(
                crop
            )

            if classification is None:
                continue

            size_bonus = min(
                0.05,
                candidate["radius"] / 1000.0,
            )

            selection_score = (
                classification["score"]
                + 0.75
                * classification["margin"]
                + size_bonus
            )

            detections.append(
                {
                    "candidate": candidate,
                    "box": box,
                    "classification": classification,
                    "selection_score": float(
                        selection_score
                    ),
                }
            )

        if not detections:
            return None, candidates

        detections.sort(
            key=lambda item: item[
                "selection_score"
            ],
            reverse=True,
        )

        return (
            detections[0],
            candidates
        )

    def update_stability(
        self,
        raw_label,
        raw_confidence
    ):
        if raw_label is None:
            self.pending_label = None
            self.pending_count = 0

            self.lost_frames += 1

            if (
                self.lost_frames
                >= self.lost_frames_limit
            ):
                self.stable_label = None
                self.stable_confidence = 0.0

            return self.stable_label

        self.lost_frames = 0

        if raw_label == self.pending_label:
            self.pending_count += 1

        else:
            self.pending_label = raw_label
            self.pending_count = 1

        if (
            self.pending_count
            >= self.stable_frames
        ):
            if raw_label != self.stable_label:
                self.get_logger().info(
                    "Kararlı tabela algılandı: "
                    f"{raw_label}"
                )

            self.stable_label = raw_label
            self.stable_confidence = (
                raw_confidence
            )

        return self.stable_label

    def start_image_viewer(self):
        """
        rqt_image_view penceresini yalnızca bir kere açar.

        Açılan pencerede doğrudan
        /teknofest/sign_debug_image topic'i gösterilir.
        """

        # İki saniyelik timer yalnızca bir kere çalışsın.
        if self.image_viewer_timer is not None:
            self.image_viewer_timer.cancel()
            self.image_viewer_timer = None

        # Viewer zaten açıksa tekrar açma.
        if (
            self.image_viewer_process is not None
            and self.image_viewer_process.poll() is None
        ):
            return

        # Grafik ekranı bulunamazsa RQT açılamaz.
        if not os.environ.get("DISPLAY"):
            self.get_logger().warning(
                "DISPLAY bulunamadı. "
                "rqt_image_view açılamadı."
            )
            return

        try:
            self.image_viewer_process = subprocess.Popen(
                [
                    "ros2",
                    "run",
                    "rqt_image_view",
                    "rqt_image_view",
                    "/teknofest/sign_debug_image",
                ],
                start_new_session=True,
            )

            self.get_logger().info(
                "rqt_image_view otomatik açıldı: "
                "/teknofest/sign_debug_image"
            )

        except FileNotFoundError:
            self.get_logger().error(
                "ros2 komutu bulunamadı. "
                "ROS 2 ortamının source edildiğini kontrol et."
            )

        except Exception as exc:
            self.get_logger().error(
                "rqt_image_view açılamadı: "
                f"{exc}"
            )

    def stop_image_viewer(self):
        """
        Sign detector kapanırken bu node tarafından açılan
        rqt_image_view penceresini de kapatır.
        """

        if self.image_viewer_process is None:
            return

        if self.image_viewer_process.poll() is not None:
            self.image_viewer_process = None
            return

        try:
            process_group = os.getpgid(
                self.image_viewer_process.pid
            )

            os.killpg(
                process_group,
                signal.SIGTERM,
            )

            self.image_viewer_process.wait(
                timeout=3.0
            )

        except subprocess.TimeoutExpired:
            try:
                process_group = os.getpgid(
                    self.image_viewer_process.pid
                )

                os.killpg(
                    process_group,
                    signal.SIGKILL,
                )

            except ProcessLookupError:
                pass

        except ProcessLookupError:
            pass

        except Exception as exc:
            self.get_logger().warning(
                "rqt_image_view kapatılırken "
                f"hata oluştu: {exc}"
            )

        finally:
            self.image_viewer_process = None

    def template_by_label(
        self,
        label
    ):
        for template in self.templates:
            if template["label"] == label:
                return template

        return None

    def publish_result(self):
        stage_msg = Int32()
        stop_msg = Bool()
        label_msg = String()
        confidence_msg = Float32()

        if self.stable_label:
            template = self.template_by_label(
                self.stable_label
            )

        else:
            template = None

        if template is None:
            stage_msg.data = 0
            stop_msg.data = False
            label_msg.data = "none"
            confidence_msg.data = 0.0

        else:
            stage_msg.data = int(
                template["stage_id"]
            )

            stop_msg.data = (
                template["label"]
                == "stop"
            )

            label_msg.data = str(
                template["label"]
            )

            confidence_msg.data = float(
                self.stable_confidence
            )

        self.stage_pub.publish(
            stage_msg
        )

        self.stop_pub.publish(
            stop_msg
        )

        self.label_pub.publish(
            label_msg
        )

        self.conf_pub.publish(
            confidence_msg
        )

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )

        except Exception as exc:
            self.get_logger().error(
                "Kamera görüntüsü çevrilemedi: "
                f"{exc}"
            )
            return

        debug = frame.copy()

        best_detection, all_candidates = (
            self.choose_best_detection(
                frame
            )
        )

        raw_label = None
        raw_confidence = 0.0

        best_label = "none"
        second_label = "none"

        margin = 0.0

        if best_detection is not None:
            classification = best_detection[
                "classification"
            ]

            best_label = classification[
                "label"
            ]

            second_label = classification[
                "second_label"
            ]

            margin = classification[
                "margin"
            ]

            raw_confidence = classification[
                "score"
            ]

            if classification["accepted"]:
                raw_label = classification[
                    "label"
                ]

        self.update_stability(
            raw_label,
            raw_confidence
        )

        self.publish_result()

        # Bulunan bütün adayları ince gri daireyle göster.
        for candidate in all_candidates:
            cv2.circle(
                debug,
                (
                    candidate["cx"],
                    candidate["cy"]
                ),
                candidate["radius"],
                (
                    160,
                    160,
                    160
                ),
                1,
            )

        # En iyi aday turuncu daire ve kutuyla gösterilir.
        if best_detection is not None:
            candidate = best_detection[
                "candidate"
            ]

            x1, y1, x2, y2 = best_detection[
                "box"
            ]

            cv2.circle(
                debug,
                (
                    candidate["cx"],
                    candidate["cy"]
                ),
                candidate["radius"],
                (
                    0,
                    200,
                    255
                ),
                2,
            )

            cv2.rectangle(
                debug,
                (
                    x1,
                    y1
                ),
                (
                    x2,
                    y2
                ),
                (
                    0,
                    200,
                    255
                ),
                2,
            )

            label_text = (
                f"best={best_label} "
                f"{raw_confidence:.2f} "
                f"2nd={second_label} "
                f"d={margin:.3f}"
            )

            cv2.putText(
                debug,
                label_text,
                (
                    x1,
                    max(
                        20,
                        y1 - 7
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (
                    0,
                    200,
                    255
                ),
                1,
                cv2.LINE_AA,
            )

        status = (
            f"best={best_label} "
            f"stable={self.stable_label or 'none'} "
            f"score={raw_confidence:.2f} "
            f"margin={margin:.3f}"
        )

        cv2.putText(
            debug,
            status,
            (
                12,
                25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (
                0,
                255,
                0
            ),
            2,
            cv2.LINE_AA,
        )

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(
                debug,
                encoding="bgr8",
            )

            debug_msg.header = msg.header

            self.debug_pub.publish(
                debug_msg
            )

        except Exception as exc:
            self.get_logger().warning(
                "Debug görüntüsü yayınlanamadı: "
                f"{exc}"
            )


def main(args=None):
    rclpy.init(args=args)

    node = SignDetector()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.stop_image_viewer()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
