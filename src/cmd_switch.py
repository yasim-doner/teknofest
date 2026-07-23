#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Int32


class CmdSwitchNode(Node):
    """
    Aktif stage'e göre doğru cmd_vel kaynağını /rover/cmd_vel'e aktarır.

    Stage 5  -> /cone_avoid/cmd_vel
    Stage 6  -> /dynamic_obstacle/cmd_vel
    Stage 8  -> /laser_target/cmd_vel
    Diğerleri -> /fallow_corridor/cmd_vel
    """

    def __init__(self):
        super().__init__("cmd_switch")

        self.declare_parameter("initial_stage", 0)
        self.active_stage = int(
            self.get_parameter("initial_stage").value
        )
        self.released_stage = None
        self.final_stop_latched = False

        self.stage_topics = {
            5: "/cone_avoid/cmd_vel",
            6: "/dynamic_obstacle/cmd_vel",
            8: "/laser_target/cmd_vel",
        }

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            "/rover/cmd_vel",
            10,
        )

        self.fallow_sub = self.create_subscription(
            Twist,
            "/fallow_corridor/cmd_vel",
            self.fallow_callback,
            10,
        )

        self.cone_sub = self.create_subscription(
            Twist,
            "/cone_avoid/cmd_vel",
            self.cone_callback,
            10,
        )

        self.dynamic_sub = self.create_subscription(
            Twist,
            "/dynamic_obstacle/cmd_vel",
            self.dynamic_callback,
            10,
        )

        self.laser_sub = self.create_subscription(
            Twist,
            "/laser_target/cmd_vel",
            self.laser_callback,
            10,
        )

        self.stage_sub = self.create_subscription(
            Int32,
            "/teknofest/stage_id",
            self.stage_callback,
            10,
        )

        self.release_sub = self.create_subscription(
            Int32,
            "/teknofest/release",
            self.release_callback,
            10,
        )

        self.final_stop_sub = self.create_subscription(
            Bool,
            "/teknofest/final_stop",
            self.final_stop_callback,
            10,
        )

        # Final duruş aktif olduğunda başka node'ların eski hız komutları
        # roverı tekrar hareket ettirmesin diye sıfır hız sürekli yayınlanır.
        self.final_stop_timer = self.create_timer(
            0.1,
            self.enforce_final_stop,
        )

        active_topic = self.stage_topics.get(
            self.active_stage,
            "/fallow_corridor/cmd_vel",
        )

        self.get_logger().info(
            "=== Cmd Switch Node Initialized ==="
        )
        self.get_logger().info(
            f"Başlangıç stage: {self.active_stage}"
        )
        self.get_logger().info(
            f"Başlangıç aktif kaynak: {active_topic}"
        )

    def fallow_callback(self, msg: Twist):
        if self.final_stop_latched:
            return

        if self.active_stage not in self.stage_topics:
            self.cmd_vel_pub.publish(msg)

    def cone_callback(self, msg: Twist):
        if self.final_stop_latched:
            return

        if self.active_stage == 5:
            self.cmd_vel_pub.publish(msg)

    def dynamic_callback(self, msg: Twist):
        if self.final_stop_latched:
            return

        if self.active_stage == 6:
            self.cmd_vel_pub.publish(msg)

    def laser_callback(self, msg: Twist):
        if self.final_stop_latched:
            return

        if self.active_stage == 8:
            self.cmd_vel_pub.publish(msg)

    def stage_callback(self, msg: Int32):
        stage_id = int(msg.data)

        if self.final_stop_latched:
            return

        # sign_detect henüz hazır değilken 0 yayınlasa bile
        # doğrudan stage testi bozulmasın.
        if stage_id <= 0:
            return

        # Bir görev kontrolü bıraktıktan sonra sign_detect bir süre daha aynı
        # stage'i yayınlayabilir. Aynı stage tekrar kontrolü ele almasın.
        if self.released_stage is not None:
            if stage_id == self.released_stage:
                return

            if stage_id > self.released_stage:
                self.get_logger().info(
                    f"Yeni stage {stage_id} geldi; "
                    f"Stage {self.released_stage} kilidi kaldırıldı."
                )
                self.released_stage = None

        # Doğrudan stage:=8 testinde algılayıcının geçici olarak stage_1
        # üretmesi Stage 8 kontrolünü düşürmesin. Normal parkurda stage'ler
        # ileri yönde ilerlediği için geriye giden stage'ler reddedilir.
        if stage_id < self.active_stage:
            self.get_logger().warning(
                f"Geriye giden stage reddedildi: "
                f"{self.active_stage} -> {stage_id}"
            )
            return

        if stage_id == self.active_stage:
            return

        self.active_stage = stage_id
        target_topic = self.stage_topics.get(
            stage_id,
            "/fallow_corridor/cmd_vel",
        )

        self.get_logger().info(
            f"Stage {stage_id} algılandı. "
            f"Aktif kontrol kaynağı: {target_topic}"
        )

    def release_callback(self, msg: Int32):
        released_stage = int(msg.data)

        if self.final_stop_latched:
            return

        if released_stage != self.active_stage:
            return

        self.get_logger().info(
            f"Stage {released_stage} kontrolü bıraktı. "
            "Fallow corridor'a dönülüyor."
        )

        # Kritik: sign_detect aynı stage'i her kamera karesinde yayınlamaya
        # devam eder. Bırakılan stage kaydedilmezse görev 0.2 saniye sonra
        # tekrar etkinleşir ve rover aynı yerde kalır.
        self.released_stage = released_stage
        self.active_stage = 0

        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)

        self.get_logger().info(
            f"Stage {released_stage} kilitlendi; "
            "daha yüksek bir stage gelene kadar yeniden etkinleşmeyecek."
        )


    def final_stop_callback(self, msg: Bool):
        if not bool(msg.data):
            return

        if self.final_stop_latched:
            return

        self.final_stop_latched = True
        self.active_stage = 10
        self.released_stage = None

        self.cmd_vel_pub.publish(Twist())

        self.get_logger().warning(
            "FINAL STOP alındı. Parkur tamamlandı; "
            "/rover/cmd_vel kalıcı olarak sıfırda tutulacak."
        )

    def enforce_final_stop(self):
        if not self.final_stop_latched:
            return

        self.cmd_vel_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = CmdSwitchNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if rclpy.ok():
            node.cmd_vel_pub.publish(Twist())

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
