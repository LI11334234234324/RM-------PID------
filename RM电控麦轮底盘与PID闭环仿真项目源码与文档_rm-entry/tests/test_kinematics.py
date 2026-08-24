"""
单元测试：麦克纳姆轮底盘正/逆运动学解算器 (tests/test_kinematics.py)
"""

import unittest
import numpy as np
from src.kinematics import MecanumKinematics


class TestMecanumKinematics(unittest.TestCase):

    def setUp(self):
        # 初始化标准步兵底盘：轴距 0.4m, 轮距 0.4m, 轮半径 0.076m, 最大轮速 30 rad/s
        self.kin = MecanumKinematics(
            wheel_base=0.40,
            track_width=0.40,
            wheel_radius=0.076,
            max_wheel_speed=30.0
        )
        self.R = 0.076
        self.L = (0.40 / 2.0) + (0.40 / 2.0)  # 0.40 m

    def test_round_trip_consistency(self):
        """测试正逆运动学往返一致性: FK(IK(vx, vy, w)) == (vx, vy, w)"""
        test_cases = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 2.0),
            (0.5, -0.8, 1.2),
            (-1.2, 0.6, -0.9),
            (0.0, 0.0, 0.0)
        ]
        for vx, vy, w in test_cases:
            wheel_speeds = self.kin.inverse_kinematics(vx, vy, w, scale_if_saturated=False)
            res_vx, res_vy, res_w = self.kin.forward_kinematics(wheel_speeds)
            self.assertAlmostEqual(res_vx, vx, places=6, msg=f"vx 不一致 for ({vx}, {vy}, {w})")
            self.assertAlmostEqual(res_vy, vy, places=6, msg=f"vy 不一致 for ({vx}, {vy}, {w})")
            self.assertAlmostEqual(res_w, w, places=6, msg=f"omega 不一致 for ({vx}, {vy}, {w})")

    def test_pure_forward(self):
        """测试纯前进: 四轮转速相同且均为 vx / R"""
        vx = 1.52
        w_fl, w_fr, w_rl, w_rr = self.kin.inverse_kinematics(vx, 0.0, 0.0, scale_if_saturated=False)
        expected_w = vx / self.R  # 1.52 / 0.076 = 20.0 rad/s
        self.assertAlmostEqual(w_fl, expected_w, places=6)
        self.assertAlmostEqual(w_fr, expected_w, places=6)
        self.assertAlmostEqual(w_rl, expected_w, places=6)
        self.assertAlmostEqual(w_rr, expected_w, places=6)

    def test_pure_strafe_left(self):
        """测试纯左横移: 左前/右后反转，右前/左后正转"""
        vy = 0.76
        w_fl, w_fr, w_rl, w_rr = self.kin.inverse_kinematics(0.0, vy, 0.0, scale_if_saturated=False)
        expected_w = vy / self.R  # 10.0 rad/s
        self.assertAlmostEqual(w_fl, -expected_w, places=6)
        self.assertAlmostEqual(w_fr,  expected_w, places=6)
        self.assertAlmostEqual(w_rl,  expected_w, places=6)
        self.assertAlmostEqual(w_rr, -expected_w, places=6)

    def test_pure_rotation_ccw(self):
        """测试纯逆时针原地自转: 右侧轮正转，左侧轮反转"""
        w = 2.0
        w_fl, w_fr, w_rl, w_rr = self.kin.inverse_kinematics(0.0, 0.0, w, scale_if_saturated=False)
        expected_lin = self.L * w  # 0.40 * 2.0 = 0.8 m/s
        expected_w = expected_lin / self.R
        self.assertAlmostEqual(w_fl, -expected_w, places=6)
        self.assertAlmostEqual(w_fr,  expected_w, places=6)
        self.assertAlmostEqual(w_rl, -expected_w, places=6)
        self.assertAlmostEqual(w_rr,  expected_w, places=6)

    def test_diagonal_45_deg(self):
        """测试 45° 斜向平移: 对角线车轮转动，另一对角线车轮静止"""
        # vx = 1.0, vy = 1.0 (向左前方 45° 平移)
        # v_fl = vx - vy = 0 -> FL 轮静止
        # v_rr = vx - vy = 0 -> RR 轮静止
        # v_fr = vx + vy = 2.0 -> FR 轮正转
        # v_rl = vx + vy = 2.0 -> RL 轮正转
        w_fl, w_fr, w_rl, w_rr = self.kin.inverse_kinematics(1.0, 1.0, 0.0, scale_if_saturated=False)
        self.assertAlmostEqual(w_fl, 0.0, places=6)
        self.assertAlmostEqual(w_rr, 0.0, places=6)
        self.assertAlmostEqual(w_fr, 2.0 / self.R, places=6)
        self.assertAlmostEqual(w_rl, 2.0 / self.R, places=6)

    def test_speed_saturation_and_scaling(self):
        """测试超速等比缩放: 保持合成运动方向比例不失真"""
        # 请求一个极高速度: vx = 3.0, vy = 3.0, w = 10.0
        # 轮速肯定超过 max_wheel_speed (30 rad/s)
        w_scaled = self.kin.inverse_kinematics(3.0, 3.0, 10.0, scale_if_saturated=True)
        max_actual_w = np.max(np.abs(w_scaled))
        self.assertLessEqual(max_actual_w, self.kin.max_wheel_speed + 1e-6)
        
        # 验证缩放后由正运动学解算出的 (vx, vy, w) 仍严格保持原始比例
        scaled_vx, scaled_vy, scaled_w = self.kin.forward_kinematics(w_scaled)
        self.assertAlmostEqual(scaled_vx / scaled_vy, 3.0 / 3.0, places=5)
        self.assertAlmostEqual(scaled_w / scaled_vx, 10.0 / 3.0, places=5)

    def test_world_body_velocity_transform(self):
        """测试世界系与机体系坐标系旋转转换"""
        # 当底盘航向为 90° (pi/2) 时:
        # 世界系向 X (正东) 运动 1.0 m/s 对应机体系向 -Y (右侧) 运动 1.0 m/s
        vx_w, vy_w = 1.0, 0.0
        yaw = np.pi / 2.0
        vx_b, vy_b = self.kin.world_to_body_velocity(vx_w, vy_w, yaw)
        self.assertAlmostEqual(vx_b, 0.0, places=6)
        self.assertAlmostEqual(vy_b, -1.0, places=6)

        # 逆变换回世界系
        rec_vx_w, rec_vy_w = self.kin.body_to_world_velocity(vx_b, vy_b, yaw)
        self.assertAlmostEqual(rec_vx_w, vx_w, places=6)
        self.assertAlmostEqual(rec_vy_w, vy_w, places=6)


if __name__ == '__main__':
    unittest.main()
