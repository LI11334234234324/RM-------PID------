"""
单元测试：一阶直流电机与编码器模型 (tests/test_motor.py)
"""

import unittest
import numpy as np
from src.motor import DCMotorModel


class TestDCMotorModel(unittest.TestCase):

    def setUp(self):
        # 默认参数: tau=0.050s, K=1.0, max_speed=35.0 rad/s, dt=0.005s
        self.motor = DCMotorModel(
            tau=0.050,
            gain_k=1.0,
            max_speed=35.0,
            deadband=0.0,
            enable_encoder_noise=False,
            dt=0.005
        )

    def test_step_response_steady_state(self):
        """测试电机阶跃响应稳态收敛值: u=20.0, 经过充分时间 (200步=1.0s, 20*tau) 后收敛至 K * u = 20.0"""
        u_step = 20.0
        for _ in range(200):
            self.motor.step(u_step)

        self.assertAlmostEqual(self.motor.actual_speed, u_step, places=4)

    def test_time_constant_tau(self):
        """测试时间常数特性: 在 t = tau (0.05s, 10步) 时，转速达到 (1 - 1/e) * u 约 63.2%"""
        u_step = 10.0
        self.motor.reset()
        
        # 推进 10 步 (10 * 0.005s = 0.050s = tau)
        for _ in range(10):
            self.motor.step(u_step)

        expected_speed = u_step * (1.0 - np.exp(-1.0))  # 6.3212 rad/s
        self.assertAlmostEqual(self.motor.actual_speed, expected_speed, places=3)

    def test_speed_saturation(self):
        """测试电机物理最大转速饱和限制"""
        self.motor.reset()
        huge_u = 100.0  # 远超 max_speed (35.0)
        for _ in range(100):
            self.motor.step(huge_u)

        self.assertEqual(self.motor.actual_speed, 35.0)

    def test_deadband(self):
        """测试电机死区特性: 输入幅值小于 deadband 时电机保持不动"""
        motor_dead = DCMotorModel(tau=0.05, deadband=2.0, dt=0.005)
        for _ in range(50):
            motor_dead.step(1.5)  # 1.5 < 2.0 死区

        self.assertEqual(motor_dead.actual_speed, 0.0)

        # 当输入超过死区时，应产生动作
        for _ in range(200):
            motor_dead.step(5.0)  # 有效应为 5.0 - 2.0 = 3.0
        self.assertAlmostEqual(motor_dead.actual_speed, 3.0, places=4)

    def test_encoder_quantization_toggle(self):
        """测试编码器量化开关特性"""
        motor_noisy = DCMotorModel(tau=0.05, enable_encoder_noise=True, encoder_cpr=8192, dt=0.005)
        motor_noisy.actual_speed = 10.0
        measured = motor_noisy._sample_encoder_speed()
        # 验证有测量值返回且在 10.0 附近
        self.assertAlmostEqual(measured, 10.0, delta=0.5)


if __name__ == '__main__':
    unittest.main()
