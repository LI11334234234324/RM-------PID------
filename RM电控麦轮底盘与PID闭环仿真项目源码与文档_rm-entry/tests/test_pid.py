"""
单元测试：PID 控制器及抗积分饱和特性 (tests/test_pid.py)
"""

import unittest
import numpy as np
from src.pid import PIDController
from src.motor import DCMotorModel


class TestPIDController(unittest.TestCase):

    def setUp(self):
        self.dt = 0.005

    def test_pure_p_steady_state_error(self):
        """
        数学断言测试：纯 P 控制器在有阻力负载系统上必定存在不可消除的稳态误差
        
        理论证明:
            电机模型稳态: omega_ss = K * u - friction * omega_ss
            控制律: u = Kp * (target - omega_ss)
            代入得: omega_ss = (K * Kp * target) / (1 + friction + K * Kp)
            稳态误差: e_ss = target - omega_ss = target * (1 + friction) / (1 + friction + K * Kp) > 0
        """
        target_speed = 20.0
        # 带有 0.2 阻尼系数的电机
        motor = DCMotorModel(tau=0.05, gain_k=1.0, friction_factor=0.2, dt=self.dt)
        pid_p = PIDController(kp=2.0, ki=0.0, kd=0.0, max_output=50.0, dt=self.dt)

        for _ in range(200):  # 仿真 1.0 秒
            u = pid_p.update(target_speed, motor.actual_speed)
            motor.step(u)

        steady_error = abs(target_speed - motor.actual_speed)
        # 纯 P 的稳态误差理论值: 20 * (1 + 0.2) / (1 + 0.2 + 2.0) = 24 / 3.2 = 7.5
        self.assertGreater(steady_error, 1.0, msg="纯 P 控制器应该存在显著稳态误差")
        self.assertAlmostEqual(steady_error, 7.5, delta=0.5)

    def test_pi_eliminates_steady_state_error(self):
        """测试 PI 控制器积分项能够彻底消除稳态误差 (e_ss -> 0)"""
        target_speed = 20.0
        motor = DCMotorModel(tau=0.05, gain_k=1.0, friction_factor=0.2, dt=self.dt)
        pid_pi = PIDController(kp=2.0, ki=15.0, kd=0.0, max_output=50.0, max_integral=30.0, dt=self.dt)

        for _ in range(400):  # 仿真 2.0 秒
            u = pid_pi.update(target_speed, motor.actual_speed)
            motor.step(u)

        steady_error = abs(target_speed - motor.actual_speed)
        self.assertLess(steady_error, 1e-3, msg="PI 控制器稳态误差应收敛至 0")

    def test_anti_windup_bounds_integral(self):
        """测试抗积分饱和 (Anti-Windup) 机制: 强行堵转或大误差饱和时，积分量严格有界"""
        pid = PIDController(
            kp=5.0,
            ki=20.0,
            kd=0.0,
            max_output=30.0,
            max_integral=10.0,
            dt=self.dt,
            anti_windup=True
        )

        # 模拟电机堵转 (反馈恒为 0，给大阶跃 50.0)
        target = 50.0
        feedback = 0.0

        for _ in range(1000):  # 持续 5 秒的大误差
            pid.update(target, feedback)

        # 验证输出严格被限幅在 max_output
        self.assertEqual(pid.output, 30.0)
        # 验证积分项没有无限制膨胀 (windup)，被约束在 max_integral 内
        self.assertLessEqual(abs(pid.i_out), 10.0 + 1e-6)

    def test_output_clamping(self):
        """测试总输出绝对值不超过 max_output"""
        pid = PIDController(kp=100.0, ki=100.0, kd=0.0, max_output=25.0, dt=self.dt)
        out_pos = pid.update(100.0, 0.0)
        self.assertEqual(out_pos, 25.0)

        out_neg = pid.update(-100.0, 0.0)
        self.assertEqual(out_neg, -25.0)

    def test_reset_behavior(self):
        """测试控制器 reset 后各状态清零"""
        pid = PIDController(kp=2.0, ki=5.0, dt=self.dt)
        pid.update(10.0, 0.0)
        self.assertNotEqual(pid.integral, 0.0)
        
        pid.reset()
        self.assertEqual(pid.integral, 0.0)
        self.assertEqual(pid.prev_error, 0.0)
        self.assertEqual(pid.output, 0.0)
        self.assertTrue(pid.is_first_run)


if __name__ == '__main__':
    unittest.main()
