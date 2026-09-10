#pragma once

#include "arx_r5_src/interfaces/InterfacesThread.hpp"

namespace arx::r5
{
    class InterfacesPy
    {
    public:
        InterfacesPy(const std::string& urdf_path, const std::string &can_name);
        ~InterfacesPy();

        void set_arm_status(int state);

        void set_joint_positions(std::vector<double> pos);
        void set_ee_pose(std::vector<double> pose);

        void set_catch(double pos);

        void mit_joint_control(int id, double kp, double kd, double pos, double vel, double torque);

        /// @brief 获取关节位置
        /// @return 7维vector
        std::vector<double> get_joint_positions();

        /// @brief 获取关节速度
        /// @return 7维vector
        std::vector<double> get_joint_velocities();

        /// @brief 获取关节电流
        /// @return 7维vector
        std::vector<double> get_joint_currents();

        /// @brief 获取关节扭矩
        /// @return 7维vector
        std::vector<double> get_joint_torques();

        /// @brief 获取末端位姿
        /// @return 三维矢量+四元数组成的7维vector
        std::vector<double> get_ee_pose();

        /// @brief 获取末端位姿xyzrpy
        /// @return xyzrpy组成的6维vector
        std::vector<double> get_ee_pose_xyzrpy();
        void arx_x(double arx1, double arx2, double arx3);

        /// @brief 获取末端外力估计（广义动量观测器）
        /// @return 6维vector [Fx, Fy, Fz, Tx, Ty, Tz]，单位 N / N·m
        std::vector<double> get_ext_wrench();

        /// @brief 设置末端外力指令，在重力补偿模式下叠加到前馈扭矩
        /// @param wrench 6维vector [Fx, Fy, Fz, Tx, Ty, Tz]，单位 N / N·m
        void set_ext_wrench(std::vector<double> wrench, double k = 1.0);

        /// @brief 获取夹爪估计外力扭矩
        /// @return 夹爪外力扭矩，速度大于阈值时返回0
        double get_gripper_ext_torque();

        /// @brief 设置夹爪外力扭矩指令
        /// @param torque 扭矩值
        void set_gripper_ext_torque(double torque);
        void set_slave_gripper_position(double pos);

    private:
        class impl;
        std::unique_ptr<impl> pimpl;
    };
}

namespace arx::solve
{
    Eigen::Isometry3d Xyzrpy2Isometry(double input[6]);
    std::vector<double> Isometry2Xyzrpy(Eigen::Isometry3d pose);
}
