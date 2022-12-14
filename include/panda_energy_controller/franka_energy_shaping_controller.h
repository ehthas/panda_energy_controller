// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <mutex>

#include <controller_interface/multi_interface_controller.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/PoseStamped.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <Eigen/Dense>

#include <franka_example_controllers/compliance_paramConfig.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <franka_hw/franka_cartesian_command_interface.h>

namespace panda_energy_controller {

class FrankaEnergyShapingController : public controller_interface::MultiInterfaceController<
                                                franka_hw::FrankaModelInterface,
                                                hardware_interface::EffortJointInterface,
                                                franka_hw::FrankaPoseCartesianInterface,
                                                franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  franka_hw::FrankaPoseCartesianInterface* cartesian_pose_interface_;
  std::unique_ptr<franka_hw::FrankaCartesianPoseHandle> cartesian_pose_handle_;
  ros::Duration elapsed_time_;
  std::array<double, 16> initial_pose_{};
  typedef std::vector<double> state_type;

  Eigen::Matrix<double, 7, 1> tau_J_d;  
  Eigen::Matrix<double, 7, 1> q_dot;
  Eigen::Matrix<double, 7, 1> u;

  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

  //double filter_params_{0.005};
  //double nullspace_stiffness_{20.0};
  //double nullspace_stiffness_target_{20.0};
  //const double delta_tau_max_{1.0};
  //Eigen::Matrix<double, 6, 6> cartesian_stiffness_;
  //Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_;
  //Eigen::Matrix<double, 6, 6> cartesian_damping_;
  //Eigen::Matrix<double, 6, 6> cartesian_damping_target_;
  //Eigen::Matrix<double, 7, 1> q_d_nullspace_;
  //Eigen::Vector3d position_d_;
  //Eigen::Quaterniond orientation_d_;
  //Eigen::Vector3d position_d_target_;
  //Eigen::Quaterniond orientation_d_target_;

  static void my_system ( const state_type &s0 , state_type &dsdt , const double t); 

/*
  // Dynamic reconfigure
  std::unique_ptr<dynamic_reconfigure::Server<franka_example_controllers::compliance_paramConfig>>
      dynamic_server_compliance_param_;
  ros::NodeHandle dynamic_reconfigure_compliance_param_node_;
  void complianceParamCallback(franka_example_controllers::compliance_paramConfig& config,
                               uint32_t level);

  // Equilibrium pose subscriber
  ros::Subscriber sub_equilibrium_pose_;
  void equilibriumPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg);
*/
};

}  // namespace panda_energy_controller
