#include <panda_energy_controller/franka_energy_shaping_controller.h>

#include <cmath>
#include <memory>

#include <stdexcept>
#include <string>

#include <boost/type_traits/is_same.hpp>
#include <boost/numeric/odeint/stepper/stepper_categories.hpp>
#include <boost/numeric/odeint/integrate/null_observer.hpp>
#include <boost/numeric/odeint/integrate/detail/integrate_adaptive.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/skewness.hpp>


#include <franka_hw/franka_cartesian_command_interface.h>
#include <hardware_interface/hardware_interface.h>

#include <XmlRpc.h>
#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>


using namespace boost::numeric::odeint;
using namespace boost::accumulators;

typedef std::vector<double> state_type;
typedef runge_kutta4<state_type> rk4;

namespace panda_energy_controller {

bool FrankaEnergyShapingController::init(hardware_interface::RobotHW* robot_hw,
                                               ros::NodeHandle& node_handle) {
  //std::vector<double> cartesian_stiffness_vector;    
  //std::vector<double> cartesian_damping_vector;

  cartesian_pose_interface_ = robot_hw->get<franka_hw::FrankaPoseCartesianInterface>();
  if (cartesian_pose_interface_ == nullptr) {
    ROS_ERROR(
        "FrankaEnergyShapingController: Could not get Cartesian Pose "
        "interface from hardware");
    return false;
  }

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("FrankaEnergyShapingController: Could not get parameter arm_id");
    return false;
  }

  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "FrankaEnergyShapingController: Invalid or no joint_names parameters provided, "
        "aborting controller init!");
    return false;
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "FrankaEnergyShapingController: Error getting model interface from hardware");
    return false;
  }


  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }


  try {
    cartesian_pose_handle_ = std::make_unique<franka_hw::FrankaCartesianPoseHandle>(
        cartesian_pose_interface_->getHandle(arm_id + "_robot"));
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM(
        "FrankaEnergyShapingController: Exception getting Cartesian handle: " << e.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR("FrankaEnergyShapingController: Could not get state interface from hardware");
    return false;
  }

  try {
    auto state_handle = state_interface->getHandle(arm_id + "_robot");

    std::array<double, 7> q_start{{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
    for (size_t i = 0; i < q_start.size(); i++) {
      if (std::abs(state_handle.getRobotState().q_d[i] - q_start[i]) > 0.1) {
        ROS_ERROR_STREAM(
            "FrankaEnergyShapingController: Robot is not in the expected starting position for "
            "running this example. Run `roslaunch franka_example_controllers move_to_start.launch "
            "robot_ip:=<robot-ip> load_gripper:=<has-attached-gripper>` first.");
        return false;
      }
    }
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM(
        "FrankaEnergyShapingController: Exception getting state handle: " << e.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "FrankaEnergyShapingController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "FrankaEnergyShapingController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  return true;
}

void FrankaEnergyShapingController::starting(const ros::Time& /*time*/) {
  
  initial_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_d;   // initial_pose_

  elapsed_time_ = ros::Duration(0.0);
}

void FrankaEnergyShapingController::update(const ros::Time& /*time*/,
                                                 const ros::Duration& period) {

  typedef std::vector<double> state_type;
  typedef runge_kutta4<state_type> rk4;


  // generating cartesian pose
  elapsed_time_ += period;

  double radius = 0.3;
  double angle = M_PI / 4 * (1 - std::cos(M_PI / 5.0 * elapsed_time_.toSec()));
  double delta_x = radius * std::sin(angle);
  double delta_z = radius * (std::cos(angle) - 1);
  std::array<double, 16> H_v0_vec = initial_pose_;   
  H_v0_vec[12] -= delta_x;
  H_v0_vec[14] -= delta_z;

  Eigen::Map<Eigen::Matrix<double, 4, 4>> H_v0(H_v0_vec.data());


  // initial values
  double ko = 20.0;             //rotational stiffness constant
  double kt = 2000.0;            //translational stiffness constant
  double b = 10.0;               //Damping coefficient
  double epsilon = 0.001;      //Minimum energy in tank
  double Emax = 1.0;             //Maximum allowed energy
  double Pmax = 2.0;             //Maximum allowed power
   
  Eigen::MatrixXd I3(3,3);
  I3 << 1,0,0,0,1,0,0,0,1;
  Eigen::MatrixXd I7(7,7);
  I7 << 1,0,0,0,0,0,0,
         0,1,0,0,0,0,0,
         0,0,1,0,0,0,0,
         0,0,0,1,0,0,0,
         0,0,0,0,1,0,0,
         0,0,0,0,0,1,0,
         0,0,0,0,0,0,1;
            
  // initial equations
  Eigen::MatrixXd Bi = b * I7;        // I7 is equivalent to eye(7) where eye refers to identity matrix and 6 refers to size of matrix
  Eigen::MatrixXd Ko = ko * I3;
  Eigen::MatrixXd Kt = kt * I3; 
  Eigen::MatrixXd Kc(3,3);
  Kc << 0,0,0,0,0,0,0,0,0;

  Eigen::MatrixXd Goi = 0.5*Ko.trace()*I3 - Ko;
  Eigen::MatrixXd Gti = 0.5*Kt.trace()*I3 - Kt;          // trace refers to tensor space operator
  Eigen::MatrixXd Gci = 0.5*Kc.trace()*I3 - Kc;

  double gamma = sqrt(2*epsilon);                     // square root

  // Integration parameters
  double t0 = 0.0;
  double t1 = 10.0;
  double d_t = 1.0;

  // get state variables
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
  std::array<double, 49> mass_array = model_handle_->getMass();
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);

  // convert to Eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 7>> Mass(mass_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_dot(robot_state.dq.data());
  //Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(  // NOLINT (readability-identifier-naming)
      //robot_state.tau_J_d.data());

  Eigen::Map<Eigen::Matrix<double, 4, 4>> H_t0(robot_state.O_T_EE.data());
  // H_t0 is current eef transform
  // H_t0 is read as eef current config in frame{t} wrt base frame{0}
  Eigen::MatrixXd R_t0 = H_t0.block(0,0,3,3);
  Eigen::VectorXd p_t0 = H_t0.block(0,3,3,1);

  Eigen::MatrixXd H_0t(4,4);
  H_0t.fill(0);

  H_0t.block(0,0,3,3) = R_t0.transpose();
  H_0t.block(0,3,3,1) = -R_t0.transpose()*p_t0;
  H_0t.block(3,3,1,1) << 1.0;  
  // H_v0 is desired eef transform
  // H_v0 is read as eef desired config in frame{v} wrt base frame{0}

  Eigen::MatrixXd H_vt = H_0t*H_v0;             
    
  // extracting rotaional Rvt and translational pvt part from Hvt for further calculating wrench 
  Eigen::MatrixXd R_vt = H_vt.block(0,0,3,3);

  Eigen::VectorXd p_vt = H_vt.block(0,3,3,1);

  Eigen::MatrixXd R_tv = R_vt.transpose(); 

  // converting position vector to skew-symmetric matrix
  //MatrixXd tilde_p_vt = [0 -p_vt(3,1) p_vt(2,1);p_vt(3,1) 0 -p_vt(1,1);-p_vt(2,1) p_vt(1,1) 0];
  Eigen::MatrixXd tilde_p_vt(3,3); 
  tilde_p_vt << 0,-p_vt(2,0),p_vt(1,0),p_vt(2,0),0,-p_vt(0,0),-p_vt(1,0),p_vt(0,0),0;
            
                        
  // Safety layer (PE, KE and total Energy of the system)
  double Vti = (-1/4*(tilde_p_vt*Gti*tilde_p_vt).trace())-(1/4*(tilde_p_vt*R_vt*Gti*R_tv*tilde_p_vt).trace());
  double Voi = -((Goi*R_vt).trace());
  double Vci = ((Gci*R_tv*tilde_p_vt).trace());


  double V_pi = Vti + Voi + Vci;         // initial potential energy

  double T_k = 1/2*q_dot.transpose() * Mass * q_dot;        // transpose of qdot x M x qdot

  double E_tot = T_k + V_pi;               // initial energy of the system
     
  double lamb_da;

  if (E_tot > Emax)  
      lamb_da = (Emax - T_k)/ V_pi;
  else
      lamb_da = 1;
  return;
  

  // calculation of new co-stiffness matrices and corresponding potential energy
  Eigen::MatrixXd Go = lamb_da * Goi;           // new co-stiffness matrices
  Eigen::MatrixXd Gt = lamb_da * Gti;
  Eigen::MatrixXd Gc = lamb_da * Gci;
     
  double Vt = (-1/4*(tilde_p_vt*Gt*tilde_p_vt).trace())-(1/4*(tilde_p_vt*R_vt*Gt*R_tv*tilde_p_vt).trace());
  double Vo = -((Go*R_vt).trace());
  double Vc = ((Gc*R_tv*tilde_p_vt).trace());


  double V_p = Vt + Vo + Vc;         // potential energy
  E_tot = T_k + lamb_da*V_pi;            // total energy of the system
                

  // Rotational part of wrench
  Eigen::MatrixXd tilde_m_t = - 2* 1/2*(Go*R_vt-(Go*R_vt).transpose())-1/2*(Gt*R_tv*tilde_p_vt*tilde_p_vt*R_vt-    (Gt*R_tv*tilde_p_vt*tilde_p_vt*R_vt).transpose())-2*1/2*(Gc*tilde_p_vt*R_vt-(Gc*tilde_p_vt*R_vt).transpose());
  Eigen::VectorXd m_t(3,1); 
  //m_t = [tilde_m_t(3,2); tilde_m_t(1,3); tilde_m_t(2,1)];   in matlab
  m_t << tilde_m_t(2,1),tilde_m_t(0,2),tilde_m_t(1,0);

  // Translational part of wrench
  Eigen::MatrixXd tilde_f_t = -R_tv * 1/2*(Gt*tilde_p_vt- (Gt*tilde_p_vt).transpose())*R_vt-1/2*(Gt*R_tv*tilde_p_vt*R_vt-(Gt*R_tv*tilde_p_vt*R_vt).transpose())-2*1/2*(Gc*R_vt-(Gc*R_vt).transpose());
  Eigen::VectorXd f_t(3,1);
  //f_t = [tilde_f_t(3,2); tilde_f_t(1,3); tilde_f_t(2,1)];   in matlab
  f_t << tilde_f_t(2,1),tilde_f_t(0,2),tilde_f_t(1,0);
            
  Eigen::VectorXd Wt(6,1);           // wrench vector initialization
  Wt << 0,0,0,0,0,0; 
    
  Wt.block(0,0,3,1) = f_t;          // Wsn0(1:3,1) = t ... t represented with small m here;
  Wt.block(3,0,3,1) = m_t;          // Wsn0(4:6,1) = f; 

  Eigen::MatrixXd R_0t = H_0t.block(0,0,3,3);
  //p_0t = H_0t.block(0,3,3,1);

  //MatrixXd tilde_p_0t(3,3);
  //tilde_p_0t << 0,-p_0t(2,0),p_0t(1,0),p_0t(2,0),0,-p_0t(0,0),-p_0t(1,0),p_0t(0,0),0;
  Eigen::MatrixXd tilde_p_t0(3,3); 
  tilde_p_t0 << 0,-p_t0(2,0),p_t0(1,0),p_t0(2,0),0,-p_t0(0,0),-p_t0(1,0),p_t0(0,0),0;

  Eigen::MatrixXd AdjT_H_t0(6,6);
  AdjT_H_t0.fill(0);

  AdjT_H_t0.block(0,0,3,3) = R_0t;
  AdjT_H_t0.block(0,3,3,3) = -R_0t*tilde_p_t0;
  AdjT_H_t0.block(3,3,3,3) = R_0t;

  Eigen::VectorXd W0(6,1);           // wrench vector transformation of 
  W0 = -AdjT_H_t0* Wt;     

  // Power of the System 
     
  double P_c = ((jacobian.transpose() * W0 - Bi * q_dot).transpose()) * q_dot ;  // Initial power of the controller
     
  double beta;
  if (P_c > Pmax) 
        beta = (((((jacobian.transpose()) * W0).transpose())*q_dot) - Pmax)/ ((q_dot.transpose())*Bi*q_dot);
  else
        beta = 1; 
  return;

  // New joint damping matrix using scaling parameter beta
  Eigen::MatrixXd B = beta * Bi;

  // New power of the controller using new joint damping matrix
  Eigen::VectorXd tau_cmd(7);
  tau_cmd = (jacobian.transpose()) * W0 - B * q_dot;     // Controller force

  P_c = tau_cmd.transpose() * q_dot ;      // Power of the controller

  Eigen::VectorXd s0(7);
  Eigen::VectorXd tauc(7);
  Eigen::VectorXd H(7);
  Eigen::VectorXd u_vec(7);
  Eigen::VectorXd u(7);
 
  s0[0] = sqrt(3*10);
  s0[1] = sqrt(3*10);
  s0[2] = sqrt(3*10);
  s0[3] = sqrt(3*10);
  s0[4] = sqrt(3*10);
  s0[5] = sqrt(3*10);
  s0[6] = sqrt(3*10);

  state_type integrate_adaptive();

  s0[0] = s0[0];
  s0[1] = s0[1];     
  s0[2] = s0[2];
  s0[3] = s0[3];
  s0[4] = s0[4];
  s0[5] = s0[5];
  s0[6] = s0[6];


  // Controller torques for each joint
  tauc[0] = tau_cmd[0];
  tauc[1] = tau_cmd[1];
  tauc[2] = tau_cmd[2];
  tauc[3] = tau_cmd[3];
  tauc[4] = tau_cmd[4];
  tauc[5] = tau_cmd[5];
  tauc[6] = tau_cmd[6];
          

  // Energy in each tank, an energy tank is modeled as a spring with const stiffness k = 1
  // connected to robot through a transmission unit 
  // so H = 0.5*k*s^2 ==> H = 0.5*s^2 
  H[0] = 0.5*s0[0]*s0[0];
  H[1] = 0.5*s0[1]*s0[1];
  H[2] = 0.5*s0[2]*s0[2];
  H[3] = 0.5*s0[3]*s0[3];
  H[4] = 0.5*s0[4]*s0[4];
  H[5] = 0.5*s0[5]*s0[5];
  H[6] = 0.5*s0[6]*s0[6];           

  H << H[0] + H[1] + H[2] + H[3] + H[4] + H[5] + H[6];     // Total energy in tanks
          
        
  // transmission unit allows power flow from controller to robot and it is regulated by ratio u
  // here u is transmission variable
  if ((H[0] > epsilon))    
         u_vec[0] = -tauc[0]/s0[0];
  else
         u_vec[0] = (-tauc[0]/gamma*gamma)*s0[0];
  return;

  if ((H[1] > epsilon))   
        u_vec[1] = -tauc[1]/s0[1];
  else
        u_vec[1] = ((-tauc[1])/gamma*gamma)*s0[1];
  return;

  if ((H[2] > epsilon))  
        u_vec[2] = -tauc[2]/s0[2];
  else
        u_vec[2] = (-tauc[2]/gamma*gamma)*s0[2];
  return;

  if ((H[3] > epsilon))   
       u_vec[3] = -tauc[3]/s0[3];
  else
      u_vec[3] = (-tauc[3]/gamma*gamma)*s0[3];
  return;

  if ((H[4] > epsilon))   
      u_vec[4] = -tauc[4]/s0[4];
  else
      u_vec[4] = (-tauc[4]/gamma*gamma)*s0[4]; 
  return;

  if ((H[5] > epsilon))  
     u_vec[5] = -tauc[5]/s0[5];
  else
     u_vec[5] = (-tauc[5]/gamma*gamma)*s0[5];
  return;

  if ((H[6] > epsilon)) 
     u_vec[6] = -tauc[6]/s0[6];
  else
    u_vec[6] = (-tauc[6]/gamma*gamma)*s0[6];
  return;

  //VectorXd W0(6,1);
  u << u_vec[0],u_vec[1],u_vec[2],u_vec[3],u_vec[4],u_vec[5],u_vec[6];
  H << H[0],H[1],H[2],H[3],H[4],H[5],H[6]; 
  s0 << s0[0] + s0[1] + s0[2] + s0[3] + s0[4] + s0[5] + s0[6]; 
  
  Eigen::VectorXd tau_J_d(7);  
  tau_J_d << -u * s0;   

  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(tau_J_d(i));
  }

}

template< class Stepper , class System , class State , class Time > 
       
state_type integrate_adaptive( Stepper stepper_type , System my_system , State s0 , Time t0 , Time t1 , Time d_t , null_observer() )
{
  return integrate_adaptive( rk4() , my_system , s0 , t0 , t1 , d_t , null_observer() );
}

void my_system ( const state_type &s0 , state_type &dsdt , const double t) 
{  
  
  Eigen::VectorXd q_dot(7);
  Eigen::VectorXd u(7);
  for (size_t i=0; i<7; ++i)
  {
     dsdt[i] = q_dot[i]*u[i];
  }  
                    
}

}  // namespace panda_energy_controller

PLUGINLIB_EXPORT_CLASS(panda_energy_controller::FrankaEnergyShapingController,
                       controller_interface::ControllerBase)
