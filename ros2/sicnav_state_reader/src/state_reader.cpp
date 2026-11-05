#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include <gz/plugin/Register.hh>

#include <gz/sim/System.hh>
#include <gz/sim/EntityComponentManager.hh>

#include <gz/sim/components/Actor.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Pose.hh>

using namespace gz;
using namespace gz::sim;


struct PrevState
{
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double time = 0.0;
  bool valid = false;
};


class SicnavStateReader :
  public System,
  public ISystemConfigure,
  public ISystemPreUpdate
{
public:

  void Configure(
    const Entity&,
    const std::shared_ptr<const sdf::Element>&,
    EntityComponentManager&,
    EventManager&) override
  {
    if (!rclcpp::ok())
    {
      rclcpp::init(0, nullptr);
    }

    _node = std::make_shared<rclcpp::Node>(
      "sicnav_gazebo_state_reader");

    _human_pub =
      _node->create_publisher<std_msgs::msg::String>(
        "/sicnav/human_states",
        10);

    _robot_pub =
      _node->create_publisher<std_msgs::msg::String>(
        "/sicnav/robot_state",
        10);

    _frame_pub =
      _node->create_publisher<std_msgs::msg::String>(
        "/sicnav/state_frame",
        10);

    RCLCPP_INFO(
      _node->get_logger(),
      "SICNav Gazebo state reader loaded");
  }


  void PreUpdate(
    const UpdateInfo& info,
    EntityComponentManager& ecm) override
  {
    rclcpp::spin_some(_node);

    if (info.paused)
      return;

    const double sim_time =
      std::chrono::duration<double>(
        info.simTime).count();

    // JMID input용 4 Hz sampling
    if (sim_time - _last_publish_time < 0.25)
      return;

    _last_publish_time = sim_time;

    _last_humans_json.clear();
    _last_robot_json.clear();

    publish_humans(ecm, sim_time);
    publish_robot(ecm, sim_time);
    publish_state_frame(sim_time);
  }


private:

  static bool is_agent_name(
    const std::string& name)
  {
    if (name.rfind("agent", 0) != 0)
      return false;

    if (name.size() <= 5)
      return false;

    for (size_t i = 5; i < name.size(); ++i)
    {
      if (!std::isdigit(name[i]))
        return false;
    }

    return true;
  }


  static double wrap_angle(double a)
  {
    while (a > M_PI)
      a -= 2.0 * M_PI;

    while (a < -M_PI)
      a += 2.0 * M_PI;

    return a;
  }


  void publish_humans(
    EntityComponentManager& ecm,
    double sim_time)
  {
    struct Human
    {
      int id;
      double x;
      double y;
      double vx;
      double vy;
    };

    std::vector<Human> humans;


    ecm.Each<
      components::Actor,
      components::Name,
      components::TrajectoryPose>(
      [&](const Entity&,
          const components::Actor*,
          const components::Name* name_comp,
          const components::TrajectoryPose* traj_comp)
          -> bool
      {
        const std::string name =
          name_comp->Data();

        if (!is_agent_name(name))
          return true;

        const auto& pose =
          traj_comp->Data();

        const double x =
          pose.Pos().X();

        const double y =
          pose.Pos().Y();

        double vx = 0.0;
        double vy = 0.0;

        auto& prev = _previous[name];

        if (prev.valid)
        {
          const double dt =
            sim_time - prev.time;

          if (dt > 1e-5)
          {
            vx = (x - prev.x) / dt;
            vy = (y - prev.y) / dt;
          }
        }

        prev.x = x;
        prev.y = y;
        prev.time = sim_time;
        prev.valid = true;

        const int id =
          std::stoi(name.substr(5));

        humans.push_back({
          id,
          x,
          y,
          vx,
          vy
        });

        return true;
      });


    std::sort(
      humans.begin(),
      humans.end(),
      [](const Human& a, const Human& b)
      {
        return a.id < b.id;
      });


    std::ostringstream ss;

    ss << std::fixed
       << std::setprecision(4);

    ss << "[";

    for (size_t i = 0; i < humans.size(); ++i)
    {
      const auto& h = humans[i];

      ss
        << "{"
        << "\"id\":" << h.id << ","
        << "\"x\":" << h.x << ","
        << "\"y\":" << h.y << ","
        << "\"vx\":" << h.vx << ","
        << "\"vy\":" << h.vy
        << "}";

      if (i + 1 < humans.size())
        ss << ",";
    }

    ss << "]";


    _last_humans_json = ss.str();

    std_msgs::msg::String msg;
    msg.data = _last_humans_json;

    _human_pub->publish(msg);


    if (!humans.empty())
    {
      RCLCPP_INFO_THROTTLE(
        _node->get_logger(),
        *_node->get_clock(),
        2000,
        "Detected %zu moving crowd agents",
        humans.size());
    }
  }


  void publish_robot(
    EntityComponentManager& ecm,
    double sim_time)
  {
    bool found = false;

    ecm.Each<
      components::Model,
      components::Name,
      components::Pose>(
      [&](const Entity&,
          const components::Model*,
          const components::Name* name_comp,
          const components::Pose* pose_comp)
          -> bool
      {
        const std::string name =
          name_comp->Data();

        if (name != "turtlebot4")
          return true;

        const auto& pose =
          pose_comp->Data();

        const double x =
          pose.Pos().X();

        const double y =
          pose.Pos().Y();

        const double yaw =
          pose.Rot().Yaw();

        double vx = 0.0;
        double vy = 0.0;
        double omega = 0.0;

        auto& prev =
          _previous[name];

        if (prev.valid)
        {
          const double dt =
            sim_time - prev.time;

          if (dt > 1e-5)
          {
            vx = (x - prev.x) / dt;
            vy = (y - prev.y) / dt;

            omega =
              wrap_angle(
                yaw - prev.yaw) / dt;
          }
        }

        const double v =
          std::cos(yaw) * vx
          + std::sin(yaw) * vy;


        prev.x = x;
        prev.y = y;
        prev.yaw = yaw;
        prev.time = sim_time;
        prev.valid = true;


        std::ostringstream ss;

        ss << std::fixed
           << std::setprecision(4);

        ss
          << "{"
          << "\"x\":" << x << ","
          << "\"y\":" << y << ","
          << "\"yaw\":" << yaw << ","
          << "\"v\":" << v << ","
          << "\"omega\":" << omega
          << "}";


        _last_robot_json = ss.str();

        std_msgs::msg::String msg;
        msg.data = _last_robot_json;

        _robot_pub->publish(msg);

        found = true;

        return false;
      });


    if (!found)
    {
      RCLCPP_WARN_THROTTLE(
        _node->get_logger(),
        *_node->get_clock(),
        3000,
        "turtlebot4 not found");
    }
  }


  void publish_state_frame(double sim_time)
  {
    if (_last_humans_json.empty() ||
        _last_robot_json.empty())
    {
      return;
    }

    std::ostringstream ss;

    ss << std::fixed
       << std::setprecision(6);

    ss
      << "{"
      << "\"sim_time\":" << sim_time << ","
      << "\"humans\":" << _last_humans_json << ","
      << "\"robot\":" << _last_robot_json
      << "}";

    std_msgs::msg::String msg;
    msg.data = ss.str();

    _frame_pub->publish(msg);
  }


private:

  rclcpp::Node::SharedPtr _node;

  rclcpp::Publisher<
    std_msgs::msg::String>::SharedPtr
    _human_pub;

  rclcpp::Publisher<
    std_msgs::msg::String>::SharedPtr
    _robot_pub;

  rclcpp::Publisher<
    std_msgs::msg::String>::SharedPtr
    _frame_pub;

  std::string _last_humans_json;
  std::string _last_robot_json;

  std::unordered_map<
    std::string,
    PrevState> _previous;

  double _last_publish_time = -1.0;
};


GZ_ADD_PLUGIN(
  SicnavStateReader,
  System,
  SicnavStateReader::ISystemConfigure,
  SicnavStateReader::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(
  SicnavStateReader,
  "sicnav_state_reader")
