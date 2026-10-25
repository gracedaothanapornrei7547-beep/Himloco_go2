// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/devices/keyboard/keyboard.h"

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    auto cfg = env->cfg["observations"]["joint_pos_rel"];
    if(cfg["params"]["asset_cfg"]["joint_ids"].IsDefined())
    {
        auto joint_ids = cfg["params"]["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]] - asset->data.default_joint_pos[joint_ids[i]];
        }
    }
    else
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    auto & data = asset->data.joint_vel;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

REGISTER_OBSERVATION(velocity_commands)
{
    std::vector<float> obs(3);
    auto & joystick = env->robot->data.joystick;

    auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    obs[0] = std::clamp(joystick->ly(), cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    obs[1] = std::clamp(-joystick->lx(), cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    obs[2] = std::clamp(-joystick->rx(), cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());

    return obs;
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = env->cfg["observations"]["gait_phase"]["params"]["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);
    return obs;
}

REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    static Keyboard keyboard;
    static std::vector<float> velocity(3, 0.0f);
    
    keyboard.update();
    
    auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];
    float max_lin_vel_x = cfg["lin_vel_x"][1].as<float>();
    float min_lin_vel_x = cfg["lin_vel_x"][0].as<float>();
    float max_lin_vel_y = cfg["lin_vel_y"][1].as<float>();
    float min_lin_vel_y = cfg["lin_vel_y"][0].as<float>();
    float max_ang_vel_z = cfg["ang_vel_z"][1].as<float>();
    float min_ang_vel_z = cfg["ang_vel_z"][0].as<float>();
    
    float speed_step = 0.1f;
    
    std::string key = keyboard.key();
    
    if(key == "w" || key == "up") {
        velocity[0] = std::min(velocity[0] + speed_step, max_lin_vel_x);
    }
    else if(key == "s" || key == "down") {
        velocity[0] = std::max(velocity[0] - speed_step, min_lin_vel_x);
    }
    
    if(key == "a" || key == "left") {
        velocity[1] = std::min(velocity[1] + speed_step, max_lin_vel_y);
    }
    else if(key == "d" || key == "right") {
        velocity[1] = std::max(velocity[1] - speed_step, min_lin_vel_y);
    }
    
    if(key == "q") {
        velocity[2] = std::min(velocity[2] + speed_step, max_ang_vel_z);
    }
    else if(key == "e") {
        velocity[2] = std::max(velocity[2] - speed_step, min_ang_vel_z);
    }
    
    if(key == " ") {
        velocity[0] = 0.0f;
        velocity[1] = 0.0f;
        velocity[2] = 0.0f;
    }
    
    velocity[0] = std::clamp(velocity[0], min_lin_vel_x, max_lin_vel_x);
    velocity[1] = std::clamp(velocity[1], min_lin_vel_y, max_lin_vel_y);
    velocity[2] = std::clamp(velocity[2], min_ang_vel_z, max_ang_vel_z);
    
    return velocity;
}
}
}