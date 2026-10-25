// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <unordered_set>
#include <deque>
#include "isaaclab/manager/manager_term_cfg.h"
#include <iostream>
#include <spdlog/spdlog.h>

namespace isaaclab
{

using ObsMap = std::map<std::string, ObsFunc>;

inline ObsMap& observations_map() {
    static ObsMap instance;
    return instance;
}

#define REGISTER_OBSERVATION(name) \
    inline std::vector<float> name(ManagerBasedRLEnv* env); \
    inline struct name##_registrar { \
        name##_registrar() { observations_map()[#name] = name; } \
    } name##_registrar_instance; \
    inline std::vector<float> name(ManagerBasedRLEnv* env)


class ObservationManager
{
public:
    ObservationManager(YAML::Node cfg, ManagerBasedRLEnv* env)
    :cfg(cfg), env(env)
    {
        // Read global history length
        history_length = cfg["history_length"] ? cfg["history_length"].as<int>() : 1;
        _prapare_terms();
        _initialize_history_buffer();
    }

    void reset()
    {
        for(auto & term : obs_term_cfgs)
        {
            term.reset(term.func(this->env));
        }
        // Clear and reinitialize history buffer with zeros
        // Buffer stores [current, history1, ..., historyN-1] in flattened form
        obs_history_buffer.assign(num_one_step_obs * history_length, 0.0f);
    }

    std::vector<float> compute()
    {
        // Step 1: Compute current observations (without history in individual terms)
        std::vector<float> current_obs;
        for(auto & term : obs_term_cfgs)
        {
            auto term_obs = term.func(this->env);  // Get raw observation
            
            // Apply scaling
            if (term.scale.size() > 0) {
                for (size_t i = 0; i < term_obs.size(); ++i) {
                    term_obs[i] *= term.scale[i];
                }
            }
            
            // Apply clipping
            if (term.clip.size() == 2) {
                for (auto& val : term_obs) {
                    val = std::max(term.clip[0], std::min(val, term.clip[1]));
                }
            }
            
            current_obs.insert(current_obs.end(), term_obs.begin(), term_obs.end());
        }
        
        // Step 2: Update flattened history buffer
        // Shift old data: [current, hist1, hist2, ..., histN-1] -> [_, hist1, hist2, ..., histN-1]
        // Then insert new current at front
        if (history_length > 1) {
            std::copy(obs_history_buffer.begin(), 
                     obs_history_buffer.begin() + num_one_step_obs * (history_length - 1),
                     obs_history_buffer.begin() + num_one_step_obs);
        }
        std::copy(current_obs.begin(), current_obs.end(), obs_history_buffer.begin());
        
        return obs_history_buffer;
    }

    YAML::Node cfg;
    ManagerBasedRLEnv* env;
    int history_length;
    std::vector<float> obs_history_buffer;  // [current, prev, prev-1, ...]
    size_t num_one_step_obs;  // Dimension of one step observation
    
    // Get observation dimension (sum of all term dimensions)
    size_t get_obs_dim() const
    {
        return num_one_step_obs * history_length;
    }
    
private:
    void _initialize_history_buffer()
    {
        // Get the observation dimension from terms
        num_one_step_obs = 0;
        for (const auto& term_cfg : obs_term_cfgs) {
            auto sample_obs = term_cfg.func(env);
            num_one_step_obs += sample_obs.size();
        }
        
        // Pre-fill flattened buffer with zeros
        obs_history_buffer.assign(num_one_step_obs * history_length, 0.0f);

        spdlog::info("History buffer initialized: history_length={}, one_step_dim={}, total_dim={}", 
                     history_length, num_one_step_obs, get_obs_dim());
    }

    void _prapare_terms()
    {
        // Get observations config from the full config
        auto obs_cfg = cfg["observations"];
        
        if (!obs_cfg) {
            throw std::runtime_error("No 'observations' key found in config");
        }
        
        for(auto it = obs_cfg.begin(); it != obs_cfg.end(); ++it)
        {
            auto term_yaml_cfg = it->second;
            ObservationTermCfg term_cfg;
            // Don't set history_length for individual terms - we handle it globally
            term_cfg.history_length = 1;

            auto term_name = it->first.as<std::string>();
            if(observations_map()[term_name] == nullptr)
            {
                throw std::runtime_error("Observation term '" + term_name + "' is not registered.");
            }
            term_cfg.func = observations_map()[term_name];   

            auto obs = term_cfg.func(this->env);
            term_cfg.reset(obs);
            term_cfg.scale = term_yaml_cfg["scale"].as<std::vector<float>>();
            if(!term_yaml_cfg["clip"].IsNull()) {
                term_cfg.clip = term_yaml_cfg["clip"].as<std::vector<float>>();
            }

            this->obs_term_cfgs.push_back(term_cfg);
            spdlog::info("Successfully loaded observation term: {}", term_name);
            
        }
    }
    
    std::vector<ObservationTermCfg> obs_term_cfgs;
};

};