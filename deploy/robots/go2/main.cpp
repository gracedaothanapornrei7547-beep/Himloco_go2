#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "isaaclab/devices/keyboard/keyboard.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    // Initialize keyboard input
    FSMState::keyboard = std::make_shared<Keyboard>();

    // Get joystick reference
    auto& joy = FSMState::lowstate->joystick;

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(new State_Passive(FSMMode::Passive));
    
    // Transition from Passive to FixStand: Keyboard [Z] OR Joystick [L2 + A]
    fsm->states.back()->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ 
                bool keyboard_input = FSMState::keyboard->key() == "z" && FSMState::keyboard->on_pressed;
                bool joystick_input = joy.LT.pressed && joy.A.on_pressed;
                return keyboard_input || joystick_input;
            }, 
            (int)FSMMode::FixStand
        )
    );
    
    fsm->add(new State_FixStand(FSMMode::FixStand));
    
    // Transition from FixStand to Velocity: Keyboard [X] OR Joystick [Start]
    fsm->states.back()->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ 
                bool keyboard_input = FSMState::keyboard->key() == "x" && FSMState::keyboard->on_pressed;
                bool joystick_input = joy.start.on_pressed;
                return keyboard_input || joystick_input;
            }, 
            FSMMode::Velocity
        )
    );
    
    // Transition from Velocity back to Passive: Keyboard [C] OR Joystick [Back]
    fsm->add(new State_RLBase(FSMMode::Velocity, "Velocity"));
    fsm->states.back()->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ 
                bool keyboard_input = FSMState::keyboard->key() == "c" && FSMState::keyboard->on_pressed;
                bool joystick_input = joy.back.on_pressed;
                return keyboard_input || joystick_input;
            }, 
            (int)FSMMode::Passive
        )
    );

    std::cout << "\n=== Control Methods ===" << std::endl;
    std::cout << "\n--- Keyboard Control ---" << std::endl;
    std::cout << "  [Z]     - Enter FixStand mode" << std::endl;
    std::cout << "  [X]     - Start RL control" << std::endl;
    std::cout << "  [C]     - Stop RL control (return to Passive)" << std::endl;
    
    std::cout << "\n--- Joystick Control ---" << std::endl;
    std::cout << "  [L2 + A]    - Enter FixStand mode" << std::endl;
    std::cout << "  [Start]     - Start RL control" << std::endl;
    std::cout << "  [Back]      - Stop RL control (return to Passive)" << std::endl;
    std::cout << "=======================\n" << std::endl;

    while (true)
    {
        FSMState::keyboard->update();
        sleep(1);
    }
    
    return 0;
}

