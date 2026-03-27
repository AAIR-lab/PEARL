def get_env_builder(env_name): 

    if env_name == "office": 
        from environments.envs.office_param_actions import OfficeParamActionsEnv
        return OfficeParamActionsEnv
    
    elif env_name == "logistics":
        from environments.envs.logistics_param_actions import MulticityParamActionsEnv
        return MulticityParamActionsEnv
    
    elif env_name == "pinball": 
        from environments.envs.pinball_param_actions import PinballEnv
        return PinballEnv    

    elif env_name == "goal":  
        from environments.envs.goal_param_actions import GoalWrapperEnv
        return GoalWrapperEnv


