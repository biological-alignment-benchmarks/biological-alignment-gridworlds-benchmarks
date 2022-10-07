import gym


def cleanup_gym_envs(keyword='savanna'):
    registry_keys = list(gym.envs.registration.registry.keys())
    for env in registry_keys:
        # print(env)
        if keyword in env:
            print(f"Removing {env} from registry")
            del gym.envs.registration.registry[env]