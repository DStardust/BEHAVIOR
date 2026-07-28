import json
import numpy as np
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject

gm.USE_GPU_DYNAMICS = True 
gm.ENABLE_TRANSITION_RULES = True 

def instantiate_deltasg_env(json_path, env_index=0):
    """
    根据 DeltaSG JSON 数据在 OmniGibson 中实例化 3D 场景
    """
    # 1. 加载 dataset.json 数据
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    env_data = dataset["task_environments"][env_index]
    print(f"Loading Environment: {env_data['env_id']} - {env_data['task']['instruction']}")

    # 2. 构建基础 Config
    scene_model = env_data["base_scene"]["scene_model"]
    robot_model = env_data["robot"]["model"].capitalize()
    robot_name = env_data["robot"]["robot_id"]
    robot_pos = env_data["robot"]["pose"]["position"]
    robot_ori = env_data["robot"]["pose"]["orientation_xyzw"]

    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
        },
        "robots": [
            {
                "type": robot_model,
                "name": robot_name,
                "position": robot_pos,
                "orientation": robot_ori,
            }
        ],
        # 新增一个空的 objects 列表，用于存放我们自定义的任务物品
        "objects": [] 
    }

    # 3. 将 added_objects 提前打包注入到 Config 中 (声明式加载)
    print(f"Injecting {len(env_data['added_objects'])} added objects into config...")
    for obj_data in env_data["added_objects"]:
        cfg["objects"].append({
            "type": "DatasetObject",
            "name": obj_data["object_name"],
            "category": obj_data["category"],
            "model": obj_data["model"],
            "position": obj_data["pose"]["position"],
            "orientation": obj_data["pose"]["orientation_xyzw"],
            # "kinematic_only": True,
        })
        
    # for obj_data in cfg["objects"]:
    #     obj_data["kinematic_only"] = True
    
    # 4. 启动 OmniGibson 仿真环境 (底层将一次性完美分配所有 GPU 张量)
    print("Initializing OmniGibson Environment (Scene + Robot + Objects)...")
    env = og.Environment(configs=cfg)

    # 5. 仿真预热 (Warm-up)
    # 此时场景、机器人、物品都已经就位且张量完好，跑空 step 消除初始接触应力
    print("Warming up the physics engine for 50 steps...")
    robot = env.robots[0]
    dummy_action = np.zeros(robot.action_dim)
    
    for _ in range(50):
        env.step(dummy_action)

    print("Environment Instantiation Complete! Ready for Expert Solver.")
    return env

if __name__ == "__main__":
    # 指向你的 json 文件路径
    JSON_FILE_PATH = "demo_data/dataset.json"
    
    # 实例化第 1 个场景 (做饼干任务 - online_env_a_0000)
    sim_env = instantiate_deltasg_env(JSON_FILE_PATH, env_index=0)
    
    # 如果想测试其他环境，修改 env_index 即可，例如:
    # 摆放调料: env_index=1
    # 整理美术用品: env_index=2
    # 准备早餐: env_index=3
    # 买杂货: env_index=4

    # 保持仿真器开启以供观察（或接入你的专家求解器进行 Navigation / Interaction）
    try:
        while True:
            sim_env.step(np.zeros(sim_env.robots[0].action_dim))
    except KeyboardInterrupt:
        print("Shutting down simulation...")
        og.shutdown()