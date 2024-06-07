import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

# 设置 NuScenes 数据集的路径和版本
# nusc_1 = NuScenes(version='v1.0-mini', dataroot='../../data/nuscenes/nuscenes', verbose=True)
nusc = NuScenes(version='v1.0-trainval', dataroot='../../data/nuscenes/nuscenes', verbose=True)

# 读取 tokens 文件
tokens_file_path = 'train.txt'
with open(tokens_file_path, 'r') as file:
    tokens = file.readlines()

# mini 数据集的场景
mini_scenes = set(create_splits_scenes()['mini_val'])

# 输出文件
output_file_path = 'mini_val.txt'
with open(output_file_path, 'w') as out_file:
    for token in tokens:
        token = token.strip()
        # 查询该 token 的 scene
        sample = nusc.get('sample', token)
        scene_token = sample['scene_token']
        scene = nusc.get('scene', scene_token)
        scene_name = scene['name']

        # 检查 scene 是否属于 mini 数据集
        if scene_name in mini_scenes:
            out_file.write(f"{token}\n")

print("筛选完成，mini 数据集的 tokens 已保存到 mini_tokens.txt 文件中。")
