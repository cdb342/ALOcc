
# alocc:
# self.ttt_layer_func_after_his -> self.scene_his_func_after_vox_his
# self.ttt_q -> self.scene_his_q
# self.ttt_k -> self.scene_his_k
# self.ttt_v -> self.scene_his_v
# self.ttt_learned_lr -> self.scene_his_learned_lr
# self.ttt_learned_lr_bias -> self.scene_his_learned_lr_bias
# self.ttt_iter -> self.scene_his_iter
# self.ttt_warm_up_iter -> self.scene_his_warm_up_iter
# self.ttt_warm_up_start -> self.scene_his_warm_up_start
# self.ttt_base_eta -> self.scene_his_base_eta
# self.ttt_learnable_weight -> self.scene_his_learnable_weight
# depth_net -> geomery_net
# forward_projection -> view_transformer


# depth_net.depth2occ_hidden_weight_net -> depth_net.depth2occ_inter_weight_net


import torch
import os

# 输入原始文件路径和输出文件路径
# ckpt_path = "/home/yc37439/code/alocc2/ww/gdfusion/alocc_3d_256x704_bevdet_preatrain_gt_depth_gdfusion/iter_96060_ema.pth"
# out_path = "/home/yc37439/code/alocc2/ww/gdfusion/alocc_3d_256x704_bevdet_preatrain_gt_depth_gdfusion/iter_96060_ema_change.pth"
# ckpt_path = "/home/yc37439/code/alocc2/ww/alocc_2d_256x704_bevdet_preatrain/iter_96060_ema.pth"
# out_path = "/home/yc37439/code/alocc2/ww/alocc_2d_256x704_bevdet_preatrain/iter_96060_ema_change.pth"

# ckpt_path="/home/yc37439/code/alocc2/ww/alocc/aloccflow_2d_256x704_bevdet_preatrain/iter_144090_ema.pth"
# out_path="/home/yc37439/code/alocc2/ww/alocc/aloccflow_2d_256x704_bevdet_preatrain/iter_144090_ema_change.pth"

# ckpt_path='/home/yc37439/code/alocc2/ww/causalocc/alocc_3d_256x704_bevdet_preatrain_causalocc/iter_96060_ema.pth'
# out_path='/home/yc37439/code/alocc2/ww/causalocc/alocc_3d_256x704_bevdet_preatrain_causalocc/iter_96060_ema_change.pth'

# ckpt_path='/home/yc37439/code/alocc2/work_dir/alocc/alocc_3d_256x704_bevdet_preatrain/iter_96060_ema.pth'
# out_path='/home/yc37439/code/alocc2/work_dir2/'+ckpt_path.split('/')[-2]+'.pth'

# ckpt_path='/home/yc37439/code/alocc2/work_dir/alocc/alocc_3d_256x704_bevdet_preatrain_gt_depth/iter_96060_ema.pth'

# ckpt_path='/home/yc37439/code/alocc2/work_dir/alocc/alocc_2d_256x704_bevdet_preatrain/iter_96060_ema.pth'
# ckpt_path='/home/yc37439/code/alocc2/work_dir/causalocc/alocc_3d_256x704_bevdet_preatrain_causalocc/iter_96060_ema.pth'

# ckpt_path='/home/yc37439/code/alocc2/work_dir/gdfusion/alocc_3d_256x704_bevdet_preatrain_gdfusion/iter_96060_ema.pth'

# ckpt_path='/home/yc37439/code/alocc2/work_dir/gdfusion/alocc_3d_256x704_bevdet_preatrain_gt_depth_gdfusion/iter_96060_ema.pth'

# ckpt_path='/home/yc37439/code/alocc2/work_dir/fbocc/iter_80040_ema.pth'

# ckpt_path='/home/yc37439/code/alocc2/work_dir/gdfusion/fbocc_gdfusion/iter_80040_ema.pth'

# ckpt_path='/home/yc37439/code/alocc2/work_dir/alocc/alocc_3d_256x704_bevdet_preatrain_wo_mask/iter_96060_ema.pth'

# ckpt_path='/home/yc37439/code/alocc2/work_dir/alocc/alocc_2d_256x704_bevdet_preatrain_wo_mask/iter_96060_ema.pth'

# ckpt_path='/home/yc37439/code/alocc2/work_dir/bevdetocc/iter_96060_ema.pth'

# ckpt_path='/home/yc37439/code/alocc2/work_dir/causalocc/alocc_3d_256x704_bevdet_preatrain_causalocc_wo_his/epoch_24_ema.pth'

# ckpt_path='/home/yc37439/code/alocc2/work_dir/alocc/ada_occ_2d_mini_256x704_bevdet_preatrain/iter_96060_ema.pth'

# ckpt_path='/home/yc37439/code/alocc2/work_dir/alocc/ada_occ_2d_mini_256x704_bevdet_preatrain/iter_96060_ema.pth'

# ckpt_path='/home/yc37439/code/alocc2/work_dir/alocc/aloccflow_2d_256x704_bevdet_preatrain/flow2d_iter_160100_ema.pth'
# ckpt_path='/home/yc37439/code/alocc2/work_dir/alocc/aloccflow_3d_256x704_bevdet_preatrain/flow_3d_iter_160100_ema.pth'
# ckpt_path='/home/yc37439/code/alocc2/work_dir/alocc/alocc_3d_900x1600_bevdet_preatrain_surroundocc/epoch_24_ema.pth'

# ckpt_path='/home/yc37439/code/alocc2/work_dir/gdfusion/alocc_3d_900x1600_bevdet_preatrain_gdfusion_surroundocc/iter_96060_ema.pth'
# ckpt_path='/home/yc37439/code/alocc2/work_dir/causalocc/bevdetocc_r50_256x704_bevdet_pretrain_causalocc_wo_his/epoch_24_ema.pth'
ckpt_path='/home/yc37439/code/alocc2/work_dir/gdfusion/bevdetocc_r50_256x704_bevdet_pretrain_gdfusion/iter_96060_ema.pth'

out_dir='/home/yc37439/code/alocc2/work_dir2/'+ckpt_path.split('/')[6]
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
out_path=os.path.join(out_dir,ckpt_path.split('/')[-2]+'.pth')

# 定义替换规则（旧字符串 -> 新字符串）
replace_rules = {
    "depth_net.depth2occ_hidden_weight_net": "depth_net.depth2occ_inter_weight_net",
    "depth_net.depth2occ_hidden_bias_net": "depth_net.depth2occ_inter_bias_net",
    "forward_projection": "view_transformer",
    "ttt_layer_func_after_his.ttt_learnable_weight" : "scene_his_func_after_vox_his.scene_his_learnable_weight",
    "ttt_layer_func_after_his.ttt_learnable_bias":"scene_his_func_after_vox_his.scene_his_learnable_bias",
    "ttt_layer_func_after_his.ttt_norm_weight":"scene_his_func_after_vox_his.scene_his_norm_weight",
    "ttt_layer_func_after_his.ttt_norm_bias":"scene_his_func_after_vox_his.scene_his_norm_bias",
    "ttt_layer_func_after_his.ttt_q":"scene_his_func_after_vox_his.scene_his_q",
    "ttt_layer_func_after_his.ttt_k":"scene_his_func_after_vox_his.scene_his_k",
    "ttt_layer_func_after_his.ttt_v":"scene_his_func_after_vox_his.scene_his_v",
    "maskformerocc_head":"alocc_head",
    # 可以继续添加更多规则
    # "xxx.old": "xxx.new",
}
# import pdb;pdb.set_trace()
# 加载 checkpoint
state_dict_ = torch.load(ckpt_path, map_location="cpu")
state_dict=state_dict_['state_dict']
# 创建新的 dict 并做替换
new_state_dict = {}
changed = []  # 记录被替换过的 key
# import pdb;pdb.set_trace()
for k, v in state_dict.items():
    new_k = k
    for old, new in replace_rules.items():
        if old in new_k:
            new_k = new_k.replace(old, new)
    new_state_dict[new_k] = v
    if new_k != k:
        changed.append((k, new_k))

# 打印对照表
if changed:
    print("以下 key 已被替换：")
    for old_k, new_k in changed:
        print(f"{old_k}  ->  {new_k}")
else:
    print("没有匹配到需要替换的 key。")

# 保存更新后的权重
state_dict_['state_dict'] = new_state_dict
torch.save(state_dict_, out_path)
print(f"\n转换完成，保存到 {out_path}")
