import torch
import typing
import rlbench
from scipy.spatial.transform import Rotation as R

class BCZDataset(torch.utils.data.Dataset):
    def __init__(self, variations, demos_config,action_mode, task_embeds, views, action_scale, action_annealed):
       self.data = None
       self.variations = variations
       self.demos_config = demos_config
       self.action_mode = action_mode
       self.task_embeds = task_embeds
       self.views = views
       self.action_scale = action_scale
       self.action_annealed = action_annealed
       
       self.len = sum([sum([len(demo) for demo in rlbench.utils.get_stored_demos(**demos_config, variation_number=variation)]) for variation in variations])

    def __len__(self) -> int:
        return self.len
    
    def __getitem__(self, index):
        if self.data is None:
            self.data = self._load_data()
        
        images, task_embed, xyz, axangle, gripper = self.data[index]
        images = {view: [Image.open(frame) for frame in frames] for view, frames in images.items()}
        images = {view: ([rlbench.backend.utils.image_to_float_array(frame, rlbench.backend.const.DEPTH_SCALE)[None, ...].astype(np.float32) for frame in frames]
                         if 'depth' in view else
                         [np.array(frame.convert('RGB')).transpose(2, 0, 1).astype(np.float32) / 255.0 for frame in frames])
                         for view, frames in images.items()}
        images = {view: np.stack(frames, axis=0) for view, frames in images.items()}

        return images, task_embed, xyz, axangle, gripper

    def _load_data(self) -> typing.List[typing.Tuple[torch.Tensor, torch.Tensor]]:
        data = []
        for variation in self.variations:
            # TODO
            demos = rlbench.utils.get_stored_demos(variation_number= variation, **self.demos_config)
            for i_demo, demo in enumerate(demos):
                images = {view: [[]] for view in self.views}
                xyzs = []
                axangles = []
                grippers = []
                xyz_deltas = []
                axangles_deltas = []
                gripper_deltas = []
                for i_obs in range(len(demo)):
                    xyz, axangle, gripper, xyz_delta, axangle_delta, gripper_delta = self._get_state_delta(demo, i_obs)
                    xyzs.append(xyz)
                    axangles.append(axangle)
                    grippers.append(gripper)
                    xyz_deltas.append(xyz_delta)
                    axangle_deltas.append(axangle_delta)
                    gripper_deltas.append(gripper_delta)
                    for view in self.views:
                        images[view][0].append(demo[i_obs].__dict__[view])
                xyzs = np.stack(xyzs, axis=0).astype('float32')
                axangles = np.stack(axangles, axis=0).astype('float32')
                grippers = np.stack(grippers, axis=0).astype('float32')
                xyz_deltas = np.stack(xyz_deltas, axis=0).astype('float32')
                axangle_deltas = np.stack(axangle_deltas, axis=0).astype('float32')
                gripper_deltas = np.stack(gripper_deltas, axis=0).astype('float32')
                
                for i_obs in range(len(demo)):
                    curr_act_indices = np.arange(i_obs, i_obs+10)
                    curr_act_indices = np.clip(curr_act_indices, 0, len(demo) - 1)

                    this_xyzs = xyz_deltas[curr_act_indices] + xyzs[curr_act_indices] - xyzs[i_obs]
                    this_axangles = axangle_deltas[curr_act_indices] + axangles[curr_act_indices] - axangles[i_obs]
                    this_grippers = gripper_deltas[curr_act_indices]

                    this_xyzs = this_xyzs * self.action_scale[0]
                    this_axangles = this_axangles * self.action_scale[1]

                    this_images = {view: [images[view][0][i_obs]] for view in self.views}
                    data.append((this_images, self.task_embeds[self.demos_config.task_name][variation],
                                 this_xyzs, this_axangles, this_grippers))
        return data
    
    def _get_state_delta(self, demo, i_curr):
        i_curr = min(i_curr, len(demo)-1)
        arm_curr = demo[i_curr].gripper_pose
        xyz_curr = arm_curr[:3]
        axangle_curr = R.from_quat(arm_curr[3:]).as_rotvec()
        gripper = [demo[i_curr].gripper_open]
        
        i_next = min(i_curr+1, len(demo)-1)
        
        xyz_mag = 0
        
        while i_next < len(demo) and xyz_mag < 0.01:
            arm_next = demo[i_next].gripper_pose
            xyz_next = arm_next[:3]
            axangle_next = R.from_quat(arm_next[3:]).as_rotvec()
            xyz_delta = xyz_next - xyz_curr
            axangle_delta = axangle_next - axangle_curr
            gripper_delta = [demo[i_next].gripper_open]

            i_next += 1
            xyz_mag = np.linalg.norm(xyz_delta)
            if not self.action_annealed:
                break
        
        return xyz_curr, axangle_curr, gripper, xyz_delta, axangle_delta, gripper_delta