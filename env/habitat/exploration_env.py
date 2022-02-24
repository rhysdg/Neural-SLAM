from collections import deque
import math
import os
import pickle
from pickletools import float8
import sys
from cv2 import threshold
import time

import magnum as mn

import gym
import habitat_sim
import matplotlib
import numpy as np
import quaternion
import skimage.morphology
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import habitat
from habitat import logger
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)


from env.utils.map_builder import MapBuilder
from env.utils.fmm_planner import FMMPlanner

from env.habitat.utils.noisy_actions import CustomActionSpaceConfiguration
import env.habitat.utils.pose as pu
import env.habitat.utils.visualizations as vu
from env.habitat.utils.supervision import HabitatMaps

from model import get_grid

from env.habitat.utils.semantic_prediction import SemanticPredMaskRCNN
import math

from habitat_sim.nav import NavMeshSettings



def _preprocess_depth(depth):
    depth = depth[:, :, 0]*1
    mask2 = depth > 0.99
    depth[mask2] = 0.

    for i in range(depth.shape[1]):
        depth[:,i][depth[:,i] == 0.] = depth[:,i].max()

    mask1 = depth == 0
    depth[mask1] = np.NaN
    depth = depth*1000.
    return depth


class Exploration_Env(habitat.RLEnv):

    def __init__(self, args, rank, config_env, config_baseline, dataset):
        if args.visualize:
            plt.ion()
        if args.print_images or args.visualize:
            self.figure, self.ax = plt.subplots(1,2, figsize=(6*16/9, 6),
                                                facecolor="whitesmoke",
                                                num="Thread {}".format(rank))

        self.args = args
        self.num_actions = 3
        self.dt = 10

        self.rank = rank

        self.sem_pred = SemanticPredMaskRCNN(args)


        
        self.x = None
        self.y = None
        self.o = None
        self.sim_map = None
        #self._initialize_object()


        self.navmesh_settings = NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_radius = 0.1
        self.navmesh_settings.agent_height = 1.5


        self.pose_loss = deque(maxlen=1000)

        self.so = 0  # number of static objects thats potenially movable after certain ts.


        self.sensor_noise_fwd = \
                pickle.load(open("noise_models/sensor_noise_fwd.pkl", 'rb'))
        self.sensor_noise_right = \
                pickle.load(open("noise_models/sensor_noise_right.pkl", 'rb'))
        self.sensor_noise_left = \
                pickle.load(open("noise_models/sensor_noise_left.pkl", 'rb'))

        #habitat.SimulatorActions.extend_action_space("NOISY_FORWARD")
        #habitat.SimulatorActions.extend_action_space("NOISY_RIGHT")
        #habitat.SimulatorActions.extend_action_space("NOISY_LEFT")
        HabitatSimActions.extend_action_space("NOISY_FORWARD")
        HabitatSimActions.extend_action_space("NOISY_RIGHT")
        HabitatSimActions.extend_action_space("NOISY_LEFT")


        config_env.defrost()
        config_env.SIMULATOR.ACTION_SPACE_CONFIG = \
                "CustomActionSpaceConfiguration"
        config_env.SIMULATOR.ENABLE_PHYSICS = True
        config_env.freeze()


        super().__init__(config_env, dataset)

        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.observation_space = gym.spaces.Box(0, 255,
                                                (3, args.frame_height,
                                                    args.frame_width),
                                                dtype='uint8')

        self.mapper = self.build_mapper()

        self.episode_no = 0

        self.res = transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize((args.frame_height, args.frame_width),
                                      interpolation = Image.NEAREST)])
        self.scene_name = None
        self.maps_dict = {}

        #self._initialize_object()



    def randomize_env(self):
        self._env._episode_iterator._shuffle_iterator()

    def save_trajectory_data(self):
        if "replica" in self.scene_name:
            folder = self.args.save_trajectory_data + "/" + \
                        self.scene_name.split("/")[-3]+"/"
        else:
            folder = self.args.save_trajectory_data + "/" + \
                        self.scene_name.split("/")[-1].split(".")[0]+"/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = folder+str(self.episode_no)+".txt"
        with open(filepath, "w+") as f:
            f.write(self.scene_name+"\n")
            for state in self.trajectory_states:
                f.write(str(state)+"\n")
            f.flush()

    def save_position(self):
        self.agent_state = self._env.sim.get_agent_state()
        self.trajectory_states.append([self.agent_state.position,
                                       self.agent_state.rotation])



    def _get_sem_pred(self, rgb, use_seg=True):
        #! to debug not using detectron for now
        # use_seg = self.args.use_seg
        # self.use_seg = use_seg
        if use_seg:
            semantic_pred, self.rgb_vis = self.sem_pred.get_prediction(rgb)
            semantic_pred = semantic_pred.astype(np.float32)        #(128,128,16)
            #pass
        else:
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
            self.rgb_vis = rgb[:, :, ::-1]
        return semantic_pred

    def _preprocess_obs(self, rgb, depth, semantic, use_seg):

        # if args.use_seg == False:
        #     self.semantic = obs['semantic']


        # #!TODO get the correct semantic segmantation
        # #sem_seg_pred.shape(128,128,16)
        # sem_seg_pred = self._get_sem_pred(
        #     rgb.astype(np.uint8), use_seg=True)


        # # Preprocess semantic observations
        # # First downscaling 
        # ds = args.env_frame_width // args.frame_width  # Downscaling factor
        # if ds != 1:
        #     depth_sem = depth[ds // 2::ds, ds // 2::ds]
        # depth_sem = np.expand_dims(depth_sem, axis=2)


        # #depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)
        # state = np.concatenate((rgb,depth_sem,sem_seg_pred),axis=2).transpose(2,0,1)

        args = self.args
        #depth = _preprocess_depth(depth)
        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            #rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            semantic = semantic[ds // 2::ds, ds // 2::ds]
            #semantic = np.array(self.res(semantic.astype(np.uint8)))
        #obs = obs.transpose(1, 2, 0)
        # rgb = obs[:, :, :3]
        # depth = obs[:, :, 3:4]
        #sem_seg_pred = semantic

        if use_seg == 1: # use gt segmantation
            #objects_sem_id = self.objects_semantic_id
            objects_sem_id = 10
            mask_semantic = semantic == objects_sem_id

            # apply semantic mask on to rgb and depth obs, before feeding to n-slam.
            rgb[mask_semantic] = 0.
            # self.obs = rgb
            # self.sem = mask_semantic
            self.obs = mask_semantic
            depth[mask_semantic] = 0.
            depth = np.expand_dims(depth, axis=2)
            semantic = np.expand_dims(semantic, axis=2)
            state = np.concatenate((rgb, depth, semantic),
                                axis=2).transpose(2, 0, 1)
        
        elif use_seg == 2:  # use mask rcnn segmantation (detectron)
            sem_seg_pred = self._get_sem_pred(
                rgb.astype(np.uint8), use_seg=True)

            # if ds != 1:
            #     sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

            depth = np.expand_dims(depth, axis=2)
            semantic = np.expand_dims(semantic, axis=2)
            self.sem = sem_seg_pred
            state = np.concatenate((rgb, depth, sem_seg_pred),
                                axis=2).transpose(2, 0, 1)

        else:   # do not use any segmantation
            depth = np.expand_dims(depth, axis=2)
            semantic = np.expand_dims(semantic, axis=2)
            state = np.concatenate((rgb, depth, semantic),
                                axis=2).transpose(2, 0, 1)

        return state

    def reconfigure(self, config):
        super().reconfigure(config)
        print('debug in reconfigure')


    def reset(self):
        args = self.args
        self.episode_no += 1
        self.timestep = 0
        self._previous_action = None
        self.trajectory_states = []

        self.sim_count = 0
        self.sim_object_to_objid_mapping = {}
        self.object_positions = []
        self.object_ids = []
        self.objects = []
        self.object_scene_nodes = []
        self.path_step = 0
        self.path = []
        self.object_cur_loc = None
        self.obj_num = 10
        self.step_count = 0
        self.path_step = []
        self.direction = []
        for i in range(self.obj_num):
            self.path_step.append(0)
            self.direction.append(1)


        if args.randomize_env_every > 0:
            if np.mod(self.episode_no, args.randomize_env_every) == 0:
                self.randomize_env()


        #self._initialize_object()

        # Get Ground Truth Map
        self.explorable_map = None
        while self.explorable_map is None:
            obs = super().reset()

            #some variables to init objects 
            #-----objects related initialzation-----#

            #self._initialize_object()


            full_map_size = args.map_size_cm//args.map_resolution
            self.explorable_map = self._get_gt_map(full_map_size)
            #self.reconfigure()
        self.prev_explored_area = 0.

        # Preprocess observations
        #! here obs is an Observation object, need to first convert to numpy then transpose
        rgb = obs['rgb'].astype(np.uint8)
        self.obs = rgb # For visualization
        if self.args.frame_width != self.args.env_frame_width:
            rgb = np.asarray(self.res(rgb))
        depth = _preprocess_depth(obs['depth'])
        semantic = obs['semantic'].astype(np.uint8)
        #state = rgb.transpose(2, 0, 1)
        
        #state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        use_seg = self.args.use_seg
        
        state = self._preprocess_obs(rgb, depth, semantic, use_seg=use_seg)


        # if args.use_seg == False:
        #     self.semantic = obs['semantic']


        # #!TODO get the correct semantic segmantation
        # #sem_seg_pred.shape(128,128,16)
        # sem_seg_pred = self._get_sem_pred(
        #     rgb.astype(np.uint8), use_seg=True)


        # # Preprocess semantic observations
        # # First downscaling 
        # ds = args.env_frame_width // args.frame_width  # Downscaling factor
        # if ds != 1:
        #     depth_sem = depth[ds // 2::ds, ds // 2::ds]
        # depth_sem = np.expand_dims(depth_sem, axis=2)


        # #depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)
        # state = np.concatenate((rgb,depth_sem,sem_seg_pred),axis=2).transpose(2,0,1)


        # Initialize map and pose 
        self.map_size_cm = args.map_size_cm
        self.mapper.reset_map(self.map_size_cm)
        self.curr_loc = [self.map_size_cm/100.0/2.0,
                         self.map_size_cm/100.0/2.0, 0.]            # [12, 12, 0]
        self.curr_loc_gt = self.curr_loc
        self.last_loc_gt = self.curr_loc_gt
        self.last_loc = self.curr_loc
        self.last_sim_location = self.get_sim_location()


        #! Initializa object map

        # # try to get object position
        # obj_sim_pos = self.path[0].points[0]
        # dxo_gt, dyo_gt, doo_gt = pu.get_rel_pose_change(self.last_sim_location, self.get_object_location(obj_sim_pos))
        # self.object_cur_loc = pu.get_new_pose(self.curr_loc_gt,
        #                        (dxo_gt, dyo_gt, doo_gt))

        # Convert pose to cm and degrees for mapper
        mapper_gt_pose = (self.curr_loc_gt[0]*100.0,
                          self.curr_loc_gt[1]*100.0,
                          np.deg2rad(self.curr_loc_gt[2]))

        # Update ground_truth map and explored area
        fp_proj, self.map, fp_explored, self.explored_map = \
            self.mapper.update_map(depth, mapper_gt_pose)

        # Initialize variables
        #self.scene_name = self.habitat_env.sim.config.SCENE
        self.scene_name = self.habitat_env.sim.config.sim_cfg.scene_id
        self.visited = np.zeros(self.map.shape)
        self.visited_vis = np.zeros(self.map.shape)
        self.visited_gt = np.zeros(self.map.shape)
        self.collison_map = np.zeros(self.map.shape)
        self.col_width = 1

        # Set info
        self.info = {
            'time': self.timestep,
            'fp_proj': fp_proj,
            'fp_explored': fp_explored,
            'sensor_pose': [0., 0., 0.],
            'pose_err': [0., 0., 0.],
            'pose_loss':  0.,
        }

        self.save_position()


        return state, self.info
        #return obs, self.info


    def step(self, action):

        args = self.args
        self.timestep += 1
    
        full_map_size = args.map_size_cm//args.map_resolution
        
        # count step
        self.step_count += 1

        # print('========================================')
        # print(f'step count is {self.step_count}')
        # print('========================================')


        if self.step_count % 10 == 0:
            self.vel_control.linear_velocity = -1 * self.vel_control.linear_velocity

        if self.step_count > 300:
            self.so = 1

        elif self.step_count > 600:
            self.so = 2

        # moving object within each step
        if args.dynamic:
            for i in range(len(self.objects) - self.so):
                if self.direction[i]==1 and self.path_step[i]  < len(self.path[i].points):
                    self.objects[i].translation = np.array(self.path[i].points[self.path_step[i]])
                    self.path_step[i] += 1
                elif self.direction[i]==-1 and self.path_step[i] > 0:
                    self.objects[i].translation = np.array(self.path[i].points[self.path_step[i]])
                    self.path_step[i] -= 1
                elif self.path_step[i] ==0:
                    self.direction[i] = -1 * self.direction[i]
                elif self.path_step[i] == len(self.path[i].points):
                    self.direction[i] = -1 * self.direction[i]
                    self.path_step[i] -= 1


                # if i == 0:
        
        #print("object id: 0 is at the location %s" %([self.objects[0].translation.r, self.objects[0].translation.b]))

            # print(f'object sim location {}')

            #! takes too much time, FPS drops down lower than 1.
            # obj_x, obj_y = self._get_topdown_point(full_map_size, i)

        # Action remapping
        if action == 2: # Forward
            action = 1
            #noisy_action = habitat.SimulatorActions.NOISY_FORWARD
            noisy_action = HabitatSimActions.NOISY_FORWARD
        elif action == 1: # Right
            action = 3
            #noisy_action = habitat.SimulatorActions.NOISY_RIGHT
            noisy_action = HabitatSimActions.NOISY_RIGHT
        elif action == 0: # Left
            action = 2
            #noisy_action = habitat.SimulatorActions.NOISY_LEFT
            noisy_action =  HabitatSimActions.NOISY_LEFT

        self.last_loc = np.copy(self.curr_loc)
        self.last_loc_gt = np.copy(self.curr_loc_gt)
        self._previous_action = action

        if args.noisy_actions:
            #obs, rew, done, info = super().step(noisy_action)
            obs, rew, done, info = super().step(action)
        else:
            obs, rew, done, info = super().step(action)


        rgb = obs['rgb'].astype(np.uint8)
        depth = _preprocess_depth(obs['depth'])
        semantic = obs['semantic'].astype(np.uint8)

        self.obs = rgb # For visualization
        if self.args.frame_width != self.args.env_frame_width:
            rgb = np.asarray(self.res(rgb))

        #state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        use_seg = self.args.use_seg
        
        state = self._preprocess_obs(rgb, depth, semantic, use_seg=use_seg)

        

        #obs = obs.transpose(1, 2, 0)
        # rgb = obs[:, :, :3]
        #self.obs = state.transpose(1,2,0)[:,:,3]


        # #TODO concatenate depth layer and semantic seg
        # depth = _preprocess_depth(obs['depth'])

        # #sem_seg_pred.shape(128,128,16)
        # sem_seg_pred = self._get_sem_pred(
        #     rgb.astype(np.uint8), use_seg=self.args.use_seg)


        # Preprocess semantic observations
        # First downscaling 
        # ds = args.env_frame_width // args.frame_width  # Downscaling factor
        # if ds != 1:
        #     depth_sem = depth[ds // 2::ds, ds // 2::ds]
        # depth_sem = np.expand_dims(depth_sem, axis=2)


        # #depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)
        # state = np.concatenate((rgb,depth_sem,sem_seg_pred),axis=2).transpose(2,0,1)


        #state = rgb.transpose(2, 0, 1)

        #depth = _preprocess_depth(obs['depth'])

        # Get base sensor and ground-truth pose
        dx_gt, dy_gt, do_gt = self.get_gt_pose_change()
        dx_base, dy_base, do_base = self.get_base_pose_change(
                                        action, (dx_gt, dy_gt, do_gt))

        self.curr_loc = pu.get_new_pose(self.curr_loc,
                               (dx_base, dy_base, do_base))

        self.curr_loc_gt = pu.get_new_pose(self.curr_loc_gt,
                               (dx_gt, dy_gt, do_gt))

        if not args.noisy_odometry:
            self.curr_loc = self.curr_loc_gt
            dx_base, dy_base, do_base = dx_gt, dy_gt, do_gt

        # Convert pose to cm and degrees for mapper
        mapper_gt_pose = (self.curr_loc_gt[0]*100.0,
                          self.curr_loc_gt[1]*100.0,
                          np.deg2rad(self.curr_loc_gt[2]))


        # Update ground_truth map and explored area
        fp_proj, self.map, fp_explored, self.explored_map = \
                self.mapper.update_map(depth, mapper_gt_pose)


        # Update collision map
        if action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, t2 = self.curr_loc
            if abs(x1 - x2)< 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                self.col_width = min(self.col_width, 9)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold: #Collision
                length = 2
                width = self.col_width
                buf = 3
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05*((i+buf) * np.cos(np.deg2rad(t1)) + \
                                        (j-width//2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05*((i+buf) * np.sin(np.deg2rad(t1)) - \
                                        (j-width//2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r*100/args.map_resolution), \
                               int(c*100/args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                    self.collison_map.shape)
                        self.collison_map[r,c] = 1

        # Set info
        self.info['time'] = self.timestep
        self.info['fp_proj'] = fp_proj
        self.info['fp_explored']= fp_explored
        self.info['sensor_pose'] = [dx_base, dy_base, do_base]
        self.info['pose_err'] = [dx_gt - dx_base,
                                 dy_gt - dy_base,
                                 do_gt - do_base]
        self.info['pose_loss'] = 0.

        if self.timestep%args.num_local_steps==0:
            area, ratio = self.get_global_reward()
            self.info['exp_reward'] = area
            self.info['exp_ratio'] = ratio
        else:
            self.info['exp_reward'] = None
            self.info['exp_ratio'] = None

        self.save_position()

        if self.info['time'] >= args.max_episode_length:
            done = True
            self.step_count = 0
            if self.args.save_trajectory_data != "0":
                self.save_trajectory_data()
        else:
            done = False

        
        return state, rew, done, self.info

    def get_reward_range(self):
        # This function is not used, Habitat-RLEnv requires this function
        return (0., 1.0)

    def get_reward(self, observations):
        # This function is not used, Habitat-RLEnv requires this function
        return 0.

    def get_global_reward(self):
        curr_explored = self.explored_map*self.explorable_map
        curr_explored_area = curr_explored.sum()

        reward_scale = self.explorable_map.sum()
        m_reward = (curr_explored_area - self.prev_explored_area)*1.
        m_ratio = m_reward/reward_scale
        m_reward = m_reward * 25./10000. # converting to m^2
        self.prev_explored_area = curr_explored_area

        m_reward *= 0.02 # Reward Scaling

        return m_reward, m_ratio

    def get_done(self, observations):
        # This function is not used, Habitat-RLEnv requires this function
        return False

    def get_info(self, observations):
        # This function is not used, Habitat-RLEnv requires this function
        info = {}
        return info

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def get_spaces(self):
        return self.observation_space, self.action_space

    def build_mapper(self):
        params = {}
        params['frame_width'] = self.args.env_frame_width
        params['frame_height'] = self.args.env_frame_height
        params['fov'] =  self.args.hfov
        params['resolution'] = self.args.map_resolution
        params['map_size_cm'] = self.args.map_size_cm
        params['agent_min_z'] = 25
        params['agent_max_z'] = 150
        params['agent_height'] = self.args.camera_height * 100
        params['agent_view_angle'] = 0
        params['du_scale'] = self.args.du_scale
        params['vision_range'] = self.args.vision_range
        params['visualize'] = self.args.visualize
        params['obs_threshold'] = self.args.obs_threshold
        self.selem = skimage.morphology.disk(self.args.obstacle_boundary /
                                             self.args.map_resolution)
        mapper = MapBuilder(params)
        return mapper


    def get_sim_location(self):
        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o


    def get_object_location(self, points):
        x = -points[2]
        y = -points[0]
        o = np.pi
        
        return x, y, o


    def get_gt_pose_change(self):
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do


    def get_base_pose_change(self, action, gt_pose_change):
        dx_gt, dy_gt, do_gt = gt_pose_change
        if action == 1: ## Forward
            x_err, y_err, o_err = self.sensor_noise_fwd.sample()[0][0]
        elif action == 3: ## Right
            x_err, y_err, o_err = self.sensor_noise_right.sample()[0][0]
        elif action == 2: ## Left
            x_err, y_err, o_err = self.sensor_noise_left.sample()[0][0]
        else: ##Stop
            x_err, y_err, o_err = 0., 0., 0.

        x_err = x_err * self.args.noise_level
        y_err = y_err * self.args.noise_level
        o_err = o_err * self.args.noise_level
        return dx_gt + x_err, dy_gt + y_err, do_gt + np.deg2rad(o_err)


    def get_short_term_goal(self, inputs):

        args = self.args

        # Get Map prediction
        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']

        grid = np.rint(map_pred)
        explored = np.rint(exp_pred)

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]


        # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0/args.map_resolution - gx1),
                      int(c * 100.0/args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, grid.shape)

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0/args.map_resolution - gx1),
                 int(c * 100.0/args.map_resolution - gy1)]
        start = pu.threshold_poses(start, grid.shape)
        #TODO: try reducing this

        self.visited[gx1:gx2, gy1:gy2][start[0]-2:start[0]+3,
                                       start[1]-2:start[1]+3] = 1

        steps = 25
        for i in range(steps):
            x = int(last_start[0] + (start[0] - last_start[0]) * (i+1) / steps)
            y = int(last_start[1] + (start[1] - last_start[1]) * (i+1) / steps)
            self.visited_vis[gx1:gx2, gy1:gy2][x, y] = 1

        # Get last loc ground truth pose
        last_start_x, last_start_y = self.last_loc_gt[0], self.last_loc_gt[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0/args.map_resolution),
                      int(c * 100.0/args.map_resolution)]
        last_start = pu.threshold_poses(last_start, self.visited_gt.shape)

        # Get ground truth pose
        start_x_gt, start_y_gt, start_o_gt = self.curr_loc_gt
        r, c = start_y_gt, start_x_gt
        start_gt = [int(r * 100.0/args.map_resolution),
                    int(c * 100.0/args.map_resolution)]
        start_gt = pu.threshold_poses(start_gt, self.visited_gt.shape)
        #self.visited_gt[start_gt[0], start_gt[1]] = 1





        steps = 25
        for i in range(steps):
            x = int(last_start[0] + (start_gt[0] - last_start[0]) * (i+1) / steps)
            y = int(last_start[1] + (start_gt[1] - last_start[1]) * (i+1) / steps)
            self.visited_gt[x, y] = 1


        # Get goal
        goal = inputs['goal']
        goal = pu.threshold_poses(goal, grid.shape)


        # Get intrinsic reward for global policy
        # Negative reward for exploring explored areas i.e.
        # for choosing explored cell as long-term goal
        self.extrinsic_rew = -pu.get_l2_distance(10, goal[0], 10, goal[1])
        self.intrinsic_rew = -exp_pred[goal[0], goal[1]]

        # Get short-term goal
        stg = self._get_stg(grid, explored, start, np.copy(goal), planning_window)

        # Find GT action
        if self.args.eval or not self.args.train_local:
            gt_action = 0
        else:
            gt_action = self._get_gt_action(1 - self.explorable_map, start,
                                            [int(stg[0]), int(stg[1])],
                                            planning_window, start_o)

        (stg_x, stg_y) = stg
        relative_dist = pu.get_l2_distance(stg_x, start[0], stg_y, start[1])
        relative_dist = relative_dist*5./100.
        angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                stg_y - start[1]))
        angle_agent = (start_o)%360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal)%360.0
        if relative_angle > 180:
            relative_angle -= 360

        def discretize(dist):
            dist_limits = [0.25, 3, 10]
            dist_bin_size = [0.05, 0.25, 1.]
            if dist < dist_limits[0]:
                ddist = int(dist/dist_bin_size[0])
            elif dist < dist_limits[1]:
                ddist = int((dist - dist_limits[0])/dist_bin_size[1]) + \
                    int(dist_limits[0]/dist_bin_size[0])
            elif dist < dist_limits[2]:
                ddist = int((dist - dist_limits[1])/dist_bin_size[2]) + \
                    int(dist_limits[0]/dist_bin_size[0]) + \
                    int((dist_limits[1] - dist_limits[0])/dist_bin_size[1])
            else:
                ddist = int(dist_limits[0]/dist_bin_size[0]) + \
                    int((dist_limits[1] - dist_limits[0])/dist_bin_size[1]) + \
                    int((dist_limits[2] - dist_limits[1])/dist_bin_size[2])
            return ddist

        output = np.zeros((args.goals_size + 1))

        output[0] = int((relative_angle%360.)/5.)       # 50
        output[1] = discretize(relative_dist)                # 4.0
        output[2] = gt_action                                                     # 0.0

        self.relative_angle = relative_angle

        if args.visualize or args.print_images:
            dump_dir = "{}/dump/{}/".format(args.dump_location,
                                                args.exp_name)
            ep_dir = '{}/episodes/{}/{}/'.format(
                            dump_dir, self.rank+1, self.episode_no)
            if not os.path.exists(ep_dir):
                os.makedirs(ep_dir)

            if args.vis_type == 1: # Visualize predicted map and pose
                #! TESTING
                #! TODO
                # 1. caculate the rel distance between agent sim location and object sim location
                #           at EACH timestep before feeded into vu.get_color_map
                #goal[0], goal[1], _ = self.object_cur_loc

                        # try to get object position
                #obj_sim_pos = self.path[0].points[0]
                obj_x, obj_y, obj_z = self.objects[0].translation.b, \
                    self.objects[0].translation.g ,self.objects[0].translation.r
                obj_sim_pos = [obj_x, obj_y, obj_z]
                dxo_gt, dyo_gt, doo_gt = pu.get_rel_pose_change(self.last_sim_location, self.get_object_location(obj_sim_pos))
                object_cur_loc = pu.get_new_pose(self.curr_loc_gt,
                                    (dxo_gt, dyo_gt, doo_gt))

                goal_ = [0, 0]
                #goal_[0], goal_[1], _ = self.object_cur_loc
                goal_[0], goal_[1], _ = object_cur_loc
                goal_[0] = int(goal_[0])
                goal_[1] = int(goal_[1])
                vis_grid = vu.get_colored_map(np.rint(map_pred),
                                self.collison_map[gx1:gx2, gy1:gy2],
                                self.visited_vis[gx1:gx2, gy1:gy2],
                                self.visited_gt[gx1:gx2, gy1:gy2],
                                goal,
                                #goal_,
                                self.explored_map[gx1:gx2, gy1:gy2],
                                self.explorable_map[gx1:gx2, gy1:gy2],
                                self.map[gx1:gx2, gy1:gy2] *
                                    self.explored_map[gx1:gx2, gy1:gy2])
                vis_grid = np.flipud(vis_grid)      # (240,240,3)
                a_0 = start_x - gy1*args.map_resolution/100.0
                a_1 = start_y - gx1*args.map_resolution/100.0
                a_2 = start_o
                x_,y_,z_ = self.objects[0].translation.b, self.objects[0].translation.g,self.objects[0].translation.r
                bounds = self.habitat_env.sim.pathfinder.get_bounds()
                px = (x_ - bounds[0][0]) * 24
                py = (y_ - bounds[0][2]) * 24
                vis_grid_ = vis_grid[:,:,::-1]      #(240,240,2), first two channels of vis_grid
                self.predicted_pose = torch.from_numpy(np.array([start_x - gy1*args.map_resolution/100.0,
                                start_y - gx1*args.map_resolution/100.0]
                                ))
                self.gt_pose = torch.from_numpy(np.array([start_x_gt - gy1*args.map_resolution/100.0,
                                start_y_gt - gx1*args.map_resolution/100.0]
                                ))
                pose_loss = torch.nn.MSELoss()(
                        self.predicted_pose,
                        self.gt_pose
                )
                self.info['pose_loss'] = pose_loss

                if self.step_count % self.args.log_interval == 0:
                    print("Loss at time step {} is {:.4f}".format(self.step_count, pose_loss.item()))
                
                vu.visualize(self.figure, self.ax, self.obs, vis_grid[:,:,::-1],
                            (start_x - gy1*args.map_resolution/100.0,
                            start_y - gx1*args.map_resolution/100.0,
                            start_o),
                            (start_x_gt - gy1*args.map_resolution/100.0,
                            start_y_gt - gx1*args.map_resolution/100.0,
                            start_o_gt),
                            dump_dir, self.rank, self.episode_no,
                            self.timestep, args.visualize,
                            args.print_images, args.vis_type)


            else: # Visualize ground-truth map and pose
                vis_grid = vu.get_colored_map(self.map,
                                self.collison_map,
                                self.visited_gt,
                                self.visited_gt,
                                (goal[0]+gx1, goal[1]+gy1),
                                self.explored_map,
                                self.explorable_map,
                                self.map*self.explored_map)
                vis_grid = np.flipud(vis_grid)
                vu.visualize(self.figure, self.ax, self.obs, vis_grid[:,:,::-1],
                            (start_x_gt, start_y_gt, start_o_gt),
                            (start_x_gt, start_y_gt, start_o_gt),
                            dump_dir, self.rank, self.episode_no,
                            self.timestep, args.visualize,
                            args.print_images, args.vis_type)

        return output

    def _get_gt_map(self, full_map_size):
        #self.scene_name = self.habitat_env.sim.config.SCENE
        self.scene_name = self.habitat_env.sim.config.sim_cfg.scene_id
        logger.error('Computing map for %s', self.scene_name)

        # Get map in habitat simulator coordinates
        self.map_obj = HabitatMaps(self.habitat_env)
        if self.map_obj.size[0] < 1 or self.map_obj.size[1] < 1:
            logger.error("Invalid map: {}/{}".format(
                            self.scene_name, self.episode_no))
            return None

        agent_y = self._env.sim.get_agent_state().position.tolist()[1]*100.
        sim_map = self.map_obj.get_map(agent_y, -50., 50.0)

        sim_map[sim_map > 0] = 1.       #(287,187)

        self.sim_map = sim_map



        # Transform the map to align with the agent
        min_x, min_y = self.map_obj.origin/100.0        # (-8.53, -12.04)  sim map origin in sim coordinates
        x, y, o = self.get_sim_location()       # return initial agent sim position (7.23, -1.55,0.68)
        x, y = -x - min_x, -y - min_y       # (1.29, 13.59)
        if self.x is None:
            self.x, self.y, self.o = x, y, o

        #! print initial agent position and  origin in topdown map
        agent_state = self._env.sim.get_agent_state(0)
        agent_x, agent_y = -agent_state.position[2], -agent_state.position[0]
        agent_x, agent_y = (-agent_x - min_x) * 20, (-agent_y - min_y) * 20
        sim_map[
            int(agent_y) - 5: int(agent_y) + 5,
            int(agent_x) - 5: int(agent_x) + 5
        ] = 5


        ori_x, ori_y = self.map_obj.origin/100.0
        ori_x, ori_y = (-ori_x - min_x) * 20, (-ori_y - min_y) * 20     #(226,548)
        sim_map[
            int(ori_y) - 5: int(ori_y) + 5,
            int(ori_x) - 5: int(ori_x) + 5
        ] = 10



        map_dir = "data/results/neural_slam"
        np.save(
             os.path.join(map_dir,'sim_map.npy'),
             sim_map
        )


        # self.map_obj.max = ([81,205]) origin([-853,-1204])
        range_x, range_y = self.map_obj.max/100. - self.map_obj.origin/100.     #(9.34, 14.09) map size in meters?

        map_size = sim_map.shape
        scale = 2.
        grid_size = int(scale*max(map_size))      # 564
        grid_map = np.zeros((grid_size, grid_size))     # initialize map as (564, 564)

        

        grid_map[(grid_size - map_size[0])//2:
                 (grid_size - map_size[0])//2 + map_size[0],
                 (grid_size - map_size[1])//2:
                 (grid_size - map_size[1])//2 + map_size[1]] = sim_map


        if map_size[0] > map_size[1]:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale) \
                             * map_size[1] * 1. / map_size[0],
                    (y - range_y/2.) * 2. / (range_y * scale),
                    180.0 + np.rad2deg(o)
                ]])         # agent pose at t0
        else:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale),
                    (y - range_y/2.) * 2. / (range_y * scale) \
                            * map_size[0] * 1. / map_size[1],
                    180.0 + np.rad2deg(o)
                ]])

        rot_mat, trans_mat = get_grid(st, (1, 1,
            grid_size, grid_size), torch.device("cpu"))

        grid_map = torch.from_numpy(grid_map).float()
        grid_map = grid_map.unsqueeze(0).unsqueeze(0)
        translated = F.grid_sample(grid_map, trans_mat)
        rotated = F.grid_sample(translated, rot_mat)

        episode_map = torch.zeros((full_map_size, full_map_size)).float()       # full_map_zise = 480
        if full_map_size > grid_size:
            episode_map[(full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size,
                        (full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size] = \
                                rotated[0,0]
        else:
            episode_map = rotated[0,0,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size]



        episode_map = episode_map.numpy()
        episode_map[episode_map > 0] = 1.

        


        return episode_map


    def _get_stg(self, grid, explored, start, goal, planning_window):

        [gx1, gx2, gy1, gy2] = planning_window

        x1 = min(start[0], goal[0])
        x2 = max(start[0], goal[0])
        y1 = min(start[1], goal[1])
        y2 = max(start[1], goal[1])
        dist = pu.get_l2_distance(goal[0], start[0], goal[1], start[1])
        buf = max(20., dist)
        x1 = max(1, int(x1 - buf))
        x2 = min(grid.shape[0]-1, int(x2 + buf))
        y1 = max(1, int(y1 - buf))
        y2 = min(grid.shape[1]-1, int(y2 + buf))

        rows = explored.sum(1)
        rows[rows>0] = 1
        ex1 = np.argmax(rows)
        ex2 = len(rows) - np.argmax(np.flip(rows))

        cols = explored.sum(0)
        cols[cols>0] = 1
        ey1 = np.argmax(cols)
        ey2 = len(cols) - np.argmax(np.flip(cols))

        ex1 = min(int(start[0]) - 2, ex1)
        ex2 = max(int(start[0]) + 2, ex2)
        ey1 = min(int(start[1]) - 2, ey1)
        ey2 = max(int(start[1]) + 2, ey2)

        x1 = max(x1, ex1)
        x2 = min(x2, ex2)
        y1 = max(y1, ey1)
        y2 = min(y2, ey2)

        traversible = skimage.morphology.binary_dilation(
                        grid[x1:x2, y1:y2],
                        self.selem) != True
        traversible[self.collison_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
                    int(start[1]-y1)-1:int(start[1]-y1)+2] = 1

        if goal[0]-2 > x1 and goal[0]+3 < x2\
            and goal[1]-2 > y1 and goal[1]+3 < y2:
            traversible[int(goal[0]-x1)-2:int(goal[0]-x1)+3,
                    int(goal[1]-y1)-2:int(goal[1]-y1)+3] = 1
        else:
            goal[0] = min(max(x1, goal[0]), x2)
            goal[1] = min(max(y1, goal[1]), y2)

        def add_boundary(mat):
            h, w = mat.shape
            new_mat = np.ones((h+2,w+2))
            new_mat[1:h+1,1:w+1] = mat
            return new_mat

        traversible = add_boundary(traversible)

        planner = FMMPlanner(traversible, 360//self.dt)

        reachable = planner.set_goal([goal[1]-y1+1, goal[0]-x1+1])

        stg_x, stg_y = start[0] - x1 + 1, start[1] - y1 + 1
        for i in range(self.args.short_goal_dist):
            stg_x, stg_y, replan = planner.get_short_term_goal([stg_x, stg_y])
        if replan:
            stg_x, stg_y = start[0], start[1]
        else:
            stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y)


    def _get_gt_action(self, grid, start, goal, planning_window, start_o):

        [gx1, gx2, gy1, gy2] = planning_window

        x1 = min(start[0], goal[0])
        x2 = max(start[0], goal[0])
        y1 = min(start[1], goal[1])
        y2 = max(start[1], goal[1])
        dist = pu.get_l2_distance(goal[0], start[0], goal[1], start[1])
        buf = max(5., dist)
        x1 = max(0, int(x1 - buf))
        x2 = min(grid.shape[0], int(x2 + buf))
        y1 = max(0, int(y1 - buf))
        y2 = min(grid.shape[1], int(y2 + buf))

        path_found = False
        goal_r = 0
        while not path_found:
            traversible = skimage.morphology.binary_dilation(
                            grid[gx1:gx2, gy1:gy2][x1:x2, y1:y2],
                            self.selem) != True
            traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
            traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
                        int(start[1]-y1)-1:int(start[1]-y1)+2] = 1
            traversible[int(goal[0]-x1)-goal_r:int(goal[0]-x1)+goal_r+1,
                        int(goal[1]-y1)-goal_r:int(goal[1]-y1)+goal_r+1] = 1
            scale = 1
            planner = FMMPlanner(traversible, 360//self.dt, scale)

            reachable = planner.set_goal([goal[1]-y1, goal[0]-x1])

            stg_x_gt, stg_y_gt = start[0] - x1, start[1] - y1
            for i in range(1):
                stg_x_gt, stg_y_gt, replan = \
                        planner.get_short_term_goal([stg_x_gt, stg_y_gt])

            if replan and buf < 100.:
                buf = 2*buf
                x1 = max(0, int(x1 - buf))
                x2 = min(grid.shape[0], int(x2 + buf))
                y1 = max(0, int(y1 - buf))
                y2 = min(grid.shape[1], int(y2 + buf))
            elif replan and goal_r < 50:
                goal_r += 1
            else:
                path_found = True

        stg_x_gt, stg_y_gt = stg_x_gt + x1, stg_y_gt + y1
        angle_st_goal = math.degrees(math.atan2(stg_x_gt - start[0],
                                                stg_y_gt - start[1]))
        angle_agent = (start_o)%360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal)%360.0
        if relative_angle > 180:
            relative_angle -= 360

        if relative_angle > 15.:
            gt_action = 1
        elif relative_angle < -15.:
            gt_action = 0
        else:
            gt_action = 2

        return gt_action


    # method to initialize moving objects.
    def _initialize_object(self):
        object_path = "data/objects"
        sim = self.habitat_env.sim

        # allow sliding and physics
        sim.config.sim_cfg.allow_sliding = True
        sim.config.sim_cfg.enable_physics = True
        #sim.sim_cfg.sim_cfg.enable_physics = True
        sim.sim_config.sim_cfg = sim.config.sim_cfg


        #first remove all existing objects
        existing_object_ids = sim.get_existing_object_ids()
        if len(existing_object_ids)>0:
            for obj_id in existing_object_ids:
                sim.remove_object(obj_id)

        # add random path, path distance must be longer than 15.
        for i in range(self.obj_num):
            self.path.append(habitat_sim.ShortestPath())
            found_valid_path = False
            # area = sim.pathfinder.navigable_area
            # cs = round(1.4 * math.sqrt(area), 2)
            while not found_valid_path:
                self.path[i].requested_start = sim.pathfinder.get_random_navigable_point()
                self.path[i].requested_end = sim.pathfinder.get_random_navigable_point()
                found_path = sim.pathfinder.find_path(self.path[i])
                if found_path and len(self.path[i].points) > 12:
                    found_valid_path = True
                    print(f"found valid path with distance {self.path[i].geodesic_distance}")
                else:
                    print('failed to find a long path, try another one..')
                    print(self.path[i].geodesic_distance)
            self.object_positions.append(self.path[i].requested_start)
            self.object_positions.append(self.path[i].requested_end)
            # self.path[i].requested_start = sim.pathfinder.get_random_navigable_point()
            # while not found_valid_path:
            #     self.path[i].requested_end = sim.pathfinder.get_random_navigable_point()
            #     found_path = sim.pathfinder.find_path(self.path[i])
            #     if found_path and self.path[i].geodesic_distance > 5:
            #         found_


        # load object templates
        obj_templates_mgr = sim.get_object_template_manager()
        rigid_obj_mgr = sim.get_rigid_object_manager()
        locobot_template_id = obj_templates_mgr.load_object_configs(
            "data/objects/locobot_merged"
        )[0]
        locobot_orientation = mn.Quaternion.rotation(mn.Deg(90.0), [-1.0, 0.0, 0.0])

        sphere_template_id = obj_templates_mgr.load_configs(
            "data/objects/sphere"
        )[0]

        # add objects in initial position
        #! version 0.0 of adding objects
        # for i in range(self.obj_num):
        #     locobot_template = obj_templates_mgr.get_template_by_id(locobot_template_id)
        #     locobot_template.semantic_id = 10
        #     obj_templates_mgr.register_template(locobot_template)
        #     self.objects.append(
        #         rigid_obj_mgr.add_object_by_template_id(locobot_template_id)
        #     )
        #     self.objects[i].translation = self.object_positions[2*i]
        #     self.objects[i].collidable = True
        #     print(f"added objects with objects id {self.objects[i].object_id}")
        #     self.object_scene_nodes.append(self.objects[i].root_scene_node)

        #     self.object_ids.append(self.objects[i].object_id)
        #! version 0.0 end

        #! version 1.0 of adding objects
        for i in range(self.obj_num):
            self.objects.append(
                rigid_obj_mgr.add_object_by_template_id(locobot_template_id)
            )
            self.objects[i].semantic_id = 10
            self.objects[i].translation = self.object_positions[2 * i]
            #self.objects[i].rotation = locobot_orientation
            print(f"added objects with objects id {self.objects[i].object_id}")
            self.object_ids.append(self.objects[i].object_id)

            if self.objects[i].object_id == -1:
                print("something is wrong")
                continue

        sphere_obj = rigid_obj_mgr.add_object_by_template_id(sphere_template_id)
        print(f"added object with objects id {sphere_obj.object_id}")
        #sphere_obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        # sim.set_object_motion_type(
        #     habitat_sim.physics.MotionType.STATIC, sphere_obj.object_id
        # )
        sphere_obj.translation = sim.agents[0].state.position + [0.0, 1.0, 1.0]
        self.vel_control = sim.get_object_velocity_control(sphere_obj.object_id)
        self.vel_control.linear_velocity = np.array([0, -1.0, 0.0])
        self.vel_control.controlling_lin_vel = True
        # target_diretion = mn.Vector3([0.0,0.0,-1.0])
        # sphere_obj.linear_velocity = target_diretion * 3
        # sphere_obj.angular_velocity = [0.0, -1.0, 0.0]
        
    

        sim.recompute_navmesh(sim.pathfinder, self.navmesh_settings, True)


    def get_objects_and_paths(self):
        # sim = self.habitat_env.sim
        # objs = []
        return self.objects, self.path


    def _get_topdown_point(self, full_map_size, obj_id):

        # self.scene_name = self.habitat_env.sim.config.sim_cfg.scene_id
        # logger.error('Computing map for %s', self.scene_name)

        # # # Get map in habitat simulator coordinates
        time_stamp_0 = time.time()
        # self.map_obj = HabitatMaps(self.habitat_env)
        # if self.map_obj.size[0] < 1 or self.map_obj.size[1] < 1:
        #     logger.error("Invalid map: {}/{}".format(
        #                     self.scene_name, self.episode_no))
        #     return None

        # agent_y = self._env.sim.get_agent_state().position.tolist()[1]*100.
        # sim_map = self.map_obj.get_map(agent_y, -50., 50.0)

        # sim_map[sim_map > 0] = 1.
        # # sim_map[sim_map > 0] = 0.

        sim_map = self.sim_map

        time_stamp_1 = time.time()

        print("section 1 runtime is %.2f" % (time_stamp_1 - time_stamp_0))

        # get object sim location
        obj_x, obj_y = -self.objects[obj_id].translation.b, -self.objects[obj_id].translation.r
        #obj_x_, obj_y_ = -self.objects[obj_id].translation.b, -self.objects[obj_id].translation.r
        agent_state = self._env.sim.get_agent_state(0)
        #obj_x, obj_y = -agent_state.position[2], -agent_state.position[0]

        obj_sim_pos = []
        for point in self.path[obj_id].points:
            position = np.array([-point[2], -point[0]])
            obj_sim_pos.append(position)

        # object_sim_pos = []
        # for point in self.path[0].points:
        #     position = np.array([-point[2], -point[0]])
        #     object_sim_pos.append(position)

        # # convert points to topdown map
        # points_topdown = []
        # min_x, min_y = self.map_obj.origin/100.0
        # for pos in object_sim_pos:
        #     px = (- pos[0] - min_x) * 20
        #     py = (- pos[1] - min_y) * 20
        #     points_topdown.append(np.array([px, py]))


            
        # for point in points_topdown:
        #     sim_map[
        #         int(point[1]) - 2: int(point[1]) + 2,
        #         int(point[0]) - 2: int(point[0]) + 2
        #     ] = 2

        # convert points to topdown map
        min_x, min_y = self.map_obj.origin/100.0
        obj_x, obj_y = (-obj_x -min_x) * 20 , (-obj_y - min_y) * 20
       # obj_x_, obj_y_ =  (-obj_x_ -min_x) * 20 , (-obj_y_ - min_y) * 20 

        if int(obj_x) <= 1:
            obj_x += 1
        if int(obj_y) <= 1:
            obj_y += 1

        sim_map[
            int(obj_y),
            int(obj_x)
        ] = 10


        # Transform the map to align with the agent
        min_x, min_y = self.map_obj.origin/100.0
        if self.x is not None:
            x, y, o = self.x, self.y, self.o
        else:
            x, y, o = self.get_sim_location()
        x, y = -x - min_x, -y - min_y
        range_x, range_y = self.map_obj.max/100. - self.map_obj.origin/100.

        map_size = sim_map.shape
        scale = 2.
        grid_size = int(scale*max(map_size))
        grid_map = np.zeros((grid_size, grid_size))

        grid_map[(grid_size - map_size[0])//2:
                 (grid_size - map_size[0])//2 + map_size[0],
                 (grid_size - map_size[1])//2:
                 (grid_size - map_size[1])//2 + map_size[1]] = sim_map


        time_stamp_2 = time.time()
        print("section 2 runtime is %.2f" % (time_stamp_2 - time_stamp_1))


        if map_size[0] > map_size[1]:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale) \
                             * map_size[1] * 1. / map_size[0],
                    (y - range_y/2.) * 2. / (range_y * scale),
                    180.0 + np.rad2deg(o)
                ]])

        else:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale),
                    (y - range_y/2.) * 2. / (range_y * scale) \
                            * map_size[0] * 1. / map_size[1],
                    180.0 + np.rad2deg(o)
                ]])

        rot_mat, trans_mat = get_grid(st, (1, 1,
            grid_size, grid_size), torch.device("cpu"))

        time_stamp_3 = time.time()

        grid_map = torch.from_numpy(grid_map).float()
        grid_map = grid_map.unsqueeze(0).unsqueeze(0)
        translated = F.grid_sample(grid_map, trans_mat)
        rotated = F.grid_sample(translated, rot_mat)

        print("section 3 runtime is %.2f" % (time_stamp_3 - time_stamp_2))

        episode_map = torch.zeros((full_map_size, full_map_size)).float()
        if full_map_size > grid_size:
            episode_map[(full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size,
                        (full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size] = \
                                rotated[0,0]
        else:
            episode_map = rotated[0,0,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size]



        episode_map = episode_map.numpy()
        
        cor_x, cor_y = int(np.where(episode_map==episode_map.max())[0]), int(np.where(episode_map==episode_map.max())[1])

        time_stamp_4 = time.time()
        print("section 4 runtime is %.2f" % (time_stamp_4 - time_stamp_3))


        return cor_x, cor_y



class Sem_Exp_Env(Exploration_Env):
    """ The semantic exploration env class, inhereted from Explore_Env with extended functions like obs pre-processsing.
    """

    def __init__(self, args, rank, config_env, config_baseline, dataset):
        super().__init__(args, rank, config_env, config_baseline, dataset)
