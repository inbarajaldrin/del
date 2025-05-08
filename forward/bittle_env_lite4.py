import gym
import numpy as np
import pybullet as p
import pybullet_data

# Constants to define training and visualisation.
GUI_MODE = True            # Set "True" to display pybullet in a window
EPISODE_LENGTH = 250        # Number of steps for one training episode
MAXIMUM_LENGTH = 1.5e6        # Number of total steps for entire training

# Factors to weight rewards and penalties.
PENALTY_STEPS = 2e6         # Increase of penalty by step_counter/PENALTY_STEPS
FAC_MOVEMENT = 5000         # Reward movement in x-direction
FAC_STABILITY = 0.2         # Punish body roll and pitch velocities
FAC_Z_VELOCITY = 0.2        # Punish z movement of body
FAC_SLIP = 0.0              # Punish slipping of paws
FAC_ARM_CONTACT = 0.01      # Punish crawling on arms and elbows
FAC_SMOOTH_1 = 1.0          # Punish jitter and vibrational movement, 1st order
FAC_SMOOTH_2 = 1.0          # Punish jitter and vibrational movement, 2nd order
FAC_CLEARANCE = 0.0         # Factor to enfore foot clearance to PAW_Z_TARGET
PAW_Z_TARGET = 0.005        # Target height (m) of paw during swing phase

BOUND_ANG = 110             # Joint maximum angle (deg)
STEP_ANGLE = 11             # Maximum angle (deg) delta per step
ANG_FACTOR = 0.1            # Improve angular velocity resolution before clip.

GAIT_FACTOR = 0.3           # Weight for diagonal gait pattern reward

# Values for randomization, to improve sim to real transfer.
RANDOM_GYRO = 0             # Percent
RANDOM_JOINT_ANGS = 0       # Percent
RANDOM_MASS = 0            # Percent, currently inactive
RANDOM_FRICTION = 0        # Percent, currently inactive

LENGTH_RECENT_ANGLES = 3    # Buffer to read recent joint angles
LENGTH_JOINT_HISTORY = 30   # Number of steps to store joint angles.

# Size of oberservation space is set up of:
# [LENGTH_JOINT_HISTORY, quaternion, gyro]
SIZE_OBSERVATION = LENGTH_JOINT_HISTORY * 8 + 6     

class OpenCatGymEnv(gym.Env):
    """ Gymnasium environment (stable baselines 3) for OpenCat robots.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.step_counter = 0
        self.step_counter_session = 0
        self.state_history = np.array([])
        self.angle_history = np.array([])
        self.bound_ang = np.deg2rad(BOUND_ANG)

        if GUI_MODE:
            p.connect(p.GUI)
            # Uncommend to create a video.
            #video_options = ("--width=960 --height=540"
            #                + "--mp4="training.mp4" --mp4fps=60")
            #p.connect(p.GUI, options=video_options)
        else:
            # Use for training without visualisation (significantly faster).
            p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                    cameraYaw=-170,
                                    cameraPitch=-40,
                                    cameraTargetPosition=[0.4,0,0])

        # The action space are the 8 joint angles.
        self.action_space = gym.spaces.Box(np.array([-1]*8), np.array([1]*8))

        # The observation space are the torso roll, pitch and the
        # angular velocities and a history of the last 30 joint angles.
        self.observation_space = gym.spaces.Box(np.array([-1]*SIZE_OBSERVATION),
                                                np.array([1]*SIZE_OBSERVATION))

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        last_position = p.getBasePositionAndOrientation(self.robot_id)[0][0]
        joint_angs = np.asarray(p.getJointStates(self.robot_id, self.joint_id), dtype=object)[:,0]
        ds = np.deg2rad(STEP_ANGLE)  # Maximum change of angle per step
        joint_angs += action * ds  # Change per step including agent action

        # Apply joint boundaries individually
        min_ang = -self.bound_ang
        max_ang = self.bound_ang
        joint_angs[0] = np.clip(joint_angs[0], min_ang, max_ang)  # shoulder_left
        joint_angs[1] = np.clip(joint_angs[1], min_ang, max_ang)  # elbow_left
        joint_angs[2] = np.clip(joint_angs[2], min_ang, max_ang)  # shoulder_right
        joint_angs[3] = np.clip(joint_angs[3], min_ang, max_ang)  # elbow_right
        joint_angs[4] = np.clip(joint_angs[4], min_ang, max_ang)  # hip_right
        joint_angs[5] = np.clip(joint_angs[5], min_ang, max_ang)  # knee_right
        joint_angs[6] = np.clip(joint_angs[6], min_ang, max_ang)  # hip_left
        joint_angs[7] = np.clip(joint_angs[7], min_ang, max_ang)  # knee_left

        # Transform angle to degree and perform rounding
        joint_angsDeg = np.rad2deg(joint_angs.astype(np.float64))
        joint_angsDegRounded = joint_angsDeg.round()
        joint_angs = np.deg2rad(joint_angsDegRounded)

        p.stepSimulation()

        # Check for friction of paws
        paw_contact = []
        paw_idx = [3, 6, 9, 12]
        for idx in paw_idx:
            paw_contact.append(True if p.getContactPoints(bodyA=self.robot_id, linkIndexA=idx) else False)

        paw_slipping = 0
        for in_contact in np.nonzero(paw_contact)[0]:
            paw_slipping += np.linalg.norm(
                (p.getLinkState(self.robot_id, linkIndex=paw_idx[in_contact], computeLinkVelocity=1)[0][0:1]))

        # Read clearance of paw from ground
        paw_clearance = 0
        for idx in paw_idx:
            paw_z_pos = p.getLinkState(self.robot_id, linkIndex=idx)[0][2]
            paw_clearance += (paw_z_pos-PAW_Z_TARGET)**2 * np.linalg.norm(
                (p.getLinkState(self.robot_id, linkIndex=idx, computeLinkVelocity=1)[0][0:1]))**0.5

        # Check if elbows or lower arm are in contact with ground
        arm_idx = [1, 2, 4, 5]
        for idx in arm_idx:
            if p.getContactPoints(bodyA=self.robot_id, linkIndexA=idx):
                self.arm_contact += 1

        # Set new joint angles
        p.setJointMotorControlArray(self.robot_id,
                                self.joint_id,
                                p.POSITION_CONTROL,
                                joint_angs,
                                forces=np.ones(8)*0.2)
        p.stepSimulation()

        # Normalize joint_angs
        joint_angs_norm = joint_angs / self.bound_ang

        if self.step_counter % 2 == 0:
            self.angle_history = np.append(self.angle_history, self.randomize(joint_angs_norm, RANDOM_JOINT_ANGS))
            self.angle_history = np.delete(self.angle_history, np.s_[0:8])

        self.recent_angles = np.append(self.recent_angles, joint_angs_norm)
        self.recent_angles = np.delete(self.recent_angles, np.s_[0:8])

        joint_angs_prev = self.recent_angles[8:16]
        joint_angs_prev_prev = self.recent_angles[0:8]

        # Read robot state
        state_pos, state_ang = p.getBasePositionAndOrientation(self.robot_id)
        p.stepSimulation()
        state_vel = np.asarray(p.getBaseVelocity(self.robot_id)[1])
        state_vel = state_vel[0:2]*ANG_FACTOR
        state_vel_clip = np.clip(state_vel, -1, 1)
        self.state_robot = np.concatenate((state_ang, state_vel_clip))
        current_position = p.getBasePositionAndOrientation(self.robot_id)[0][0]

        # Calculate rewards
        smooth_movement = np.sum(
            FAC_SMOOTH_1*np.abs(joint_angs_norm-joint_angs_prev)**2
            + FAC_SMOOTH_2*np.abs(joint_angs_norm - 2*joint_angs_prev + joint_angs_prev_prev)**2)

        z_velocity = p.getBaseVelocity(self.robot_id)[0][2]

        body_stability = (FAC_STABILITY * (state_vel_clip[0]**2 + state_vel_clip[1]**2)
                        + FAC_Z_VELOCITY * z_velocity**2)

        movement_forward = current_position - last_position
                    
        # DIAGONAL GAIT REWARD CALCULATION
        # Get current joint states (position and velocity)
        joint_states = [p.getJointState(self.robot_id, joint) for joint in self.joint_id]
        current_pos = np.array([state[0] for state in joint_states])/self.bound_ang
        current_vel = np.array([state[1] for state in joint_states])

        # Normalize velocities
        max_vel = np.pi * 10
        current_vel = current_vel / max_vel

        # Diagonal pair similarity calculations (keep existing)
        diag1_pos_sim = 1.0 - 0.5*(np.abs(current_pos[0] - current_pos[4]) + 
                                np.abs(current_pos[1] - current_pos[5]))
        diag1_vel_sim = 1.0 - 0.5*(np.abs(current_vel[0] - current_vel[4]) +
                                np.abs(current_vel[1] - current_vel[5]))
        diag2_pos_sim = 1.0 - 0.5*(np.abs(current_pos[2] - current_pos[6]) + 
                                np.abs(current_pos[3] - current_pos[7]))
        diag2_vel_sim = 1.0 - 0.5*(np.abs(current_vel[2] - current_vel[6]) +
                                np.abs(current_vel[3] - current_vel[7]))

        # Combined similarity reward
        similarity_reward = 0.5 * (diag1_pos_sim * diag1_vel_sim + 
                                 diag2_pos_sim * diag2_vel_sim)

        # New alternation reward calculation
        # Get activity levels for both diagonal pairs
        pair1_activity = np.mean(np.abs(current_vel[[0, 1, 4, 5]]))  # Left front/Right back
        pair2_activity = np.mean(np.abs(current_vel[[2, 3, 6, 7]]))  # Right front/Left back
        
        # Alternation reward encourages one pair to be active while the other is not
        alternation_reward = (pair1_activity * (1 - pair2_activity) + 
                            pair2_activity * (1 - pair1_activity))

        # Combine both rewards
        gait_reward = (similarity_reward + alternation_reward) * GAIT_FACTOR

        # Final reward calculation
        reward = (FAC_MOVEMENT * movement_forward
                + gait_reward
                - self.step_counter_session/PENALTY_STEPS * (
                    smooth_movement + body_stability
                    + FAC_CLEARANCE * paw_clearance
                    + FAC_SLIP * paw_slipping**2
                    + FAC_ARM_CONTACT * self.arm_contact))
       
        # Termination conditions
        terminated = False
        truncated = False
        info = {}

        self.step_counter += 1
        if self.step_counter > EPISODE_LENGTH:
            self.step_counter_session += self.step_counter
            terminated = False
            truncated = True
        elif self.is_fallen():
            self.step_counter_session += self.step_counter
            reward = 0
            terminated = True
            truncated = False

        # if self.step_counter > EPISODE_LENGTH:
        #     self.step_counter = 0  # just reset counter, don't end episode
        # elif self.is_fallen():
        #     pass  # optional: let it keep going even if it falls

        self.observation = np.hstack((self.state_robot, self.angle_history))

        done = terminated or truncated
        return (np.array(self.observation).astype(np.float32),
                reward, done, info)

    def reset(self, seed=None, options=None):
        self.step_counter = 0
        self.arm_contact = 0
        p.resetSimulation()
        # Disable rendering during loading.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        p.setGravity(0,0,-9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")

        start_pos = [0,0,0.08]
        start_orient = p.getQuaternionFromEuler([0,0,0])

        urdf_path = "models/"
        self.robot_id = p.loadURDF(urdf_path + "bittle_esp32.urdf",
                                start_pos, start_orient,
                                flags=p.URDF_USE_SELF_COLLISION)

        # Initialize urdf links and joints - ONLY STORE THE 8 MOVABLE JOINTS
        self.joint_id = []
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            joint_type = info[2]
            
            # Only include revolute joints (skip fixed joints)
            if joint_type == p.JOINT_REVOLUTE:
                self.joint_id.append(j)
                p.changeDynamics(self.robot_id, j, maxJointVelocity=np.pi*10)

        # Verify we have exactly 8 joints
        if len(self.joint_id) != 8:
            raise ValueError(f"Expected 8 movable joints, found {len(self.joint_id)}")

        # Setting start position - use only 8 values since we have 8 joints
        joint_angs = np.deg2rad(np.array([1, 1, 1, 1, 1, 1, 1, 1]) * 27)

        # Reset joint states - only for our 8 movable joints
        for i, j in enumerate(self.joint_id):
            p.resetJointState(self.robot_id, j, joint_angs[i])

        # Normalize joint angles (for observation space)
        joint_angs_norm = joint_angs / self.bound_ang

        # Read robot state (pitch, roll and their derivatives of the torso)
        state_ang = p.getBasePositionAndOrientation(self.robot_id)[1]
        state_vel = np.asarray(p.getBaseVelocity(self.robot_id)[1])
        state_vel = state_vel[0:2]*ANG_FACTOR
        self.state_robot = np.concatenate((state_ang, np.clip(state_vel, -1, 1)))

        # Initialize robot state history with reset position
        self.angle_history = np.tile(joint_angs_norm, LENGTH_JOINT_HISTORY)
        self.recent_angles = np.tile(joint_angs_norm, LENGTH_RECENT_ANGLES)
        self.observation = np.concatenate((self.state_robot, self.angle_history))
        
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        info = {}
        return np.array(self.observation).astype(np.float32)
        
    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()

    def is_fallen(self):
        """ Check if robot is fallen. It becomes "True",
            when pitch or roll is more than 1.3 rad.
        """
        pos, orient = p.getBasePositionAndOrientation(self.robot_id)
        orient = p.getEulerFromQuaternion(orient)
        is_fallen = (np.fabs(orient[0]) > 1.3
                    or np.fabs(orient[1]) > 1.3)

        # 检测主干和地面碰撞
        torso_idx = 0
        if p.getContactPoints(bodyA=self.robot_id, linkIndexA=torso_idx, bodyB=self.plane_id):
            is_fallen = True
            
        return is_fallen

    def randomize(self, value, percentage):
        """ Randomize value within percentage boundaries.
        """
        percentage /= 100
        value_randomized = value * (1 + percentage*(2*np.random.rand()-1))

        return value_randomized
