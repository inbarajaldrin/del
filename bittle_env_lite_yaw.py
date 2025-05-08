import gym
import numpy as np
import pybullet as p
import pybullet_data

# Constants to define training and visualisation.
GUI_MODE = True            # Set "True" to display pybullet in a window
EPISODE_LENGTH = 250        # Number of steps for one training episode
MAXIMUM_LENGTH = 2e6        # Number of total steps for entire training

# Factors to weight rewards and penalties.
PENALTY_STEPS = 2e6         # Increase of penalty by step_counter/PENALTY_STEPS
FAC_LATERAL = 1500          # Reward movement in y-direction (right)
FAC_STABILITY = 0.5         # Punish body roll and pitch velocities
FAC_Z_VELOCITY = 0.0        # Punish z movement of body
FAC_SLIP = 0.0              # Punish slipping of paws
FAC_ARM_CONTACT = 0.01      # Punish crawling on arms and elbows
FAC_SMOOTH_1 = 1.0          # Punish jitter and vibrational movement, 1st order
FAC_SMOOTH_2 = 1.0          # Punish jitter and vibrational movement, 2nd order
FAC_CLEARANCE = 0.0         # Factor to enforce foot clearance to PAW_Z_TARGET
PAW_Z_TARGET = 0.005        # Target height (m) of paw during swing phase

BOUND_ANG = 110             # Joint maximum angle (deg)
STEP_ANGLE = 11             # Maximum angle (deg) delta per step
ANG_FACTOR = 0.1            # Improve angular velocity resolution before clip.

GAIT_FACTOR = 0.5           # Weight for diagonal gait pattern reward

# Values for randomization, to improve sim to real transfer.
RANDOM_GYRO = 0             # Percent
RANDOM_JOINT_ANGS = 0       # Percent
RANDOM_MASS = 0             # Percent, currently inactive
RANDOM_FRICTION = 0         # Percent, currently inactive

LENGTH_RECENT_ANGLES = 3    # Buffer to read recent joint angles
LENGTH_JOINT_HISTORY = 30   # Number of steps to store joint angles.

# Size of observation space
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
        else:
            p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                    cameraYaw=-170,
                                    cameraPitch=-40,
                                    cameraTargetPosition=[0.4,0,0])

        self.action_space = gym.spaces.Box(np.array([-1]*8), np.array([1]*8))
        self.observation_space = gym.spaces.Box(np.array([-1]*SIZE_OBSERVATION),
                                                np.array([1]*SIZE_OBSERVATION))

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        last_position = p.getBasePositionAndOrientation(self.robot_id)[0][1]  # Y-coordinate
        joint_angs = np.asarray(p.getJointStates(self.robot_id, self.joint_id), dtype=object)[:,0]
        ds = np.deg2rad(STEP_ANGLE)
        joint_angs += action * ds

        # Apply joint boundaries
        min_ang = -self.bound_ang
        max_ang = self.bound_ang
        joint_angs = np.clip(joint_angs, min_ang, max_ang)

        # Transform angle to degree and perform rounding
        joint_angsDeg = np.rad2deg(joint_angs.astype(np.float64)).round()
        joint_angs = np.deg2rad(joint_angsDeg)

        p.stepSimulation()

        # Contact and slipping checks
        paw_contact = []
        paw_idx = [3, 6, 9, 12]
        for idx in paw_idx:
            paw_contact.append(True if p.getContactPoints(bodyA=self.robot_id, linkIndexA=idx) else False)

        paw_slipping = 0
        for in_contact in np.nonzero(paw_contact)[0]:
            paw_slipping += np.linalg.norm(
                (p.getLinkState(self.robot_id, linkIndex=paw_idx[in_contact], computeLinkVelocity=1)[0][0:1]))

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

        # Read robot state
        state_pos, state_ang = p.getBasePositionAndOrientation(self.robot_id)
        state_vel = np.array(p.getBaseVelocity(self.robot_id)[1][0:2]) * ANG_FACTOR
        state_vel_clip = np.clip(state_vel, -1, 1)
        self.state_robot = np.concatenate((state_ang, state_vel_clip))
        current_position = p.getBasePositionAndOrientation(self.robot_id)[0][1]  # Y-coordinate

        # Calculate rewards
        movement_right = current_position - last_position
        
        # Diagonal gait reward calculation
        joint_states = [p.getJointState(self.robot_id, joint) for joint in self.joint_id]
        current_pos = np.array([state[0] for state in joint_states])/self.bound_ang
        current_vel = np.array([state[1] for state in joint_states]) / (np.pi * 10)

        # Diagonal pair coordination
        diag1_sim = 1.0 - 0.5*(np.abs(current_pos[0] - current_pos[4]) + np.abs(current_vel[0] - current_vel[4]))
        diag2_sim = 1.0 - 0.5*(np.abs(current_pos[2] - current_pos[6]) + np.abs(current_vel[2] - current_vel[6]))
        gait_reward = 0.5 * (diag1_sim + diag2_sim)

        # Final reward calculation
        reward = (
            FAC_LATERAL * movement_right
            + GAIT_FACTOR * gait_reward
            - self.step_counter_session / PENALTY_STEPS * (
                np.sum(FAC_SMOOTH_1 * np.abs(joint_angs_norm - self.recent_angles[-16:-8])**2)
                + FAC_STABILITY * (state_vel_clip[0]**2 + state_vel_clip[1]**2)
                + FAC_ARM_CONTACT * self.arm_contact
            )
        )


        # Termination conditions
        terminated = False
        truncated = False
        info = {}

        self.step_counter += 1
        if self.step_counter > EPISODE_LENGTH:
            self.step_counter_session += self.step_counter
            truncated = True
        elif self.is_fallen():
            self.step_counter_session += self.step_counter
            reward = 0
            terminated = True

        self.observation = np.hstack((self.state_robot, self.angle_history))

        return (
            np.array(self.observation).astype(np.float32),
            reward,
            terminated or truncated,
            False,  # This is the 'truncated' flag (different from the 'truncated' variable above)
            info    # Make sure this is defined earlier in your step method
        )

    def reset(self, seed=None, options=None):
        self.step_counter = 0
        self.arm_contact = 0
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        p.setGravity(0,0,-9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")

        start_pos = [0,0,0.08]
        start_orient = p.getQuaternionFromEuler([0,0,0])

        self.robot_id = p.loadURDF("models/bittle_esp32.urdf",
                                start_pos, start_orient,
                                flags=p.URDF_USE_SELF_COLLISION)

        # Initialize joints
        self.joint_id = []
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            if info[2] == p.JOINT_REVOLUTE:
                self.joint_id.append(j)
                p.changeDynamics(self.robot_id, j, maxJointVelocity=np.pi*10)

        # Reset joint states
        joint_angs = np.deg2rad(np.array([1, 1, 1, 1, 1, 1, 1, 1]) * 27)
        for i, j in enumerate(self.joint_id):
            p.resetJointState(self.robot_id, j, joint_angs[i])

        # Initialize observations
        joint_angs_norm = joint_angs / self.bound_ang
        state_ang = p.getBasePositionAndOrientation(self.robot_id)[1]
        state_vel = np.clip(
            np.asarray(p.getBaseVelocity(self.robot_id)[1][0:2]) * ANG_FACTOR,
            -1,
            1
        )
        
        self.state_robot = np.concatenate((state_ang, state_vel))
        self.angle_history = np.tile(joint_angs_norm, LENGTH_JOINT_HISTORY)
        self.recent_angles = np.tile(joint_angs_norm, LENGTH_RECENT_ANGLES)
        
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        
        # Return both observation and info dictionary
        observation = np.array(np.concatenate((self.state_robot, self.angle_history))).astype(np.float32)
        return observation, {}
                
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
