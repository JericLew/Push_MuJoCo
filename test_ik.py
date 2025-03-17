# reference: https://github.com/Phylliade/ikpy/blob/master/tutorials/Quickstart.ipynb

import ikpy.chain
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image

import os

np.set_printoptions(precision=3, suppress=True, linewidth=100)

cwd = os.getcwd()
panda_chain = ikpy.chain.Chain.from_urdf_file(os.path.join(cwd, "franka_emika_panda/panda.URDF"), base_elements=["link0"])
# print("panda_chain: ", panda_chain.links)

# TODO: add end-effector
# [joint_0 (for base_link, always 0), joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]
initial_joints = [0, 0, 0, 0, -1.57079, 0, 1.57079, -0.7853] # 'home' from scene_push.xml
print(f"FK result of initial joints: \n{panda_chain.forward_kinematics(initial_joints)}")

# # result of above
# matrix = np.array([
#     [0.707,  0.707,  0.0,    0.554],
#     [0.707, -0.707, -0.0,   -0.0],
#     [-0.0,    0.0,  -1.0,    0.732],
#     [0.0,     0.0,   0.0,    1.0]
# ])

matrix = np.array([
    [0.707,  0.707,  0.0,    0.554],
    [0.707, -0.707, -0.0,   -0.0],
    [-0.0,    0.0,  -1.0,    0.732],
    [0.0,     0.0,   0.0,    1.0]
])

guess = [0, 0, 0, 0, -1.57079, 0, 1.57079, -0.7853] # 'home' from scene_push.xml
# guess = [0] * 8 # NOTE: this dont work. need a good enuf initial guess

result = panda_chain.inverse_kinematics_frame(target=matrix, initial_position=guess)
print(f"IK result: \n{result}")
# [ 0.     0.    -0.001  0.    -1.571 -2.002  1.57  -0.785]
# ^ hmmm need to debug why index 6 got problem

# NOTE: both same result
# result = ikpy.inverse_kinematics.inverse_kinematic_optimization(chain=panda_chain, target_frame=matrix, starting_nodes_angles=guess)
# print(f"IK result (optimization): \n{result}")

verify_result = panda_chain.forward_kinematics(result)
print(f"verify result: \n{verify_result}")

# NOTE: the red box floating is [0.554, 0, 0.732] FK pos result of initial_joints
xml_path = "franka_emika_panda/scene_push.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
data.qpos[:7] = result[1:]
with mujoco.Renderer(model) as renderer:
  mujoco.mj_forward(model, data)
  renderer.update_scene(data)
  image = renderer.render()
  pil_image = Image.fromarray(image)
  pil_image.save('test_ik.png')
