# reference: https://github.com/Phylliade/ikpy/blob/master/tutorials/Quickstart.ipynb
# urdf from: https://cobra.cps.cit.tum.de/robots

import ikpy.chain
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image

np.set_printoptions(precision=3, suppress=True, linewidth=100)

panda_chain = ikpy.chain.Chain.from_urdf_file("panda.URDF", base_elements=["panda_link0"])

target_position = [0.35, 0, 0.25]

# default x0: [0. 0. 0. 0. 0. 0. 0. 0. 0.]
# lb: [  -inf -2.967 -1.833 -2.967 -3.142 -2.967 -0.087 -2.967   -inf]
# ub: [   inf  2.967  1.833  2.967 -0.4    2.967  3.822  2.967    inf]
# default x0 will result in ValueError: `x0` is infeasible. because out of bounds
# so need initial_guess
# index 0, 7, 8 is fixed joint
initial_guess = [0, 0, 0, 0, -1.571, 0, 0, 0, 0]

result = panda_chain.inverse_kinematics(target_position, initial_position=initial_guess)
print("The angles of each joints are : ", result)
# result = [ 0.    -0.168  0.365 -0.179 -2.77   2.004  0.587  0.     0.   ]

verify_result = panda_chain.forward_kinematics(result)
print("The forward kinematics result is : ", verify_result)


xml_path = "franka_emika_panda/scene_push.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
data.qpos[:7] = [0, -0.168, 0.365, -0.179, -2.77, 2.004, 0.587]
with mujoco.Renderer(model) as renderer:
  mujoco.mj_forward(model, data)
  renderer.update_scene(data)
  image = renderer.render()
  pil_image = Image.fromarray(image)
  pil_image.save('test_ik.png')
