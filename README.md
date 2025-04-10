### Create environment
```
conda env create -f environment.yml
conda activate push_mujoco
```

# Todo
## Env
- Top view camera (DONE)
- complete get obs (DONE)
- complete get rewards (DONE)
- complete check  done (DONE)
- use IK for action space (IK Works, not using for now)
- Side view camera (DONE)
- render_mode = "human" only (DONE)
- Image input to 0-1 (DONE) Not really normalised but is okay for now
- Start positions to be easier (DONE)
    - block always infront (DONE)
    - start at correct level (DONE)
- Add height penalty (DONE)
- More Termination cases to speed up convergence
    - Too far from object 0.25 in xy plane (DONE)
    - Too far from object 0.3 in z height (DONE)
- Delta actions (makes sense since intial actions close to 0) (DONE)
- Privileged Actor

## PPO
- PPO base code with two NN (DONE)
- Integrate Actor & Critic into Single NN (DONE)
- Integrate with PushNN (DONE)
- Integrate with Custom Env (DONE)
- Use vectorised environment (DONE, sync though)
- ASYNC Vectorised environment (DONE some issue with OpenGL, run $MUJOCO_GL=egl first)
- Entropy Loss (DONE)
- Simple Logging (DONE)
- Saving Models (DONE)
- Saving Images (DONE)
- wandb logging

## Network
- Make a basic network (DONE)
- Maybe having a seperate Actor and Critic is better (DONE)
- Two image input (DONE)
- Parameter log_std (DONE)
- Two image network (DONE)
- Increased state encoder depth (DONE)
- verify if the action and log probs are accurate
