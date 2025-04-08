### Create environment
```
conda env create -f environment.yml
conda activate push_mujoco
```

# Todo
## Env
- add camera (DONE)
- complete get obs (DONE)
- complete get rewards (DONE)
- complete check  done (DONE)
- use IK for action space (IK Works, not using for now)
- More Termination cases to speed up convergence
- Delta actions? either from start qpos or current qpos
- Change viewer to render OR add renderer option
    - Save Images during training

## PPO
- PPO base code with two NN (DONE)
- Integrate Actor & Critic into Single NN (DONE)
- Integrate with PushNN (DONE)
- Integrate with Custom Env (DONE)
- Use vectorised environment (DONE, sync though)
- ASYNC Vectorised environment (some issue with OpenGL)
- Entropy Loss
- Logging/Wandb
- Saving Models

## Network
- Make a basic network (DONE)
- Maybe having a seperate Actor and Critic is better
- Two image input (but this is expensive space wise, could have lower res image)
- verify if the action and log probs are accurate
