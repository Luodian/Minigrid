uv venv --python=3.13
source .venv/bin/activate

uv pip install -e .
uv pip install minigrid
uv pip install 'stable-baselines3[extra]'
uv pip install "gymnasium[box2d]"
uv pip install "gymnasium[atari,accept-rom-license]" shimmy
uv pip install ale-py autorom
AutoROM --accept-license

python -c "import gymnasium as gym; import ale_py; gym.register_envs(ale_py); print('Available Atari envs:'); [print(env_id) for env_id in gym.registry.keys() if 'Breakout' in env_id]"