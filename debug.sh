# Need to py3.9 since newer lp nightly builds are only on py3.9
conda create -n mava-new python=3.9 -y
conda activate mava-new
pip install -r requirements.txt
# We need a newer version of numpy, rlax forces <1.23, but we need a newer version for tf nightly etc.
pip install numpy -U 
# Remove dm-launchpad - we need only nightly one.
pip uninstall dm-launchpad -y
# Re-install lp nightly
pip install --upgrade --force-reinstall dm-launchpad-nightly
# Ensure we have correct protobuff
pip install protobuf==3.19.6
python examples/debugging/simple_spread/feedforward/decentralised/run_idqn.py