# META
- Get into some chats and stuff

# IDEAS
Check notes

# NEW PLAN
Get basics right and be competitive with them.
Train in Cloud.

The maindifficulty in training a DNN is then associated with the scheduling of the learning rate and the
amount of L2 weight decay regularization employed. 

Snapmix adaptation? Mixing with null class?

# b4 learning
rate max: 5e-3 ???

# Train on instances!
FP16?

# Practical stuff
tmux
docker and pycharm?
fp16
automatic setup of instances?

# Look at augmentations
Really look at them!

# Balanced sampling

# Class weights

# Other modifications to loss functions, other loss functions

# What seemed to work in the runs
The smaller wd seemed to work well, with augs.
The cosine run seemed to work well.
Play some more with learning rates, wd.

Does batch size also have an impact on weight decay choice?

# Balanced validation set?

# Another scheduling variant
CosineAnnealingLR

# Try 1cycle as per Leslie paper

# Proper learning rate finder as per Leslie paper

# Also tune momentum

# Augmentations
Grid looks promising?
Snapmix / Mosiac augment

Combine grid by using the color information, or by using activation maps
Maybe treat the strange images with few leaf patches differently?

# The patches method with small model was not even that bad!

# Properly validate augs
on/off runs