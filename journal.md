# January 29, 2021
Updating the scheduler has side effects on optimizer, beyound changing learning rate.
Also consider momentum, initial_lr and possibly others.

# February 1, 2021
- Modifying the heads does not seem to really matter, e.g. adding more dense layers,
  play with global pooling strategies.
    
- Resnext: Possibly a good starting point next to EfficientNet as well.
Heroseo uses resnext50_32x4d and tf_efficientnet_b3_ns

- When chaining learning rate schedulers, updating the optimizer can be tricky, including updating the momentum.

- 
