# [IP1] Pendulum Acrobatics

Integrated Project WiSe 2022/23 by Florian Wolf, supervised by Pascal Klink and Kai Ploeger.

## Abstract
For decades, cart-pole systems have been an important benchmark problem for dynamics and control theory.
Due to their high level of nonlinearity, instability and underactuation, they require fast-reactive
controllers and are widely used to test emerging control approaches. However, the possible motion
profiles of the classical cart-pole system are limited by the rectilinear cart motion. Furthermore, the
system is typically investigated in isolation, directly allowing the cart to be displaced. In this work,
we investigate a three-dimensional spherical cart-pole system that is realized at the end-effector of a
Barret WAM robotic arm, allowing for more challenging motions to be generated while simultaneously
introducing a kinematic subtask in the control problem. We benchmark different MPC control schemes on
both simple setpoint reaches as well as the generation of circular and spiral trajectories. The best
performing method, that we implement on top of the Crocoddyl library, delivers convincing simulated
results on all investigated trajectories.
