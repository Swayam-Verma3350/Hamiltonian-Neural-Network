# Hamiltonian Neural Networks for Learning Physical Dynamics

## Abstract

Neural networks can learn the equations of motion of a physical system from data, but a network that predicts time derivatives directly has no reason to respect conservation laws, so simulations built from it slowly gain or lose energy. A Hamiltonian Neural Network (HNN) instead learns a single scalar function H(q, p) and derives the motion from Hamilton's equations by automatic differentiation, so the learned energy is conserved exactly by construction. This project compares an HNN with a standard baseline network of identical size, data and training procedure on four problems: an ideal mass-spring, an ideal pendulum, damped versions of both (with a sweep over the friction strength), and a two-body gravitational orbit with an 8-dimensional state. On the frictionless systems the two models fit single-step derivatives about equally well, but the HNN is a much better simulator: its energy error is about 140 times lower on the mass-spring, 1.7 times lower on the pendulum, and about 70,000 times lower on the two-body problem. With friction the advantage disappears: on the damped pendulum the HNN's energy error is about 600 times and its coordinate error about 50 times higher than the baseline's. The results show that building the right physical structure into a model helps exactly as long as the structure matches the system.

## 1. Introduction

Learning dynamics from data is a basic problem in scientific machine learning. The usual approach is to train a network f so that f(state) approximates d(state)/dt, then simulate by integrating f forward in time. This works for short horizons, but small errors in f accumulate. For an isolated physical system the total energy is constant, yet a learned f has no mechanism that keeps the simulated energy constant, so long simulations drift and spiral.

Hamiltonian Neural Networks (Greydanus et al., 2019) address this by changing what the network represents. Instead of the vector field itself, the network represents the energy function, and the vector field is obtained from it through the structure of Hamiltonian mechanics. The question investigated here is simple: does this change improve the learned model, when does it help most, and when does it hurt?

The project makes the following contributions:

- A from-scratch PyTorch implementation of a baseline network and an HNN that share one architecture, one dataset and one training loop, so the comparison isolates a single design choice.
- Evaluation on three kinds of metric: single-step derivative error, long-horizon energy error, and long-horizon trajectory error.
- Experiments on four physical systems of increasing difficulty, including a deliberate test of a case where the HNN's assumption is false (friction).
- A parameter study over the damping strength γ, and a larger two-body problem with an 8-dimensional state.

## 2. Background

### 2.1 Hamiltonian mechanics

A conservative system with n degrees of freedom has positions q = (q1, ..., qn) and conjugate momenta p = (p1, ..., pn). Its dynamics are determined by one scalar function, the Hamiltonian H(q, p), which for the systems here is the total energy. Hamilton's equations are

- dq/dt = ∂H/∂p
- dp/dt = −∂H/∂q

Two consequences matter for this project. First, H is constant along any solution, since dH/dt = (∂H/∂q)(dq/dt) + (∂H/∂p)(dp/dt) = 0. Second, the vector field (∂H/∂p, −∂H/∂q) has zero divergence, so the flow preserves phase-space volume. These properties hold for any smooth function H, which is what makes them easy to build into a model.

### 2.2 Baseline network

The baseline is a multilayer perceptron that takes the state x = (q, p) and outputs a vector of the same size, trained to match the true derivatives (dq/dt, dp/dt). It is flexible but unconstrained: nothing forces the learned field to conserve anything.

### 2.3 Hamiltonian Neural Network

The HNN uses the same perceptron but with a single scalar output, interpreted as the energy H(q, p). The derivatives are computed by automatic differentiation of that scalar with respect to the input and rearranged as in Hamilton's equations: (dq/dt, dp/dt) = (∂H/∂p, −∂H/∂q). The training loss is the same mean squared error between predicted and true derivatives as for the baseline, so the HNN receives no energy label and no extra penalty. It must discover an energy function whose Hamiltonian flow reproduces the observed motion.

Because the learned field is exactly Hamiltonian, its trajectories conserve the learned H and preserve phase-space volume. This is conservation of the learned energy, not of the true energy: the two agree only to the extent that the network has learned H correctly (up to an additive constant).

## 3. Methods

### 3.1 Physical systems

| System            | Hamiltonian                              | Parameters and data settings                                                                                                              |
| ----------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Ideal mass-spring | H = p²/(2m) + ½ k q²                     | m = 0.5, k = 2 (angular frequency 2). Start radius in (q, p) from 0.1 to 1.0. 30 points per trajectory. Evaluation start (q, p) = (1, 0). |
| Ideal pendulum    | H = p²/(2 m l²) + m g l (1 − cos q)      | m = 0.5, l = 1, g = 9.8. Start radius 1.3 to 2.3. 45 points per trajectory. Evaluation start (2.1, 0).                                    |
| Damped versions   | same H, with an extra term −γ p in dp/dt | γ = 0.1 in the main experiment. Sweep over γ ∈ {0, 0.05, 0.1, 0.2, 0.4}.                                                                  |
| Two-body problem  | H = ½ Σ \|p\_i\|² − 1/\|q1 − q2\|        | Equal masses and G equal to 1. Eight-dimensional state (qx1, qx2, qy1, qy2, px1, px2, py1, py2).                                          |

The damped systems are the one case where the HNN's central assumption is false: friction removes energy, so no Hamiltonian vector field can describe them. They are included on purpose to test where the method breaks.

### 3.2 Data generation

For the mass-spring, pendulum and damped systems, 50 trajectories were generated. Each starts at a random direction in the (q, p) plane at a radius drawn uniformly from the system's radius range, and is integrated for 3 time units with an adaptive Runge-Kutta solver (relative tolerance 1e-10). The exact derivatives from the equations of motion are the labels. Gaussian noise with standard deviation 0.1 is then added to the input states only, to imitate noisy measurements, while the labels stay exact. The first half of the pooled points (the first 25 trajectories) is used for training and the second half for testing, so no trajectory appears in both. This gives 750 training and 750 test points for the mass-spring and 1125 and 1125 for the pendulum. The random seed is 0.

For the two-body problem, 1000 orbits of 50 points over 20 time units were generated, with exact derivatives and no input noise. Each orbit starts from a near-circular configuration: body 1 at a random position with both coordinates in \[0.5, 1.5\], body 2 mirrored through the origin (zero centre of mass and zero total momentum), and a speed equal to the circular-orbit speed multiplied by 1 + 0.05 ε with ε standard normal, which makes the orbit slightly elliptical. The first 20% of points (10,000) are the test set and the remaining 40,000 the training set.

### 3.3 Models

Both models use the same multilayer perceptron: two hidden layers of 200 units with tanh activations, orthogonal weight initialisation, and a final linear layer without bias. The only difference is the output size, which is the state dimension for the baseline and 1 for the HNN. The HNN's derivatives come from torch.autograd.grad with create\_graph enabled, so the training loss can be differentiated through them.

### 3.4 Training

| Setting | Mass-spring, pendulum, damped | Two-body |
| --- | --- | --- |
| Optimiser | Adam, learning rate 1e-3 | Adam, learning rate 1e-3 |
| Weight decay | 1e-4 | none |
| Steps | 2000, full batch | 10,000, minibatch of 200 |
| Loss | mean squared error on derivatives | same |
| Seed | 0 | 0 |

### 3.5 Evaluation metrics

1. **Single-step error.** Mean squared error between predicted and true derivatives on train and test points, with the standard error over elements.
2. **Energy MSE.** The learned vector field is integrated from 15 start points (100 time points up to t = 20 for the one-degree-of-freedom systems; 500 points up to t = 25 for the two-body problem, which is longer than the training orbits). The metric is the mean squared difference between the true energy H evaluated along the model's path and along the true path. Standard errors come from five groups of three start points. For the one-degree-of-freedom systems the start points have coordinates with magnitude between 0.2 and 1.0. For the damped systems the true path loses energy, so this metric also checks whether a model reproduces that loss.
3. **Coordinate MSE.** The mean squared distance between the model's path and the true path in (q, p), up to t = 20, from the single fixed start point listed in Section 3.1. It is not computed for the two-body problem.

Energy MSE separates structure from accuracy: a model can fit single-step derivatives well and still produce a simulation whose energy wanders.

## 4. Results

All numbers are raw values (not scaled), with standard errors where available. The full table is in Appendix A, and the figure files are listed in Appendix B.

### 4.1 Ideal mass-spring

Both models fit the single-step derivatives similarly: test MSE 0.0394 ± 0.0020 for the baseline and 0.0359 ± 0.0018 for the HNN. The difference is about 1.3 combined standard errors, so it is not clearly significant. Train and test errors are almost equal for both, so neither overfits. The error is dominated by the input noise, which no model can remove.

As simulators the two models differ strongly:

| Metric | Baseline | HNN | Ratio |
| --- | --- | --- | --- |
| Energy MSE | 0.0414 ± 0.0102 | 0.000301 ± 0.000118 | about 140 times lower for the HNN |
| Coordinate MSE | 0.371 | 0.0596 | about 6 times lower for the HNN |

Figure 1 shows the baseline's path spiralling inward: its total energy falls from 1.0 to about 0.6 by t = 20 and its coordinate error grows to about 1.0. The HNN stays on the true ellipse with a total energy within about 3% of 1.0, and its coordinate error reaches only about 0.18 at t = 20.

### 4.2 Ideal pendulum

The single-step errors are indistinguishable (test MSE 0.0617 ± 0.0045 baseline, 0.0618 ± 0.0046 HNN). The simulator metrics favour the HNN, but by less than on the spring:

| Metric | Baseline | HNN | Ratio |
| --- | --- | --- | --- |
| Energy MSE | 0.00245 ± 0.00041 | 0.00142 ± 0.00013 | about 1.7 times lower for the HNN |
| Coordinate MSE | 0.637 | 0.182 | about 3.5 times lower for the HNN |

The energy gap is only about two combined standard errors, so it should be read as suggestive. The coordinate gap comes from one start point and one trained model per method. Figure 2 shows where it arises: the two coordinate errors are comparable (the baseline's slightly lower) until about t = 12, after which the baseline's grows quickly to about 3.7 at t = 20 while the HNN's stays below about 0.7. The baseline's total energy falls from about 7.4 to about 6.5, whereas the HNN's stays near 7.4 to 7.5, slightly above the true constant of about 7.37, so its path sits on an orbit whose true energy is slightly off.

### 4.3 Damped mass-spring and pendulum (γ = 0.1)

Here the HNN's assumption fails, and the results change accordingly. The single-step errors are still close for both systems, so a single-step comparison would give no warning. The simulator metrics show the problem:

| System | Metric | Baseline | HNN |
| --- | --- | --- | --- |
| Damped mass-spring | Energy MSE | 0.00823 ± 0.00204 | 0.425 ± 0.107 |
| Damped mass-spring | Coordinate MSE | 0.102 | 0.105 |
| Damped pendulum | Energy MSE | 0.00185 ± 0.00017 | 1.11 ± 0.24 |
| Damped pendulum | Coordinate MSE | 0.0660 | 3.43 |

On the damped spring the HNN's energy error is about 50 times the baseline's, while the coordinate errors are about equal (0.102 against 0.105). On the damped pendulum the HNN is much worse on both: about 600 times higher energy error and about 50 times higher coordinate error. In both cases the HNN's energy error is large because the true energy decays and the HNN, which conserves its learned energy, cannot follow it. Figures 3 and 4 show this directly. In both, the HNN's path stays on the outer orbit and its energy stays constant, while the true energy decays: on the spring to about 0.13 of its initial value by t = 20 (close to the roughly 14% that exp(−γ t) predicts for a harmonic oscillator at γ = 0.1), and on the pendulum from about 7.4 to about 1.1. The baseline follows the decay closely on the pendulum and decays a little too fast on the spring (to about 0.07). On the spring the HNN's coordinate error is lower than the baseline's until about t = 14, then overtakes it and reaches about 0.25 at t = 20, which is why the averaged errors are about equal. On the pendulum the HNN's coordinate error oscillates with peaks up to about 10, while the baseline's stays below about 0.5.

### 4.4 Parameter study: damping strength

The sweep retrains both models for γ ∈ {0, 0.05, 0.1, 0.2, 0.4} on each base system and records the coordinate MSE. At γ = 0 the setting is the frictionless one of Sections 4.1 and 4.2, where the HNN is better; at γ = 0.1 the pendulum shows the HNN far worse (Section 4.3). Figure 5 plots the coordinate MSE of both models against γ on a logarithmic scale (values read from the plot are approximate; exact values are in gamma\_sweep.csv). For the mass-spring, the baseline's error falls steadily with γ, from about 0.37 at γ = 0 to about 0.002 at γ = 0.4, while the HNN's is about 0.06 at γ = 0, dips to about 0.03 at γ = 0.05, then rises to about 0.4 at γ = 0.4. The curves cross at γ ≈ 0.1, where both are about 0.10: the HNN is better below that value and worse above it, by about a factor of 10 at γ = 0.2 and more than 200 at γ = 0.4. For the pendulum the crossover comes much earlier. The HNN is better at γ = 0 (about 0.18 against 0.64), but at γ = 0.05 its error jumps to about 3.4 and stays between about 3 and 3.5 for all larger γ, while the baseline's stays between about 0.01 and 0.08. The baseline's error also falls as γ grows partly because strongly damped paths decay quickly toward rest, which makes coordinate errors small for any model that decays too. The dip in the HNN's spring error at γ = 0.05 is a reminder that each point is a single training run.

### 4.5 Two-body gravitational problem

This is the most demanding test: an 8-dimensional state, strongly nonlinear forces, clean inputs, and a longer training schedule.

| Metric | Baseline | HNN | Ratio |
| --- | --- | --- | --- |
| Train MSE | 3.59e-5 ± 1.3e-6 | 4.63e-6 ± 9.7e-8 | about 8 times lower for the HNN |
| Test MSE | 3.22e-5 ± 8.3e-7 | 4.36e-6 ± 5.8e-8 | about 7 times lower for the HNN |
| Energy MSE | 0.464 ± 0.170 | 6.52e-6 ± 4.6e-7 | about 70,000 times lower for the HNN |

Unlike the one-degree-of-freedom systems, the HNN is also clearly better on the single-step error here, probably because the Hamiltonian is a much simpler object to learn than an eight-component vector field with a singular force. In simulation the HNN's total energy stays flat while the baseline's wanders. For the single start point in Figure 6, the HNN's orbits and its kinetic and potential energy follow the ground truth closely and its total energy stays at about −0.28, whereas the baseline's orbits spiral and its total energy falls below −0.75 within about 5 time units. Figure 7 repeats the energy comparison for five start points over 25 time units, longer than the training orbits, with true total energies between about −0.13 and −0.28. The HNN stays on the true value in all five. The baseline shows sharp excursions around t = 15 to 19: down to about −1.9 and −2.0 in two cases, to about −0.75 in another, and above zero (up to about 1.7 and 1.2) in two, where the two bodies would be unbound (6 and 7).

### 4.6 Summary of findings

| System | Single-step error | Energy error | Coordinate error |
| --- | --- | --- | --- |
| Ideal mass-spring | similar | HNN far better | HNN better |
| Ideal pendulum | similar | HNN better (about 2 standard errors) | HNN better |
| Damped mass-spring | similar | baseline far better | about equal |
| Damped pendulum | similar | baseline far better | baseline far better |
| Two-body | HNN better | HNN far better | not measured |

## 5. Discussion

**Structure, not accuracy, is what the HNN adds.** On the frictionless one-degree-of-freedom systems the HNN and the baseline fit single-step derivatives equally well, because with noisy inputs both are limited by the same noise floor. What the HNN adds is a guarantee about how errors behave over time. A small error in the baseline's field can push the energy up or down every step, while a small error in the HNN's learned H only changes which energy surface the motion follows, never whether energy is conserved. This is why single-step error is a poor indicator of simulation quality, and why the energy and trajectory metrics are needed.

**The gain is larger when learning H is easier than learning the field.** The two-body result supports this. The HNN has to learn one scalar function of eight variables, while the baseline has to learn eight coupled outputs including a singular force, and the HNN is the better fit even for single-step error.

**The pendulum gain is smaller.** A plausible reason, not tested here, is that the energy-evaluation start points have coordinates between 0.2 and 1.0 (radius from about 0.28 to 1.41), whereas the pendulum's training radii are 1.3 to 2.3, so most evaluation starts lie in a region the models never saw. Both models then have to extrapolate, which limits what conservation by construction can fix. The mass-spring training radii (0.1 to 1.0) cover most of the evaluation region.

**The failure under friction is expected and informative.** The damped experiments confirm that the HNN is a hard constraint, not a regulariser. When the constraint matches the physics it helps greatly, and when it does not it cannot be overridden by more data, since the HNN has no way to represent energy loss. The damped spring also shows a limit of the coordinate metric: the averaged errors are similar (0.102 against 0.105), but the HNN is better until about t = 14 and worse after, and the energy metric exposes a 50-fold structural difference. The damping sweep shows how quickly the assumption breaks: on the spring the HNN stays competitive up to γ ≈ 0.1, while on the pendulum even γ = 0.05 makes it more than 100 times worse on coordinate error.

**Role of noise.** Input noise was included to test robustness. It sets the floor for the single-step error (train and test errors are nearly equal). It does not affect the HNN's conservation property, because that property does not depend on the data.

## 6. Conclusion

Building Hamiltonian structure into a neural network does not change how well it fits single-step derivatives on these systems, but it changes how the model behaves as a simulator: with the same data, architecture and training, the HNN reduced long-horizon energy error by a factor of about 140 on the mass-spring and about 70,000 on the two-body problem, and reduced trajectory error on the ideal systems. The benefit disappears when the physics breaks the assumption, as the damped systems show, where the HNN becomes far worse than the unconstrained baseline. The practical lesson is to match the model's built-in structure to the system, and to judge learned dynamics by long-horizon behaviour and not by single-step error alone.

## 7. Reproducibility

The full notebook, trained weights, figures and metric tables are in the repository linked at the top. Running the notebook from top to bottom regenerates every result with seed 0. The settings are the ones in Section 3: noise 0.1 on the inputs of the spring, pendulum and damped systems, 2000 full-batch steps for the one-degree-of-freedom systems, and 10,000 steps with batch size 200 for the two-body problem.

## References

1. S. Greydanus, M. Dzamba and J. Yosinski, "Hamiltonian Neural Networks", Advances in Neural Information Processing Systems (NeurIPS), 2019.

## Appendix A. Full metrics

Mean ± standard error. Train and test are derivative MSE; energy and coordinate MSE are from simulated paths.

| System | Model | Train MSE | Test MSE | Energy MSE | Coordinate MSE |
| --- | --- | --- | --- | --- | --- |
| Ideal mass-spring | Baseline | 0.0402 ± 0.0021 | 0.0394 ± 0.0020 | 0.0414 ± 0.0102 | 0.371 |
| Ideal mass-spring | HNN | 0.0369 ± 0.0019 | 0.0359 ± 0.0018 | 0.000301 ± 0.000118 | 0.0596 |
| Ideal pendulum | Baseline | 0.0618 ± 0.0041 | 0.0617 ± 0.0045 | 0.00245 ± 0.00041 | 0.637 |
| Ideal pendulum | HNN | 0.0610 ± 0.0041 | 0.0618 ± 0.0046 | 0.00142 ± 0.00013 | 0.182 |
| Damped mass-spring (γ=0.1) | Baseline | 0.0371 ± 0.0019 | 0.0365 ± 0.0019 | 0.00823 ± 0.00204 | 0.102 |
| Damped mass-spring (γ=0.1) | HNN | 0.0375 ± 0.0020 | 0.0364 ± 0.0019 | 0.425 ± 0.107 | 0.105 |
| Damped pendulum (γ=0.1) | Baseline | 0.0665 ± 0.0044 | 0.0660 ± 0.0047 | 0.00185 ± 0.00017 | 0.0660 |
| Damped pendulum (γ=0.1) | HNN | 0.0697 ± 0.0044 | 0.0681 ± 0.0047 | 1.11 ± 0.24 | 3.43 |
| Two-body | Baseline | 3.59e-5 ± 1.3e-6 | 3.22e-5 ± 8.3e-7 | 0.464 ± 0.170 | n/a |
| Two-body | HNN | 4.63e-6 ± 9.7e-8 | 4.36e-6 ± 5.8e-8 | 6.52e-6 ± 4.6e-7 | n/a |

## Appendix B. Figure index

The plots are in the repository folder results/figures. For each one-degree-of-freedom system the figure has four panels: the predicted phase-space path, the coordinate MSE over time, the energy function learned by the HNN evaluated along each path (flat for the HNN by construction), and the true total energy along each path.

| Figure | File | Content |
| --- | --- | --- |
| 1 | Ideal\_mass\_spring.png | Ideal mass-spring: ground truth, baseline and HNN |
| 2 | Ideal\_pendulum.png | Ideal pendulum |
| 3 | Damped\_mass\_spring\_0\_1.png | Damped mass-spring, γ = 0.1 |
| 4 | Damped\_pendulum\_0\_1.png | Damped pendulum, γ = 0.1 |
| 5 | gamma\_sweep.png | Coordinate MSE against damping γ, both systems (panel titles name the undamped base system) |
| 6 | two\_body\_energy\_and\_trajectories.png | Kinetic, potential and total energy and the two orbits, for one start point |
| 7 | two\_body\_energy\_grid.png | Total energy against time for five start points |
