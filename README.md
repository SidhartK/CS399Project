# CS399Project

## Experiment 1
### Background

In this experiment, we will be working with a simple POMDP which has a task that runs for `T` timesteps. However, at each timestep there is a `p < 1` probability of failure meaning essentially that the job will error. However, the job will not immediately restart. Instead we must make take an observation action to observe the state of the system. Such an observation action has a fixed cost of `c`. If we observe that the job has failed, then we will restart that job starting from the next timestep (i.e. if we make the observation that the job has errored at timestep `t`, then the job will restart at timestep `t+1` and )

### Hypothesis
