# quadruped-fall-recovery-rl
Hierarchical reinforcement learning framework for robust fall recovery and locomotion in quadruped robots.

<table>
  <thead>
    <tr>
      <th>Paper</th>
      <th>Year</th>
      <th>Robot</th>
      <th>Task</th>
      <th>Method</th>
      <th>State</th>
      <th>Action</th>
      <th>Reward</th>
      <th>Key Idea</th>
      <th>Strength</th>
      <th>Weakness</th>
      <th>Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <a href="https://arxiv.org/pdf/2110.05457">
        Legged Robots that Keep on Learning
        </a>
      </td>
      <td>2021</td>
      <td>Unitree A1</td>
      <td>Real-world fine-tuning + recovery</td>
      <td>Off-policy RL (REDQ)</td>
      <td>
        Proprioception (IMU orientation, joint angles, prev actions) + goal (target root pose, joint angles)
      </td>
      <td>Joint torque</td>
      <td>
        Imitation-based: tracking joint pose/velocity, end-effector position, root motion
      </td>
      <td>
        Autonomous real-world fine-tuning enabled by learned recovery for continuous policy improvement
      </td>
      <td>
        autonomous, sample-efficient (REDQ), stable off-policy RL, real-world, automated recovery, multi-task
      </td>
      <td>
        environment-specific fine-tuning, limited generalization
      </td>
      <td>
        <a href="https://github.com/lauramsmith/fine-tuning-locomotion">Code</a>
      </td>
    </tr>
  </tbody>
</table>

