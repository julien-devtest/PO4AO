## PO4AO Project Summary

**Purpose:**
The PO4AO (Predictive Optimized Adaptive Optics) project implements an advanced adaptive optics control system, likely for the GHOST spectrograph on the Gemini South telescope. It uses a model-based reinforcement learning (RL) approach to predict and correct atmospheric turbulence, aiming to improve the image quality provided to the instrument.

**Architecture & Key Components:**

The system is built around a closed-loop AO architecture with RL-driven control:

1.  **Core Control Loop (`po4ao/po4ao.py`):**
    *   **Environment Interaction:** Communicates with an external AO system (real or simulated, possibly via "COSMIC" and "Cacao" interfaces) using shared memory (`commands_from_cosmic`, `commands_to_cosmic`) and GPU buffers for speed.
    *   **State Definition:** The state comprises current and a history (`n_history`) of past wavefront sensor (WFS) measurements (projected to DM space) and past DM commands (actions).
    *   **Action Execution:** The RL policy determines the optimal DM commands (voltages). These commands are adjusted for system leak, clamped, and sent through a FIFO buffer to account for control delays before being applied to the DM.
    *   **WFS Readout:** Receives new WFS data after applying commands.

2.  **Reinforcement Learning Agent:**
    *   **Policy Model (`ConvPolicyFastFast` in `po4ao/po4ao_models.py`):** A convolutional neural network that takes the current state (including history) and outputs the DM command image. It incorporates a KL mode projection (`KL_projection`) and clamps outputs.
    *   **Dynamics Model (`EnsembleDynamicsFast` in `po4ao/po4ao_models.py`):** An ensemble of convolutional neural networks that predicts the next WFS state given the current state and action. This model allows the policy to "plan" by simulating the outcomes of its actions.
    *   **Experience Replay (`EfficientExperienceReplay` in `po4ao/po4ao_util.py`):** Stores (state, action, next_state) tuples from interactions. It has separate buffers for general experience and "warmup" experience. Data is stored on CUDA devices.
    *   **Loss Function (`loss_fn` in `po4ao.py`):** Guides policy training by penalizing residual wavefront error (the predicted next state) and the magnitude of control actions.

3.  **Training Process (`po4ao.py`, `po4ao_util.py`):**
    *   **Warmup Phase:** Initializes the system by running episodes with a simpler controller (e.g., an integrator with added noise, `run_episode_warmup`) to populate the replay buffers.
    *   **Parallel Training (`training_thread`):** The dynamics and policy models are trained in a separate process on a designated GPU (`device1`). This allows the main control loop (on `device0`) to continue operating.
        *   **Dynamics Training (`train_dynamics`):** The dynamics model is trained to accurately predict `next_state` from `(state, action)` pairs sampled from replay buffers.
        *   **Policy Training (`train_policy`):** The policy is trained to minimize the cumulative loss over a `planning_horizon`, using the dynamics model to predict future states resulting from its actions.
    *   **Optimizer (`SharedAdam`):** A version of the Adam optimizer modified to share its internal state across processes, crucial for the parallel training setup.
    *   **Model Synchronization:** The policy model used in the control loop (`policy_copy` on `device0`) is periodically updated with the weights from the policy being trained (`policy` on `device1`).

4.  **Configuration (`po4ao_config.py`):**
    *   A central dictionary holds all critical parameters for RL (episode lengths, learning rates, noise levels), model architectures (filter counts), MDP settings (history length, planning horizon), replay buffer sizes, integrator settings, and save/load options.

5.  **Interfaces & Data:**
    *   **COSMIC/Cacao:** External interfaces for interacting with the AO hardware or a high-fidelity simulation environment.
    *   **DM Coordinates & KL Modes:** Static data (`dm_coord.mat`, `KL_PTT_ifun_BMC492_cap6.fits`) defining DM actuator geometry and Karhunen-Loève modes for modal control/projection.
    *   **Atmospheric Data (`1-stage-wind-profiles.txt`):** Provides realistic wind and turbulence profiles (Cn2), likely for simulation or testing the system's performance under various atmospheric conditions.

**Workflow:**

1.  **Initialization:**
    *   Load configuration.
    *   Initialize SHMs and GPU buffers for communication with the AO system.
    *   Load DM geometry, KL modes.
    *   Initialize policy and dynamics models, optimizers, and replay buffers.
    *   Optionally load pre-trained models or replay buffers.

2.  **Warmup (if not loading buffers/models):**
    *   Run `warmup_episodes` using `run_episode_warmup`. This involves a basic controller (e.g., integrator) with exploration noise.
    *   Populate `replay_warmup` and `replay` buffers.
    *   Optionally pre-train dynamics and policy models using the warmup data.

3.  **Main RL Loop:**
    *   The control loop (`run_episode_policy`) runs for a defined number of `iterations`:
        *   The current policy (`policy_copy`) observes the state from the AO system.
        *   It computes and applies DM commands.
        *   The resulting `(obs, action, next_obs)` is stored in the `replay` buffer.
    *   Concurrently, the `training_thread`:
        *   Waits for new data to be available in the replay buffers.
        *   Samples batches from `replay` and `replay_warmup`.
        *   Trains the `dynamics` model.
        *   Trains the `policy` model using the `dynamics` model for predictions.
        *   Periodically, the `policy_copy` in the control loop is updated with the latest trained `policy`.

4.  **Saving:**
    *   Periodically or at the end, models, replay buffers, and reward data are saved.

This system aims to achieve superior AO correction by learning a predictive model of the AO system's behavior and optimizing its control policy through RL, tailored for the specific conditions and hardware of the GHOST instrument.
