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

# Additional Details and Q&A

This document contains answers to specific questions asked about the PO4AO repository and its components.

## Q: More details about EnsembleDynamicsFast

Okay, let's dive deeper into `EnsembleDynamicsFast`.

**From `po4ao/po4ao_models.py`:**

```python
class EnsembleDynamicsFast(nn.Module):

    def __init__(self, mask, n_history, n_models=5):
        super().__init__()
        self.n_models = n_models  # Number of individual dynamics models in the ensemble
        self.models = nn.ModuleList([]) # PyTorch list to hold the models

        # Create n_models instances of ConvDynamicsFast
        for _ in range(n_models):
            self.models.append(ConvDynamicsFast(mask, n_history))

    def forward(self, feats):
        next_states = []
        
        # Pass the input features through each model in the ensemble
        for model in self.models:
            next_states.append(model.forward(feats)) 
        
        # Concatenate the predictions from all models along a new dimension (dim=1)
        # If feats has shape (batch_size, channels, H, W)
        # and each model outputs (batch_size, 1, H, W)
        # then the concatenated output will be (batch_size, n_models, H, W)
        return torch.cat(next_states, dim=1)

class ConvDynamicsFast(nn.Module):
    def __init__(self, mask, n_history):
        super().__init__()
        
        self.mask = mask # A boolean mask, likely indicating valid actuator locations on the DM

        self.n_history = n_history # Number of past states and actions to consider

        # The core neural network
        self.net = nn.Sequential(
            # Input channels: n_history * 2 (for past states and past actions)
            # The input 'feats' to EnsembleDynamicsFast.forward, and thus to ConvDynamicsFast.forward,
            # is expected to be pre-concatenated history, current state, and current action.
            # So, the input to this Conv2d is (n_history-1 states) + (n_history-1 actions) + (current_state) + (current_action)
            # which is (n_history-1)*2 + 1 + 1 = 2*n_history channels.
            nn.Conv2d(n_history * 2, n_filt, 3, padding=1), 
            nn.LeakyReLU(),
            
            nn.Conv2d(n_filt, n_filt, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(n_filt, 1, 3, padding=1) # Output is a single channel (predicted next state)
            #nn.Tanh() # Optional Tanh activation
        )

        self._hidden = None # Not used in this "Fast" version

    def forward(self, feats):
        # 'feats' is the concatenated tensor of [history_states, history_actions, current_state, current_action]
        out = self.net(feats)

        # Apply the mask: only keep predictions for valid DM locations
        ret = torch.zeros_like(out) 
        ret[:,:, self.mask] = out[:,:,self.mask] 
        
        return out # Should probably be 'ret' if masking is intended for the output
                   # Current code returns 'out', meaning the masking operation is done but not used for the final output of this model.
                   # However, the EnsembleDynamicsFast will return the direct output of this.
                   # Looking at ConvDynamics (non-fast), it returns 'out' as well, but the masking there is `ret[:,:, self.mask] = out[:,:,self.mask]`, then `return out;`. This is consistently odd.
                   # The most probable intent is that `out` should only have non-zero values where `mask` is true.
                   # A more standard way would be `out = self.net(feats); out = out * self.mask.unsqueeze(0).unsqueeze(0); return out;` if mask is broadcastable.
                   # Or `...; result = torch.zeros_like(out); result[self.mask] = out[self.mask]; return result`.
                   # The current code `ret[:,:, self.mask] = out[:,:,self.mask]` populates `ret` correctly, but `out` is returned.
                   # This implies the `mask` parameter in `ConvDynamicsFast` might not be effectively used as intended in the returned value.
                   # Let's assume the structure of `ConvDynamicsFast` is as written and the mask is handled by the caller or implicitly.
```

**Key Features and Purpose:**

1.  **Ensemble Method:**
    *   `EnsembleDynamicsFast` is a wrapper that holds multiple instances (`n_models`, defaulting to 5) of a base dynamics model, `ConvDynamicsFast`.
    *   **Purpose:** Ensembling is a common technique in machine learning and RL to improve robustness and reduce uncertainty.
        *   **Reduces Overfitting:** Different models in the ensemble might learn slightly different aspects of the data or make different errors, and their combined output can be more generalized.
        *   **Uncertainty Estimation:** The variance in predictions across the ensemble members can be used as a measure of model uncertainty. This can be valuable for exploration in RL (e.g., preferring actions leading to states where the model is uncertain) or for risk-aware control.
        *   **Improved Prediction Accuracy:** Averaging or combining predictions from multiple models often leads to better performance than a single model.

2.  **Base Model (`ConvDynamicsFast`):**
    *   This is a Convolutional Neural Network (CNN) designed to predict the *next state* (likely a WFS image) given a history of previous states and actions, plus the current state and action.
    *   **Input (`feats`):** The `forward` method of `ConvDynamicsFast` takes a single argument `feats`. As seen in `train_dynamics` in `po4ao.py`:
        ```python
        history = torch.cat([states_unfolded[:,:-1].squeeze(2), actions_unfolded[:,:-1].squeeze(2)], dim=1)    
        pred = bs_model(torch.cat([history, state, action], dim=1)) # bs_model is an instance of ConvDynamicsFast
        ```
        So, `feats` is a tensor concatenated along the channel dimension, containing:
        *   `history`: Past `n_history-1` states and `n_history-1` actions.
        *   `state`: The current state.
        *   `action`: The current action.
        The total number of input channels to the first `Conv2d` layer is `(n_history-1) + (n_history-1) + 1 + 1 = 2 * n_history`.
    *   **Architecture:** It's a simple 3-layer CNN:
        1.  `Conv2d(n_history * 2, n_filt, 3, padding=1)` + `LeakyReLU`
        2.  `Conv2d(n_filt, n_filt, 3, padding=1)` + `LeakyReLU`
        3.  `Conv2d(n_filt, 1, 3, padding=1)` (Outputs a single channel, representing the predicted next state image).
    *   **Mask (`mask`):**
        *   The `mask` parameter is initialized during construction. In `main()` of `po4ao.py`:
            ```python
            mask = np.zeros((24,24))
            mask[dm_x, dm_y] = 1
            mask = np.array(mask, dtype = bool)
            # ...
            dynamics = EnsembleDynamicsFast(mask, n_history).to(device1)
            ```
            This `mask` identifies the valid actuator locations on the 24x24 DM grid.
        *   The line `ret[:,:, self.mask] = out[:,:,self.mask]` in `ConvDynamicsFast` *suggests* an intention to apply this mask, potentially zeroing out predictions for non-actuator locations. However, the model returns `out`, not `ret`. This is a bit unusual. If `out` is not modified in-place by this operation (which it typically wouldn't be for boolean mask assignment unless the mask makes it a view), then the `mask` might not be having the intended effect on the output of `ConvDynamicsFast` itself. However, the training loss will still primarily be driven by the valid (masked) areas if the target `next_states` are also effectively masked or zero elsewhere.

3.  **Forward Pass of `EnsembleDynamicsFast`:**
    *   When `EnsembleDynamicsFast` is called (e.g., `dynamics(...)` in `train_policy`), it iterates through each `ConvDynamicsFast` model in its `self.models` list.
    *   It passes the same input `feats` to each of these individual models.
    *   Each model produces a prediction for the next state (shape `[batch_size, 1, H, W]`).
    *   The predictions from all `n_models` are then concatenated along `dim=1`. This results in an output tensor of shape `[batch_size, n_models, H, W]`.

**How it's Used (in `po4ao.py`):**

*   **Initialization:**
    ```python
    dynamics = EnsembleDynamicsFast(mask, n_history).to(device1).share_memory()
    ```
    An instance is created, moved to `device1` (GPU for training), and its memory is shared (for multiprocessing).

*   **Training the Dynamics Model (`train_dynamics`):**
    ```python
    for bs_model in dynamics.models: # Iterate through each ConvDynamicsFast model in the ensemble
        # ... (prepare data: history, state, action, next_states) ...
        pred = bs_model(torch.cat([history, state, action], dim=1)) # Get prediction from one model
        pred_loss = (next_states - pred).pow(2).mean() # Calculate loss for this model
        loss += pred_loss # Accumulate loss
    loss.backward() # Backpropagate the total loss
    optimizer.step() # Update parameters of ALL models in the ensemble
    ```
    Each model in the ensemble is trained independently on the same batch of data to predict the true `next_states`. The total loss is the sum of individual model losses.

*   **Using for Policy Training (`train_policy`):**
    ```python
    # ... (prepare initial state, past_obs, past_act) ...
    for t in range(0, planning_horizon):
        history = torch.cat([past_obs, past_act], dim=1)
        action = policy(torch.cat([state, history], dim=1))
        
        # Get predictions from the *entire ensemble*
        next_state_ensemble_preds = dynamics(torch.cat([history, state, action], dim=1)) 
        # next_state_ensemble_preds has shape [batch_size, n_models, H, W]
        
        # The loss_fn uses the first model's prediction (index 0) from the ensemble
        losses += loss_fn(next_state_ensemble_preds[:, 0], action) 

        # For rolling the state forward, the *mean* of the ensemble predictions is used
        next_state_mean = torch.mean(next_state_ensemble_preds, dim=1, keepdim=True) 
        state = next_state_mean 
    ```
    *   When the policy needs to predict future states during its own training, it calls the `EnsembleDynamicsFast` instance.
    *   This returns a tensor of shape `[batch_size, n_models, H, W]`.
    *   Interestingly, for calculating the `loss_fn` during policy training, only the prediction from the *first model* in the ensemble (`next_state_ensemble_preds[:, 0]`) is used.
    *   However, when rolling out the trajectory (i.e., setting the `state` for the next step of the planning horizon), the `torch.mean` of the predictions across all models in the ensemble is used. This is a common way to get a more stable estimate from an ensemble.

**In Summary:**

`EnsembleDynamicsFast` provides a way to use multiple `ConvDynamicsFast` models together. This can lead to more robust and accurate predictions of the AO system's dynamics. During its own training, each model in the ensemble learns independently. When used by the policy for planning, the mean prediction of the ensemble is typically used to advance the simulated state, while the loss for policy training might (as implemented here) rely on the prediction of a single member of the ensemble (or could be modified to use the mean). The "Fast" designation likely refers to the `ConvDynamicsFast` base model taking a pre-concatenated `feats` tensor as input, potentially simplifying its internal logic compared to a version that takes separate history, state, and action tensors. The handling of the `mask` within `ConvDynamicsFast` is slightly ambiguous from the code, but its intent is to focus on valid DM actuator regions.

## Q: If all dynamic models are trained on the same set of data, why do they give different outputs?

That's an excellent question that gets to the heart of why ensembles are effective! Even when trained on the exact same dataset, individual models within an ensemble (like the `ConvDynamicsFast` instances within `EnsembleDynamicsFast`) can produce different outputs due to several factors:

1.  **Random Initialization of Weights:**
    *   Neural network weights are typically initialized with small random values before training begins. Since each `ConvDynamicsFast` model in the ensemble is a separate instance, its weights will be initialized independently with different random values.
    *   This different starting point means that even though they see the same data and use the same optimization algorithm, they will likely traverse different paths in the weight space during training and converge to different local minima of the loss function. These different local minima represent slightly different learned functions.

2.  **Stochasticity in the Training Process (though less common here if not explicitly added):**
    *   **Data Shuffling:** If the order of training data batches is shuffled differently for each model (though in this specific `train_dynamics` function, it appears they process the *same sampled batch* for each model within a single training step, this could apply if training was fully independent across epochs).
    *   **Stochastic Optimization Algorithms:** Some optimization algorithms have inherent randomness. While Adam (used here via `SharedAdam`) is mostly deterministic given the same data and initialization, if there were other sources of randomness in gradients (e.g., dropout, though not explicitly visible in `ConvDynamicsFast`), it would contribute.
    *   **Parallelism and Floating-Point Arithmetic:** When training on GPUs, the order of some floating-point operations can be non-deterministic, especially in parallel computations. This can lead to tiny variations that accumulate over time, causing models to diverge slightly. While `torch.set_deterministic(True)` (commented out in `po4ao.py`) can help mitigate this, it's not active by default.

3.  **Differences in Model Capacity vs. Data Complexity:**
    *   If the models have high capacity (many parameters) relative to the complexity captured in the training data, there can be many different parameter settings (local minima) that achieve similarly low training error. The random initialization helps them find these different solutions.

4.  **No Explicit Synchronization of All Random Seeds (Potentially):**
    *   While the overall training process is controlled, not every single source of randomness that PyTorch or CUDA might use internally is always perfectly seeded to be identical for each model instance unless extreme care is taken.

**Why is this difference desirable?**

The fact that they produce different outputs is precisely what makes the ensemble useful:

*   **Error Reduction:** If individual models make slightly different, uncorrelated errors, averaging their predictions can cause these errors to cancel out, leading to a more accurate overall prediction.
*   **Improved Generalization:** By exploring different parts of the solution space, the ensemble is less likely to overfit to the specific noise or idiosyncrasies of the training data compared to a single, potentially over-optimized model.
*   **Uncertainty Quantification:** The disagreement (variance) between the predictions of the ensemble members can be used as a proxy for the model's uncertainty about its prediction. If all models agree, confidence is high. If they disagree significantly, confidence is low.

In the `po4ao.py` script:
*   During `train_dynamics`, the loss for each model is calculated based on its own prediction, and the optimizer updates each model's unique weights.
*   During `train_policy`, the `torch.mean(next_state_ensemble_preds, dim=1, ...)` line explicitly leverages these differences by averaging the predictions to get a more stable estimate of the next state for planning.

So, the independent random initializations are the primary driver for the diverse outputs from the ensemble members, even with identical training data and architecture.

## Q: More details about the policy model

Let's break down the policy model, `ConvPolicyFastFast`, from `po4ao/po4ao_models.py` and see how it's used in `po4ao.py`.

**From `po4ao/po4ao_models.py`:**

```python
class ConvPolicyFastFast(nn.Module):
    def __init__(self, xvalid, yvalid, F, n_history):
        super().__init__()
        
        self.xvalid = xvalid  # x-coordinates of valid DM actuators
        self.yvalid = yvalid  # y-coordinates of valid DM actuators
        # These define the locations on the 24x24 grid where the output action is meaningful.

        self.n_history = n_history # Number of past observations/actions to include in the input

        # 'F' is likely a projection matrix, possibly for KL modes or another basis transformation.
        # It's registered as a buffer, meaning it's part of the model's state but not a trainable parameter.
        # It's unsqueezed to facilitate batch operations.
        self.register_buffer('F', F.unsqueeze(0)) 

        # The neural network core
        self.net = nn.Sequential(
            # Input channels:
            # The input 'feats' to ConvPolicyFastFast.forward is concatenated:
            # (current_state) + (n_history-1 past_obs) + (n_history-1 past_act)
            # Total channels = 1 (current_state) + (n_history-1) (past_obs) + (n_history-1) (past_act)
            # = 1 + 2*(n_history-1) = 2*n_history - 1 channels.
            nn.Conv2d(n_history * 2 -1, n_filt, 3, padding=1), # n_filt from config
            nn.LeakyReLU(),
            nn.Conv2d(n_filt, n_filt, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(n_filt, 1, 3, padding=1), # Outputs a single channel image (raw action)
            # nn.Tanh() # Optional Tanh, commented out. Output clamping is done separately.
        )
   
    def forward(self, feats):
        # 'feats' is the concatenated tensor of [current_state, past_observations, past_actions]
        out = self.net(feats) # Raw action image, shape [batch_size, 1, H, W]
        
        # Clamp the output values to a specific range.
        # This limits the magnitude of the requested DM command.
        # The values -0.08 and 0.08 are likely related to voltage limits or normalized action space.
        out = out.clamp(-0.08, 0.08) 

        # Apply the projection matrix 'F' to the valid actuator locations.
        # 1. out[:, :, self.xvalid, self.yvalid]: Selects the elements of 'out' corresponding to valid actuators.
        #    This flattens the selected part into a vector for each item in the batch.
        #    Shape becomes [batch_size, num_valid_actuators].
        # 2. .squeeze(1).unsqueeze(2): Reshapes for matrix multiplication.
        #    - squeeze(1): If the single channel dim is still there, removes it. Shape [batch_size, num_valid_actuators].
        #    - unsqueeze(2): Adds a dimension for matmul: [batch_size, num_valid_actuators, 1].
        # 3. torch.matmul(self.F, ...): Performs matrix multiplication.
        #    - self.F has shape [1, num_output_modes, num_valid_actuators] (assuming F projects to modes).
        #    - Or, if F is [1, num_valid_actuators, num_valid_actuators] for a direct transformation.
        #    - Given `KL_projection` is used for F, it's likely [1, num_valid_actuators, num_valid_actuators].
        #    - The result will have shape [batch_size, num_valid_actuators_after_proj, 1].
        # 4. .squeeze(-1).unsqueeze(1): Reshapes back.
        #    - squeeze(-1): Removes the last dimension: [batch_size, num_valid_actuators_after_proj].
        #    - unsqueeze(1): Adds channel dimension back: [batch_size, 1, num_valid_actuators_after_proj].
        # 5. out[:, :, self.xvalid, self.yvalid] = ... : Places the transformed values back into the original 'out' tensor
        #    at the valid actuator locations.
        # This step effectively takes the network's direct output at actuator locations, projects/transforms it using F,
        # and then puts the result of that projection back into the output action image.
        out[:, :, self.xvalid, self.yvalid] = torch.matmul(self.F, out[:, :, self.xvalid, self.yvalid].squeeze(1).unsqueeze(2)).squeeze(-1).unsqueeze(1)
    
        return out # The final action image, shape [batch_size, 1, H, W]
```

**Key Features and Purpose:**

1.  **Actor in RL:**
    *   The `ConvPolicyFastFast` model acts as the "actor" in an actor-critic RL setup (though the "critic" part is implicitly handled by the dynamics model and the loss function during policy training). Its job is to decide what action to take (i.e., what voltages to apply to the DM) given the current state of the environment.

2.  **Input (`feats`):**
    *   Similar to the dynamics model, it takes a concatenated tensor `feats` as input. In `po4ao.py`, this is constructed as:
        ```python
        # From run_episode_policy:
        action = policy(torch.cat([obs.unsqueeze(0).unsqueeze(0), past_obs, past_act],dim = 1))
        # From train_policy:
        action = policy(torch.cat([state, history], dim=1)) # where history is [past_obs, past_act]
        ```
        The input `feats` consists of:
        *   The current observation/state (`obs` or `state`): A 2D image, likely WFS data.
        *   `past_obs`: A history of `n_history-1` previous observations.
        *   `past_act`: A history of `n_history-1` previous actions.
        The total number of input channels to the first `Conv2d` is `1 (current_state) + (n_history-1) (past_obs) + (n_history-1) (past_act) = 2*n_history - 1`.

3.  **Network Architecture:**
    *   A 3-layer Convolutional Neural Network (CNN), structurally similar to the `ConvDynamicsFast` model but with a different number of input channels.
    *   It outputs a single-channel image of the same spatial dimensions as the input state, representing the raw DM command.

4.  **Output Processing:**
    *   **Clamping:** The raw output of the CNN is clamped to a range (`-0.08` to `0.08`). This ensures the commanded actions stay within permissible physical or operational limits.
    *   **Projection/Transformation (`F`):**
        *   The matrix `F` is applied specifically to the output values at the valid DM actuator locations (`self.xvalid`, `self.yvalid`).
        *   In `main()` of `po4ao.py`, `F` is initialized from `KL_projection`:
            ```python
            KL_projection = m2v_data[:,:nmodes] @ np.linalg.pinv(m2v_data[:,:nmodes])
            KL_projection = torch.from_numpy(np.asarray(KL_projection)).float()
            # ...
            policy = ConvPolicyFastFast(xvalid1, yvalid1, KL_projection, n_history).to(device1)
            policy_copy = ConvPolicyFastFast(xvalid0, yvalid0, KL_projection, n_history).to(device0)
            ```
            `m2v_data` is loaded from `KL_PTT_ifun_BMC492_cap6.fits`. This `KL_projection` is a matrix that projects data onto a subspace defined by Karhunen-Loève (KL) modes (a form of Principal Component Analysis).
        *   **Purpose:** This step likely ensures that the actions applied to the DM conform to certain desired spatial characteristics (the KL modes). It might be used to:
            *   Generate smoother DM shapes.
            *   Prioritize correction of dominant aberration modes.
            *   Ensure the output lies in the space spanned by the chosen KL modes, effectively filtering out other components.
            The network learns to output values at actuator locations which, *after projection by F*, result in the desired overall DM command.

**How it's Used (in `po4ao.py`):**

*   **Initialization:**
    *   Two instances are created:
        *   `policy`: Resides on `device1` (training GPU), its parameters are trained.
        *   `policy_copy`: Resides on `device0` (control GPU), its parameters are periodically updated from `policy`. It's set to `eval()` mode and doesn't track gradients. This is the instance used for generating actions in the control loop.
    *   Both are initialized with the same `KL_projection` matrix.

*   **Generating Actions (`run_episode_policy`):**
    ```python
    action = policy_copy(torch.cat([obs.unsqueeze(0).unsqueeze(0), past_obs, past_act],dim = 1))
    next_obs = step(action) # Apply the action to the environment
    ```
    During an episode, `policy_copy` takes the current observation and history, and outputs the DM command image (`action`) to be sent to the AO system.

*   **Training the Policy (`train_policy`):**
    ```python
    # ... (sample data, prepare current state and history) ...
    # Loop over planning_horizon:
    history_for_policy = torch.cat([past_obs, past_act], dim=1) # Note: 'past_obs' here is from the start of the planning horizon
    current_state_for_policy = state # 'state' is rolled forward using dynamics model

    action = policy(torch.cat([current_state_for_policy, history_for_policy], dim=1)) # Get action from the policy being trained

    # Predict next state using the dynamics model
    next_state_pred_ensemble = dynamics(torch.cat([history_for_policy, current_state_for_policy, action], dim=1))
    
    # Calculate loss based on predicted next state and action
    losses += loss_fn(next_state_pred_ensemble[:, 0], action) 
    
    # ... (roll state and history for next step of planning horizon) ...

    loss = losses.mean()
    loss.backward() # Backpropagate gradients through the policy network
    policy_optimizer.step() # Update policy network parameters
    ```
    *   The `policy` (on `device1`) is trained to produce actions that minimize a loss function (`loss_fn`).
    *   This loss function penalizes predicted future states (from the `dynamics` model) and the magnitude of the actions themselves.
    *   The training involves "imagining" trajectories over a `planning_horizon` using the learned `dynamics` model.

*   **Synchronization:**
    ```python
    # In main loop, after training step is finished:
    if training_finished:
        training = False
        policy_copy.load_state_dict(policy.state_dict()) # Update policy_copy with new weights
    ```
    The weights from the `policy` (being trained) are copied to `policy_copy` (used for control).

**In Summary:**

`ConvPolicyFastFast` is a CNN that learns the control policy for the AO system. It takes the current WFS image and a history of past observations and actions as input. Its output is a DM command image. Key processing steps include clamping the output to a valid range and projecting the commands at actuator locations using a KL-mode projection matrix (`F`). This ensures actions are physically plausible and potentially structured according to dominant aberration modes. It's trained using model-based RL, where a learned dynamics model helps predict the consequences of its actions over a planning horizon. Two instances are used: one for training and one for control, with weights periodically synchronized.

## Q: How can I speed up eval of policy_copy?

Speeding up the evaluation of `policy_copy` (i.e., the forward pass during control) is critical for real-time adaptive optics. Here are several strategies, ranging from general PyTorch optimizations to model-specific considerations:

**1. Ensure No Gradient Tracking:**
*   This is already correctly done in the provided code:
    ```python
    policy_copy = ConvPolicyFastFast(xvalid0, yvalid0, KL_projection, n_history).to(device0).share_memory().eval()
    # ...
    for p in policy_copy.parameters():
        p.grad = None
    ```
    And operations are typically within a `@torch.no_grad()` context for functions like `run_episode_policy` (implicitly, as it's not training).
*   **Why:** `torch.no_grad()` and `.eval()` mode prevent PyTorch from building a computation graph for backpropagation and turn off layers like Dropout or BatchNorm if they behave differently during training vs. inference. This significantly reduces overhead.

**2. Optimize Input Data Preparation:**
*   The primary input is `torch.cat([obs.unsqueeze(0).unsqueeze(0), past_obs, past_act], dim=1)`.
    *   **Pre-allocate and Update:** Instead of re-concatenating from scratch every time, if `past_obs` and `past_act` are managed in a rolling buffer (like a `collections.deque` of tensors or a larger pre-allocated tensor that you index into), you might save some overhead from repeated `torch.cat` operations on potentially new tensor objects. The current implementation with `torch.cat` on existing tensors is likely efficient, but for extreme optimization, this could be a point.
    *   **Tensor Creation Overhead:** Ensure `obs` is already a tensor on `device0`. If it's coming from CPU (e.g., a NumPy array from a sensor) and then moved to GPU and unsqueezed every step, that transfer and manipulation has a cost. Minimize CPU-GPU transfers within the loop. The current code seems to handle `obs` as a GPU tensor already.

**3. Model Architecture Simplification (If Possible and Acceptable):**
*   This is a more involved change and depends on performance/accuracy trade-offs.
    *   **Reduce `n_filt` (Number of Filters):** Fewer filters in `nn.Conv2d` layers directly reduce computation (FLOPs) and memory bandwidth. This is the most impactful architectural change for speed.
    *   **Reduce Network Depth:** Fewer convolutional layers. The current model is already quite shallow (3 conv layers).
    *   **Kernel Size:** Using smaller kernels (e.g., 1x1 where appropriate, if 3x3 isn't strictly necessary for receptive field) can reduce FLOPs, but 3x3 is standard for capturing local spatial features.
    *   **Trade-off:** These changes might reduce the model's accuracy or its ability to represent complex policies. This requires experimentation.

**4. Quantization:**
*   **Concept:** Convert model weights and/or activations from 32-bit floating point (FP32) to lower precision formats like 16-bit floating point (FP16/BF16) or 8-bit integers (INT8).
*   **Benefits:**
    *   Faster computation on compatible hardware (e.g., Tensor Cores on NVIDIA GPUs for FP16).
    *   Reduced memory footprint and bandwidth requirements.
*   **PyTorch Support:** PyTorch has tools for quantization-aware training and post-training quantization.
    *   **Automatic Mixed Precision (AMP):** `torch.cuda.amp` can automatically use FP16 for many operations, often providing speedup with minimal accuracy loss. This is relatively easy to try:
        ```python
        # In run_episode_policy or the main loop section calling it
        with torch.cuda.amp.autocast(enabled=True): # Check if enabled=True is needed or if it's auto
            action = policy_copy(torch.cat([obs.unsqueeze(0).unsqueeze(0), past_obs, past_act],dim = 1))
        # Note: The output 'action' might be FP16. Ensure 'step()' can handle it or cast it back.
        ```
    *   **Static/Dynamic Quantization (INT8):** More complex, might require calibration data, and could have a higher impact on accuracy if not done carefully.
*   **Hardware Dependency:** Speedups are most significant on GPUs with good support for lower precision.

**5. Model Compilation / Optimization Frameworks:**
*   **TorchScript (JIT Compilation):**
    *   Convert your PyTorch model into TorchScript, a statically analyzable and optimizable representation.
    ```python
    # After policy_copy is initialized and moved to device
    example_input = torch.randn(1, 2 * n_history - 1, data_shape, data_shape).to(device0) # Batch size 1
    traced_policy_copy = torch.jit.trace(policy_copy, example_input)
    # Now use traced_policy_copy(...) for inference
    ```
    *   **Benefits:** Can fuse operations, optimize graph execution, and reduce Python overhead.
*   **ONNX (Open Neural Network Exchange) and Runtimes:**
    *   Export the model to ONNX format.
    *   Use an optimized runtime like ONNX Runtime, TensorRT (NVIDIA), or OpenVINO (Intel) for inference. TensorRT can provide significant speedups on NVIDIA GPUs by performing layer fusion, precision calibration, and kernel auto-tuning.
    *   This is more involved as it requires an export step and integration with a different runtime.
    ```python
    # Example for ONNX export (simplified)
    # torch.onnx.export(policy_copy, example_input, "policy_copy.onnx", ...)
    # Then load and run with ONNX Runtime or convert to TensorRT engine.
    ```

**6. CUDA Kernel Fusion (Advanced):**
*   If specific sequences of operations are bottlenecks, it's theoretically possible to write custom CUDA kernels that fuse these operations. This is highly complex and usually a last resort. PyTorch's JIT or TensorRT often try to do this automatically.

**7. Hardware:**
*   **Faster GPU:** A more powerful GPU will naturally execute the model faster.
*   **Sufficient GPU Memory Bandwidth:** Ensure the GPU isn't bottlenecked by memory speed, especially for a conv-heavy model.

**Practical Steps to Try First:**

1.  **Confirm No Gradients/Eval Mode:** Already good.
2.  **TorchScript (`torch.jit.trace`):** This is often a good first step for PyTorch-native optimization with relatively low effort.
3.  **Automatic Mixed Precision (`torch.cuda.amp`):** If your GPU supports it well (most modern NVIDIA GPUs do), this can give good speedups. Be mindful of potential small accuracy changes and ensure subsequent operations can handle FP16 if necessary.

**Profiling:**
*   Before optimizing, it's crucial to know *where* the time is spent. Use PyTorch's profiler:
    ```python
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.profiler.record_function("policy_inference"):
            # Your policy_copy forward pass here
            action = policy_copy(torch.cat([obs.unsqueeze(0).unsqueeze(0), past_obs, past_act],dim = 1))

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # Or save the trace and view it in Chrome (chrome://tracing) or a profiler UI
    # prof.export_chrome_trace("trace.json")
    ```
    This will tell you which operations are taking the most time on the GPU and CPU, guiding your optimization efforts. For instance, it can reveal if `torch.cat` is more expensive than expected, or if a specific conv layer is the bottleneck.

Start with profiling to understand bottlenecks, then try TorchScript and AMP as they are generally easier to implement. Architectural changes and quantization are more involved but can yield larger gains if needed.

