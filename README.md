# DePass Attribution Toolkit

This repository provides an implementation of DePass, a modular and extensible attribution framework for analyzing transformer-based language models. DePass supports arbitrary-granularity attribution by allowing initialization and propagation of attribution signals from any component within a transformer model. In this implementation, we demonstrate several representative use cases including token-level, neuron-level, module-level (MLP or attention), head-level, and custom subspace-level attributions, enabling fine-grained interpretability of internal mechanisms across attention and feedforward layers.

## Environment Setup

Tested with the following major packages:

-`torch==2.4.1+cu121`

-`transformers==4.44.2`

-`numpy==1.26.3`

Ensure GPU support (CUDA 12.1) is available for best performance.

```bash

pip install torch==2.4.1+cu121 transformers==4.44.2 numpy==1.26.3

```

You may also need `tqdm` for progress bar visualization.


## Quick Start

In `Demo.ipynb`, the typical workflow includes:

1.**Model and Tokenizer Loading**:

   Load a pretrained model (e.g., LLaMA, Qwen) and tokenizer using HuggingFace `transformers`.

2.**Attribution Manager Instantiation**:

```python

DecomposedStateManager = decomposed_state_manager(model, tokenizer, mlp_decomposed_function="softmax")

```

3.**Token-Level Attribution**:

```python

attr_state, states = DecomposedStateManager.get_last_layer_decomposed_state(prompt)

```

4.**Module-Level Attribution (e.g., MLP layer)**:

```python

attr_state_module = DecomposedStateManager.get_layer_module_decomposed_state(prompt, start_layer_idx=5, type="mlp")

```

5.**Subspace-Level Attribution**:

   Users can define a custom initialization tensor for a given layer and propagate it:

```python

attribute_state = DecomposedStateManager.get_subspace_decomposed_state(prompt,start_layer_idx=layer_idx-1,attribute_state=attribute_state)

```

## Attribution Outputs

The output attribution tensors produced by DePass vary by use case but follow the general format:

```

(N, *, D)

```

Where:

-`N`: sequence length (number of tokens)

-`*`: dimension determined by decomposition granularity:

-`M`: Number of user-defined components (e.g., selected neurons, module parts, or embedding subspaces)

-`N`: full token-to-token attribution (when analyzing inter-token propagation)

-`D`: hidden size of the model

This flexible structure enables arbitrary initialization and propagation schemes across the transformer layers.


## File Structure

- **`DePass/manager.py`**  
  Core implementation of the `DecomposedStateManager` class, providing main functionalities for DePass decomposition.

- **`DePass/utils.py`**  
  Utility functions supporting DePass operations.

- **`Demo/Demo.ipynb`**  
  Demonstrates DePass usage with HuggingFace-compatible LLaMA and Qwen models, including:
  - Token-level attribution  
  - Model component-level attribution (e.g., MLP, attention，neurons)  
  - Subspace-level attribution with custom initialization  

---

### Input-Level DePass Evaluation

#### `Input-Level-DePass-Evaluation/Output-Input-Attribution`
Experiments for **4.1.1 Token-Level Output Attribution via DePass**, analyzing input contributions to model outputs.
- `get_importance_score.py`: Computes importance scores using different attribution methods.  
- `get_patch_result.py`: Performs ablation based on importance scores to measure probability changes.  
- `result_analysis.ipynb`: Visualizes attribution and ablation results.  

#### `Input-Level-DePass-Evaluation/Subspace-Input-Attribution`
Experiments for **4.1.2 Token-Level Subspace Attribution via DePass**, analyzing input attribution within hidden subspaces.
- `classifier-training/train_classifier.py`: Trains the *truthful* subspace classifier.  
- `subspace-input-experiment/get_model_answer.py`: Conducts ablation experiments based on importance scores.  

---

### Model Component-Level DePass Evaluation

Experiments for **4.2 Model Component-Wise DePass**, decomposing model components such as attention heads and MLP neurons.
- `Model-Component-Level-DePass-Evaluation/attention-head-attribution/get_mask_head_answer.py`: Evaluates importance of attention heads.  
- `Model-Component-Level-DePass-Evaluation/mlp-neuron-attribution/get_mask_neuron_answer.py`: Evaluates importance of MLP neurons.  

---

### Subspace-Level DePass Evaluation

Experiments for **4.3 Subspace-Level Decomposition**, focusing on language subspace analysis with DePass.
- `Subspace-Level-DePass-Evaluation/language_probing/train_classifier.py`: Trains language classifiers for subspace probing.  
- `Subspace-Level-DePass-Evaluation/get_embedding.py`: Performs DePass-based subspace analysis.  


## Notes

- Internally uses PyTorch hooks to capture intermediate activations and control attention behavior.
