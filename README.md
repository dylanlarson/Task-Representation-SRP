# Task Representation
This repository uses data and environments from https://github.com/HumanCompatibleAI/overcooked_ai
The poster is attached, let me know if you have any questions.

## Setup
Two environments are used.
1. Environment as per https://github.com/HumanCompatibleAI/overcooked_ai.
   This is only required to render the images, otherwise it is not required. The version 1 train/validation/test images are provided in 3 zip files (in /version_1/data/imgs/ folder) for convenience.
2. Environment with pytorch and other additional libraries as per requirements.txt (I've been working on a few different machines so hopefully it is complete, main things are pytorch lightning, pandas, torchmetrics, torchvision, shap, grad-cam, install as required).

## Recreation procedure
Seeds were not used so results may vary slightly.
The procedures are based on version 1 as version 2(cross validation) files were not cleaned and are incomplete (May be easier to adapt the cleaned files instead if you want to crossvalidate more). 
Version 0 contains uncleaned files but may be easier to edit or follow as functions are included within he notebook and not in separate python files.

Caution of lines which generate, save or load data/models. Comment relevant sections to save data and reduce re-running of sections but be careful of overwriting data.

### Split data into training, validation and test splits.
  
Using Environment 1 and data_splitting.ipynb, all frames are loaded in and split in train, test and validation sets. The directories are not self creating so you may need to create folders (./data/imgs/[train|test|val]/[0|1|2|3|4]/).  
All frames are duplicated to provide both players perspectives. The frames that have no "useful" action are removed (Next player state is the same - this removing idle actions, interacts with nothing and trying to move into benches).
  
The data is split by trial id, such that frames from the same trial are all in the same train/val/test set (Prevent interleaved frames (or same frames but opposite player perspective) with near identical frames from being in both sets). This is done randomly a number of times (specify this) and the split with the lowest average variance (across distributions of both actions and map layouts) is used for the most balanced dataset (The dataset is skewed but hopefully balanced enough, I obtained a minimum of 13% of each maps/actions across all sets).
  
For crossvalidation, the data is split into 5 sets and concatenated to form different sets (Could be optimised to reduce saving multiple copies of data with smarter indexing, etc.)
### Model Training
Using  Environment 2 and Predictor_XXX.ipynb notebooks. The network architectures can be found in networks.py. The encoding and data preparation functions are in data_handler.py. Model analysis functions are in helper.py.  Functions aren't well documented so let me know if you have any questions. Checkpoints are used for the models as they have the best val loss.
The notebooks are hopefully self explanatory, you may need to update some directories/file names (such as checkpoint names). Directories need to be created for results and models (not for checkpoints and logs, ./[results|models]/[v1|v2|v2_2|v3|v4]/). The saliency functions have abilities to plot in the notebook or save figures of specified indices.  
#### Encoding Notes
MLP features uses relative coordinates to the player. The pot, onion source, dish source, etc. are only the closest tile to the player (if there is a draw it randomly selects one of the two).
There are labels for MLP features and Encoded CNN in the encoding function and saliency_rank.ipynb for graphs/ranks.

### Model Analysis
All analysis should be performed within environment 2.  
There is some analysis within the Predictor_XXX.ipynb notebooks where access to the model is required. Saliency analysis through backpropagation is performed there. They are backpropagated with respect to the predicted class/action. The gradient data can be found as grad_abs in the respective saliency functions in helper.py. The ranking of tiles is generated within these files and saved as .npy files for analysis in the saliency_rank.ipynb notebook.  
The MLP also has Shapley values implemented within the notebook. The order of computation for this is 2^n (where n is number of input features) so it is only really feasible for the MLP implementation. Not sure if any analysis for the encoded CNN is valid however internal comparisons between similarly encoded channels hopefully is valid (i.e. P1 vs. P2, or between held different held items).
Using saliency_rank.ipynb the rankings of permanent tiles (in tile_label) within each frame are combined using the Schluze method.  This forms a combined ranking of tile saliency, however this is pretty dependent on the test set so potentially more resampling of data is needed (Resampled a couple times and similar results are made but some rankings shift, sometimes CNN2 pot and P2 swap).
