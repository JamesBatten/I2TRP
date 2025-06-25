# I2TRP: Image to Tree Recursive Predictor

This is the code repo for the paper Image to Tree with Recursive Prompting: https://arxiv.org/abs/2301.00447

I2TRP is a deep learning model designed to predict properties of a tree structure directly from a visual grid input (e.g., an image with multiple feature channels). The model combines a powerful vision backbone with a Transformer-based decoder architecture to learn the relationship between a global visual context and specific local queries.

The architecture is built with PyTorch and uses a modular design, allowing for easy configuration and extension of its components.

## Model Architecture

The model's architecture can be understood as a pipeline with four main stages: a **Visual Encoder** to process the main image, a **Query Encoder** to process local patches, a **Transformer Decoder** to fuse information, and a final **Predictor** to generate outputs.

### 1. Visual Encoder (Memory Generation)

The Visual Encoder's role is to process the main input grid and compress it into a sequence of contextualized feature vectors that serve as the `memory` for the decoder.

-   **Input**: A high-dimensional visual grid (e.g., an image with channels for RGB, coordinates, etc.).
-   **Core Components**: An optional `UNet` followed by a `ViT` (Vision Transformer).
-   **Processing**:
    -   The input grid is first passed through a `UNet`. This step serves to extract hierarchical features and adjust the feature map's channel depth for the subsequent ViT. This component is configurable and can be excluded.
    -   The resulting feature map is then divided into non-overlapping patches. Each patch is linearly projected into an embedding, and learnable positional encodings are added.
    -   This sequence of patch embeddings is processed by a **`TransformerEncoder`** (`ViT`). The self-attention mechanism allows the model to capture global context and long-range dependencies across the entire visual input.
-   **Output**: The encoder's output is a sequence of rich visual embeddings (`memory`) that represents the entire input grid.

### 2. Patch Encoder (Target Generation)

The Patch Encoder generates feature representations for specific locations of interest, which will be used to query the visual memory.

-   **Input**: A batch of small image patches, where each patch corresponds to a specific node in the tree being predicted.
-   **Core Component**: A `PatchEncoder` module, which internally uses a standard CNN `Backbone` (e.g., ResNet, SEResNeXt).
-   **Processing**: The `PatchEncoder` processes each image patch independently, encoding it into a fixed-size feature vector.
-   **Output**: A sequence of feature vectors (`target` or `queries`), where each vector corresponds to one of the input patches.

### 3. Transformer Decoder

The Decoder takes the global visual `memory` and the specific `target` queries to produce refined representations for each query location.

-   **Input**:
    1.  The `memory` sequence from the Visual Encoder.
    2.  The `target` sequence from the Query Encoder.
-   **Core Component**: A standard PyTorch `TransformerDecoder`.
-   **Processing**: The decoder uses cross-attention. Each `target` vector attends to the entire `memory` sequence. This allows the model to "ask questions" about the global context (e.g., "What visual information is relevant to this specific patch?") and enrich its local representation.
-   **Output**: A sequence of refined feature vectors, one for each query, that now contains both local information and global context.

### 4. Predictor

The Predictor is a multi-headed module that takes the refined features from the decoder and generates the final outputs for different tasks.

-   **Input**: The sequence of refined feature vectors from the Transformer Decoder.
-   **Core Component**: A `Predictor` module containing multiple prediction "heads".
-   **Processing**: The refined features are passed in parallel to several MLP-based heads (`CFC` or `CMLP2`).
-   **Output**: A dictionary of predictions. Key heads include:
    -   `selection`: Predicts a selection probability for each query node.
    -   `topology`: Predicts structural or topological properties for each node.

## Project Structure

-   `i2trp/model/i2trp.py`: Contains the main `I2TRP` module, which integrates the full pipeline.
-   `i2trp/model/unet.py`: Defines the U-Net for initial feature extraction from the input grid.
-   `i2trp/model/vit.py`: The Vision Transformer that creates the global context `memory`.
-   `i2trp/model/patch_encoder.py`: The CNN-based encoder for generating `query` vectors from local patches.
-   `i2trp/model/predictor.py`: The final multi-headed prediction module.
-   `i2trp/model/backbone.py`: A flexible wrapper for various `torchvision` and `timm` CNN backbones used by the `PatchEncoder`.
-   `i2trp/model/{down_block.py, up_block.py, conv.py}`: Reusable building blocks for constructing the `UNet`.
-   `i2trp/model/{mlp2.py, fc.py, norm.py, nonlinearity.py, weight_init.py}`: A collection of helper modules for building robust and configurable PyTorch network layers.
-   `i2trp/common/utils.py`: Provides various utility functions for tensor and array manipulation.