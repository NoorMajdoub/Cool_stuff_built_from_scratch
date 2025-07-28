
My notes and explanation of how the CLIP arch is , i wont focus on the embeddings stuff i am interested in how the constructive learning happen 

--
What is CLIP?
--
CLIP (Contrastive Language–Image Pretraining) is a model trained to associate images and texts based on their semantic meaning. Rather than generating text or images, CLIP's job is matching: given a text and an image, it tells how well they correspond.

It’s a zero-shot classifier — this means it can classify images into arbitrary textual categories without having seen labeled examples of those categories during training.

CLIP can be used for:

Image classification

Semantic similarity

Image captioning evaluation

Text-to-image search and vice versa

--
How does CLIP work?
--
CLIP consists of:

An image encoder (usually ViT or ResNet)

A text encoder (usually a Transformer like BERT)

Two projection layers (W_i and W_t) that map image and text embeddings into a shared space 

The model is trained so that matching image-text pairs are close together in this space, and mismatched ones are far apart.

--
InfoNCE Loss: The Engine of CLIP
--
CLIP uses a contrastive loss called InfoNCE, which encourages:

High similarity between positive pairs (e.g., an image of a dog and the caption "a dog").

Low similarity between negative pairs (e.g., the same image and the caption "a car").

The formula:
<img width="563" height="138" alt="image" src="https://github.com/user-attachments/assets/19449ddc-6dc1-4fae-93ee-f6095a5ceefa" />

q: the query (e.g., image embedding)

k⁺: the positive key (correct text)

k_i: all text embeddings (positive + negatives)

sim(q, k): similarity function (usually cosine similarity)

τ: temperature scaling factor

How is this implemented in the infoNce we use in CLIP

