
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

How is this implemented in the infoNce we use in CLIP:

similarity = image_proj @ text_proj.T  # shape: (B, B) : 2D array having each image in our batch and the sim score it gives with the texts 

=| the True label is on the diagonal :The @ (dot product) is how we compute the similarity scores, which are fed into InfoNCE (via softmax + cross-entropy).
loss_i2t = F.cross_entropy(similarity / temperature, labels)
loss_t2i = F.cross_entropy(similarity.T / temperature, labels)
loss = (loss_i2t + loss_t2i) / 2

--
 Why normalize ?
 --
Before computing similarity, we normalize the image and text vectors:


image_proj = F.normalize(image_proj, dim=-1)
text_proj = F.normalize(text_proj, dim=-1)
similarity = image_proj @ text_proj.T  # shape: (B, B)


cause what we care about is what direction the embedding vecotr is going in , not how long ,big it is

Cosine similarity becomes a dot product after normalization.

Helps stabilize training and avoid dominating effects of vector length.

--
what are we training here?
--
what will be changing during training is the projection matrices wi and wt those learn neeed to learn to project the image and text of same semantic meaning in the same direction 
Because even though the image encoder and text encoder might be pretrained (e.g., ViT and BERT), their features aren’t aligned in any meaningful way across modalities.
The weights of the projection layers learn how to map features into a space where semantic relationships are reflected geometrically (through vector direction and distance).

The projection weights are trained (via InfoNCE loss) to rotate/scale/transform the original features in a way that:

dog image ↔ "a dog" → vectors point in the same direction

cat image ↔ "a cat" → same

dog image ↔ "a cat" → vectors point in different directions

--
 Training Flow
 --
Encode B images → image_embeddings → project to shared space

Encode B texts → text_embeddings → project to shared space

Normalize both sets of embeddings

Compute similarity matrix S = image_proj @ text_proj.T → shape (B, B)

Apply cross-entropy loss row-wise and column-wise:

The correct image-text pair is on the diagonal of the matrix

Off-diagonal = mismatches

Each row is treated like a classification problem:

"Which of these B text prompts matches this image?"

And same for columns:

"Which of these B images matches this text?"

--
Why Cross Entropy?
--
Cross-entropy forces the model to rank the true match highest.

It doesn’t just increase similarity blindly — it penalizes relative similarity:

If the wrong text is more similar to the image than the correct one, the model is punished.

This competitive nature improves robustness and generalization.

This matrix is fed to a cross-entropy loss, treating each row and column as a classification problem where only the diagonal should have high sim.



