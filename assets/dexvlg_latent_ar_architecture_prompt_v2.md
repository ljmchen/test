# Image-generation prompt: DexVLG architecture-focused v2

Use case: infographic-diagram
Asset type: publication-ready model architecture figure for a top-tier computer vision / robotics paper

Primary request: Create a clean scientific diagram titled **“DexVLG: Reason-then-Flow Architecture”**. Explain the current `latent_ar` model with sequential per-hand conditional flow generation. Focus almost entirely on the neural architecture and information flow. Make the figure simpler and more immediately readable than a dense implementation diagram.

Style and composition:
- Wide 16:9 landscape canvas, white background, crisp vector-like scientific illustration.
- One strong left-to-right reading direction with generous whitespace and aligned modules.
- Flat restrained palette: blue for geometry, amber for language, violet for fusion, teal for reasoning, coral for flow generation, green for outputs.
- Dark sans-serif typography, large readable labels, thin borders, orthogonal arrows, no crossing connectors.
- Use solid arrows for the main forward path and one curved teal arrow only for autoregressive feedback.
- No 3D scene, no realistic robot hands, no large hand renderings. Represent outputs as two small abstract pose cards labeled “Pose 1” and “Pose 2”. A tiny minimal hand glyph is acceptable only inside each pose card, never as a focal visual.
- Do not fill the figure with tensor dimensions, equations, loss weights, data statistics, benchmarks, or training configuration. The only dimensional detail allowed is the final pose decomposition.

Required architecture, from left to right:

1. **Inputs & Encoders**
- Two compact parallel input streams.
- Upper stream: a small point-cloud silhouette labeled exactly **“Object Point Cloud”** → blue block **“PointNet++”** → small block **“Projector”**.
- Lower stream: a speech bubble labeled exactly **“Language Instruction”** → amber block **“ModernBERT”**.
- Keep these input icons small; the encoder blocks should be visually dominant.

2. **Multimodal Fusion**
- Merge the geometry and language streams into a violet block labeled exactly **“Fusion Transformer”**.
- Add a small attached violet module labeled **“Implicit Affordance”** to show task-relevant pooling over geometric tokens.
- Produce a prominent token-stack block labeled exactly **“Fused Multimodal Memory”**.
- Draw the fused memory feeding both the reasoner and the flow generator through clear arrows.

3. **Latent Autoregressive Reasoner**
- Use one large teal container labeled exactly **“Latent Autoregressive Reasoner”**.
- Inside, show a short horizontal chain labeled **“Continuous Latent Thinking”** with six circles: **“z₁ → z₂ → z₃ → z₄ → z₅ → z₆”**.
- After the latent chain, show **“Hand Step 1”** and **“Hand Step 2 (optional)”** as two compact decision cards.
- Each decision card has exactly two small heads: **“Presence”** and **“Side (L/R)”**.
- The reasoner cross-attends to **“Fused Multimodal Memory”**.

4. **Sequential Shared Flow Generator**
- Place a coral container below and to the right of the reasoner, labeled exactly **“Shared Conditional Flow Transformer”**.
- Subtitle: **“AdaLN Time Conditioning + Cross-Attention”**.
- Inside, show the simple process **“Gaussian Noise → Velocity Field → Euler ODE → Grasp Pose”**.
- Show two small reuse tags, **“Pass 1”** and **“Pass 2”**, inside the same container to make it obvious that both hand steps reuse the same flow network and weights.
- Arrows into this flow container come from three sources: **“Fused Multimodal Memory”**, the latent thinking tokens, and the current hand-step state.
- Flow output from **“Pass 1”** becomes **“Pose 1”**.
- Draw one clear curved feedback arrow labeled exactly **“Previous-Pose Feedback”** from **“Pose 1”** back into **“Hand Step 2 (optional)”**.
- Flow output from **“Pass 2”** becomes **“Pose 2”**. Make Pose 2 visibly optional, using a dashed outline around only that output card.

5. **Structured Output**
- At the far right, show two small clean output cards: **“Pose 1”** and optional **“Pose 2”**.
- Beneath them, one concise label: **“1–2 Dexterous Grasp Poses”**.
- Show the only dimensional annotation in the entire figure, exactly once:
  **“Pose = [ Translation (3) | 6D Rotation (6) | Joint Angles (22) ]”**.

6. Minimal footer
- A narrow unobtrusive footer with only two phrases:
  **“Training: Conditional Flow Matching + Presence/Side Supervision”**
  **“Inference: Autoregressive Decisions + Euler Flow Sampling”**
- Do not add equations or additional loss details.

Scientific correctness constraints:
- The only model inputs are point cloud and language instruction.
- Geometry is encoded by PointNet++, language by ModernBERT, then fused by a Fusion Transformer with implicit affordance pooling.
- The reasoner performs six continuous latent-thinking steps, then predicts hand presence and left/right side.
- The current mainline generates hands sequentially, not through joint two-query denoising.
- Pose 1 is embedded back as context for the optional second hand.
- A single shared conditional flow transformer generates continuous grasp poses from Gaussian noise using an Euler-integrated velocity field.

Text constraints: Render every quoted label verbatim with correct spelling and capitalization. Use no extra paragraphs or invented module names.

Avoid: dense tensor-shape annotations, detailed hand anatomy, large hand-object scenes, training-loss equations, excessive callouts, tiny text, tangled arrows, decorative neural-network nodes, gradients, shadows, dark background, pseudo-code, logos, watermarks, citations, and the obsolete fixed-two-query or optional joint-denoising architectures.
