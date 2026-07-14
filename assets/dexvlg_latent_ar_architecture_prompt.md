# Image-generation prompt: DexVLG latent_ar architecture

Use case: infographic-diagram
Asset type: publication-ready model-architecture figure for a top-tier computer vision / robotics paper

Primary request: Create a precise, elegant scientific architecture diagram titled **“Reason-then-Flow for Language-Guided Variable-Hand Dexterous Grasping”**. Depict the current mainline configuration of this repository: `latent_ar` with sequential per-hand conditional flow matching (`joint_hand_denoise = false`). Do not depict the obsolete fixed-two-query legacy architecture and do not depict the optional joint-denoising variant.

Canvas and visual language:
- Wide 16:9 landscape canvas, pure white background, generous margins, crisp vector-like rendering, publication-ready at high resolution.
- Build a clean left-to-right computational pipeline with four aligned macro stages: **Inputs & Encoders → Multimodal Memory → Latent Autoregressive Reasoner → Conditional Flow & Structured Output**.
- Use restrained flat colors: geometry blue, language amber, fused memory violet, latent reasoning teal, flow generation coral, final grasp output green. Use dark charcoal text, thin gray module borders, subtle rounded rectangles, no gradients, no shadows, no glossy 3D effects.
- Use solid arrows for inference/data flow, dashed arrows for training-only supervision, and one small legend explaining these two arrow styles.
- Typography must be a clean sans-serif similar to Helvetica/Arial. Strong hierarchy: title, stage headers, module names, then compact tensor annotations. Keep every label horizontal and readable; avoid tiny text.
- Use small, tasteful scientific icons only where helpful: an object-centered point cloud, a text instruction bubble, six latent circles, and one/two stylized Shadow Hands around an object. Icons must never replace the labeled computational blocks.

Exact architecture and layout:

1. Left column — **Inputs & Encoders**
- Upper geometry stream: an object-centered point-cloud icon labeled exactly **“Object Point Cloud  N = 4096, xyz only”**.
- Arrow into a blue block labeled **“PointNet++ Set Abstraction”** with the compact annotation **“4096 → 512 → 128 → 64 points”** and output tag **“P ∈ ℝ⁶⁴ˣ²⁵⁶”**.
- Arrow into a small blue-violet block labeled **“Projector MLP”**, annotation **“256 → 768, LN + GELU”**, and output **“P′ ∈ ℝ⁶⁴ˣ⁷⁶⁸”**.
- Lower language stream: a speech bubble containing the example instruction **“Grasp the object with both hands”**.
- Arrow into an amber block labeled **“ModernBERT-base”**, annotation **“max 128 tokens; pad mask”**, and output **“L ∈ ℝ¹²⁸ˣ⁷⁶⁸”**.

2. Center-left column — **Multimodal Memory**
- Merge `P′` and `L` through a small node labeled **“Modality Embeddings + Concatenation”**.
- Feed this into a large violet block labeled **“Fusion Transformer”** with exact sublabels **“4 encoder layers • 8 heads • d = 768”** and **“masked self-attention”**.
- Mark its output **“M ∈ ℝ¹⁹²ˣ⁷⁶⁸”**.
- From the fused point-cloud portion, draw a short side branch into **“Implicit Affordance Attention”**, annotation **“MLP score → softmax-weighted summary”**, producing one token **“a ∈ ℝ¹ˣ⁷⁶⁸”**.
- Show `a` appended to `M`, yielding a prominent compact memory stack labeled **“Fused Memory  M⁺ = [M; a] ∈ ℝ¹⁹³ˣ⁷⁶⁸”**. The padding mask continues with this memory.

3. Center-right upper panel — **Latent Autoregressive Reasoner**
- Draw a teal container labeled **“Latent Autoregressive Reasoner”**, annotation **“4-layer causal decoder • 8 heads • d = 512”**.
- Show a cross-attention arrow from `M⁺` into the whole reasoner container.
- Inside, show the exact continuous latent sequence:
  **“[BOT] → z₁ → z₂ → z₃ → z₄ → z₅ → z₆”**
  with small curved feedback arrows and caption **“Coconut-style continuous feedback”**.
- Continue the sequence to two action steps **“h₀”** and **“h₁”**.
- Under each action step, place two small classifier heads labeled **“Presence”** and **“Side: Left / Right”**.
- From generated hand 0, draw a clear autoregressive feedback arrow to hand step 1 through a small token block labeled exactly **“e₀ = SideEmb(s₀) + PoseMLP(x̂₀)”**.
- Add a compact note beside this feedback: **“train: GT pose + σ = 0.1 noise; test: sampled pose”**.
- Add another small note: **“slot 0 always emitted; slot 1 if p > 0.5; sides mutually exclusive”**.

4. Center-right lower panel — **Per-Hand Conditional Flow Matching**
- Show one reusable coral flow module applied at `s = 0, 1`, not two unrelated networks. Title it **“Shared Per-Hand Flow-Matching Transformer”**.
- At its condition input, show the exact token composition:
  **“Cₛ = [M⁺; Proj(z₁:₆); Proj(hₛ); Proj(e₀) if s = 1]”**.
- Show the noisy pose query **“ε ∼ 𝒩(0, I), 31D”** entering the flow module together with **“time t”** and **“Side Embedding”**.
- Inside the flow module, label **“6 DiT-style blocks • 8 heads • d = 256”** and list the block sequence compactly as **“AdaLN(t) → Self-Attn → Cross-Attn(Cₛ) → FFN”**.
- The module predicts **“vθ(xₜ, t, Cₛ)”**. Show a short unrolled integration arrow labeled **“Euler ODE ×10,  t: 0 → 1”** leading to **“x̂ₛ ∈ ℝ³¹”**.
- Make the causal dependence unmistakable: first generate `x̂₀`; then feed `e₀` into step 1 and generate `x̂₁`.

5. Right column — **Structured Grasp Output**
- Show an object with either one or two clean stylized Shadow Hand silhouettes, one blue-left and one coral-right, without photorealism.
- Place the exact formula prominently:
  **“x̂ₛ = [ T (3) | R₆D (6) | θ (22) ] ∈ ℝ³¹”**
- Add **“1–2 hands + predicted side”** and a small output row **“Left • Right • Hand-over • Simultaneous BiDex”**.
- Add a small final normalization note: **“translation scaled by object length Lobj; joints by Shadow-Hand ROM; R₆D → SO(3)”**.

6. Bottom strip — **Training Objective vs. Inference**
- Use a thin, full-width bottom band divided into two balanced sections.
- Training section, dashed connectors to the flow/reasoner heads, with these equations rendered exactly:
  **“xₜ = (1 − t)ε + t x₁,    v* = x₁ − ε”**
  **“L = LCFM + 0.5 Lpresence + 0.5 Lside + 0.5 Lgeo”**
  Add compact notes **“component weights T:R:θ = 3:6:1”**, **“LGBiDex: rotation ×2”**, **“Lgeo only for LGBiDex, t > 0.3”**, and **“loss masked to present hands”**.
- Inference section with solid connectors and the exact statement **“Gaussian noise → 10-step Euler flow → normalized pose → denormalized world-frame grasp”**.
- Clearly tag **“task type: training-only loss routing, never a model input”** so the graphic cannot imply a task tag enters the network.

Scientific correctness constraints:
- The only model inputs are the object point cloud and natural-language instruction.
- The current mainline uses ModernBERT-base, not classic BERT; PointNet++ consumes xyz only even though RGB may exist in the dataloader.
- Fused memory has 64 point tokens + 128 language tokens + 1 affordance summary token.
- The reasoner performs six continuous latent-thinking steps, predicts presence and left/right side, and generates at most two hands.
- The current mainline generates hands sequentially; hand 1 is conditioned on the sampled pose and side of hand 0. Do not draw cross-hand joint denoising.
- Each hand pose is one 31D vector: translation 3, continuous 6D rotation 6, Shadow Hand joints 22.
- Conditional flow matching predicts a velocity field and samples via a deterministic Euler ODE, not a diffusion noise scheduler.

Text constraints: Render all quoted labels and equations verbatim, with correct spelling, subscripts, superscripts, mathematical symbols, and capitalization. Do not invent module names, tensor sizes, loss terms, performance numbers, citations, logos, branding, decorative captions, or extra text.

Avoid: clutter, crossing arrows, ambiguous feedback direction, pseudo-code, code screenshots, dark background, gradients, perspective distortion, decorative neural-network node clouds, dense paragraph text, illegible equations, misspelled labels, watermarks, logos, captions outside the canvas, and any implication that `task_type` is an input.
