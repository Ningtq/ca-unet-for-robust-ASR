# Introduction: Bridging Front-End and Back-End for Robust ASR via Cross-Attention-Based U-Net

## Background & Problem

There is a **mismatch** between speech enhancement (SE) as the front-end and automatic speech recognition (ASR) as the back-end:

- SE focuses on speech quality but may introduce artifacts and over-suppression
- ASR requires preservation of critical information for recognition
- Existing linear fusion methods (Observation Adding, OA) are limited in complex acoustic environments

## Proposed Solution

Introduces a **Cross-Attention-Based U-Net Module**:

- **Architecture**: Dual-branch U-Net structure processing enhanced and noisy speech separately
- **Core Mechanism**: Cross-attention enables interactive feature fusion between enhanced and noisy speech
- **Key Feature**: Operates with frozen front-end and back-end models, avoiding high computational costs of joint fine-tuning

## Architecture Details

1. **Cross-Attention Fusion**: Dynamically learns complementary information from both enhanced and noisy speech inputs.
2. **U-Net Encoding-Decoding**: Enables multi-scale feature extraction and reconstruction for robust acoustic representation.
3. **Gated Fusion Module**: Adaptively integrates outputs from dual branches to balance information flow.
4. **Self-Attention Enhancement**: Effectively refines post-residual features, improving downstream recognition performance, as validated by ablation studies.
5. **Frozen Parameter Training**: Compatible with pre-trained front-end and back-end models, enabling flexible and practical deployment.

## Experimental Results

The full experimental details can be found at **results/README.md**
