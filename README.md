# gfx906-diffusers-test

## Hardware / Software

- **GPU:** AMD gfx906 (Radeon VII / Instinct MI50/MI60 32gb)
- **ROCm version:** 6.3.0
- **Python:** 3.11.15
- **PyTorch:** ROCm build (`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3`)
- **diffusers:** 0.38.0
- **transformers:** 4.44.2

See `requirements.txt` for the full dependency list.

## Files

| File | Description |
|---|---|
| `sd1.5_img_gen.py` | Image generation / benchmark script for SD1.5-based checkpoints (e.g. ChilloutMix) |
| `dit_img_gen.py` | Image generation / benchmark script for DiT-based (transformer) diffusion models — currently testing `PixArt-alpha/PixArt-XL-2-1024-MS` |
| `sd1.5-sample/` | Sample outputs from the SD1.5 script |
| `dit-example/` | Sample outputs from the DiT script |
| `requirements.txt` | Python dependencies |

## Usage

```bash
# SD1.5 / ChilloutMix
python sd1.5_img_gen.py

# DiT-based models (PixArt-alpha, etc.)
python dit_img_gen.py
```

Output images are saved to their respective sample directories.

## Benchmark Results

| Model | Resolution | Steps | CFG | Time per image | Peak VRAM |
|---|---|---|---|---|---|
| ChilloutMix (SD1.5) | 512×512 | 25 | 7.5 | ~6–8s | 4.40 GB |
| PixArt-alpha/PixArt-XL-2-1024-MS | 1024×1024 | 20 | 3.0 | ~55–56s | 16.82 GB |
