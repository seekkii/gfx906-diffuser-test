import os
import torch
import time
from diffusers import PixArtAlphaPipeline

device = "cuda"

pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    torch_dtype=torch.float16,
).to(device)

# sdpa_backend = torch.backends.cuda.sdp_kernel(
#     enable_flash=False, 
#     enable_math=True, 
#     enable_mem_efficient=True
# )

prompts = [
    "Realistic photo, wide shot, solo, one person, alone, an attractive Japanese woman in her late 20s standing beneath a massive blooming cherry blossom tree at night, rule of thirds composition, subject positioned in the lower third of frame. graceful relaxed posture. Petals drifting and glowing softly in the night air, soft round face, beautiful natural double eyelids, wearing a purple kimono. Deep depth of field, every branch and petal in sharp focus, soft natural lighting, photorealistic, DSLR photo, wide angle lens"
    """,
    anime key visual, wide shot, solo woman, late 20s Japanese woman under massive blooming sakura tree at night, lower third composition, relaxed graceful pose, purple kimono, soft expressive face, almond eyes, subtle blush.
    night sakura scene, glowing drifting petals, moonlit atmosphere, cinematic lighting, gentle rim light, highly detailed sakura branches filling frame, painterly background, cel shading, soft gradients, ultra-detailed anime style, masterpiece, 4k""",
    "Beautiful anime style illustration, wide establishing shot, a young person with vibrant hair wearing a high-tech jacket crouching down to pet a fluffy stray cat, positioned small in frame in the lower third, rule of thirds composition. The scene is dominated by a narrow city alleyway stretching into the distance, lined with glowing storefronts, neon signs, and hanging cables. Stunning evening sky above the rooftops with dramatic clouds breaking apart after a rainstorm, streaks of orange and purple twilight bleeding through the gaps. Neon and storefront lights cast soft, colorful reflections across the wet pavement, puddles mirroring the sky and signage. Deep depth of field, every detail of the alley, sky, and reflections in sharp focus, highly detailed anime background art, cinematic wide-angle framing, painterly atmospheric lighting"
    "A 3D character render in Unreal Engine 5 style showing a futuristic nomad crouching next to a small stray cat on a wet asphalt street. The narrow alley is crowded with glowing holographic advertisements and neon piping, creating hyper-detailed ray-traced reflections in the rain puddles. Crisp textures, volumetric foggy atmosphere, subsurface scattering on skin, clean cinematic framing."

]
negative_prompt = (
    # "close-up, portrait, character focus, blurry background, shallow depth of field, two people, multiple people, flat lighting, small scale"
    "photorealistic, DSLR, portrait, close-up, blurry, shallow depth of field, multiple people, bad anatomy, deformed hands, extra fingers, text, watermark"
)
steps = 20
def generate_image(prompt):
    generator = torch.Generator(device=device).manual_seed(40)
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=3.0,
            generator=generator,
        ).images[0]
    return image

torch.cuda.synchronize()
start_time = time.time()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

os.makedirs("dit_example", exist_ok=True)
print(f"Starting benchmark ({steps} steps at 1024x1024)...")

for i, prompt in enumerate(prompts):
    image = generate_image(prompt)
    image.save(f"dit_example/pixart_{i}.png")
    

torch.cuda.synchronize()
end_time = time.time()

total_time = end_time - start_time
total_steps = steps * len(prompts)
it_per_sec = total_steps / total_time
sec_per_it = total_time / total_steps
peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) # Convert bytes to GB

print("\n" + "-"*40)
print("          BENCHMARK RESULTS          ")
print("-"*40)
print(f"Total Inference Time : {total_time:.2f} seconds")
print(f"Performance Score    : {it_per_sec:.2f} it/s ({sec_per_it:.2f} s/it)")
print(f"Peak VRAM Allocated  : {peak_vram:.2f} GB")
print("-"*40 + "\n")

