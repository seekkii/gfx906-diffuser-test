import time
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import DDIMScheduler
torch.set_float32_matmul_precision('high')

prompt_1 = "1girl, detailed face, from side, profile view, looking away, holding a coffee cup with both hands, wearing stylish casual clothing, sitting in a modern coffee shop, natural morning lighting, depth of field, ulzzang-6500, portrait photography, masterpiece, best quality, ultra high resolution, highly detailed, photorealistic, (ulzzang-6500:1.0)"
prompt_2 = "1girl, detailed face, from above, three-quarter view, looking down, reading a book, gentle smile, wearing stylish casual clothing, sitting in a modern coffee shop, soft window light, depth of field, upper body, ulzzang-6500, portrait photography, masterpiece, best quality, ultra high resolution, highly detailed, photorealistic, (ulzzang-6500:1.0)"
prompt_3 = "1girl, detailed face, looking at the viewer, wearing stylish casual clothing, sitting in a modern coffee shop, natural morning lighting, depth of field, ulzzang-6500, portrait photography, masterpiece, best quality, ultra high resolution, highly detailed, photorealistic, (ulzzang-6500:1.0)"
prompt_4 = "1girl, detailed face, from behind, looking back over shoulder, looking at viewer, hand adjusting hair, soft smile, wearing stylish casual clothing, sitting in a modern coffee shop, warm ambient lighting, depth of field, upper body, ulzzang-6500, portrait photography, masterpiece, best quality, ultra high resolution, highly detailed, photorealistic, (ulzzang-6500:1.0)"
neg_prompt_1 = "EasyNegative, paintings, sketches, worst quality, low quality, normal quality, lowres, monochrome, grayscale, bad hand, blurry, bad feet, distorted, text, watermark, signature, extra limbs, asymmetrical eyes, deformed face"
neg_prompt_2 = """bad anatomy, extra limbs, deformed, unnatural proportions, oversmoothed skin, "
    "bad eyes, asymmetrical face, deformed face, mutated hands, long neck, missing fingers, extra arms, blurry details"
    "low quality, worst quality, jpeg artifacts"""
neg_prompt_3 = "paintings, sketches, worst quality, low quality, normal quality, lowres, monochrome, grayscale, skin spots, acnes, skin blemishes, age spot, glans, extra fingers, fewer fingers, watermark, white letters, multi nipples, bad anatomy, bad hands, text, error, missing fingers, missing arms, missing legs, extra digit, fewer digits, cropped, worst quality, jpeg artifacts, signature, watermark, username, bad feet, Multiple people, blurry, poorly drawn hands, poorly drawn face, mutation, deformed, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed, mutated hands, polar lowres, bad body, bad proportions, gross proportions, wrong feet bottom render, abdominal stretch, briefs, knickers, kecks, thong, fused fingers, bad body,\nbad-picture-chill-75v,  ng_deepnegative_v1_75t, EasyNegative, bad proportion body to legs, wrong toes, extra toes, missing toes, weird toes, 2 body, 2 pussy, 2 upper, 2 lower, 2 head, 3 hand, 3 feet, extra long leg, super long leg, mirrored image, mirrored noise, aged up, old"
width = 512
height = 512
steps = 25   
guidance_scale = 8.0
seed = 42

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

print("Loading pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    "emilianJR/chilloutmix_NiPrunedFp32Fix",
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

pipe = pipe.to("cuda")

# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_textual_inversion(
    "/home/tungf/data/ulzzang-6500.pt"
)

# Good for vision-style models; often helps a bit.
try:
    pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)
except Exception:
    pass


# pipe.set_progress_bar_config(disable=True)

# Optional: compile UNet for repeated runs.
# First run will be slower because compilation happens then.
# USE_COMPILE = True
# if USE_COMPILE:
#     pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")

def generate(prompt, negative_prompt,run_seed: int):
    # generator = torch.Generator(device="cuda").manual_seed(run_seed)

    sync()
    t0 = time.perf_counter()
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        # generator=generator,
    ).images[0]

    sync()
    t1 = time.perf_counter()
    return image, t1 - t0

for i, prompt in enumerate([prompt_3], start=1):
    image, elapsed = generate(prompt, neg_prompt_3, seed)
    image.save(f"sd1.5-example/output_{i}.png")
    print(f"Saved sd1.5-example/output_{i}.png")
    print(f"1 image took {elapsed:.2f} seconds")


