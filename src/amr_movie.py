import glob
from PIL import Image

max_frames = 199
frames = []
for path in sorted(glob.glob('plots/amr/f_*.png'))[:max_frames]:
    frames.append(Image.open(path).copy())  # .copy() to avoid lazy loading issues

frames[0].save(
    'plots/amr_evolution.gif',
    save_all=True,
    append_images=frames[1:],
    duration=100,   # ms per frame — increase to slow down
    loop=0          # 0 = loop forever
)
print(f'GIF saved with {len(frames)} frames')