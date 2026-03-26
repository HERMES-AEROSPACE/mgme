import glob
from PIL import Image

frames = []
for path in sorted(glob.glob('plots/f_*.png')):
    frames.append(Image.open(path).copy())  # .copy() to avoid lazy loading issues

frames[0].save(
    'plots/amr_evolution.gif',
    save_all=True,
    append_images=frames[1:],
    duration=200,   # ms per frame — increase to slow down
    loop=0          # 0 = loop forever
)
print(f'GIF saved with {len(frames)} frames')