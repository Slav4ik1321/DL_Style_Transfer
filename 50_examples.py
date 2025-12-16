import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import requests
import os
import random
import copy
import warnings
import json
import io
from dataclasses import dataclass, field
from typing import List

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================
@dataclass
class Config:
    seed: int = 42
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir: str = './data'
    output_dir: str = './final_report_plots'

    imsize: int = 224
    num_steps: int = 250

    content_weight: float = 10.0
    style_weight: float = 1e6
    tv_weight: float = 1e-3

    #content_layers: List[str] = field(default_factory=lambda: ['conv_10']) # Mid
    #style_layers: List[str] = field(default_factory=lambda: ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']) # DenseStart
    content_layers: List[str] = field(default_factory=lambda: ['conv_14']) # Deep
    style_layers: List[str] = field(default_factory=lambda: ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']) # DenseStart

    num_images: int = 50
    grid_cols: int = 5

cfg = Config()

CIFAR_CLASSES = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}

# ==========================================
# 2. Утилиты
# ==========================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

def get_imagenet_labels():
    path = os.path.join(cfg.data_dir, "imagenet_class_index.json")
    if not os.path.exists(cfg.data_dir): os.makedirs(cfg.data_dir)
    if not os.path.exists(path):
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        try:
            r = requests.get(url)
            with open(path, 'wb') as f: f.write(r.content)
        except: return {}
    with open(path, 'r') as f:
        class_idx = json.load(f)
    return {int(k): v[1] for k, v in class_idx.items()}

class QuickDrawDataset(Dataset):
    def __init__(self, root_dir, category='cat', transform=None):
        self.root_dir = root_dir; self.category = category; self.transform = transform
        self.file_path = os.path.join(root_dir, f'{category}.npy')
        self._download()
        try: self.data = np.load(self.file_path)
        except:
            if os.path.exists(self.file_path): os.remove(self.file_path)
            self._download(); self.data = np.load(self.file_path)
    def _download(self):
        if os.path.exists(self.file_path): return
        os.makedirs(self.root_dir, exist_ok=True)
        url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{self.category}.npy'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try: requests.get(url, stream=True)
            except: pass
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img_arr = self.data[idx].reshape(28, 28)
        img = Image.fromarray(img_arr).convert('RGB')
        if self.transform: img = self.transform(img)
        return img

def get_loaders():
    t = transforms.Compose([transforms.Resize((cfg.imsize, cfg.imsize)), transforms.ToTensor()])
    print("[INFO] Loading CIFAR-10...")
    cifar = datasets.CIFAR10(root=cfg.data_dir, train=True, download=True, transform=t)
    qd = QuickDrawDataset(root_dir=cfg.data_dir, category='cat', transform=t)
    return cifar, qd

# ==========================================
# 3. Модель и Losses
# ==========================================
class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super().__init__()
        self.target = target.detach(); self.weight = weight
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target) * self.weight
        return input

class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super().__init__()
        self.target = self.gram_matrix(target).detach(); self.weight = weight
    def gram_matrix(self, input):
        a, b, c, d = input.size(); features = input.view(a * b, c * d)
        return torch.mm(features, features.t()).div(a * b * c * d)
    def forward(self, input):
        self.loss = nn.functional.mse_loss(self.gram_matrix(input), self.target) * self.weight
        return input

class TVLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def forward(self, input):
        self.loss = self.weight * (
            torch.sum(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) +
            torch.sum(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
        )
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean.view(-1, 1, 1))
        self.register_buffer('std', std.view(-1, 1, 1))
    def forward(self, img): return (img - self.mean) / self.std

def get_nst_model(cnn, style_img, content_img):
    cnn = copy.deepcopy(cnn)
    norm = Normalization(cnn_normalization_mean, cnn_normalization_std).to(cfg.device)
    c_losses, s_losses = [], []
    model = nn.Sequential(norm)
    tv_loss = TVLoss(cfg.tv_weight)
    model.add_module("tv_loss", tv_loss)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d): i += 1; name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU): name = f'relu_{i}'; layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d): name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d): name = f'bn_{i}'
        else: continue
        model.add_module(name, layer)
        if name in cfg.content_layers:
            target = model(content_img).detach()
            cl = ContentLoss(target, cfg.content_weight)
            model.add_module(f"c_loss_{i}", cl); c_losses.append(cl)
        if name in cfg.style_layers:
            target = model(style_img).detach()
            sl = StyleLoss(target, cfg.style_weight)
            model.add_module(f"s_loss_{i}", sl); s_losses.append(sl)

    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)): break
    return model[:i+1], s_losses, c_losses, tv_loss

def run_transfer(cnn, content, style):
    input_img = content.clone()
    input_img.requires_grad_(True)
    model, s_losses, c_losses, tv_loss = get_nst_model(cnn, style, content)
    model.requires_grad_(False)
    optimizer = optim.LBFGS([input_img])

    # === ИСТОРИЯ И ЛУЧШИЙ РЕЗУЛЬТАТ ===
    history = {'style': [], 'content': [], 'tv': [], 'total': []}

    # Инициализируем лучшее состояние текущим
    best_state = {
        'loss': float('inf'),
        'img': input_img.clone().detach()
    }

    run = [0]
    while run[0] <= cfg.num_steps:
        def closure():
            with torch.no_grad(): input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)

            s_score = sum(sl.loss for sl in s_losses)
            c_score = sum(cl.loss for cl in c_losses)
            tv_score = tv_loss.loss

            loss = s_score + c_score + tv_score
            loss.backward()

            loss_val = loss.item()

            history['style'].append(s_score.item())
            history['content'].append(c_score.item())
            history['tv'].append(tv_score.item())
            history['total'].append(loss_val)

            # --- СОХРАНЕНИЕ ЛУЧШЕГО ---
            if loss_val < best_state['loss']:
                best_state['loss'] = loss_val
                # Копируем текущий тензор и отсоединяем от графа
                best_state['img'] = input_img.clone().detach()
            # --------------------------

            run[0] += 1
            return loss
        optimizer.step(closure)

    # Возвращаем именно лучшую картинку, а не последнюю
    final_img = best_state['img']
    with torch.no_grad(): final_img.clamp_(0, 1)

    return final_img, history

def get_class_pred(model, img):
    model.eval()
    with torch.no_grad():
        norm = transforms.Normalize(cnn_normalization_mean, cnn_normalization_std)
        inp = norm(img.squeeze(0)).unsqueeze(0)
        return model(inp).argmax(dim=1).item()

# ==========================================
# 4. Визуализация и Графики
# ==========================================

def generate_loss_plot(history, title_idx):
    """Создает PIL картинку с графиком лоссов"""
    plt.figure(figsize=(4, 3))

    plt.plot(history['style'], label='Style', color='red', linewidth=1)
    plt.plot(history['content'], label='Content', color='blue', linewidth=1)
    plt.plot(history['tv'], label='TV', color='green', linewidth=1)

    plt.yscale('log')
    plt.title(f"Losses Img {title_idx}", fontsize=10)
    plt.xlabel("Step")
    plt.ylabel("Loss (Log Scale)")
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def create_labeled_grid(results, labels_map, cols=5):
    n = len(results)
    rows = (n + cols - 1) // cols
    thumb_size = 128
    text_height = 40
    padding = 10

    cell_w = thumb_size * 3
    cell_h = thumb_size + text_height
    grid_img = Image.new('RGB', (cols * (cell_w + padding) + padding, rows * (cell_h + padding) + padding), (250, 250, 250))
    draw = ImageDraw.Draw(grid_img)
    try: font = ImageFont.truetype("arial.ttf", 11)
    except: font = ImageFont.load_default()

    for i, (c, s, o, true_idx, p_bef, p_aft) in enumerate(results):
        row = i // cols
        col = i % cols
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + padding)

        true_name = CIFAR_CLASSES.get(true_idx, "???")
        vgg_bef = labels_map.get(p_bef, str(p_bef)).split(',')[0][:12]
        vgg_aft = labels_map.get(p_aft, str(p_aft)).split(',')[0][:12]

        draw.text((x + 5, y + 2), f"True: {true_name}", fill="black", font=font)
        is_stable = (p_bef == p_aft)
        color = (0, 100, 0) if is_stable else (200, 0, 0)
        draw.text((x + 5, y + 16), f"VGG: {vgg_bef} -> {vgg_aft}", fill=color, font=font)

        y_img = y + text_height
        c = c.resize((thumb_size, thumb_size))
        s = s.resize((thumb_size, thumb_size))
        o = o.resize((thumb_size, thumb_size))
        grid_img.paste(c, (x, y_img)); grid_img.paste(s, (x + thumb_size, y_img)); grid_img.paste(o, (x + thumb_size * 2, y_img))

    return grid_img

def create_loss_grid(plot_images, cols=5):
    n = len(plot_images)
    rows = (n + cols - 1) // cols
    w, h = plot_images[0].size
    padding = 10
    grid_w = cols * (w + padding) + padding
    grid_h = rows * (h + padding) + padding
    grid_img = Image.new('RGB', (grid_w, grid_h), color=(255, 255, 255))

    for i, img in enumerate(plot_images):
        row = i // cols
        col = i % cols
        x = padding + col * (w + padding)
        y = padding + row * (h + padding)
        grid_img.paste(img, (x, y))

    return grid_img

# ==========================================
# 5. Main
# ==========================================
def main():
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    labels_map = get_imagenet_labels()

    print("[INFO] Loading VGG19...")
    full_vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).to(cfg.device).eval()
    cnn_features = full_vgg.features
    for p in full_vgg.parameters(): p.requires_grad = False

    cifar, qd = get_loaders()
    indices = list(range(len(cifar)))
    random.shuffle(indices)
    subset = indices[:cfg.num_images]

    print(f"[INFO] Processing {cfg.num_images} images with history tracking...")
    print(f"       Weights: C={cfg.content_weight}, S={cfg.style_weight:.0e}, TV={cfg.tv_weight:.0e}")

    img_results = []
    plot_results = []

    to_pil = transforms.ToPILImage()

    for i, idx in enumerate(subset):
        print(f"Processing [{i+1}/{cfg.num_images}]...", end='\r')

        content = cifar[idx][0].unsqueeze(0).to(cfg.device)
        true_label_idx = cifar[idx][1]

        style_idx = (idx * 3) % len(qd)
        style = qd[style_idx].unsqueeze(0).to(cfg.device)

        # Получаем картинку И историю
        output, history = run_transfer(cnn_features, content, style)

        # Данные для картинки
        p_bef = get_class_pred(full_vgg, content)
        p_aft = get_class_pred(full_vgg, output)
        c_pil = to_pil(content.squeeze(0).cpu())
        s_pil = to_pil(style.squeeze(0).cpu()).convert('RGB')
        o_pil = to_pil(output.squeeze(0).cpu())
        img_results.append((c_pil, s_pil, o_pil, true_label_idx, p_bef, p_aft))

        # Данные для графика
        plot_pil = generate_loss_plot(history, i+1)
        plot_results.append(plot_pil)

        del content, style, output, history
        torch.cuda.empty_cache()

    print("\n[INFO] Saving grids...")

    # Сетка картинок
    grid_imgs = create_labeled_grid(img_results, labels_map, cols=cfg.grid_cols)
    grid_imgs.save(os.path.join(cfg.output_dir, 'images_grid.jpg'), quality=95)

    # Сетка графиков
    grid_plots = create_loss_grid(plot_results, cols=cfg.grid_cols)
    grid_plots.save(os.path.join(cfg.output_dir, 'losses_grid.jpg'), quality=95)

    print(f"[DONE] Saved to {cfg.output_dir}")

if __name__ == "__main__":
    main()