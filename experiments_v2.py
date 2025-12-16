"""
Style Transfer Research Script
==============================

Этот скрипт реализует перенос стиля (Neural Style Transfer) с использованием
VGG19. В качестве контента используются изображения из CIFAR-10, в качестве
стиля — наброски из датасета QuickDraw.

Особенности:
1. Автоматическая загрузка данных (CIFAR-10, QuickDraw, ImageNet labels).
2. Расчет метрик стабильности (Hard/Soft Stability) на основе предсказаний VGG.
3. Сравнение различных комбинаций слоев для Content и Style loss.
4. Генерация итогового отчета в консоль и сохранение визуализаций.

Зависимости:
    pip install torch torchvision numpy pillow requests tqdm
"""

import os
import copy
import json
import random
import warnings
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

# ==========================================
# 1. Конфигурация и Константы
# ==========================================

# Стандартные значения нормализации для моделей, обученных на ImageNet
CNN_NORMALIZATION_MEAN = torch.tensor([0.485, 0.456, 0.406])
CNN_NORMALIZATION_STD = torch.tensor([0.229, 0.224, 0.225])


@dataclass
class Config:
    """Конфигурация эксперимента."""
    seed: int = 42
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Параметры изображений
    imsize: int = 224
    data_dir: str = './data'
    output_dir: str = './results_labeled'

    # Параметры оптимизации
    num_steps: int = 300
    style_weight: float = 1e6
    content_weight: float = 10

    # Параметры эксперимента
    num_examples_per_exp: int = 10
    force_grayscale: bool = True  # Преобразование контента в ч/б (3 канала)


cfg = Config()


# ==========================================
# 2. Утилиты воспроизводимости и IO
# ==========================================

def set_seed(seed: int) -> None:
    """
    Фиксирует random seed для Python, NumPy и Torch для обеспечения
    воспроизводимости результатов (Критерий: Воспроизводимость).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_imagenet_labels() -> Dict[int, str]:
    """
    Загружает словарь классов ImageNet для интерпретации предсказаний модели.

    Returns:
        Dict[int, str]: Словарь {index: label_name}
    """
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    path = os.path.join(cfg.data_dir, "imagenet_class_index.json")

    if not os.path.exists(cfg.data_dir):
        os.makedirs(cfg.data_dir)

    if not os.path.exists(path):
        print("[INFO] Downloading ImageNet labels...")
        try:
            r = requests.get(url)
            with open(path, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            warnings.warn(f"Failed to download labels: {e}. Predictions will utilize IDs.")
            return {}

    try:
        with open(path, 'r') as f:
            class_idx = json.load(f)
        # Формат JSON: {"0": ["n01440764", "tench"], ...} -> берем второе значение (human readable)
        return {int(k): v[1] for k, v in class_idx.items()}
    except Exception:
        return {}


def save_debug_image(
        content_tensor: torch.Tensor,
        style_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        label_before: str,
        label_after: str,
        is_stable: bool,
        soft_stab: float,
        exp_id: int,
        exp_name: str
) -> None:
    """
    Создает и сохраняет коллаж (Content | Style | Result) с аннотациями.
    """
    to_pil = transforms.ToPILImage()

    # Денормализация и конвертация в PIL
    # Примечание: тензоры уже в диапазоне [0, 1] после clamp, transform.ToPILImage обрабатывает это.
    content_img = to_pil(content_tensor.squeeze(0).cpu())

    # Обработка стиля (если он grayscale 1xHxW -> повторяем до 3xHxW для визуализации)
    s_vis = style_tensor.squeeze(0).cpu()
    if s_vis.shape[0] == 1:
        s_vis = s_vis.repeat(3, 1, 1)
    style_img = to_pil(s_vis)

    output_img = to_pil(output_tensor.squeeze(0).cpu())

    # Создание холста
    width, height = content_img.size
    header_height = 40
    combined = Image.new('RGB', (width * 3, height + header_height), color='white')

    combined.paste(content_img, (0, header_height))
    combined.paste(style_img, (width, header_height))
    combined.paste(output_img, (width * 2, header_height))

    draw = ImageDraw.Draw(combined)

    # Текст аннотации
    text_labels = f"{label_before} -> {label_after}"
    text_stats = f"Hard: {int(is_stable)} | Soft: {soft_stab:.2f}"
    color = (0, 100, 0) if is_stable else (200, 0, 0)  # Зеленый если стабильно, красный если нет

    # Пытаемся использовать стандартный шрифт, если нет - дефолтный
    try:
        font = ImageFont.load_default()
    except IOError:
        font = None

    draw.text((10, 5), text_labels, fill=color, font=font)
    draw.text((10, 20), text_stats, fill=color, font=font)

    filename = f"{exp_id:02d}_{exp_name}.png"
    filepath = os.path.join(cfg.output_dir, filename)
    combined.save(filepath)


# ==========================================
# 3. Данные (Dataset & Loaders)
# ==========================================

class QuickDrawDataset(Dataset):
    """
    Кастомный Dataset для загрузки векторных рисунков QuickDraw (.npy формат).
    Автоматически скачивает данные, если они отсутствуют.
    """

    def __init__(self, root_dir: str, category: str = 'cat', transform: Optional[callable] = None):
        self.root_dir = root_dir
        self.category = category
        self.transform = transform
        self.file_path = os.path.join(root_dir, f'{category}.npy')

        self._ensure_data()

    def _ensure_data(self) -> None:
        """Проверяет наличие файла и загружает его."""
        if not os.path.exists(self.file_path):
            self._download()

        try:
            self.data = np.load(self.file_path)
        except Exception as e:
            print(f"[WARN] File corrupted or unloadable: {e}. Redownloading...")
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
            self._download()
            self.data = np.load(self.file_path)

    def _download(self) -> None:
        """Скачивает .npy файл с Google Cloud Storage."""
        os.makedirs(self.root_dir, exist_ok=True)
        url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{self.category}.npy'

        print(f"[INFO] Downloading QuickDraw category: {self.category}...")
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(self.file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            warnings.warn(f"Failed to download QuickDraw data: {e}")
            # Создаем пустой фейковый датасет, чтобы код не падал (fallback)
            self.data = np.zeros((10, 784), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # QuickDraw изображения хранятся как плоские векторы 28x28
        img_arr = self.data[idx].reshape(28, 28)
        img = Image.fromarray(img_arr).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img


def get_loaders() -> Tuple[Dataset, Dataset]:
    """
    Подготавливает датасеты CIFAR-10 и QuickDraw с необходимыми трансформациями.
    """
    transforms_list = [transforms.Resize((cfg.imsize, cfg.imsize))]

    # Опционально: делаем контент черно-белым (но в 3 каналах),
    # чтобы уравнять условия со стилем QuickDraw
    if cfg.force_grayscale:
        transforms_list.append(transforms.Grayscale(num_output_channels=3))

    transforms_list.append(transforms.ToTensor())
    t = transforms.Compose(transforms_list)

    print("[INFO] Loading CIFAR-10...")
    cifar = datasets.CIFAR10(root=cfg.data_dir, train=True, download=True, transform=t)

    print("[INFO] Loading QuickDraw (Style Source)...")
    qd = QuickDrawDataset(root_dir=cfg.data_dir, category='cat', transform=t)

    return cifar, qd


# ==========================================
# 4. Модель и Функции Потерь (Losses)
# ==========================================

class ContentLoss(nn.Module):
    """
    Вычисляет MSE Loss между признаками текущего изображения и целевого контента.
    """

    def __init__(self, target: torch.Tensor, weight: float):
        super(ContentLoss, self).__init__()
        # target отделяем от графа вычислений, так как это константа
        self.target = target.detach()
        self.weight = weight
        self.loss = 0.0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.loss = F.mse_loss(input, self.target) * self.weight
        return input


class StyleLoss(nn.Module):
    """
    Вычисляет Style Loss используя матрицы Грама (корреляции признаков).
    """

    def __init__(self, target: torch.Tensor, weight: float):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target).detach()
        self.weight = weight
        self.loss = 0.0

    def gram_matrix(self, input: torch.Tensor) -> torch.Tensor:
        a, b, c, d = input.size()
        # a=batch size(=1), b=number of feature maps, (c,d)=dimensions of a f. map
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        # Нормализуем значения матрицы Грама, деля на количество элементов
        return G.div(a * b * c * d)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target) * self.weight
        return input


class Normalization(nn.Module):
    """
    Слой нормализации входного изображения под статистику ImageNet.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super(Normalization, self).__init__()
        # Решейп для бродкастинга: [C x 1 x 1]
        self.register_buffer('mean', mean.view(-1, 1, 1))
        self.register_buffer('std', std.view(-1, 1, 1))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean) / self.std


def get_model_and_losses(
        cnn: nn.Module,
        style_img: torch.Tensor,
        content_img: torch.Tensor,
        c_layers_names: List[str],
        s_layers_names: List[str]
) -> Tuple[nn.Sequential, List[StyleLoss], List[ContentLoss]]:
    """
    Создает модель для переноса стиля, вставляя слои потерь (Loss Layers)
    в нужные места VGG.
    """
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD).to(cfg.device)

    content_losses = []
    style_losses = []

    # Строим новую последовательную модель (Sequential)
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            # inplace=False важен для корректной работы ContentLoss/StyleLoss
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        model.add_module(name, layer)

        # Вставка Content Loss
        if name in c_layers_names:
            target = model(content_img).detach()
            content_loss = ContentLoss(target, cfg.content_weight)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        # Вставка Style Loss
        if name in s_layers_names:
            target = model(style_img).detach()
            style_loss = StyleLoss(target, cfg.style_weight)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Обрезаем модель после последнего слоя потерь, чтобы не вычислять лишнее
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    model = model[:i + 1]

    return model, style_losses, content_losses


# ==========================================
# 5. Процесс оптимизации (Style Transfer)
# ==========================================

def get_logits(model: nn.Module, img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Получает "сырые" логиты от классификатора (VGG) для расчета метрик.
    """
    model.eval()
    with torch.no_grad():
        norm = transforms.Normalize(CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD)

        # Убеждаемся, что размерность [1, 3, H, W]
        if img_tensor.dim() == 3:
            inp = img_tensor.unsqueeze(0)
        else:
            inp = img_tensor

        inp = norm(inp.squeeze(0)).unsqueeze(0)
        return model(inp)


def run_style_transfer(
        cnn: nn.Module,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        input_img: torch.Tensor,
        c_layers: List[str],
        s_layers: List[str]
) -> Tuple[torch.Tensor, float, float]:
    """
    Запускает цикл градиентного спуска (LBFGS) для переноса стиля.

    Returns:
        input_img: Оптимизированное изображение.
        final_style_score: Финальное значение Style Loss.
        final_content_score: Финальное значение Content Loss.
    """
    model, style_losses, content_losses = get_model_and_losses(
        cnn, style_img, content_img, c_layers, s_layers
    )

    # Мы оптимизируем входное изображение, а не веса модели
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_img])

    run = [0]
    final_style_score = 0.0
    final_content_score = 0.0

    while run[0] <= cfg.num_steps:
        def closure():
            nonlocal final_style_score, final_content_score

            # Ограничиваем значения пикселей [0, 1]
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            final_style_score = style_score.item()
            final_content_score = content_score.item()

            # Логирование каждые 100 шагов
            if run[0] % 100 == 0:
                print(f"\tStep {run[0]}: Style Loss: {style_score.item():.2f} Content Loss: {content_score.item():.2f}")

            return loss

        optimizer.step(closure)

    # Финальный clamp
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img, final_style_score, final_content_score


# ==========================================
# 6. Основной цикл эксперимента
# ==========================================

def calculate_metrics(
        logits_before: torch.Tensor,
        logits_after: torch.Tensor
) -> Tuple[int, int, float, bool]:
    """
    Считает метрики изменения семантики изображения.
    1. KL Divergence -> Soft Stability
    2. Top-1 Accuracy Change -> Hard Stability
    """
    # Soft Stability через KL-дивергенцию распределения вероятностей
    target_probs = F.softmax(logits_before, dim=1)
    input_log_probs = F.log_softmax(logits_after, dim=1)
    kl_div = F.kl_div(input_log_probs, target_probs, reduction='batchmean').item()
    soft_stab = np.exp(-kl_div)  # Чем выше, тем лучше сохранилась семантика

    # Hard Stability (изменился ли предсказанный класс)
    pred_idx_before = logits_before.argmax(dim=1).item()
    pred_idx_after = logits_after.argmax(dim=1).item()
    is_stable = (pred_idx_before == pred_idx_after)

    return pred_idx_before, pred_idx_after, soft_stab, is_stable


def main():
    print("=" * 60)
    print("Запуск исследования Style Transfer")
    print(f"Device: {cfg.device}")
    print("=" * 60)

    # 1. Инициализация
    set_seed(cfg.seed)
    if os.path.exists(cfg.output_dir):
        shutil.rmtree(cfg.output_dir)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 2. Загрузка ресурсов
    imagenet_labels = get_imagenet_labels()

    print("[INFO] Loading VGG19 (ImageNet weights)...")
    # Используем weights вместо pretrained (deprecated)
    full_vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(cfg.device).eval()
    cnn_features = full_vgg.features

    # Замораживаем веса VGG
    for p in full_vgg.parameters():
        p.requires_grad = False

    cifar_ds, qd_ds = get_loaders()

    # 3. Определение конфигураций слоев для эксперимента
    content_configs = {
        'C_Early': ['conv_4'],  # Поверхностные признаки
        'C_Mid': ['conv_10'],  # Средние признаки
        'C_Deep': ['conv_14']  # Глубокие семантические признаки
    }
    style_configs = {
        'S_Uniform': ['conv_1', 'conv_6', 'conv_11', 'conv_14', 'conv_16'],  # Равномерно
        'S_DenseStart': ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'],  # Текстура
        'S_DenseMid': ['conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9'],  # Узоры
        'S_DenseEnd': ['conv_12', 'conv_13', 'conv_14', 'conv_15', 'conv_16']  # Глобальный стиль
    }

    # 4. Выбор данных для экспериментов
    indices = list(range(len(cifar_ds)))
    random.shuffle(indices)
    selected_indices = indices[:cfg.num_examples_per_exp]

    print(f"[INFO] Config: Style Weight={cfg.style_weight:.0e}, Content Weight={cfg.content_weight:.0e}")

    experiments_results = []
    exp_id = 0

    # 5. Цикл по всем комбинациям настроек
    for c_name, c_layers in content_configs.items():
        for s_name, s_layers in style_configs.items():
            exp_id += 1
            exp_name = f"{c_name}_vs_{s_name}"
            print(f"\n=== Experiment {exp_id}/{len(content_configs) * len(style_configs)}: {exp_name} ===")

            exp_metrics = {
                'style_loss': [],
                'content_loss': [],
                'stable_count': 0,
                'soft_stability_sum': 0.0
            }

            for i, idx in enumerate(selected_indices):
                # Подготовка данных
                content = cifar_ds[idx][0].unsqueeze(0).to(cfg.device)

                # Выбираем стиль (циклично)
                style_idx = (idx * 3) % len(qd_ds)
                style = qd_ds[style_idx].unsqueeze(0).to(cfg.device)

                # Инициализация шумом или контентом (здесь контентом для стабильности)
                input_img = content.clone()

                # --- STYLE TRANSFER ---
                output, s_loss, c_loss = run_style_transfer(
                    cnn_features, content, style, input_img, c_layers, s_layers
                )

                # --- АНАЛИЗ (Metrics) ---
                logits_before = get_logits(full_vgg, content)
                logits_after = get_logits(full_vgg, output)

                idx_before, idx_after, soft_stab, is_stable = calculate_metrics(logits_before, logits_after)

                # Агрегация метрик
                exp_metrics['soft_stability_sum'] += soft_stab
                exp_metrics['style_loss'].append(s_loss)
                exp_metrics['content_loss'].append(c_loss)
                if is_stable:
                    exp_metrics['stable_count'] += 1

                # Сохранение визуализации для первого примера в батче
                if i == 0:
                    label_before = imagenet_labels.get(idx_before, str(idx_before))
                    label_after = imagenet_labels.get(idx_after, str(idx_after))

                    # Обрезка длинных названий
                    label_before = (label_before[:15] + '..') if len(label_before) > 15 else label_before
                    label_after = (label_after[:15] + '..') if len(label_after) > 15 else label_after

                    save_debug_image(
                        content, style, output,
                        label_before, label_after,
                        is_stable, soft_stab,
                        exp_id, exp_name
                    )

            # --- ИТОГИ ЭКСПЕРИМЕНТА ---
            avg_s = sum(exp_metrics['style_loss']) / len(exp_metrics['style_loss'])
            avg_c = sum(exp_metrics['content_loss']) / len(exp_metrics['content_loss'])

            hard_stability = exp_metrics['stable_count'] / cfg.num_examples_per_exp
            avg_soft_stability = exp_metrics['soft_stability_sum'] / cfg.num_examples_per_exp

            # Комплексный Score: поощряем высокую стабильность при минимизации style loss
            # Добавлено 1e-6 для избежания деления на ноль
            score = avg_soft_stability / (np.log1p(avg_s) + 1e-6)

            experiments_results.append({
                'name': exp_name,
                'avg_style': avg_s,
                'avg_content': avg_c,
                'hard_stab': hard_stability,
                'soft_stab': avg_soft_stability,
                'score': score
            })

    # 6. Вывод итоговой таблицы результатов
    print("\n" + "=" * 110)
    print(f"{'Experiment Name':<30} | {'S-Loss':<10} | {'HardStab':<10} | {'SoftStab':<10} | {'Score':<10}")
    print("-" * 110)

    # Сортировка по Score (чем выше, тем лучше баланс стиля и сохранения сути)
    experiments_results.sort(key=lambda x: x['score'], reverse=True)

    for res in experiments_results:
        print(
            f"{res['name']:<30} | "
            f"{res['avg_style']:<10.1f} | "
            f"{res['hard_stab']:<10.2f} | "
            f"{res['soft_stab']:<10.3f} | "
            f"{res['score']:.4f}"
        )
    print("=" * 110)
    print(f"[INFO] Results saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()