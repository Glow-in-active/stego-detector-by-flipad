# Скачать датасет и запустить FLIPAD (без DCT-пайплайна проекта)

Цель: **скачать PNG** из вашего S3 (Yandex Object Storage) на диск **в структуру, которую ждёт** `flipad/sma_deepsad.py`,
и затем запустить `flipad` на этих данных.

В этом сценарии **мы не используем** ваш HPC-бейзлайн с кэшированием DCT (`hpc/run_stego_vs_sd_dct.py`).

---

## 1) Структура папок, которую ждёт `flipad/sma_deepsad.py`

По плану проекта вы используете датасет-лейбл `coco2014train` (это ок), и две директории:

- **inlier** (ваши stego): `mas_grdh_stego`
- **outlier** (не-stego): `genimage`

Итоговая структура:

```
data/coco2014train/
  mas_grdh_stego/
    train/*.png
    val/*.png
    test/*.png
  genimage/
    train/*.png
    val/*.png
    test/*.png
```

---

## 2) Скачать поднабор из S3 в эту структуру

В репозитории есть скрипт:

- `tools/s3_download_to_flipad.py`

Он:

- листит ключи в бакете,
- детерминированно отбирает подмножества в `train/val/test`,
- скачивает PNG в `data/coco2014train/...`,
- сохраняет manifests в `data/coco2014train/_manifests/*.txt` (для воспроизводимости).

### 2.1) Переменные окружения (ключи S3)

Не храните ключи в репозитории. Перед запуском выставьте env:

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="ru-central1"
export AWS_DEFAULT_REGION="ru-central1"
export ENDPOINT="https://storage.yandexcloud.net"
export BUCKET="stegopractice"
```

### 2.2) Пилотные размеры (рекомендовано)

```bash
python3 -u tools/s3_download_to_flipad.py \
  --data-path data \
  --dataset coco2014train \
  --stego-dir mas_grdh_stego \
  --real-dir genimage \
  --seed 20260320 \
  --stego-train 5000 --stego-val 2000 --stego-test 5000 \
  --real-train 5000  --real-val 2000  --real-test 20000 \
  --workers 16
```

Если хотите сначала убедиться, что сплиты адекватные (без скачивания), добавьте `--dry-run`.

---

## 3) Запустить `flipad/sma_deepsad.py` на скачанных данных

### 3.1) Важно про `--feat`

В текущем виде `flipad/sma_deepsad.py`:

- `--feat act` требует Stable Diffusion wrapper и тяжёлые зависимости
- `--feat raw` / `--feat dct` — baseline, проще для старта

Чтобы **не использовать DCT**, начните с `--feat raw`:

```bash
python3 -u flipad/sma_deepsad.py \
  --data-path data \
  --dataset coco2014train \
  --model stablediffusion \
  --checkpoint-path "unused" \
  --train-dir mas_grdh_stego \
  --real-dir genimage \
  --feat raw \
  --net-name cifar10_biglenet \
  --num-workers 8 \
  --seed 42 \
  --num-train 5000 \
  --num-val 2000 \
  --num-test 5000 \
  --tolerable-fnr 0.01 \
  --batch-size 128 \
  --n-epochs 20 \
  --downsampling center_crop \
  --output-path output
```

Примечание: `--model stablediffusion` здесь используется только как “ярлык домена” (для resize логики). Для `--feat raw`
wrapper не инициализируется.

### 3.2) Где смотреть результат

Ищите:

- `output/.../log.log` — пороги `Thresholds: {...}` и метрики
- `output/.../results/results.json` — результаты по каждому тестовому каталогу
- `output/.../results/auc_*.npy` — AUROC

