# Запуск на суперкомпе ВШЭ (Slurm): stego (mas_GRDH) vs clean SD (Genimage/ai)

Цель: прогнать **полный датасет** на V100 через Slurm, не нарушая правило “не писать в `/tmp`”.
Мы используем `dct`‑признаки как быстрый baseline (как в Colab), а данные читаем из Yandex Object Storage (S3).

## Важное замечание про `/tmp` (по письму инженера)

- **Нельзя** создавать файлы в `/tmp`.
- Используй каталог из переменной окружения **`$TMPDIR`**.
- Перед завершением задачи нужно **очищать** свой временный каталог.

В sbatch ниже это учтено: мы создаём `$JOB_TMPDIR` внутри `$TMPDIR` и чистим его через `trap`.

## 0) Что реально по времени (оценка)

По пилоту в Colab `dct` для 2000+2000 картинок занимал ~1 минуту на фичи (локальные файлы).
На кластере будет два ограничителя:

- **S3 чтение** (много маленьких объектов): часто главный bottleneck.
- **DCT**: сравнительно лёгкая математика.

Грубая оценка (для 214k stego + 166k clean ≈ 380k изображений, V100):

- **извлечение признаков**: ~2–8 часов (зависит от S3 пропускной + параллелизма)
- **обучение DeepSAD**: обычно ~0.5–2 часа

Чтобы не ждать вслепую, сначала стоит сделать прогон на узле с меньшими числами (например 20k/5k/20k) и измерить
скорость “изображений/сек” на этапе **extract**. После этого выставлять `--time` на полный прогон.

## 1) Данные в S3 (как у тебя)

- бакет: `stegopractice`
- stego: `Stego/**/identity/*.png` (json игнорируем)
- clean SD: `Genimage/ai/*.png`

## 2) Рекомендованная стратегия для полного прогона

Главная проблема прямого запуска `flipad/sma_deepsad.py` на полном датасете — память:
если оставить признаки 128×128 float32, то `X_train` может занимать десятки гигабайт.

Поэтому для полного прогона мы делаем:

- **DCT‑признаки** на **32×32** (3 канала) и сохраняем их на диск как float16 (`.npy` memmap)
- обучаем DeepSAD сеть `cifar10_LeNet` (ожидает 3×32×32)

Это даёт 2 плюса:

- признаки в RAM становятся адекватными;
- обучение быстрее.

Если тебе принципиально оставить 128×128 — нужно либо много RAM, либо переделывать обучение на “стриминг”.

## 3) Главный запуск: `cache` (1 раз) → `cv` (K-fold)

В репозитории добавлен скрипт:

- `hpc/run_stego_vs_sd_dct.py`

Он не использует `/tmp` (если задан `$TMPDIR`) и читает изображения из S3 через `boto3`.

### 3.1) Кэширование признаков на весь датасет (cache)

Запускается на вычислительном узле (GPU не нужно). Делает:

- листинг S3 ключей,
- скачивание+декодирование PNG,
- DCT 32×32,
- запись на диск:
  - `stego_dct32.npy`, `stego_keys_used.txt`, `stego_bad_keys.txt`
  - `clean_dct32.npy`, `clean_keys_used.txt`, `clean_bad_keys.txt`

Это **самый долгий** этап. Его делаем **один раз**.

```bash
python3 -u hpc/run_stego_vs_sd_dct.py cache \
  --out-dir runs/cache_$(date +%Y%m%d_%H%M%S) \
  --size 32 \
  --workers 32 \
  --upload-prefix "Results/stego_detector/cache_$(date +%Y%m%d_%H%M%S)"
```

### 3.2) K-fold cross-validation (cv)

После того как `cache` готов, делаем K-fold CV **без повторного скачивания/пересчёта DCT**.
На каждом fold:

- train: берём `train-per-class` stego + `train-per-class` clean из (K−1) частей
- val: берём `val-size` stego из train-пула (для порога под FNR)
- test: текущий fold (stego + clean)

В итоге получаем `cv_summary.json` и `cv_results.jsonl`.

```bash
python3 -u hpc/run_stego_vs_sd_dct.py cv \
  --cache-dir runs/<cache_run> \
  --out-dir runs/<cache_run>/cv_k5 \
  --k 5 \
  --seed 20260320 \
  --val-size 10000 \
  --train-per-class 130000 \
  --net cifar10_LeNet \
  --epochs 50 \
  --batch-size 256 \
  --num-workers 8 \
  --tolerable-fnr 0.01 \
  --upload-prefix "Results/stego_detector/$(basename runs/<cache_run>)/cv_k5"
```

Результат: `runs/<cache_run>/cv_k5/cv_summary.json` (avg TPR/FPR/AUC + пофолдовое).

## 4) Готовые sbatch (просто запустить)

В репозитории уже лежат:

- `hpc/stego_cache.sbatch` (CPU, 24 часа)
- `hpc/stego_cv.sbatch` (GPU, 12 часов)

Их можно запускать напрямую.

Ниже — **шаблон**: ты вставляешь свои значения `--account`, `--time`, и путь к окружению.

## 5) Как ставить в очередь (рекомендовано)

1) На login-узле перейди в папку проекта (лучше в scratch) и подготовь окружение (см. ниже).
2) Запусти кэширование:

```bash
sbatch hpc/stego_cache.sbatch
```

3) После завершения job возьми его id, например `123456`. Запусти CV:

```bash
RUN_DIR="runs/cache_123456" sbatch hpc/stego_cv.sbatch
```

## 6) Что должно быть в python окружении

На вычислительном узле должны импортироваться пакеты:

- `boto3`, `botocore`, `Pillow`
- `numpy`, `scikit-learn`
- `torch`, `torchvision`, `torch-dct`

И должен быть доступен модуль `sad.*` из `flipad/src`:

- самый простой вариант: `export PYTHONPATH=/path/to/repo/flipad/src:$PYTHONPATH` (он уже стоит в sbatch)

