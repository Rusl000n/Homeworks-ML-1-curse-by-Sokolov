# 🎓 Машинное обучение 1 — Решения домашних заданий

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2%2B-FF6B35)](https://catboost.ai)
[![License](https://img.shields.io/badge/License-MIT-4CAF50)](LICENSE)

**Курс Евгения Соколова · НИУ ВШЭ · Весна 2026**

[Обзор](#-обзор) · [Стек технологий](#-стек-технологий) · [Домашние задания](#-домашние-задания) · [Приобретённые навыки](#-приобретённые-навыки) · [Установка](#-быстрый-старт) · [Контакты](#-контакты)

</div>

---

## 📖 Обзор

Этот репозиторий содержит мои решения домашних заданий по курсу **«Машинное обучение 1»** в НИУ ВШЭ. Каждое задание выходит за рамки поверхностного использования библиотек:

- 🔧 **Реализация с нуля** ключевых алгоритмов (градиентный спуск, деревья решений, градиентный бустинг)
- 📊 **Сравнительный анализ** собственного кода и production-библиотек
- 🧪 **Строгая оценка** с корректной кросс-валидацией, декомпозицией смещения-дисперсии и статистическим тестированием
- 📝 **Хорошо задокументированный код** с пояснениями принятых архитектурных решений

> *"Если ты не можешь реализовать это с нуля, ты не понимаешь этого по-настоящему."*

---

## 🛠 Стек технологий

| Категория | Технологии |
|:----------|:-----------|
| **Язык** | Python 3.10+ |
| **Обработка данных** | Pandas, Polars, NumPy |
| **Классический ML** | Scikit-learn, CatBoost, LightGBM, XGBoost |
| **Глубокое обучение** | PyTorch, torchaudio, librosa |
| **Feature Engineering** | category_encoders, Optuna, scipy.sparse |
| **Визуализация** | Matplotlib, Seaborn |
| **Инструменты** | Jupyter, Git, Docker |

---

## 📝 Домашние задания

### HW-01 · Обработка табличных данных
`Pandas` `Polars` `EDA` `Visualization`

Разведочный анализ данных и манипуляции с табличными данными с упором на оптимизацию производительности.

- Векторизованные операции (ни одного цикла на Python)
- Бенчмарки производительности Pandas vs Polars с ленивыми вычислениями
- Продвинутые агрегации: `groupby`, `pivot_table`, `MultiIndex`
- Стратегии обработки пропусков и поиск выбросов

### HW-02 · Оптимизация градиентным спуском
`NumPy` `Linear Algebra` `Optimization`

Пять алгоритмов градиентного спуска, реализованных **полностью с нуля** на NumPy.

- **Vanilla GD**, **SGD**, **Momentum**, **Adam**, **SAG**
- Аналитическое решение МНК через `np.linalg.solve`
- Кастомные функции потерь: MSE, Huber, LogCosh
- Расписания learning rate: TimeDecay, CosineAnnealing
- Визуализация 2D-траекторий сходимости

### HW-03 · Feature Engineering и пайплайны
`Scikit-learn` `Optuna` `Feature Engineering` `Sparse Matrices`

Feature engineering production-уровня с предотвращением утечки данных.

- Кастомные трансформеры с наследованием от `BaseEstimator`, `TransformerMixin`
- K-Fold Target Encoding без утечек
- Признаки для временных рядов: лаги, скользящие статистики, циклическое кодирование
- Корректная CV для временных рядов: `TimeSeriesSplit` с purging и embargo
- Байесовская оптимизация гиперпараметров с прунингом в Optuna

### HW-04 · Глубокое обучение и PINNs
`PyTorch` `CNN` `Audio Classification` `Physics-Informed ML`

Архитектуры нейронных сетей от базовых MLP до моделей, информированных физикой.

- Кастомные `Dataset` / `DataLoader` с кэшированием на диск
- Классификация аудио на UrbanSound8K через мел-спектрограммы
- **SPINN** (Separable Physics-Informed Neural Networks):
  - Решение дифференциальных уравнений в частных производных через CP-декомпозицию
  - Физический лосс через `torch.autograd.grad`
- LR-шедулеры, early stopping, чекпоинтинг моделей

### HW-05 · Деревья решений с нуля
`NumPy` `Recursive Algorithms` `Bias-Variance`

Полная реализация дерева решений с теоретическим анализом.

- Рекурсивное построение дерева с критериями Gini, Entropy и MSE
- Стратегии pre-pruning (`max_depth`, `min_samples_split`)
- **Декомпозиция смещения-дисперсии** через бутстрап-выборку
- Визуализация разделяющих границ на 2D-датасетах
- Важность признаков через уменьшение примеси (impurity decrease)

### HW-06 · Градиентный бустинг
`CatBoost` `XGBoost` `LightGBM` `Optuna` `Advanced Boosting`

От наивного градиентного бустинга до production-ready ансамблей.

- Реализация с нуля: обучение антиградиенту, вычисление оптимального шага (gamma)
- Современные техники: **GOSS**, **DART**, **Focal Loss**, early stopping
- Ordered Target Encoding (в стиле CatBoost)
- Методы квантования: MinEntropy, Piecewise
- Сравнение с XGBoost / LightGBM / CatBoost
- Optuna с прунингом для поиска гиперпараметров

---

## 🎯 Приобретённые навыки

### Математика и теория
Линейная алгебра · Теория вероятностей · Математическая статистика · Методы оптимизации · Декомпозиция смещения-дисперсии · Теория информации (Gini, Entropy)

### Программирование
Продвинутый Python (ООП, функциональные паттерны) · NumPy broadcasting и векторизация · Pandas/Polars для больших данных · PyTorch autograd и кастомные модули · Базовый SQL

### Машинное обучение
Линейные модели · Деревья решений и ансамбли · Внутреннее устройство градиентного бустинга · Глубокое обучение (CNN, PINNs) · Feature engineering · Оптимизация гиперпараметров · Корректная оценка моделей

### Инженерные практики
Система контроля версий Git · Воспроизводимость экспериментов (фиксация seed, фиксация версий) · Документация в Jupyter · Базовый Docker · Стиль кода PEP 8