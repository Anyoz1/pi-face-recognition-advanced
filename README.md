# Pi Face Recognition Access Control

Локальная СКУД-система на Orange Pi 5B с распознаванием лиц. Проект использует YuNet для детекции, ArcFace для построения embeddings, бинарную базу `database_bin/*.bin` и веб-админку для управления людьми и логами. Всё работает локально, без облачных сервисов.

## Возможности

- Распознавание лиц с камеры в реальном времени.
- Полностью локальная работа без отправки кадров или embeddings в облако.
- База людей в формате `database_bin/*.bin`.
- Добавление людей через admin-сайт.
- Удаление людей через admin-сайт.
- Просмотр логов работы СКУД.
- Два режима FPS: standby для ожидания и active при найденном лице.
- Early accept: быстрый пропуск при уверенном совпадении.
- Фильтр анфаса по landmarks.
- Фильтр резкости, чтобы не принимать смазанные кадры.
- Контроль CPU и температуры устройства в логах.
- Опциональный web/live режим с камерой и live frame/stream.

## Как работает pipeline

1. Камера получает кадр.
2. YuNet ищет лицо на уменьшенном изображении.
3. Из найденных лиц выбирается самое крупное.
4. Проверяется минимальный размер лица.
5. По landmarks считается `frontal_score`.
6. Если лицо повернуто боком, кадр игнорируется.
7. По 5 landmarks выполняется alignment лица.
8. Лицо приводится к размеру `112x112`.
9. ArcFace создаёт embedding: 512 значений `float32`.
10. Embedding нормализуется.
11. `database_bin/` загружается как матрица embeddings.
12. Сравнение идёт через dot product, то есть cosine similarity для нормализованных векторов.
13. Если `score >= ACCEPT_THRESHOLD`, доступ разрешён.
14. Если `REJECT_THRESHOLD <= score < ACCEPT_THRESHOLD`, результат считается сомнительным.
15. Если `score < REJECT_THRESHOLD`, человек считается неизвестным.

## Что оптимизировано

- Основной recognition loop написан на C++.
- Детектор работает на кадре `160x120`, чтобы снизить нагрузку.
- Standby FPS снижает нагрев и расход CPU в режиме ожидания.
- Active FPS включается, когда в кадре появляется лицо.
- Используется серия кадров и выбор лучшего кадра по качеству.
- Early accept завершает проверку раньше при хорошем score.
- База embeddings хранится в бинарном формате.
- Сравнение со всеми людьми выполняется одной матричной операцией.
- В логах фиксируется время этапов `det`, `arc`, `match`.
- В логах фиксируются CPU и температура.

## Модели

- `models/face_detection_yunet.onnx` — YuNet detector.
- `models/arcface.onnx` — ArcFace embedding model.

Модели работают локально через OpenCV DNN. Без файлов в `models/` проект не запустит детекцию и распознавание.

## База людей

База распознавания лежит в `database_bin/`.

Формат:

```text
database_bin/<name>.bin
```

Один `.bin` файл содержит один ArcFace embedding:

```text
512 float32 = 2048 bytes
```

Это embedding, а не фотография. Директория `dataset/` хранит исходные фото или кадры регистрации, а `database_bin/` нужен непосредственно для распознавания.

## Главные файлы

- `face_id.cpp` — основной СКУД и главный recognition loop.
- `admin_server.py` — веб-админка для базы, логов и добавления людей.
- `Makefile` — сборка C++ бинарников.
- `face_id_web.cpp` — опциональный live/web режим.
- `register_face.cpp` — CLI регистрация в `database_bin/`; важно, чтобы preprocessing регистрации совпадал с `face_id.cpp`.
- `test_speed.cpp` — тест скорости и диагностика производительности.
- `test_single.py` / `test_multi.py` — Python диагностика.
- `PROJECT_FILES.md` — подробное описание структуры проекта.
- `RUN_COMMANDS.md` — рабочие команды запуска и обслуживания.
- `archive_unused/` — старые файлы, сохранённые как архив.

## Установка зависимостей

Для Orange Pi / Debian / Armbian:

```bash
sudo apt update
sudo apt install -y build-essential pkg-config libopencv-dev python3-venv python3-opencv python3-numpy
```

Python admin server через venv:

```bash
python3 -m venv --system-site-packages .admin-venv
source .admin-venv/bin/activate
pip install fastapi uvicorn python-multipart
```

Если используется `uv`:

```bash
uv sync
```

## Сборка

```bash
make face_id
make face_id_web
make register_face
make test_speed
```

## Запуск СКУД

```bash
mkdir -p web_live
stdbuf -oL -eL ./face_id 2>&1 | tee -a web_live/face_id.log
```

## Запуск admin server

```bash
source .admin-venv/bin/activate
python admin_server.py --host 0.0.0.0 --port 8080
```

Открыть:

```text
http://<ORANGE_PI_IP>:8080
```

## Добавление человека

Рекомендуемый способ для текущей версии — через `admin_server.py`, потому что он создаёт `database_bin/<name>.bin` с preprocessing, совместимым с текущим `face_id.cpp`.

Через сайт:

1. Открыть admin server.
2. Ввести имя.
3. Загрузить одно или несколько фото.
4. Нажать создать BIN.
5. Проверить `database_bin/<name>.bin`.

## Удаление человека

Через admin-сайт: нажать `Удалить` рядом с человеком.

Через терминал:

```bash
rm database_bin/person.bin
```

## Проверка

Проверить базу:

```bash
ls -lh database_bin
stat -c '%n %s bytes' database_bin/*.bin
```

Проверить ключевые параметры в `face_id.cpp`:

```bash
grep -n "ACCEPT_THRESHOLD|REJECT_THRESHOLD|STANDBY_FPS|ACTIVE_FPS|MIN_FRONTAL_SCORE" face_id.cpp | head -30
```

## Структура проекта

Компактная структура без больших файлов:

```text
.
├── Makefile
├── README.md
├── PROJECT_FILES.md
├── RUN_COMMANDS.md
├── admin_server.py
├── face_id.cpp
├── face_id_test.cpp
├── face_id_web.cpp
├── register_face.cpp
├── test_speed.cpp
├── test_single.py
├── test_multi.py
├── pyproject.toml
├── uv.lock
├── archive_unused/
├── database_bin/
│   └── .gitkeep
├── dataset/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── web_live/
│   └── .gitkeep
└── history_logs/
    └── .gitkeep
```

## Безопасность и приватность

- `database_bin/` содержит биометрические embeddings.
- `dataset/` может содержать фотографии людей.
- `history_logs/` может содержать кадры проходов.
- Эти данные не стоит пушить в публичный GitHub.
- Система работает локально, без облачной обработки.

## Что не пушить

`.gitignore` исключает приватные и тяжёлые runtime-файлы:

- `database_bin/`
- `dataset/`
- `history_logs/`
- `web_live/`
- `.admin-venv/`
- бинарники сборки;
- ONNX модели, если они большие.

Папки сохраняются в репозитории через `.gitkeep`, но их содержимое не попадает в GitHub.
# Pi-Face-Recognition-Access-Control
