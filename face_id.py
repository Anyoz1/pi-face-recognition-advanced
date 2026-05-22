import time
import cv2
import numpy as np
import os
import psutil

# Указываем новую папку с фотографиями
DATASET_DIR = "dataset"

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return 0.0

def get_arcface_embedding(net, face_img, bbox):
    """Вырезает лицо по координатам детектора, подгоняет под 112x112 и строит ArcFace-вектор"""
    x, y, w, h = map(int, bbox[:4])
    
    # Делаем небольшой отступ, чтобы ArcFace лучше видел геометрию лица
    h_pad = int(h * 0.1)
    w_pad = int(w * 0.1)
    
    img_h, img_w = face_img.shape[:2]
    y1 = max(0, y - h_pad)
    y2 = min(img_h, y + h + h_pad)
    x1 = max(0, x - w_pad)
    x2 = min(img_w, x + w + w_pad)
    
    crop = face_img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
        
    # Нормализация строго под требования ArcFace
    blob = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
    net.setInput(blob)
    embedding = net.forward()
    cv2.normalize(embedding, embedding)
    return embedding.flatten()

def load_arcface_database(detector, arcface_net):
    database = {}
    print("┌────────────────────────────────────────────────────────┐")
    print("│ ИНИЦИАЛИЗАЦИЯ ИИ-БАЗЫ ДАННЫХ В НОВОМ ПРОЕКТЕ (ArcFace) │")
    print("├────────────────────────────────────────────────────────┤")
    
    # Создаем папку, если сервера её ещё не создали
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        
    # Читаем файлы ИМЕННО из папки dataset
    files = os.listdir(DATASET_DIR)
    valid_extensions = (".jpg", ".jpeg", ".png")
    
    for file in files:
        if file.lower().endswith(valid_extensions):
            name = os.path.splitext(file)[0]
            display_name = "Zhaksybek" if name == "master_face" else name
            
            # Собираем правильный путь к файлу внутри папки dataset
            img_path = os.path.join(DATASET_DIR, file)
            img = cv2.imread(img_path)
            
            if img is not None:
                detector.setInputSize((img.shape[1], img.shape[0]))
                _, faces = detector.detect(img)
                if faces is not None and len(faces) > 0:
                    feature = get_arcface_embedding(arcface_net, img, faces[0])
                    if feature is not None:
                        database[display_name] = feature
                        print(f"│  ► [ВЕКТОР ГОТОВ] Добавлен шаблон: {display_name:<19} │")
                else:
                    print(f"│  ⚠️ [ПРОПУСК] Лицо не обнаружено на фото: {file:<12} │")
    print("└────────────────────────────────────────────────────────┘")
    return database

def main():
    process = psutil.Process(os.getpid())
    
    det_model = "models/face_detection_yunet.onnx"
    rec_model = "models/arcface.onnx"
    
    # Инициализируем детекцию (быстрый скан на 160x120)
    detector = cv2.FaceDetectorYN.create(det_model, "", (160, 120), 0.6, 0.3, 50)
    
    try:
        arcface_net = cv2.dnn.readNetFromONNX(rec_model)
        arcface_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        arcface_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception as e:
        print(f"[ОШИБКА] Не удалось загрузить ArcFace: {e}")
        return

    face_db = load_arcface_database(detector, arcface_net)
    if not face_db:
        print(f"[ВНИМАНИЕ] Папка {DATASET_DIR} пуста. Ожидание синхронизации с сервером ноутбука.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    print("\n STATUS: Экспериментальный СКУД [ArcFace] активен.")
    frame_count = 0
    psutil.cpu_percent(interval=None)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            if frame_count % 2 == 0: continue

            # Быстрый поиск лица
            small_frame = cv2.resize(frame, (160, 120))
            detector.setInputSize((small_frame.shape[1], small_frame.shape[0]))
            status, faces = detector.detect(small_frame)

            if faces is not None and len(faces) > 0 and len(face_db) > 0:
                recognition_start = time.perf_counter()
                
                # Осветляем оригинал
                ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                channels = list(cv2.split(ycrcb))
                channels[0] = np.clip(channels[0].astype(np.int16) + 40, 0, 255).astype(np.uint8)
                frame_bright = cv2.cvtColor(cv2.merge(channels), cv2.COLOR_YCrCb2BGR)

                # Ищем точные координаты лица на осветленном оригинале
                detector.setInputSize((frame_bright.shape[1], frame_bright.shape[0]))
                _, faces_large = detector.detect(frame_bright)

                if faces_large is not None and len(faces_large) > 0:
                    # Строим тяжелый вектор
                    current_feature = get_arcface_embedding(arcface_net, frame_bright, faces_large[0])
                    
                    if current_feature is not None:
                        best_match_name = "Unknown"
                        max_score = -1.0

                        for user_name, master_feature in face_db.items():
                            cosine_score = np.dot(master_feature, current_feature)
                            if cosine_score > max_score:
                                max_score = cosine_score
                                # Порог для ArcFace ставим 0.38
                                if cosine_score > 0.38:
                                    best_match_name = user_name

                        total_time_ms = (time.perf_counter() - recognition_start) * 1000.0

                        if best_match_name != "Unknown":
                            cpu_usage = psutil.cpu_percent(interval=None)
                            ram_mb = process.memory_info().rss / (1024 * 1024)
                            current_temp = get_cpu_temp()
                            
                            print("\n┌────────────────────────────────────────────────────────┐")
                            print("│ 🔓 [ДОСТУП РАЗРЕШЕН] ТЕСТ НОВОГО АЛГОРИТМА ARCFACE     │")
                            print("├────────────────────────────────────────────────────────┤")
                            print(f"│ Пользователь:          {best_match_name:<31} │")
                            print(f"│ ArcFace Similarity:    {max_score:<31.4f} │")
                            print(f"│ Скорость инференса:    {f'{total_time_ms:.1f} мс':<31} │")
                            print("├────────────────────────────────────────────────────────┤")
                            print("│ ТЕЛЕМЕТРИЯ ОБОРУДОВАНИЯ (Orange Pi 5B)                  │")
                            print(f"│ Загрузка процессора:   {f'{cpu_usage}%':<31} │")
                            print(f"│ Выделенная память ОЗУ: {f'{ram_mb:.2f} МБ':<31} │")
                            print(f"│ Температура чипа CPU:  {f'{current_temp:.1f}°C':<31} │")
                            print("└────────────────────────────────────────────────────────┘")
                            break
                        else:
                            print(f"\n ⚠️  [ОТКАЗ] Неизвестный. ArcFace Score: {max_score:.4f}")

            print(f" [ТЕСТ] Сканирование зоны... Кадр цикла: {frame_count}", end="\r")
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n [СТОП] СКУД выключен.")
    finally:
        cap.release()

if __name__ == "__main__":
    main()
