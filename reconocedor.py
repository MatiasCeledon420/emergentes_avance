import cv2
import mediapipe as mp
import numpy as np
import socket
import threading
import time
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
from datetime import datetime
from collections import deque # --- NUEVO ---

# Configuración del servidor socket
HOST = '127.0.0.1'
PORT = 12345
servidor = None
cliente_conectado = False
mensaje_lock = threading.Lock()
ultima_letra = " "

# Configuración de ML
MODELO_PATH = "modelo_asl.pkl"
DATOS_PATH = "datos_entrenamiento.json"
modelo_ml = None
esta_entrenado = False

# Configuración de recolección de datos
modo_recoleccion = False
letra_objetivo = ""
samples_recolectados = 0
max_samples_por_letra = 50

# Inicializar MediaPipe con configuración optimizada
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7,
    model_complexity=1
)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Mejorar calidad de captura
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

def extraer_caracteristicas_avanzadas(hand_landmarks):
    """Extrae un conjunto completo de características de la mano"""
    landmarks = hand_landmarks.landmark
    caracteristicas = []
    
    for lm in landmarks:
        caracteristicas.extend([lm.x, lm.y, lm.z])
    
    puntos_clave = [0, 4, 8, 12, 16, 20]
    for i in range(len(puntos_clave)):
        for j in range(i+1, len(puntos_clave)):
            dist = calcular_distancia_3d(landmarks[puntos_clave[i]], landmarks[puntos_clave[j]])
            caracteristicas.append(dist)
    
    angulos = [
        calcular_angulo_3d(landmarks[1], landmarks[2], landmarks[3]),
        calcular_angulo_3d(landmarks[2], landmarks[3], landmarks[4]),
        calcular_angulo_3d(landmarks[5], landmarks[6], landmarks[7]),
        calcular_angulo_3d(landmarks[6], landmarks[7], landmarks[8]),
        calcular_angulo_3d(landmarks[9], landmarks[10], landmarks[11]),
        calcular_angulo_3d(landmarks[10], landmarks[11], landmarks[12]),
        calcular_angulo_3d(landmarks[13], landmarks[14], landmarks[15]),
        calcular_angulo_3d(landmarks[14], landmarks[15], landmarks[16]),
        calcular_angulo_3d(landmarks[17], landmarks[18], landmarks[19]),
        calcular_angulo_3d(landmarks[18], landmarks[19], landmarks[20])
    ]
    caracteristicas.extend(angulos)
    
    base_palma = landmarks[0]
    longitudes_dedos = [
        calcular_distancia_3d(base_palma, landmarks[4]),
        calcular_distancia_3d(base_palma, landmarks[8]),
        calcular_distancia_3d(base_palma, landmarks[12]),
        calcular_distancia_3d(base_palma, landmarks[16]),
        calcular_distancia_3d(base_palma, landmarks[20])
    ]
    caracteristicas.extend(longitudes_dedos)
    
    vector_principal = [
        landmarks[12].x - landmarks[0].x,
        landmarks[12].y - landmarks[0].y,
        landmarks[12].z - landmarks[0].z
    ]
    caracteristicas.extend(vector_principal)
    
    curvaturas = []
    dedos_indices = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20]]
    
    for dedo in dedos_indices:
        if len(dedo) >= 4:
            p1, p2, p3, p4 = [landmarks[i] for i in dedo]
            curvatura = calcular_curvatura(p1, p2, p3, p4)
            curvaturas.append(curvatura)
    
    caracteristicas.extend(curvaturas)
    
    puntos_contorno = [(lm.x, lm.y) for lm in landmarks]
    area_mano = calcular_area_poligono(puntos_contorno)
    caracteristicas.append(area_mano)
    
    return np.array(caracteristicas)

def calcular_distancia_3d(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def calcular_angulo_3d(p1, p2, p3):
    a = np.array([p1.x, p1.y, p1.z])
    b = np.array([p2.x, p2.y, p2.z])
    c = np.array([p3.x, p3.y, p3.z])
    
    ba = a - b
    bc = c - b
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0:
        return 0
    
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def calcular_curvatura(p1, p2, p3, p4):
    puntos = np.array([[p.x, p.y, p.z] for p in [p1, p2, p3, p4]])
    
    v1 = puntos[1] - puntos[0]
    v2 = puntos[2] - puntos[1]
    v3 = puntos[3] - puntos[2]
    
    ang1 = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
    ang2 = np.arccos(np.clip(np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3)), -1, 1))
    
    return (ang1 + ang2) / 2

def calcular_area_poligono(puntos):
    n = len(puntos)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += puntos[i][0] * puntos[j][1]
        area -= puntos[j][0] * puntos[i][1]
    return abs(area) / 2.0

def detectar_trayectoria_j(historial):
    """Analiza un historial de puntos para detectar una trayectoria en forma de 'J'."""
    if len(historial) < historial.maxlen:
        return False

    puntos = np.array(list(historial))
    punto_inicio = puntos[0]
    punto_final = puntos[-1]

    delta_x = punto_final[0] - punto_inicio[0]
    delta_y = punto_final[1] - punto_inicio[1]

    movimiento_vertical_suficiente = delta_y > 0.08
    curva_hacia_izquierda = delta_x < -0.03
    
    punto_medio = puntos[len(puntos) // 2]
    inicio_a_medio_dx = punto_medio[0] - punto_inicio[0]
    
    es_trayectoria_valida = movimiento_vertical_suficiente and curva_hacia_izquierda and abs(inicio_a_medio_dx) < 0.05

    return es_trayectoria_valida

def cargar_datos_entrenamiento():
    if os.path.exists(DATOS_PATH):
        with open(DATOS_PATH, 'r') as f:
            return json.load(f)
    return {"caracteristicas": [], "etiquetas": []}

def guardar_datos_entrenamiento(datos):
    with open(DATOS_PATH, 'w') as f:
        json.dump(datos, f)

def entrenar_modelo():
    global modelo_ml, esta_entrenado
    datos = cargar_datos_entrenamiento()
    
    if len(datos["caracteristicas"]) < 10:
        print("Necesitas al menos 10 samples para entrenar")
        return False
    
    X = np.array(datos["caracteristicas"])
    y = np.array(datos["etiquetas"])
    print(f"Entrenando con {len(X)} samples...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo_ml = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    modelo_ml.fit(X_train, y_train)
    
    y_pred = modelo_ml.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Precisión del modelo: {accuracy:.2f}")
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    with open(MODELO_PATH, 'wb') as f:
        pickle.dump(modelo_ml, f)
    
    esta_entrenado = True
    return True

def cargar_modelo():
    global modelo_ml, esta_entrenado
    if os.path.exists(MODELO_PATH):
        with open(MODELO_PATH, 'rb') as f:
            modelo_ml = pickle.load(f)
        esta_entrenado = True
        print("Modelo cargado exitosamente")
        return True
    return False

def predecir_letra_ml(caracteristicas):
    if modelo_ml is None or not esta_entrenado:
        return " ", 0.0
    
    caracteristicas = caracteristicas.reshape(1, -1)
    prediccion = modelo_ml.predict(caracteristicas)[0]
    probabilidades = modelo_ml.predict_proba(caracteristicas)[0]
    confianza = np.max(probabilidades)
    
    if confianza > 0.4:
        return prediccion, confianza
    else:
        return " ", confianza

def recolectar_sample(caracteristicas, letra):
    datos = cargar_datos_entrenamiento()
    datos["caracteristicas"].append(caracteristicas.tolist())
    datos["etiquetas"].append(letra)
    guardar_datos_entrenamiento(datos)
    print(f"Sample recolectado para letra '{letra}'. Total samples: {len(datos['caracteristicas'])}")

def detectar_gesto_hibrido(hand_landmarks):
    caracteristicas = extraer_caracteristicas_avanzadas(hand_landmarks)
    if esta_entrenado:
        letra_ml, confianza = predecir_letra_ml(caracteristicas)
        return letra_ml, confianza, caracteristicas
    else:
        letra_reglas = detectar_letra_por_reglas_mejoradas(hand_landmarks)
        return letra_reglas, 0.5, caracteristicas

def detectar_letra_por_reglas_mejoradas(hand_landmarks):
    landmarks = hand_landmarks.landmark
    dedos = obtener_dedos_extendidos_mejorado(landmarks)
    
    # Lógica de reglas estáticas (A, B, C, etc.)
    if sum(dedos) <= 1 and dedos[0] == 1 and landmarks[4].y > landmarks[3].y: return "A"
    distancias_entre_dedos = [calcular_distancia_3d(landmarks[8], landmarks[12]), calcular_distancia_3d(landmarks[12], landmarks[16]), calcular_distancia_3d(landmarks[16], landmarks[20])]
    if dedos[0] == 0 and sum(dedos[1:]) == 4 and all(d < 0.05 for d in distancias_entre_dedos): return "B"
    apertura = calcular_distancia_3d(landmarks[4], landmarks[8])
    if sum(dedos) <= 2 and 0.05 < apertura < 0.15: return "C"
    dist_pulgar_medio = calcular_distancia_3d(landmarks[4], landmarks[12])
    if dedos[1] == 1 and sum(dedos) <= 2 and dist_pulgar_medio < 0.08: return "D"
    distancias_a_palma = [calcular_distancia_3d(landmarks[8], landmarks[0]), calcular_distancia_3d(landmarks[12], landmarks[0]), calcular_distancia_3d(landmarks[16], landmarks[0]), calcular_distancia_3d(landmarks[20], landmarks[0])]
    if sum(dedos) == 0 and all(d < 0.12 for d in distancias_a_palma): return "E"
    dist_pulgar_indice = calcular_distancia_3d(landmarks[4], landmarks[8])
    if dedos[0] == 1 and dedos[1] == 0 and sum(dedos[2:]) >= 2 and dist_pulgar_indice < 0.04: return "F"
    angulo = calcular_angulo_3d(landmarks[2], landmarks[4], landmarks[8])
    if dedos[0] == 1 and dedos[1] == 1 and sum(dedos) == 2 and 70 < angulo < 110: return "L"
    separacion = calcular_distancia_3d(landmarks[8], landmarks[12])
    if dedos[1] == 1 and dedos[2] == 1 and sum(dedos) == 2: return "V" if separacion > 0.04 else "U"
    if dedos[1] == 1 and dedos[2] == 1 and dedos[3] == 1 and sum(dedos) == 3: return "W"
    if dedos[0] == 1 and dedos[4] == 1 and sum(dedos) == 2: return "Y"
    if dedos[4] == 1 and sum(dedos) == 1: return "I" # Devuelve 'I' estáticamente
    if sum(dedos) == 0 and 0.02 < dist_pulgar_indice < 0.06: return "O"
    
    return " "

def obtener_dedos_extendidos_mejorado(landmarks):
    dedos = []
    if landmarks[4].x > landmarks[3].x:
        dedos.append(1)
    else:
        dedos.append(0)
    
    dedos_indices = [8, 12, 16, 20]
    bases_indices = [6, 10, 14, 18]
    
    for punta, base in zip(dedos_indices, bases_indices):
        if landmarks[punta].y < landmarks[base].y:
            dedos.append(1)
        else:
            dedos.append(0)
    return dedos

def manejar_cliente(conn, addr):
    global cliente_conectado, ultima_letra
    cliente_conectado = True
    try:
        while True:
            with mensaje_lock:
                mensaje = ultima_letra
            try:
                conn.sendall(mensaje.encode('utf-8'))
                time.sleep(0.1)
            except:
                break
    except: pass
    finally:
        conn.close()
        cliente_conectado = False

def iniciar_servidor():
    global servidor
    servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    servidor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        servidor.bind((HOST, PORT))
        servidor.listen(1)
        while True:
            conn, addr = servidor.accept()
            thread = threading.Thread(target=manejar_cliente, args=(conn, addr))
            thread.daemon = True
            thread.start()
    except Exception as e:
        print(f"Error en el servidor: {e}")
    finally:
        if servidor: servidor.close()

print("=== SignVR - Detector ===")
print("Controles:")
print("- '1': Salir")
print("- 't': Entrenar modelo")
print("- 'r': Activar/desactivar modo recolección")
print("- 'a'-'z': Seleccionar letra para recolección")
print("- '0': Limpiar datos de entrenamiento")

cargar_modelo()
servidor_thread = threading.Thread(target=iniciar_servidor)
servidor_thread.daemon = True
servidor_thread.start()

# Búfer para historial de movimiento
letra_estable = " "
historial_puntos_muneca = deque(maxlen=20) 

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        letra_detectada = " "
        confianza = 0.0
        caracteristicas = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Guardar punto de la muñeca en el historial
                punto_muneca = hand_landmarks.landmark[0]
                historial_puntos_muneca.append((punto_muneca.x, punto_muneca.y))
                
                letra_detectada, confianza, caracteristicas = detectar_gesto_hibrido(hand_landmarks)
                
                if modo_recoleccion and letra_objetivo and caracteristicas is not None:
                    if samples_recolectados < max_samples_por_letra:
                        recolectar_sample(caracteristicas, letra_objetivo)
                        samples_recolectados += 1
                        if samples_recolectados >= max_samples_por_letra:
                            print(f"Recolección completa para '{letra_objetivo}'")
                            modo_recoleccion = False
                            letra_objetivo = ""
                            samples_recolectados = 0
        else:
            # Limpiar historial si no hay mano
            historial_puntos_muneca.clear()

        # Lógica de detección final
        letra_estable = letra_detectada
        
        # Lógica de detección de 'J'
        if letra_estable == "I":
            if detectar_trayectoria_j(historial_puntos_muneca):
                letra_estable = "J"

        with mensaje_lock:
            ultima_letra = letra_estable

        # UI
        estado_conexion = "Conectado" if cliente_conectado else "Esperando conexion..."
        modelo_estado = "ML Activo" if esta_entrenado else "Solo Reglas"
        modo_actual = "RECOLECCION" if modo_recoleccion else "DETECCION"
        
        cv2.putText(frame, f"Letra: {letra_estable}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"Detectado: {letra_detectada} ({confianza:.2f})", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Modelo: {modelo_estado}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(frame, f"Modo: {modo_actual}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Cliente: {estado_conexion}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if modo_recoleccion:
            cv2.putText(frame, f"Recolectando '{letra_objetivo}': {samples_recolectados}/{max_samples_por_letra}", 
                       (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Detector Avanzado ASL - INSTANTANEO", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'): break
        elif key == ord('t'):
            print("Iniciando entrenamiento...")
            entrenar_modelo()
        elif key == ord('r'):
            modo_recoleccion = not modo_recoleccion
            if not modo_recoleccion:
                letra_objetivo = ""
                samples_recolectados = 0
            print(f"Modo recolección: {'ON' if modo_recoleccion else 'OFF'}")
        elif key == ord('0'):
            guardar_datos_entrenamiento({"caracteristicas": [], "etiquetas": []})
            print("Datos de entrenamiento limpiados")
        elif ord('a') <= key <= ord('z'):
            if modo_recoleccion:
                letra_objetivo = chr(key).upper()
                samples_recolectados = 0
                print(f"Recolectando samples para letra: {letra_objetivo}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    if servidor:
        servidor.close()
    print("Programa finalizado")