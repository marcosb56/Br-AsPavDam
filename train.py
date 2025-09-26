
from ultralytics import YOLO
import yaml
import os

# Caminho do arquivo de configuração
config_file = 'config.yaml'

# Carregar config.yaml
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Construir caminhos
data_path = config['path']
train_dir = os.path.join(data_path, config['train'])
val_dir = os.path.join(data_path, config['val'])

# Verificações
assert os.path.exists(config_file), "Arquivo config.yaml não encontrado"
assert os.path.exists(train_dir), f"Pasta de treino não encontrada: {train_dir}"
assert os.path.exists(val_dir), f"Pasta de validação não encontrada: {val_dir}"

# Debug
print("Treino:", train_dir)
print("Validação:", val_dir)

# Criar modelo (pode ser .yaml ou .pt)
model = YOLO('yolov8n.pt')

# Treinar
model.train(
    data=config_file,
    epochs=100,
    imgsz=640,
    project='/content/drive/MyDrive/YOLO_Dissertacao/runs',
    name='meu_treino_yolov8'
)
