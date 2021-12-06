#!pip install -U torch torchvision
#!pip install git+https://github.com/facebookresearch/fvcore.git
#!git clone https://github.com/facebookresearch/detectron2 detectron2_repo
#!pip install -e detectron2_repo

import os
import json
import shutil
import random
import numpy as np
import pandas as pd
import cv2
import detectron2
from PIL import Image
from pathlib import Path
from pycocotools import mask as cocomask
from matplotlib import pyplot as plt
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_train_loader
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.visualizer import Visualizer

dados_imagens = pd.read_csv('MRIBrain/data.csv')

print(dados_imagens.head())
print(dados_imagens.shape)

mapa = sorted(list(Path('MRIBrain/').rglob('*tif')))

print(mapa[0:5])
print(len(mapa))

info = {"year" : 2021,
        "version" : "1.0",
        "description" : "Segmentação Tumores Cerebrais",
        "contributor" : "Fernando Feltrin",
        "url" : "https://github.com/fernandofeltrin",
        "date_created" : "2021"}

licenses = [{"id": 1,
             "name": "Attribution-NonCommercial",
             "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"}]
type = "instances"

masks = []
names = []
bboxes = []
areas = []
annotations = []
images = []
id = 0

print(len(str('MRIBrain/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_10_mask.tif')))

for im in mapa:
    if len(str(im)) == 64 or len(str(im)) == 65 :
        msk = np.array(Image.open(im).convert('L'))
        contours, _ = cv2.findContours(msk,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        crowd = 0

        for contour in contours:
            if contour.size >= 6:
                crowd += 1 
                segmentation.append(contour.flatten().tolist())

        if crowd > 1:
            iscrowd = 1
        else: 
            iscrowd = 0     
        try:
            RLEs = cocomask.frPyObjects(segmentation,
                                        msk.shape[0],
                                        msk.shape[1])
            RLE = cocomask.merge(RLEs)
            area = float(cocomask.area(RLE))
        except:
            area = []
        
        [x, y, w, h] = cv2.boundingRect(msk)
        bbox1 = [float(x), float(y), float(w), float(h)]
        areas.append(area)
        id += 1

        try:
            for s in range(len(segmentation[0]) - 1):
                segmentation[0][s] = float(segmentation[0][s])
        except:
            pass

        annotations.append({"segmentation" : segmentation,
                            "area" : area,
                            "iscrowd" : iscrowd,
                            "image_id" : id,
                            "bbox" : bbox1,
                            "category_id" :  1,
                            "id": id})
    
        images.append({"date_captured" : "2021",
                       "file_name" : str(im)[:-9]+".tif", 
                       "id" : id,
                       "license" : 1,
                       "url" : "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                       "height" : msk.shape[0],
                       "width" : msk.shape[1]})
    else: 
        mapa.remove(im)
        
print(annotations[1])
print(len(annotations))
print(annotations[46])
print(images[0])
print(len(images))

categoria_seg = [{'id': 1,
                  'name': 'tumor',
                  'supercategory': 'shape'}]
print(categoria_seg)

coco_output = {"info": info,
               "licenses": licenses ,
               "categories": categoria_seg,
               "images": [],
               "annotations": []}
print(coco_output)

for im_id in images:
    coco_output["images"].append(im_id)
    
for annotation_id in annotations:
    coco_output["annotations"].append(annotation_id)
    
with open('MRIBrain/annotation_imagens.json', 'w') as output_json_file:
    json.dump(coco_output, output_json_file)

shutil.copy ('MRIBrain/annotation_imagens.json',
             'MRIBrain/projeto/annotation_imagens.json')

register_coco_instances("medical_treino", {},
                        "MRIBrain/annotation_imagens.json",
                        "MRIBrain/")
register_coco_instances("medical_teste", {},
                        "MRIBrain/annotation_imagens.json",
                        "MRIBrain/")

treino_metadata = MetadataCatalog.get("treino")
teste_metadata = MetadataCatalog.get("teste")

print(treino_metadata)
print(teste_metadata)

treino_dict = DatasetCatalog.get("mtreino")
teste_dict = DatasetCatalog.get("teste")

print(treino_dict[0])

for item in random.sample(treino_dict, 3):
    print(item)
    imagem_nome = cv2.imread(item["file_name"])
    visualizer = Visualizer(imagem_nome[:, :, ::-1],
                            metadata = MetadataCatalog.get("treino"),
                            scale = 2)
    vis = visualizer.draw_dataset_dict(item)
    imagem_treino = vis.get_image()[:, :, ::-1]
    plt.figure(figsize = (10,6))  
    plt.imshow(imagem_treino)

config = 'COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml'

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(config))
cfg.DATASETS.TRAIN = ("treino",)
cfg.DATASETS.TEST = ("teste",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config)
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 10000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.OUTPUT_DIR = "MRIBrain/"
cfg.MODEL.DEVICE = "cuda"
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
cfg.SOLVER.WARMUP_ITERS = 3000 
cfg.SOLVER.WARMUP_METHOD = "linear"

os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)

modelo = build_model(cfg)
otimizador = build_optimizer(cfg, modelo)
scheduler = build_lr_scheduler(cfg, otimizador)
data_loader = build_detection_train_loader(cfg)
evaluator = detectron2.evaluation.COCOEvaluator("teste",
                                                cfg,
                                                distributed = True,
                                                output_dir = "MRIBrain/")
val_loader = build_detection_test_loader(cfg, "teste")

treinador = DefaultTrainer(cfg)
treinador.resume_or_load(resume = False)
treinador.train()

cfg.MODEL.WEIGHTS = os.path.join("MRIBrain/",
                                 "modelo_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85

preditor = DefaultPredictor(cfg)

cfg = get_cfg()
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85

nome_imagem = 'MRIBrain/TCGA_HT_8111_19980330/TCGA_HT_8111_19980330_10.tif'
amostra = cv2.imread(os.path.join(nome_imagem))

previsoes = preditor(amostra)

v = Visualizer(amostra[:, :, ::-1],
               MetadataCatalog.get("train"),
               scale = 2)
v = v.draw_instance_predictions(previsoes["instances"])

amostra_treino = v.get_image()[:, :, ::-1]
plt.imshow(amostra_treino)

for item in random.sample(teste_dict, 3): 
    print(item)
    imagem_nome = cv2.imread(item["file_name"])
    previsoes = preditor(imagem_nome)
    visualizer = Visualizer(imagem_nome[:, :, ::-1],
                            metadata = MetadataCatalog.get("teste"),
                            scale = 2)
    vis = visualizer.draw_instance_predictions(previsoes["instances"])
    imagem_teste = vis.get_image()[:, :, ::-1]
    plt.figure(figsize = (10,6))  
    plt.imshow(imagem_teste)
    filename = 'img' + str(int(random.randint(0, 100))) + ".jpg"
    write_res = cv2.imwrite(filename, imagem_teste)

nova_amostra = cv2.imread("TCGA_DU_6399_19830416_20.tif")
previsoes = preditor(nova_amostra)
visualizer = Visualizer(nova_amostra[:, :, ::-1],
                        metadata = MetadataCatalog.get("teste"),
                        scale = 2)                 
vis = visualizer.draw_instance_predictions(previsoes["instances"])
imagem_teste = vis.get_image()[:, :, ::-1]
plt.figure(figsize = (18,10))
plt.imshow(imagem_teste)
filename = 'img' + str(int(random.randint(0, 100))) + ".jpg"
write_res = cv2.imwrite(filename, imagem_teste)
