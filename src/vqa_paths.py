DRIVE = "/../content/drive/MyDrive"

# vqa_trainer, vqa_utils
VQA_DATA_ROOT = f'{DRIVE}/lxmert_vqa/'

# vqa_hat, lxmert_vqa
HAT_DATA = f'{VQA_DATA_ROOT}hlat/'
DATA_PTH = f'{VQA_DATA_ROOT}data/'

# lxmert_vqa, vqa_trainer_data
OBJ_PTH = f"{DATA_PTH}objects_vocab.txt"
ATTR_PTH = f"{DATA_PTH}attributes_vocab.txt"
VQA_PTH = f"{DATA_PTH}vqa-trainval_label2ans.json"
LABEL_PTH = f"{DATA_PTH}vqa-trainval_ans2label.json"
MODEL_PTH = f"{VQA_DATA_ROOT}models/"

# vqa_trainer_data
MSCOCO_IMGFEAT_ROOT = f'/../content/lxmert/data/mscoco_imgfeat/'
