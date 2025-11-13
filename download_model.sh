cd KAIR
# DnCNN模型
python main_download_pretrained_models.py --models "DnCNN" --model_dir "model_zoo"
# BSRGAN模型
python main_download_pretrained_models.py --models "BSRGAN" --model_dir "model_zoo"
# scunet
python main_download_pretrained_models.py --models "SCUNet" --model_dir "model_zoo"

# ddcolor模型
cd ../DDColor
pip install modelscope
mkdir ./modelscope
python download_model.py

# 去模糊模型 restomer
cd ../Restormer
mkdir pretrained_models
cd pretrained_models
pip install gdown
# motion_deblurring.pth
gdown 1pwcOhDS5Erzk8yfAbu7pXTud606SB4-L
# single_image_defocus_deblurring.pth
gdown 10v8BH3Gktl34TYzPy0x-pAKoRSYKnNZp
# derain
gdown 1uuejKpyo0G_5M4DAO2J9_Dijy550tjc5
