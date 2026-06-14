# densenet121_kather100k_patch_cls
模型来自：https://huggingface.co/1aurent/densenet121.tiatoolbox-kather100k
模型下载：从https://huggingface.co/1aurent/densenet121.tiatoolbox-kather100k, 下载config.json & model.safetensors放入densenet121_kather100k文件夹。

安装环境：

         conda create -n crc_cls python=3.10 -y   
         
		 conda activate crc_cls
		 
		 pip install -r requirements.txt
		 
设置PATCH_H5_DIR & WSI_ROOT & MODEL_PATH & OUT_DIR

运行：python classify_orion_crc_patches.py 
