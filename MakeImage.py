from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# アクセストークンの設定
access_tokens="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" # @param {type:"string"}
 
# モデルのインスタンス化
model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=access_tokens)
model.to("cuda")

prompt = "slightly underbaked bread" #@param {type:"string"}

# 画像出力のディレクトリ
import os
os.makedirs("outputfile", exist_ok=True)

# 画像のファイル名
import re
filename = re.sub(r'[\\/:*?"<>|,]+', '', prompt).replace(' ','_')

# 画像数
num = 4
 
for i in range(num):
  # モデルにプロンプトを入力して画像生成
  image = model(prompt, num_inference_steps=75).images[0]

  # 保存
  outputfile = f'{filename} _{i:02} .png'
  image.save(f"outputfile/{outputfile}")
 
for i in range(num):
  outputfile = f'{filename} _{i:02} .png'
  plt.imshow(plt.imread(f"outputfile/{outputfile}"))
  plt.axis('off')