import streamlit as st
from PIL import Image
import torch.nn as nn
import numpy as np
import torch
import cv2
import pandas as pd
import os

from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from transformers import AutoImageProcessor, DetrForObjectDetection
# segmentation
processor_seg = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model_seg = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
#object detection
processor_obj = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model_obj = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")


def center_image(image_path,width=700):
    st.markdown(
        f'<style>img {{ display: block; margin-left: auto; margin-right: auto; }} </style>',
        unsafe_allow_html=True
    )
    st.image(image_path,width = width)


### INTRO ###
st.header('👚 오늘 뭐입지?! 👕')
st.markdown('💬 : 🚨 **설마 너 지금.. 그렇게 입고 나가게?** 🚨')
st.markdown(' **패션센스가 2% 부족한 당신을 위해 준비했습니다!** 사진 이미지만 입력하면, 요즘 트렌디한 스타일과 여러분의 TPO를 고려하여 코디를 추천해드립니다. 무신사와 온더룩의 패셔니스타들의 코디를 지금 바로 참고해보세요! ')
center_image('./intro_img/fashionista.jpg')

st.markdown('--------------------------------------------------------------------------------------')
st.subheader('PROCESS')
center_image('./intro_img/process.png')
st.markdown('--------------------------------------------------------------------------------------')


## INPUT ###
st.subheader(' ✅ 의류 이미지 업로드 ')
input_image = st.file_uploader(" **의류 이미지를 업로드하세요. (배경이 깔끔한 사진이라면 더 좋습니다!)** ", type=['png', 'jpg', 'jpeg'])
if not input_image :
        con = st.container()
        st.stop()
center_image(input_image,400)
st.markdown('--------------------------------------------------------------------------------------')

st.subheader(' ✅ 업로드한 의류 이미지 카테고리 선택 ')
input_cat = st.radio(
    "**귀하가 업로드한 의류 이미지의 카테고리를 골라주세요.**",
    ['top👕', 'bottom👖', 'shoes👞', 'hat🧢', 'sunglasses🕶️', 'scarf🧣', 'bag👜'],
    index=None,
    horizontal = True)


if not input_cat :
        con = st.container()
        st.stop()
input_cat = input_cat[:-1]
st.write('You selected:', input_cat)
st.markdown('--------------------------------------------------------------------------------------')

st.subheader(' ✅ 추천받고 싶은 의류 카테고리 선택 ')
output_cat = st.radio(
    '**추천받고 싶은 의류 카테고리를 선택해주세요.**',
    ['top👕', 'bottom👖', 'shoes👞', 'hat🧢', 'sunglasses🕶️', 'scarf🧣', 'bag👜'],
    index=None,
    horizontal = True)

if not output_cat :
        con = st.container()
        st.write('🚫 주의: 업로드한 의류 카테고리와 다른 카테고리를 선택해주세요.')
        st.stop()
output_cat = output_cat[:-1]
st.write('You selected:', output_cat)
st.write(' ')
st.markdown('--------------------------------------------------------------------------------------')


st.subheader(' ✅ 상황 카테고리 선택 ')
situation = st.radio(
    "**상황 카테고리를 선택해주세요.**",
    ['여행🌊', '카페☕️', '전시회🖼️', '캠퍼스🏫 & 출근💼', '급추위🤧', '운동💪'],
    captions = ['(바다,여행)','(카페, 데일리)','(데이트, 결혼식)','','',''],
    index=None,
    horizontal = True)

# 선택된 상황 카테고리를 영어로 변환해서 변수 저장
situation_mapping = {
    '여행🌊': 'travel',
    '카페☕️': 'cafe',
    '전시회🖼️': 'exhibit',
    '캠퍼스🏫 & 출근💼': 'campus_work',
    '급추위🤧': 'cold',
    '운동💪': 'exercise'}

if not situation:
        con = st.container()
        st.stop()
situation= situation_mapping[situation]
st.write('You selected:', situation)

## 변수 명
# input_img
# input_cat : 입은 옷 카테고리
# output_cat  : 추천 받을 카테고리
# situation : 상황

st.markdown('--------------------------------------------------------------------------------------')


### 입력받은 이미지 segmentation & detection & vector변환 ###
image = Image.open(input_image)

# object detection & cropping 함수
def cropping(images,st = 1,
  fi = 0.0,
  step = -0.05):
  image_1 = Image.fromarray(images)
  inputs = processor_obj(images=image_1, return_tensors="pt")
  outputs = model_obj(**inputs)
  for tre in np.arange(st,fi,step):
    try:
        # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        target_sizes = torch.tensor([image_1.size[::-1]])
        results = processor_obj.post_process_object_detection(outputs, threshold=tre, target_sizes=target_sizes)[0]
        
        img = None
        for idx, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
            box = [round(i, 2) for i in box.tolist()]
            xmin, ymin, xmax, ymax = box
            img = image_1.crop((xmin, ymin, xmax, ymax))

        poss = np.array(img).sum().sum()
        return img
        break
    except:
        continue
  return images

# vector 변환 함수
default_path = './'
def image_to_vector(image,resize_size=(256,256)):  # 이미지 size 변환 resize(256,256)
    #image = Image.fromarray(image)
    #image = image.resize(resize_size)
    image = Image.fromarray(np.copy(image))
    image = image.resize(resize_size)
    image_array = np.array(image, dtype=np.float32)
    image_vector = image_array.flatten()
    return image_vector

# 전체 통합 함수
def final_image(image):    
    if len(np.array(image).shape) == 2:
        image = Image.fromarray(image).convert('RGB')
    # segmentation
    inputs = processor_seg(images=image, return_tensors="pt")
    outputs = model_seg(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    segments = torch.unique(pred_seg)
    default_path = './'
    
    for i in segments:
        if int(i) == 0:
            continue
        if int(i) == 1:
            cloth = 'hat'
            cloths = 'hat'
            mask = pred_seg == i
            image = np.array(image)
            mask_np = (mask * 255).numpy().astype(np.uint8)
            result = cv2.bitwise_and(image.astype(np.uint8), image.astype(np.uint8), mask=mask_np)
            img = cropping(result)
            img_vector = image_to_vector(img)
        elif int(i) == 3:
            cloth= 'sunglasses'
            cloths= 'sunglasses'
            mask = pred_seg == i
            image = np.array(image)
            mask_np = (mask * 255).numpy().astype(np.uint8)
            result = cv2.bitwise_and(image.astype(np.uint8), image.astype(np.uint8), mask=mask_np)
            img = cropping(result)
            img_vector = image_to_vector(img)
        elif int(i) == 4:
            cloth = 'top'
            cloths = 'top'
            mask = pred_seg == i
            image = np.array(image)
            mask_np = (mask * 255).numpy().astype(np.uint8)
            result = cv2.bitwise_and(image.astype(np.uint8), image.astype(np.uint8), mask=mask_np)
            img = cropping(result)
            img_vector = image_to_vector(img)
        elif int(i) in [5,6,7]:
            cloth= ['pants','skirt','dress']
            cloths= 'bottom'
            mask  = (pred_seg == torch.tensor(5)) | (pred_seg == torch.tensor(6)) | (pred_seg == torch.tensor(7))
            image = np.array(image)
            mask_np = (mask * 255).numpy().astype(np.uint8)
            result = cv2.bitwise_and(image.astype(np.uint8), image.astype(np.uint8), mask=mask_np)
            img = cropping(result)
            img_vector = image_to_vector(img)
        elif int(i) == 8:
            cloth = 'belt'
            cloths = 'belt'
            mask = pred_seg == torch.tensor(8)
            image = np.array(image)
            mask_np = (mask * 255).numpy().astype(np.uint8)
            result = cv2.bitwise_and(image.astype(np.uint8), image.astype(np.uint8), mask=mask_np)
            img = cropping(result)
            img_vector = image_to_vector(img)
        elif (int(i) == 9):
            cloth = 'shoes'
            cloths = 'shoes'
            mask = (pred_seg == torch.tensor(9)) | (pred_seg == torch.tensor(10))
            image = np.array(image)
            mask_np = (mask * 255).numpy().astype(np.uint8)
            result = cv2.bitwise_and(image.astype(np.uint8), image.astype(np.uint8), mask=mask_np)
            img = cropping(result)
            img_vector = image_to_vector(img)
        elif int(i) == 16:
            cloth = 'bag'
            cloths = 'bag'
            mask = pred_seg == torch.tensor(16)
            image = np.array(image)
            mask_np = (mask * 255).numpy().astype(np.uint8)
            result = cv2.bitwise_and(image.astype(np.uint8), image.astype(np.uint8), mask=mask_np)
            img = cropping(result)
            img_vector = image_to_vector(img)
        elif int(i) == 17:
            cloth = 'scarf'
            cloths = 'scarf'
            mask = pred_seg == torch.tensor(17)
            image = np.array(image)
            mask_np = (mask * 255).numpy().astype(np.uint8)
            result = cv2.bitwise_and(image.astype(np.uint8), image.astype(np.uint8), mask=mask_np)
            img = cropping(result)
            img_vector = image_to_vector(img)
        return img_vector
    
# 입력받은 이미지 전처리 완료
input_img = final_image(image)

### 유사도 분석 ###
# 하나는 이미지, 다른 하나는 경로로 받는 경우
def cosine_similarity(vec1, vec2_path):
    vec2 = np.loadtxt(vec2_path)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)   
    return similarity

# 둘 다 경로로 받는 경우
def cosine_similarity_2(vec1_path, vec2_path):
    vec1 = np.loadtxt(vec1_path)
    vec2 = np.loadtxt(vec2_path)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

with st.spinner('Wait for it...'):
    # 입력받은 이미지 & 동일 카테고리 폴더에 저장된 스타일 이미지
    sim_list = []
    file_path = './style/' + situation + '/' + input_cat + '/'   # ex) './cafe/top/'
    cloths = os.listdir('./style/' + situation + '/' + input_cat + '/')
    for cloth in cloths:
        sim_list.append(cosine_similarity(input_img, file_path + cloth))
    max_idx = np.argmax(sim_list)

    # target_image 정의
    target_image = './style/' + situation + '/' + output_cat + '/' + cloths[max_idx]
    # 유사도 분석 완료된 스타일seg 이미지와 product_seg 유사도분석
    sim_list = []
    file_path = './product/' + output_cat + '/'
    cloths = os.listdir('./product/' + output_cat + '/')
    for cloth in cloths:
        sim_list.append(cosine_similarity_2(target_image, file_path + cloth))
    max_idx = np.argmax(sim_list)
    output_name  = cloths[max_idx]
    ## 예시 출력값: 'bottom_1883.txt'

    # name 로드
    acc_name = pd.read_csv('acc_name.csv')
    bottom_name =pd.read_csv('bottom_name.csv')
    outer_name =pd.read_csv('outer_name.csv')
    shoes_name =pd.read_csv('shoes_name.csv')
    top_name =pd.read_csv('top_name.csv')

    #상품 데이터 로드
    outer = pd.read_csv('outer.csv')
    top = pd.read_csv('top.csv')
    bottom = pd.read_csv('bottom.csv')
    shoes = pd.read_csv('shoes.csv')
    acc = pd.read_csv('acc.csv')

    if output_cat == 'bottom':
        df = bottom.copy()
        df_name = bottom_name.copy()
    elif output_cat == 'top':
        df = top.copy()
        df_name = top_name.copy()
    elif output_cat == 'shoes':
        df = shoes.copy()
        df_name = shoes_name.copy()
    elif (output_cat == 'hat') or (output_cat == 'sunglasses') or (output_cat == 'scarf') or (output_cat == 'bag') or (output_cat == 'belt'):
        df = acc.copy()
        df_name = acc_name.copy()

    output_name = output_name.split('.')[0]
    file_name = df_name[df_name['index']==output_name].iloc[0,1] #3049906_16754112975667_500.jpg
    final = df[df['id'] == file_name]

    name = final['name'].values[0].split('\n')[-1] # 상품명
    price = final['price'].values[0] # 상품가격

image_path = './product/img/'

st.subheader('OUTPUT')

img = Image.open(image_path+output_cat+'/'+file_name)

col1, col2, col3 = st.columns(3)
with col1:
    st.image(img,width=400)
with col3:
    st.caption('상풍명 : ' + name)
    st.caption('가격 : ' + price)
