# 프로젝트명
> Anormal 이미지 비교로 승가 및 기타 이상행위 잡아내는 프로젝트

## 설치 방법

Docker:

```sh
intflow/de-blur-competition:HINet
```


# Utils
## `utils/unet_datamaker.py`
> normal 데이터 셋과 anormal 데이터 셋을 합치는 역할

### 실행코드 
```
source /opt/conda/bin/activate
/opt/conda/bin/python unet_datamaker.py
```



### config.yaml 설명
&#x1F4D9; : path variable
- <span style="color:orange">normal_path</span>  : normal timeseries dataset path
- <span style="color:orange">anormal_path</span> : anormal timeseries dataset path
- <span style="color:orange">out_path</span> : output path

&#x1F4D7; : option
- <span style="color:green">normal_sampling</span> : normal_path에서 몇개를 샘플링 할 것인가? 
- <span style="color:green">anormal_sampling</span> : anormal_path에서 몇개를 샘플링 할 것인가? 만약 -1이라면, 모두 선택
- <span style="color:green">only_normal</span> : Enc-Dec trainset 만들 때 사용 (정상데이터로만 만들고 싶을 때)

---

## `utils/augmentor.py`
> 주로 anormal data 증강용

### 실행코드 
```
source /opt/conda/bin/activate
/opt/conda/bin/python augmentor.py
```

### config.yaml 설명
&#x1F4D9; : path variable
- <span style="color:orange">unet_data_path</span>  : timeseries dataset path
- <span style="color:orange">output_path</span> : output path

&#x1F4D7; : option
- <span style="color:green">max_copy_img</span> : 한 source에 대해서 몇번 gen 할것인가?
- <span style="color:green">original_copy</span> : copy origin_repo to output_path
- <span style="color:green">max_algorithm</span> : select Algorithm number
- <span style="color:green">option_algorithm</span> : 비정형 Algorithm 포함할 것인가?
~ (비 뿌리기, 눈 뿌리기 등)

<br>


# Predict

## `predict/test_one.py`
> Sample 하나 넣었을 때 PSNR과 예측이미지, Anormal prob 관측

### 실행코드 
```
source /opt/conda/bin/activate
/opt/conda/bin/python test_one.py
```

### config.yaml 설명
&#x1F4D9; : path variable
- <span style="color:orange">input_path</span> : (String) One folder timeseries dataset path
- <span style="color:orange">model_path</span> : (String) Model path

&#x1F4D7; : option
- <span style="color:green">scale</span> : (Default, INT) 96 | (half_model, INT) 48
- <span style="color:green">threshold</span> : (float) anormal threshold [Not Use]
- <span style="color:green">bgr2rgb</span> : (bool) make img RGB style
---

## `predict/test_PSNR.py`
> Test dataset을 넣었을 때 anormal layer에 대한 Confusion matrix 

### 실행코드 
```
source /opt/conda/bin/activate
/opt/conda/bin/python test_PSNR.py
```

### config.yaml 설명
&#x1F4D9; : path variable
- <span style="color:orange">input_path</span> : (String) Many folder timeseries dataset path
- <span style="color:orange">model_path</span> : (String) Model path

&#x1F4D7; : option
- <span style="color:green">scale</span> : (Default, INT) 96 | (half_model, INT) 48
- <span style="color:green">threshold</span> : (float) anormal threshold [Not Use]
- <span style="color:green">bgr2rgb</span> : (bool) make img RGB style


<br>



## 개발 환경 설정

3090 X 8GPU

## 업데이트 내역

- None

## 정보

주소현 – cannonvirus@intflow.ai


<!-- Markdown link & img dfn's -->
[image_create]: https://github.com/intflow/pig_image_maker