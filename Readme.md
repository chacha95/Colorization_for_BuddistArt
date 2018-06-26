
### 불교미술 채색

- 기존에 사람이 수작업으로 채색하던 불교미술 그림에 대해, 딥러닝을 이용해 채색을 하는 프로젝트입니다.
- 핵심 채색 기능은 CycleGAN을 이용해 채색을 하였습니다.

![](https://github.com/gimikk/Colorization_for_BuddistArt/blob/master/images/%EC%B1%84%EC%83%891.PNG)

### 개발 환경
- PyQT 5
- Pytorch 0.3
- OpenCV 3.2
- Python 3.6

# CycleGAN
- CycleGAN은 Unpaired Dataset에 대한 Style transfer 기술이다. 
![](https://github.com/gimikk/Colorization_for_BuddistArt/blob/master/images/CycleGAN.PNG)
###### Work Flow
###### 1.  전처리
여러 실험 결과 딥러닝 네트워크에 들어갈 이미지를 이진화 및 노이즈 제거 연산을 하면 더 나은 결과를 보여줬다.
![](https://github.com/gimikk/Colorization_for_BuddistArt/blob/master/images/%EC%B1%84%EC%83%892.PNG)
###### 2. 딥러닝 채색
사이클 겐 모델을 사용하였으며,  불교 학술 문화원으로 받은 데이터를 input으로, Discriminator에 훈련시킬 이미지 데이터셋(실제 탱화)은 구글에 쿼리를 날려 수집하였다.
###### 3.  후보정
후보정은 PyQt 및 OpenCV를 이용해 FMM 알고리즘 및 페인팅 툴을 이용해 보정을 하였다.

# 최종 결과물

![](https://github.com/gimikk/Colorization_for_BuddistArt/blob/master/images/final_res.PNG)

![](https://github.com/gimikk/Colorization_for_BuddistArt/blob/master/images/final_res2.PNG)

