# GAN을 이용한 이미지 생성

이 프로젝트는 TensorFlow와 Keras를 사용하여 흑백 이미지를 생성하는 생성적 적대 신경망(GAN)을 구현합니다. GAN은 랜덤 노이즈로부터 이미지를 생성하는 생성자(Generator)와 생성된 이미지와 실제 이미지를 구별하는 판별자(Discriminator)로 구성됩니다.

![image](https://github.com/user-attachments/assets/eb0fc537-687a-436e-ae2b-758920941fa6)

## 주요 특징

- **생성자(Generator):** 랜덤 노이즈를 입력받아 convolution을 통해 이미지를 생성합니다.
- **판별자(Discriminator):** convolution을 사용하여 실제 이미지와 생성된 이미지를 구별합니다.
- **커스텀 GAN 학습 루프:** 생성자와 판별자를 여러번 반복하여 정확한 이미지를 생성합니다.
- **시각화:** 학습 과정에서 생성된 이미지를 저장하고, 원본 이미지와 비교할 수 있습니다.

## 요구 사항

- Python 3.7 이상
- TensorFlow 2.x
- NumPy
- Matplotlib
- tqdm
- 
## 파라미터 

-Epoch : 1000 
-batch : 1


## 결과


![image](https://github.com/user-attachments/assets/8922cc9d-596d-48e9-bf5e-018de40026f4)

