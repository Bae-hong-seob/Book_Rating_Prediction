# Book Rating Prediction with Multi-modal Recommend System

## Abstract

- 뉴스 기사나 짧은 러닝 타임의 동영상처럼 간결하게 콘텐츠를 즐길 수 있는 콘텐츠는 소비자들이 부담 없이 쉽게 선택할 수 있지만, 책 한 권을 모두 읽기 위해서는 보다 긴 물리적인 시간이 필요. 소비자 입장에서 제목, 저자, 표지, 카테고리 등 한정된 정보로 각자가 콘텐츠를 유추하고 구매 유무를 결정한다.
- 본 프로젝트는 상대적으로 선택에 더욱 신중을 가함을 고려하는 소비자들의 책 구매 결정에 대한 도움을 주기 위한 개인화된 상품 추천 프로젝트이다.

## Introduction

- 기존 추천 시스템 솔루션의 한계점은 크게 두가지로 Cold-start와 Dynamic Change 문제점이 존재. 최근 딥러닝 모델이 발전함에 따라 다양한 비선형 관계를 고려할 수 있는 솔루션으로 기존 한계점을 개선시키기 위해 연구가 활발히 진행중. 이에 제공된 데이터셋이 tableau, image, text 임에 따라 여러 종류의 데이터를 사용할 수 있는 Multi-Modal 아키텍처를 구축하고자 함.
- 아키텍처 구조는  AutoInt, CNN_FM, Deep Conn 모델의 최종 output embedding을 결합한 후 MLP(3 layer)를 거쳐 최종적인 Rating을 예측하도록 설계함.

![Untitled](https://github.com/Bae-hong-seob/Book_Rating_Prediction/assets/49437396/4536349b-3d8a-422a-8deb-cdfa0e7425c4)

- 본 프로젝트는 Naver AI Tech Recsys 교육과정을 통해 학습한 추천 시스템에 적용할 수 있는 다양한 딥러닝 모델을 직접 사용하고 튜닝해보며, 템플릿에 맞게 프로젝트를 구성하는 경험에 초점을 맞추고 진행하였음.
