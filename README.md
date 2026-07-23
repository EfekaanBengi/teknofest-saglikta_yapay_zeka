# Dinamik Varyant Siniflandirma Mimarisi

**TEKNOFEST 2026 Saglikta Yapay Zeka Yarismasi**

Genetik varyant verilerinde (MASTER, KANSER, PAH, CFTR) patolojik/benign siniflandirmasi yapan, 11 asamali tam otonom bir makine ogrenmesi pipeline'i.

---

## Proje Ozeti

Bu proje, genetik varyant verilerindeki yuksek sinif dengesizligi ve gurultu (noise) problemlerini cozmek amaciyla gelistirilmis dinamik bir pipeline mimarisini icerir. Klasik makine ogrenmesi uygulamalarindaki sabit hiperparametre kullanimi ve manuel model secimi yerine, bu sistem her veri panelinin kendi yapisina uygun kararlari istatistiksel testlere (KS-Test, Mutual Information, Cross-Validation) dayanarak dinamik olarak almaktadir.

**Temel ozellikler:**
- Veri sizintisi (Data Leakage) %100 engellenmis mimari (CategoricalEncoder + NumericalPreprocessor)
- Kolmogorov-Smirnov testi tabanli SMOTE kontrol mekanizmasi
- Panel bazli dinamik imputer, scaler ve model secimi
- ROC egrisi ve MCC/Recall dengesi uzerinden esik (threshold) optimizasyonu
- SHAP tabanli klinik aciklanabilirlik

---

## Pipeline Asamalari

| Asama | Aciklama |
|-------|----------|
| Phase 1 | Kesifci Veri Analizi (EDA) - Sinif dagilimi, korelasyon, PCA |
| Phase 2 | Biyolojik Ozellik Muhendisligi (charge_delta, hydro_delta, missing flags) |
| Phase 3 | Sizinti Denetimi (Mutual Information ile leakage tespiti) |
| Phase 5 | Uç Deger Profilleme (RobustScaler vs Winsorization karsilastirmasi) |
| Phase 7 | Ozellik Secimi (Variance Threshold + MI tabanli filtreleme) |
| Phase 4/6/8/9 | Dinamik Motor (Imputer, Resampler, Model secimi ve CV) |
| Phase 10 | SHAP Analizi (Klinik aciklanabilirlik) |
| Phase 11 | Hata Analizi (FN/FP dagilim incelemesi) |

---

## Sonuclar

### Konsolide Metrik Tablosu

| Panel  | Model              | ROC-AUC | PR-AUC | MCC    | Recall | Specificity | F1     | Threshold |
|:-------|:-------------------|:--------|:-------|:-------|:-------|:------------|:-------|:----------|
| MASTER | XGBoost            | 0.8372  | 0.9216 | 0.5011 | 0.9013 | 0.5716      | 0.8763 | 0.3595    |
| KANSER | Random Forest      | 0.9046  | 0.9370 | 0.6579 | 0.9067 | 0.7417      | 0.8967 | 0.5400    |
| PAH    | Random Forest      | 0.8010  | 0.9454 | 0.4651 | 0.9065 | 0.5645      | 0.9094 | 0.7500    |
| CFTR   | LightGBM (Ozel)    | 0.8132  | 0.9563 | 0.3491 | 0.9000 | 0.4286      | 0.8852 | 0.3601    |

Tum panellerde Recall > 0.90 hedefine ulasilmis, klinik hata (False Negative) orani minimize edilmistir.

---

## Kesifci Veri Analizi (EDA)

### Sinif Dagilimi

Her paneldeki Patojenik (1) ve Benign (0) siniflarinin dagilimi:

| MASTER | KANSER |
|:------:|:------:|
| ![MASTER Sinif Dagilimi](outputs/MASTER/phase1_class_dist.png) | ![KANSER Sinif Dagilimi](outputs/KANSER/phase1_class_dist.png) |

| PAH | CFTR |
|:---:|:----:|
| ![PAH Sinif Dagilimi](outputs/PAH/phase1_class_dist.png) | ![CFTR Sinif Dagilimi](outputs/CFTR/phase1_class_dist.png) |

### Eksik Veri Haritalari

![MASTER Eksik Veri](outputs/MASTER/phase1_missing_map.png)
![KANSER Eksik Veri](outputs/KANSER/phase1_missing_map.png)
![PAH Eksik Veri](outputs/PAH/phase1_missing_map.png)
![CFTR Eksik Veri](outputs/CFTR/phase1_missing_map.png)

### Korelasyon Haritalari

Cekirdek genetik ozelliklerin (EK_ onekli) birbirleriyle ve hedef degiskenle (Label) olan korelasyonu:

| MASTER | KANSER |
|:------:|:------:|
| ![MASTER Korelasyon](outputs/MASTER/phase1_ek_corr.png) | ![KANSER Korelasyon](outputs/KANSER/phase1_ek_corr.png) |

| PAH | CFTR |
|:---:|:----:|
| ![PAH Korelasyon](outputs/PAH/phase1_ek_corr.png) | ![CFTR Korelasyon](outputs/CFTR/phase1_ek_corr.png) |

### PCA (Temel Bilesenler Analizi)

Verinin iki boyutlu uzaydaki izdusumu ve siniflarin ayrilabilirligi:

| MASTER | KANSER |
|:------:|:------:|
| ![MASTER PCA](outputs/MASTER/phase1_pca.png) | ![KANSER PCA](outputs/KANSER/phase1_pca.png) |

| PAH | CFTR |
|:---:|:----:|
| ![PAH PCA](outputs/PAH/phase1_pca.png) | ![CFTR PCA](outputs/CFTR/phase1_pca.png) |

---

## Imputer Secim Sureci

Pipeline, her panel icin Cross-Validation testleriyle en uygun eksik veri doldurma yontemini otomatik olarak belirlenmistir.

| Panel  | Median AUC | KNN AUC | Iterative AUC | Secilen   |
|:-------|:-----------|:--------|:--------------|:----------|
| MASTER | **0.7984** | 0.7959  | 0.7632        | Median    |
| KANSER | **0.8908** | 0.8109  | 0.6649        | Median    |
| PAH    | 0.6659     | 0.6707  | **0.6866**    | Iterative |
| CFTR   | Kilitli    | -       | -             | Median    |

---

## SMOTE ve Istatistiksel Kontrol

Sentetik veri uretiminin (SMOTE) veri butunlugune etkisi, Kolmogorov-Smirnov testi ile denetlenmistir. Kolonlarin %20'sinden fazlasinin orijinal dagilimi bozulursa SMOTE reddedilmektedir.

| Panel  | SMOTE Bozulma Orani | BorderlineSMOTE Bozulma Orani | Karar       |
|:-------|:--------------------|:------------------------------|:------------|
| MASTER | %86+                | %64+                          | Reddedildi  |
| KANSER | %77.8               | %75.6                         | Reddedildi  |
| PAH    | %63.7               | %80.1                         | Reddedildi  |

Tum panellerde organik veri uzerinden egitim yapilmis, sinif dengelemesi icin `class_weight='balanced'` kullanilmistir.

---

## Model Secim Sureci

### MASTER Paneli

| Algoritma        | CV ROC-AUC | Sonuc      |
|:-----------------|:-----------|:-----------|
| **XGBoost**      | 0.8372     | Secildi    |
| LightGBM         | 0.8261     | Elendi     |
| Random Forest    | 0.8150     | Elendi     |
| Logistic Reg.    | 0.7967     | Elendi     |

### KANSER Paneli

| Algoritma        | CV ROC-AUC | Sonuc      |
|:-----------------|:-----------|:-----------|
| **Random Forest**| 0.9046     | Secildi    |
| LightGBM         | 0.9035     | Elendi     |
| XGBoost          | 0.9035     | Elendi     |
| Logistic Reg.    | 0.8878     | Elendi     |

### PAH Paneli

| Algoritma        | CV ROC-AUC | Sonuc      |
|:-----------------|:-----------|:-----------|
| **Random Forest**| 0.8010     | Secildi    |
| XGBoost          | 0.7516     | Elendi     |
| LightGBM         | 0.7503     | Elendi     |
| Logistic Reg.    | 0.6866     | Elendi     |

### CFTR Paneli

| Algoritma                       | CV ROC-AUC | Sonuc      |
|:--------------------------------|:-----------|:-----------|
| **LightGBM (Ozel CFTR)**        | 0.8132     | Secildi    |
| Random Forest                   | 0.7905     | Elendi     |
| Logistic Reg.                   | 0.7370     | Elendi     |

CFTR panelinde sadece 111 ornek bulundugu icin ozel kisitlandirilmis LightGBM versiyonu kullanilmistir (max_depth=2, reg_lambda=50.0, n_estimators=50).

---

## Performans Egrileri

### ROC Egrileri

| MASTER | KANSER |
|:------:|:------:|
| ![MASTER ROC](outputs/MASTER/phase9_roc.png) | ![KANSER ROC](outputs/KANSER/phase9_roc.png) |

| PAH | CFTR |
|:---:|:----:|
| ![PAH ROC](outputs/PAH/phase9_roc.png) | ![CFTR ROC](outputs/CFTR/phase9_roc.png) |

### Precision-Recall Egrileri

| MASTER | KANSER |
|:------:|:------:|
| ![MASTER PR](outputs/MASTER/phase9_pr.png) | ![KANSER PR](outputs/KANSER/phase9_pr.png) |

| PAH | CFTR |
|:---:|:----:|
| ![PAH PR](outputs/PAH/phase9_pr.png) | ![CFTR PR](outputs/CFTR/phase9_pr.png) |

### Confusion Matrix

| MASTER | KANSER |
|:------:|:------:|
| ![MASTER CM](outputs/MASTER/phase9_cm.png) | ![KANSER CM](outputs/KANSER/phase9_cm.png) |

| PAH | CFTR |
|:---:|:----:|
| ![PAH CM](outputs/PAH/phase9_cm.png) | ![CFTR CM](outputs/CFTR/phase9_cm.png) |

---

## SHAP Analizi (Klinik Aciklanabilirlik)

### Summary Plot

Varyant ozelliklerinin genel tahmin gucune ve yonune olan etkileri. Ozellikler yukaridan asagiya onem sirasina gore dizilmistir. Kirmizi noktalar yuksek degeri, mavi noktalar dusuk degeri temsil eder.

| MASTER | KANSER |
|:------:|:------:|
| ![MASTER SHAP](outputs/MASTER/phase10_shap_summary.png) | ![KANSER SHAP](outputs/KANSER/phase10_shap_summary.png) |

| PAH | CFTR |
|:---:|:----:|
| ![PAH SHAP](outputs/PAH/phase10_shap_summary.png) | ![CFTR SHAP](outputs/CFTR/phase10_shap_summary.png) |

### Force Plot (False Negative Vaka Analizi)

Yanlis tahmin edilen (False Negative) orneklerde modelin hangi biyolojik skorlardan etkilendigini gosteren vaka analizi:

![MASTER SHAP Force](outputs/MASTER/phase10_shap_force_fn.png)
![KANSER SHAP Force](outputs/KANSER/phase10_shap_force_fn.png)
![PAH SHAP Force](outputs/PAH/phase10_shap_force_fn.png)
![CFTR SHAP Force](outputs/CFTR/phase10_shap_force_fn.png)

---

## Hata Analizi

Modelin hatali tahmin yaptigi (False Positive ve False Negative) vakalarin olasilik skor dagilimi:

| MASTER | KANSER |
|:------:|:------:|
| ![MASTER Error](outputs/MASTER/phase11_error_dist.png) | ![KANSER Error](outputs/KANSER/phase11_error_dist.png) |

| PAH | CFTR |
|:---:|:----:|
| ![PAH Error](outputs/PAH/phase11_error_dist.png) | ![CFTR Error](outputs/CFTR/phase11_error_dist.png) |

---

## Ozellik Onem Dereceleri (Feature Importance)

LightGBM tabanli ozellik onem analizi ve Mutual Information skorlariyla karsilastirma:

| MASTER | KANSER |
|:------:|:------:|
| ![MASTER Importance](outputs/MASTER/phase7_importance.png) | ![KANSER Importance](outputs/KANSER/phase7_importance.png) |

| PAH | CFTR |
|:---:|:----:|
| ![PAH Importance](outputs/PAH/phase7_importance.png) | ![CFTR Importance](outputs/CFTR/phase7_importance.png) |

---

## Proje Yapisi

```
.
├── main.py                        # Ana calistirma dosyasi
├── pipeline.py                    # 11 asamali dinamik pipeline mimarisi
├── config.py                      # Hiperparametre gridleri ve biyokimyasal haritalar
├── create_html.py                 # Rapor HTML donusturucu
├── AKADEMIK_PIPELINE_RAPORU.md    # Detayli akademik rapor
├── AKADEMIK_PIPELINE_RAPORU.html  # Raporun HTML versiyonu
├── Final_Rapor.pdf                # Nihai sunum raporu
├── Odev_2.pdf                     # Ek dokumantasyon
└── outputs/
    ├── MASTER/                    # MASTER paneli gorselleri
    ├── KANSER/                    # KANSER paneli gorselleri
    ├── PAH/                       # PAH paneli gorselleri
    └── CFTR/                      # CFTR paneli gorselleri
```

---

## Gereksinimler

- Python 3.8+
- scikit-learn
- lightgbm
- xgboost
- shap
- imbalanced-learn
- pandas, numpy
- matplotlib, seaborn
- scipy

---

## Kullanim

```bash
python main.py
```

CSV veri dosyalarinin (YARISMA_TRAIN_MASTER.csv, YARISMA_TRAIN_KANSER.csv, YARISMA_TRAIN_PAH.csv, YARISMA_TRAIN_CFTR.csv) proje dizininde bulunmasi gerekmektedir. Pipeline calistirildiktan sonra tum gorseller ve metrikler `outputs/` klasorune kaydedilir.

---

## Yazar

**Efekaan Bengi**

TEKNOFEST 2026 Saglikta Yapay Zeka Yarismasi
