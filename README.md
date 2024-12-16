# Proje Notları

Baseline modelinde zaman ekseninde Pooling yapmak yerine LSTM kullanılmasının denenmesi

## Denenen Modeller ve Performans Sonuçları

| Model            | NM@R1   | BG@R1   | CL@R1   | Notlar                                                                                         |
|------------------|---------|---------|---------|------------------------------------------------------------------------------------------------|
| **PartLstm**     | 96.85%  | 93.31%  | 74.60%  | Parçalara ayırıp (Zaman x Hidden_dim) ekseni üzerinden LSTM                                          |
| **ConvLstm**     | 96.82%  | 92.36%  | 76.07%  | Zaman eksenini doğrudan yok etmek yerine, hem pooling hem de LSTM ile zaman boyutunda iyi bilgi çıkarılmaya çalışıldı. |
| **Baseline**     | 97.35%  | 93.24%  | 78.36%  | 4 katmanlı (1,1,1,1) ResNet9, stride uygulanmıyor; zaman ekseni max ile yok ediliyor, parçalara ayırarak mean-max yapılıyor. |
| **AttentionLstm**| 96.77%  | 92.98%  | 74.13%  | LSTM üzerine Attention katmanları eklenerek fusion denemesi.                                    |
| **BaseAttention**| -       | -       | -       | Başarısız.                                                                                     |
| **ReelResnet**   | -       | -       | -       | Başarısız. Stride uygulamaları, avg-pool+flatten ile bilgi yok oluyor.                          |

## Notlar ve Gözlemler
**Neden Baseline başarılı, ne yapmış?**  
- 4 katman (1,1,1,1) (ResNet9 kullanılıyor) ve **stride uygulanmıyor**.  
- Zaman ekseninde maksimum noktalar bulunarak zaman ekseninden kurtulunuyor.  
- HPP: h,w ekseninde 16x16 → 16’ya düşürülüyor, yani her 16 piksel için mean-max yapılıyor.  
  - Not: Demek ki veriyi parçalara ayırıp, her parçaya göre loss hesaplamak başarılı bir strateji.

**Neler denendi?**

**ReelResnet**
- Stride uygulasak ne olur?  
  - 64x64’lük veriye stride uygulasak 1x1’e düşer, yani elimizde zaman x hidden_dim kalır.
- Avg pool + flatten da yapsak?
- `n c s` elde edebiliriz.
- Örneğin: n=128, c=75 (olabilir), s=30
- Yani Baseline modelindeki zaman eksenini yok etmek yerine, zaman eksenini bir "part" gibi düşünüyoruz.
- Loss hesaplarken zaman bir parça (part) olarak alınır ve hidden_dim=75’lik vektörlerin birbirine yakınlığına bakılır.

**ConvLstm**
```
[Pooling]       [LSTM]
   |                |
   +-->  Fusion  <--+
          |
          V
[Horizontal Pooling Matching]
```
- Sadece zaman eksenini max noktalarla yok etmek yerine, zaman eksenini LSTM’den geçir.
- Zaman eksenini LSTM’den nasıl geçirirsin?
- Olabilecek senaryolar:  
  - `n c s h w -> (n h w) s c` → Başarılı  
  - `n c s h w -> (n h c) s w` → Başarısız  
  - `n c s h w -> (n w c) s h` → Başarısız  
  - `n c s h w -> (n c) s (h*w)` → Memory hatası

- LSTM sonrası elimizde `n c h w` oluyor.
- Baseline’da zaman ekseni max edildiğinde de `n c h w` elde ediliyor.
- Bunları Fusion yapıp sonra partlara ayır.
- Neden Baseline daha başarılı? Çünkü zaman ekseninde max noktaya bakmak daha iyi sonuç veriyor.

**PartLstm**
- Daha önce LSTM’den geçirip sonra partlara ayırdık.
- Şimdi önce partlara ayırıp sonra LSTM’den geçirmeyi deneyeceğiz.
- ResNet sonrası: `n, c, s, h, w`
- Partlara ayır: `n, c, s, p`
- LSTM’e hazırlık: `n c s p -> (n p) s c`
- LSTM sonrası: `n c p`
- Aslında h*w’yi ifade etmek için mean-max yapıldı ve p elde edildi, sonra bunu LSTM’e soktum. Çok başarılı olacağını düşünmüştüm...

**AttentionLstm**
- ConvLstm’deki Fusion işlemini daha profesyonel (Attention ile) yaparsak ne olur?
```
[Pooling]            [LSTM]
   |                   |
   +-->  ATTFusion  <--+
           |
           V
[Horizontal Pooling Matching]
```
- Bu fikri CSTL ile de düşünmüştüm. Orada da benzer şekilde long memory ile short memory bir nevi ATTFusion’dan geçiriliyordu.
- Başarısız oldu. Muhtemelen LSTM fikri baştan yanlış bir fikirdi...