veri =read.table (file.choose(),header=T,sep=",")

colnames(veri) <- c(
  "hasta_id",
  "yas",
  "cinsiyet",
  "gogus_agrisi",
  "dinlenme_tansiyon",
  "kolesterol",
  "aclik_kan_sekeri",
  "dinlenme_ekg",
  "maks_nabiz",
  "egzersiz_anjina",
  "oldpeak",
  "st_egimi",
  "damar_sayisi",
  "kalp_hastaligi"
)

veri$hasta_id <- NULL

veri$kalp_hastaligi <- factor(veri$kalp_hastaligi, levels = c(0,1),
                              labels = c("Yok", "Var"))

# Say??sal de??i??kenleri se??
library(caret)

set.seed(42)

egitimIndisleri <- createDataPartition(
  y = veri$kalp_hastaligi,
  p = 0.7,
  list = FALSE
)

egitimverisi <- veri[egitimIndisleri, ]
testverisi   <- veri[-egitimIndisleri, ]
sayisalDegiskenler <- sapply(egitimverisi, is.numeric)
names(egitimverisi)[sayisalDegiskenler]

# Normalizasyon modeli (ortalama = 0, sd = 1)
normModel <- preProcess(
  egitimverisi[, sayisalDegiskenler],
  method = c("center", "scale")
)

# Uygulama
egitim_norm <- egitimverisi
test_norm   <- testverisi

egitim_norm[, sayisalDegiskenler] <- predict(
  normModel,
  egitimverisi[, sayisalDegiskenler]
)

test_norm[, sayisalDegiskenler] <- predict(
  normModel,
  testverisi[, sayisalDegiskenler]
)

#karar a??ac?? modeli
karar_agaci_norm <- J48(
  kalp_hastaligi ~ .,
  data = egitim_norm,
  control = Weka_control(U = TRUE, M = 10)
)

tahmin_agac_norm <- predict(karar_agaci_norm, test_norm)

confusionMatrix(tahmin_agac_norm, test_norm$kalp_hastaligi)

#EG??T??M VE TEST BOLME

library(caret)

set.seed(1)

egitimIndisleri <- createDataPartition(
  y = veri$kalp_hastaligi,
  p = .70,
  list = FALSE
)

egitimverisi <- veri[egitimIndisleri, ]
testverisi   <- veri[-egitimIndisleri, ]


print(table(egitimverisi$kalp_hastaligi))



#KARAR AGAC?? MODELLER??

library(RWeka)
library(rJava)
library(partykit)

# Karar Agaci Modeli 1
kararagacimodeli1 <- J48(kalp_hastaligi ~ ., data = egitimverisi)
print(kararagacimodeli1)
summary(kararagacimodeli1)
plot(kararagacimodeli1)



# Karar Agaci Modeli 2
kararagacimodeli2 <- J48(kalp_hastaligi ~ ., data = egitimverisi,
                         control = Weka_control(U=TRUE, M=10))
print(kararagacimodeli2)
summary(kararagacimodeli2)
plot(kararagacimodeli2)


ongoru <- predict(kararagacimodeli2, testverisi)
ongoru[1:150]
data.frame(gercek=testverisi$kalp_hastaligi, ongoru=ongoru)[1:150,]
predict(kararagacimodeli2,ongoru)

yeni <- data.frame(
  yas = 55,
  cinsiyet = factor("Male", levels = levels(egitimverisi$cinsiyet)),
  gogus_agrisi = factor("typical", levels = levels(egitimverisi$gogus_agrisi)),
  dinlenme_tansiyon = 140,
  kolesterol = 250,
  aclik_kan_sekeri = factor("true", levels = levels(egitimverisi$aclik_kan_sekeri)),
  dinlenme_ekg = factor("normal", levels = levels(egitimverisi$dinlenme_ekg)),
  maks_nabiz = 150,
  egzersiz_anjina = factor("yes", levels = levels(egitimverisi$egzersiz_anjina)),
  oldpeak = 1.5,
  st_egimi = factor("flat", levels = levels(egitimverisi$st_egimi)),
  damar_sayisi = 1
)

predict(kararagacimodeli2, yeni)
plot(kararagacimodeli2,yeni)


# 1) Eksik verileri temizleme
egitimverisi <- na.omit(egitimverisi)
testverisi   <- na.omit(testverisi)

# 2) Hedef de??i??kenin ad??n?? belirleyelim
hedef_degisken <- "kalp_hastaligi"

# 3) E??itim ve test veri setlerini ??zellikler (X) ve hedef (y) olarak ay??rma
# ??ndeks numaras?? yerine isim kullanarak hatay?? engelliyoruz
hedef_degisken <- "kalp_hastaligi"

trainingAttributes <- egitimverisi[, setdiff(names(egitimverisi), hedef_degisken)]
trainingClass <- egitimverisi[[hedef_degisken]]

testAttributes <- testverisi[, setdiff(names(testverisi), hedef_degisken)]
testClass <- testverisi[[hedef_degisken]]

# 4) Naive Bayes modelini kurma
nb_model <- naiveBayes(trainingAttributes, trainingClass)

# 5) Test verisi ??zerinde tahmin alma
predictedClasses <- predict(nb_model, testAttributes)

# 6) Kar??????kl??k matrisi (Confusion Matrix)
karisiklikmatrisi <- table(Gerc_Deger = testClass, Tahmin = predictedClasses)
print("--- Kar??????kl??k Matrisi ---")
print(karisiklikmatrisi)

# 7) Modelin performans de??erlendirme ??l????tleri
# Matrisin yap??s??na g??re (Yok=1. sat??r, Var=2. sat??r) de??erleri ??ekiyoruz
TN <- karisiklikmatrisi[1,1] # Yok tahmin edilen Yok
FN <- karisiklikmatrisi[2,1] # Yok tahmin edilen Var
FP <- karisiklikmatrisi[1,2] # Var tahmin edilen Yok
TP <- karisiklikmatrisi[2,2] # Var tahmin edilen Var

cat("\n--- Performans ??l????tleri ---\n")
print(paste0("Dogruluk (Accuracy) = ", (TP+TN)/sum(karisiklikmatrisi)))
print(paste0("Hata Oran?? = ", 1 - ((TP+TN)/sum(karisiklikmatrisi))))
print(paste0("Hassasiyet (TPR/Recall) = ", TP/(TP+FN)))
print(paste0("??zg??ll??k (SPC) = ", TN/(FP+TN)))
print(paste0("Kesinlik (PPV) = ", TP/(TP+FP)))

