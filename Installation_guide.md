Gyógyszerfelismerő Android alkalmazás
Projekt futtatása
Követelmények
Android Studio
JDK 17
Android SDK API 36
NDK verzió: 28.2.13676358
Android 7.0 (API 24) vagy újabb készülék
SDK letöltése

A projekthez szükséges SDK innen tölthető le:

4.12.0-ás Android verzióra van szükség

https://opencv.org/releases/

A letöltött SDK-t csomagold ki, majd helyezd a projekt gyökérkönyvtárába sdk néven:

Projekt/
├── app/
├── sdk/
├── keystore/
├── gradle/
├── build.gradle
├── settings.gradle
└── gradlew

A projekt Gradle konfigurációja az SDK-t az alábbi modulként használja:

implementation project(path: ':sdk')
Futtatás
Nyisd meg a projektet Android Studioban.
Ellenőrizd, hogy a megfelelő NDK verzió van telepítve.
Futtass egy Gradle Sync-et.
Csatlakoztass egy Android készüléket USB hibakereséssel.
Indítsd el az alkalmazást a Run gombbal.
Firebase

A projekt saját debug keystore-t használ, ezért másik számítógépen történő futtatás esetén nem szükséges új SHA-1 ujjlenyomatot hozzáadni a Firebase projekthez.
