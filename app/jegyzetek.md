OCR: Google ML Kit a pontosság végett
A fizikai gyógyszereket egy PyTorch-ban tanított, ONNX-be konvertált YOLO
modellel detektálnám a 30 osztályos adaton.
Adatbázis: SQLite a felhasználó készülékén

butább mint az AI mert nem tudja egy szóról, hogy létező-e (férfi volt a képen ő férti-t olvasott)

az ML Kit tesztelése során bebizonyosodott: pusztán a vonalak felismerése nem elegendő a megbízható adatkinyeréshez.

//Google Gemini API integráció online tehát kell hozzá internet de sokkal pontosabb mint a google ml kit

https://aistudio.google.com //itt lehet API kulcsot lekérni. google-el kell belépni

adatbázisba mentett recept adatok:

    -gyógyszer neve
    -gyógyszer neve latinul is
    -fajtája
    -hányszor kell szedni / nap
    -kiváltható dátom
    -kiállítva dátum

mai megoldások:

    -Gemini API integrálása
        -képeket szövegként visszadja a megfelelő formában és helyesen
    -Firebase integrálása
        -Elmenti az adatokat helyesen(név, mennyiség, bevétel száma(napi hányszor kell), súlya, időbélyeg amikor be lett scannelve a recept)
    -Gyógyszerek összehasonlítására felkészültem(a súlyt és annak mértékegységét
    külön kezelem, illetve osztályként hivatkozok a változóra így lehetővé teszi mindezt)
    -Timestamp probléma megoldása
        -Időzónát be kellett állítani android studio keresztül, mert alapból el van csúszva 2 órával.
    -Kezdetleges LandingPage kialakítva
Problémák:

    -sok recept (Sok ugyan olyan recept)
    -Gemini API korlátját hamar elérem a sok tesztelés miatt

Fejlesztések még:

    -összehasonlító (súly szerint)
    -UI/UX fejlesztése
    -landingpage fejlesztése
    -statisztikák létrehozás (napi, heti, havi)(felhasználó, gyógyszer)
    -gyógyszer felisemrő applikáció fejelsztése
    
Second Week

Megoldott
    
    -firebase authentication   
        -email, pw - google authentication
    -laplace function - képek élességének ellenrőzése
    -Ha nem elég éles akkor visszadobja a képe és nem külde
    el az AI-nak.
    -Fizetős AI -ra váltás
    -Kontraszt, vaku lehetőségekl hozzáadása
    -Fragmentek létrehozása, újrahasznosíthatóság miatt
    -kijelentkezés
    -userID alapján történő előzmény képek szűrése
    -Előzményekben a képek nagyíthatósága

Hátra van

    -Lefényképezett gyógyszerekből adatbázist csinálni
        -Neve, előlap, hátlap oszlopokkal, esetleg bevágás (igen-nem)
    -Profil fül hozzáadása


Third week

Megoldva

    -Profil fül hozzáadása
    -firebase storage fizetős, képet csak oda lehet elmenteni ezért
    a profilkép át van alakítva egy hosszú stringé és azt mentem el az adatbázisba
    -profil kép hozzáadása és cseréje elkészül
    -elfelejtett jelszó hozzáadva - emailt ki is küldi
    -Scannelés okosabbá tétele, fejjel lefelé is betudja olvasni a receptet hibátlanul.
    -elkezdödött a gyógyszer scan feladat fejleszése
        -adatbázis létrehozása - python generálás - sqlite tábla
            -3D plot generálás
    -Előzményeknél lehet képeket törölni Swipe-to-delete technikával (Firebase-ből is törli).

4th Week

Megoldva

    -Gyógyszer scan fragment létrehozva,
    -két GEMINI model használata:
        -objektum detektálás: gemini-2.5-flash:generateContent
        -embeddignre: gemini-embedding-2:embedContent
    -A detektáló mefgkeresi a gyógyszert a képen madj köré rajzol egy Bounding Box-ot
    -Az embeddinges pedig erre a croppolt képre generál egy float értéket tartalmiazó 0-1 közötti vektrot.
    -majd ezt a vektrot összehasonlítsa az adatbázissal.
    -Majd Koszinusz hasonlósággal megnézi hoyg melyik értékhez van közelebb és azt
    -a gyógyszert visszaadja.
    -gyroscope használata - ha nincs síkba a telefon akkor ne engedjen fényképezni
    -képre egy négyzetet amin belül kell lennie a fényképnek különben ne engedje fényképezni

Feladatok:
    
    -145ös színkód mindegyik képen
    -képek mérete egyezzen meg


módosítások

    -háttércsere, detektálás megcserélése
    -szegmentálás csak annak akordinátáiból
    huzzon egy szűk bounding boxot ez t kivágjuk ez legyen a roi és ezt mentse le automiatikusan
    -magasságot 20-30cm-re maximum (arucoból )
    -timestamp legyen a mentett kép neve és ha mondjuk 1 napnál régebbi a kép akkor törölje ki.
    -ha nincs gyógyszer a képen akkor is mentse a képet amit csinál a felhasználó.
    

5th Hét

    -treshold: 0.0 temperature változón keresztül lehet állítani
    -camera engedély kérő üzenet box megjelenítése
    -képek mentése Pictures/MedScan/Pills mappába
    -keret figyelembe vétele. - az AruCo kódoknak a kereten belül kell lenniük.
    -AruCo kódok belső sarka alapján a területet kivágni
    -alkalamzás Portré módra állítva (fix álló)
    -háttérkép fix 145 145 145 színűre alakítása

6th Week
    
    -Gemini-3.6-flash használata a szegmentáláshoz 2.5-flash helyett 
    -4:3 AspectRatio kényszerítése
    -Bounding Box kivétele (csak tesztelésre van)
    -a gyógyszer scan minden folyamatáról készül kép ezt elmenit az adott
    dátum mappájába
    -Az előzményekbe menti mostmár a gyógyszert is.
    -távolsáégot mérünk - 25cm felett ne menged fényképezni az alkalamzás
    -futási idő is logolva van
    -CameraX Aspect Ratio-ja 4:3ra van korlátozva
    -1440x1080 a preferált felbontás
    -A CameraX Preview, ImageCapture és ImageAnalysis közös ViewPort használata,
    hogy a rögzített kép ugyanazt a területet mutassa, mint amit a felhasználó
    az előnézetben lát.
    
    tesztek:

    -magne b6 3\5
    -vagy 2\3
    -enterol 250mg 5\5
    -c vitamin 3\5
    -neo ferro 5\5
    -covercard 3\5

    megbeszélés
    

    7th week
    
    -adatbázis kibővítése 30 gyógyszerre
    -több gyógyszer is felismer(kb 15sec)
    -párhuzamosan futnak a folyamatok így 15ből 10sec lett.
    -minden gyógyszernél megnézi az 5 legközelebbít illetve a top3mat és ezeket kimenti
    egy txt-beha több van akkor azokat 1db txt-be írja ki.
    -maszk legyen png minden más jpg
    * Végleges mappastruktúra Android 10+ rendszeren:
    *
    * Download/MedScan/
    * ├── Images/
    * │   ├── Pills/
    * │   └── Prescriptions/
    * └── Debug/
    *     ├── Images/<sessionId>/
    *     └── Data/<sessionId>/


    Visszavan még:
    
    -fix világosság ne legyen az hogy a kép készítésénél 10másodperc alatt modnjuk elstötétül a kép
    -profiler hogy melyík függvénynél megy el a sok idő
    

    8th week

    -Recept scan-nél már a beteg nevét is beolvassuk illetve áltároljuk az adatbáisban.
    -Scannél a gombok mostmár fix helyen maradnak.
    -Szegmentálás kivágás javítása (minden formához ugye amit most csináltunk emlékszuel rá remélem)
    -összes előzmény gomb törölése létrehozvaí működik
    -hosszan nyomással is lehet törölni ugye az előzményeket cska ott ki kell jelölni melyike szeretnénk.
    -1600x1200 használata mert ezt több telefon támogatja
    -itt írjuk le ugye hogy melyik kép gondolok itt az ArUco kódos képre vagy az embedding képre mindegyikre, hogy mekkroa ugye mondjuk 512x512 ha még ne mvotl róla szó írjuk bele ezt is.
    
    smoothing
    -megbeszélés elkezdi köteteni és visszaveszi
    -bináris képeken kell

    readme az sdk telepítéséről!!!
    átméretezni mindent a régihez képest