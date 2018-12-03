# Opis dzia³ania programu rozpoznaj¹cego liœcie na obrazach
## Wymagania
Program zosta³ napisany w jêzyku programowania Python v3.5.2. Do swojego poprawnego dzia³ania wymaga zainstalowanych bibliotek NumPy v1.12.1 oraz scikit-image v0.14. Program przyjmuje dwa argumenty: œcie¿kê do katalogu z baz¹ obrazów liœci oraz nazwê pliku wynikowego. Je¿eli nazwa pliku wynikowego nie zostanie podana, to wyliczone cechy zostan¹ zapisane w pliku o domyœlnej nazwie ‘output.csv’.

<code>python3 leafFeatureIdentification.py leafsnap-subset1/ output.csv</code>

<small><b>Ryc. 1.</b> Przyk³ad poprawnego uruchomienia programu.</small>

## Detekcja liœcia

W pierwszym etapie dzia³ania funkcji _leaf_detection_ analizowany obraz najpierw zmieniany jest za pomoc¹ funkcji _color.rgb2gray_ ze zdjêcia wielobarwnego na skalê szaroœci, nastêpnie obraz progowany jest w celu usuniêcia pikseli t³a, z wykorzystaniem funkcji _filters.threshold_mean_ z biblioteki _skimage_. Funkcja ta oblicza wartoœæ progowania na podstawie œredniej jasnoœci pikseli. Kolejnym krokiem jest etykietyzacja obrazu (_measure.label_ z biblioteki _skimage_) oraz usuniêcie obszarów zawieraj¹cych siê w odleg³oœci dziesiêciu procent (wysokoœci/d³ugoœci obrazu) od dolnej lub prawej krawêdzi. Zabieg ten powoduje usuniêcie skali, która mog³aby utrudniæ rozpoznanie liœcia. Po renumeracji etykiet nastêpuje obliczenie pola powierzchni i odleg³oœci od œrodka obrazu danych obszarów. Obliczone wartoœci s¹ normalizowane - najwiêksza wartoœæ pola powierzchni oraz najmniejsza odleg³oœæ od œrodka przyjmuj¹ wartoœæ 1. Nastêpnie dla ka¿dego obszaru odpowiadaj¹ce mu wartoœci s¹ sumowane i jako liœæ uznawany jest obszar o najwiêkszej sumie powy¿szych elementów. Normalizacja oraz wybór odpowiedniej etykiety przeprowadzane s¹ w funkcji _leaf_label_identification_, która jako wynik zwraca obraz zawieraj¹cy domniemany liœæ (jasnoœæ pikseli liœcia równa 255, natomiast t³a - 0).

## Ekstrakcja cech liœcia z obrazów

Wszystkie funkcje wykorzystywane do detekcji cech liœcia pochodz¹ z biblioteki _skimage_, a dok³adniej z modu³u _measure.regionprops_. Cechy wybrane do charakteryzacji liœcia to:
<ol>
<li>area - liczba pikseli obrazu liœcia;</li>
<li>bbox area - liczba pikseli bry³y brzegowej;</li>
<li>perimeter - wielkoœæ obwodu liœcia;</li>
<li>convex area - iloœæ pikseli najmniejszego zbioru wypuk³ego zawieraj¹cego obraz liœcia;</li>
<li>ecceentricity - stosunek odleg³oœci ogniskowej (odleg³oœæ miêdzy punktami ogniskowymi) do osi g³ównej;</li>
<li>eqivalent diameter - œrednica okrêgu o tym samym obszarze co obraz liœcia;</li>
<li>extent - stosunek pikseli obrazu roœliny do liczby pikseli bry³y brzegowej;</li>
<li>major axis length - d³ugoœæ osi g³ównej elipsy, która ma t¹ sam¹ wartoœæ znormalizowan¹ drugiego momentu centralnego co obraz liœcia;</li>
<li>minor axis length - d³ugoœæ osi mniejszej elipsy, która ma t¹ sam¹ wartoœæ znormalizowan¹ drugiego momentu centralnego co obraz liœcia;</li>
<li>solidity - stosunek iloœci pikseli obrazu liœcia do iloœci pikseli najmniejszego zbioru wypuk³ego zawieraj¹cego obraz;</li>
<li>area_ratio - stosunek pola powierzchni liœcia po erozji do pola powierzchni liœcia przed erozj¹;</li>
<li>perimeter_ratio - stosunek obwodu liœcia po erozji do pola powierzchni liœcia przed erozj¹;</li>
<li>number_of_objects - liczba obszarów na jakie podzieli³ siê liœæ po erozji.</li></ol>

Trzy ostatnie cechy obliczane s¹ dla 3 ró¿nych parametrów erozji.
<br><br>
<![endif]-->

# Wybór i uczenie klasyfikatorów

## Wymagania

Program klasyfikuj¹cy liœcie poza wyró¿nionymi w punkcie pierwszym pakietów, wymaga dodatkowo zainstalowanej biblioteki scikit-learn v0.18.1. Program przyjmuje jeden argument: - œcie¿kê do katalogu z baz¹ obrazów liœci (rozszerzenie .jpg). Program na wyjœciu wypisuje dla ka¿dego pliku jego nazwê oraz przewidziany gatunek liœcia.

<code>python3 leafClassification.py leafsnap-subset1/</code>
<small><b>Ryc. 2.</b> Przyk³ad poprawnego uruchomienia programu klasyfikuj¹cego liœcie.</small>

<code>ny1079-04-4.jpg acer_campestre
wb1448-06-3.jpg ginkgo_biloba
wb1001-08-3.jpg fagus_grandifolia
pi0056-06-3.jpg ilex_opaca
pi0046-03-3.jpg carya_glabra</code>

<small><b>Ryc. 3.</b> Przyk³adowe wyniki zwracane przez program klasyfikuj¹cy liœcie.</small>

## Wybór i uczenie klasyfikatorów

Wœród dostêpnych klasyfikatorów w ramach pakietu _scikit-learn_ wybrano:

<ol><li>KNeighborsClassifier - klasyfikacja przynale¿noœci na podstawie najbli¿szego s¹siedztwa;</li>
<li>RandomForestClassifier - klasyfikacja przynale¿noœci na podstawie losowego zestawu danych (cech) z których tworzone s¹ zdekolerowane drzewa decyzyjne;</li>
<li>DecisionTreeClassifier - klasyfikacja przynale¿noœci na podstawie drzewa decyzyjnego bazuj¹cego na zestawie danych (cech);</li>
<li>MLPClassifier - klasyfikacja przynale¿noœci za pomoc¹ sieci neuronowych.</li></ol>

<img src="plots.png">

<small><b>Ryc. 4.</b> Wykres pokazuj¹cy zmianê wartoœci najlepszej klasyfikacji w zale¿noœci od iloœci najkorzystniejszych cech wybranych w oparciu o wartoœæ ANOVA F-value.</small>

Po wykonanych testach najlepszym klasyfikatorem okaza³ siê klasyfikator lasu losowego (_RandomForestClassifier_)  dla przetestowanych parametrów:
- warm_start_ - False;
- min_samples_leaf_ -  1;
- max_features_ -  log2;
- min_samples_split_ -  2;
- bootstrap_ -  False;
- n_estimators_ -  27.

Pozosta³e parametry przyjmowa³y wartoœci domyœlne. Klasyfikator wykorzystuje wszystkie dostêpne dane.

