# Opis dzia�ania programu rozpoznaj�cego li�cie na obrazach
## Wymagania
Program zosta� napisany w j�zyku programowania Python v3.5.2. Do swojego poprawnego dzia�ania wymaga zainstalowanych bibliotek NumPy v1.12.1 oraz scikit-image v0.14. Program przyjmuje dwa argumenty: �cie�k� do katalogu z baz� obraz�w li�ci oraz nazw� pliku wynikowego. Je�eli nazwa pliku wynikowego nie zostanie podana, to wyliczone cechy zostan� zapisane w pliku o domy�lnej nazwie �output.csv�.

<code>python3 leafFeatureIdentification.py leafsnap-subset1/ output.csv</code>

<small><b>Ryc. 1.</b> Przyk�ad poprawnego uruchomienia programu.</small>

## Detekcja li�cia

W pierwszym etapie dzia�ania funkcji _leaf_detection_ analizowany obraz najpierw zmieniany jest za pomoc� funkcji _color.rgb2gray_ ze zdj�cia wielobarwnego na skal� szaro�ci, nast�pnie obraz progowany jest w celu usuni�cia pikseli t�a, z wykorzystaniem funkcji _filters.threshold_mean_ z biblioteki _skimage_. Funkcja ta oblicza warto�� progowania na podstawie �redniej jasno�ci pikseli. Kolejnym krokiem jest etykietyzacja obrazu (_measure.label_ z biblioteki _skimage_) oraz usuni�cie obszar�w zawieraj�cych si� w odleg�o�ci dziesi�ciu procent (wysoko�ci/d�ugo�ci obrazu) od dolnej lub prawej kraw�dzi. Zabieg ten powoduje usuni�cie skali, kt�ra mog�aby utrudni� rozpoznanie li�cia. Po renumeracji etykiet nast�puje obliczenie pola powierzchni i odleg�o�ci od �rodka obrazu danych obszar�w. Obliczone warto�ci s� normalizowane - najwi�ksza warto�� pola powierzchni oraz najmniejsza odleg�o�� od �rodka przyjmuj� warto�� 1. Nast�pnie dla ka�dego obszaru odpowiadaj�ce mu warto�ci s� sumowane i jako li�� uznawany jest obszar o najwi�kszej sumie powy�szych element�w. Normalizacja oraz wyb�r odpowiedniej etykiety przeprowadzane s� w funkcji _leaf_label_identification_, kt�ra jako wynik zwraca obraz zawieraj�cy domniemany li�� (jasno�� pikseli li�cia r�wna 255, natomiast t�a - 0).

## Ekstrakcja cech li�cia z obraz�w

Wszystkie funkcje wykorzystywane do detekcji cech li�cia pochodz� z biblioteki _skimage_, a dok�adniej z modu�u _measure.regionprops_. Cechy wybrane do charakteryzacji li�cia to:
<ol>
<li>area - liczba pikseli obrazu li�cia;</li>
<li>bbox area - liczba pikseli bry�y brzegowej;</li>
<li>perimeter - wielko�� obwodu li�cia;</li>
<li>convex area - ilo�� pikseli najmniejszego zbioru wypuk�ego zawieraj�cego obraz li�cia;</li>
<li>ecceentricity - stosunek odleg�o�ci ogniskowej (odleg�o�� mi�dzy punktami ogniskowymi) do osi g��wnej;</li>
<li>eqivalent diameter - �rednica okr�gu o tym samym obszarze co obraz li�cia;</li>
<li>extent - stosunek pikseli obrazu ro�liny do liczby pikseli bry�y brzegowej;</li>
<li>major axis length - d�ugo�� osi g��wnej elipsy, kt�ra ma t� sam� warto�� znormalizowan� drugiego momentu centralnego co obraz li�cia;</li>
<li>minor axis length - d�ugo�� osi mniejszej elipsy, kt�ra ma t� sam� warto�� znormalizowan� drugiego momentu centralnego co obraz li�cia;</li>
<li>solidity - stosunek ilo�ci pikseli obrazu li�cia do ilo�ci pikseli najmniejszego zbioru wypuk�ego zawieraj�cego obraz;</li>
<li>area_ratio - stosunek pola powierzchni li�cia po erozji do pola powierzchni li�cia przed erozj�;</li>
<li>perimeter_ratio - stosunek obwodu li�cia po erozji do pola powierzchni li�cia przed erozj�;</li>
<li>number_of_objects - liczba obszar�w na jakie podzieli� si� li�� po erozji.</li></ol>

Trzy ostatnie cechy obliczane s� dla 3 r�nych parametr�w erozji.
<br><br>
<![endif]-->

# Wyb�r i uczenie klasyfikator�w

## Wymagania

Program klasyfikuj�cy li�cie poza wyr�nionymi w punkcie pierwszym pakiet�w, wymaga dodatkowo zainstalowanej biblioteki scikit-learn v0.18.1. Program przyjmuje jeden argument: - �cie�k� do katalogu z baz� obraz�w li�ci (rozszerzenie .jpg). Program na wyj�ciu wypisuje dla ka�dego pliku jego nazw� oraz przewidziany gatunek li�cia.

<code>python3 leafClassification.py leafsnap-subset1/</code>
<small><b>Ryc. 2.</b> Przyk�ad poprawnego uruchomienia programu klasyfikuj�cego li�cie.</small>

<code>ny1079-04-4.jpg acer_campestre
wb1448-06-3.jpg ginkgo_biloba
wb1001-08-3.jpg fagus_grandifolia
pi0056-06-3.jpg ilex_opaca
pi0046-03-3.jpg carya_glabra</code>

<small><b>Ryc. 3.</b> Przyk�adowe wyniki zwracane przez program klasyfikuj�cy li�cie.</small>

## Wyb�r i uczenie klasyfikator�w

W�r�d dost�pnych klasyfikator�w w ramach pakietu _scikit-learn_ wybrano:

<ol><li>KNeighborsClassifier - klasyfikacja przynale�no�ci na podstawie najbli�szego s�siedztwa;</li>
<li>RandomForestClassifier - klasyfikacja przynale�no�ci na podstawie losowego zestawu danych (cech) z kt�rych tworzone s� zdekolerowane drzewa decyzyjne;</li>
<li>DecisionTreeClassifier - klasyfikacja przynale�no�ci na podstawie drzewa decyzyjnego bazuj�cego na zestawie danych (cech);</li>
<li>MLPClassifier - klasyfikacja przynale�no�ci za pomoc� sieci neuronowych.</li></ol>

<img src="plots.png">

<small><b>Ryc. 4.</b> Wykres pokazuj�cy zmian� warto�ci najlepszej klasyfikacji w zale�no�ci od ilo�ci najkorzystniejszych cech wybranych w oparciu o warto�� ANOVA F-value.</small>

Po wykonanych testach najlepszym klasyfikatorem okaza� si� klasyfikator lasu losowego (_RandomForestClassifier_)  dla przetestowanych parametr�w:
- warm_start_ - False;
- min_samples_leaf_ -  1;
- max_features_ -  log2;
- min_samples_split_ -  2;
- bootstrap_ -  False;
- n_estimators_ -  27.

Pozosta�e parametry przyjmowa�y warto�ci domy�lne. Klasyfikator wykorzystuje wszystkie dost�pne dane.

