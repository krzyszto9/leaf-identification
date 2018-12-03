import sys
import os
import numpy as np
import skimage
import skimage.io as io
import skimage.color as color
import skimage.measure as measure
import skimage.filters as filters
import skimage.morphology as morphology
np.seterr(divide='ignore', invalid='ignore')

 
def save_to_file(output,results):
	results = np.array(results)
	np.savetxt(output, results, fmt='%s',delimiter='\t',header='filename\tarea\tbbox_area\tperimeter\tconvex_area\teccentricity\tequivalent_diameter\textent_number\tmajor_axis_length\tminor_axis_length\tsolidity\tarea_ratio_1\tperimeter_ratio_1\tnumber_of_objects_1\tarea_ratio_2\tperimeter_ratio_2\tnumber_of_objects_2\tarea_ratio_3\tperimeter_ratio_3\tnumber_of_objects_3')
 
def leaf_label_identification(labels,center):
	s_area = []
	distance =  []
	for region in skimage.measure.regionprops(labels):
		s_area.append(region.area)
		distance.append(np.sum((np.array(region.centroid)-center)**2))
	s_area_normed = (s_area-np.min(s_area))/(np.max(s_area)-np.min(s_area))
	distance_normed = 1- (distance-np.min(distance))/(np.max(distance)-np.min(distance))
	return (np.argmax(s_area_normed + distance_normed)) + 1

def leaf_detection(leaf):
	img = color.rgb2gray(leaf)
	#progowanie obrazu bazując na średniej jasności pikseli (nie całego obrazka)
	binary = (img < filters.threshold_mean(img[:int(img.shape[0]*0.95),:int(img.shape[1]*0.95)]))
	labels = measure.label(binary) #etykietowanie
	#usunięcie etykiet zawierajacych się w 10% dolnej lub prawej krawędzi obrazu
	bottomn = img.shape[0] *0.9
	right = img.shape[1] *0.9
	for region in skimage.measure.regionprops(labels):
		if (region.coords[:,:1]>bottomn).any() == True or (region.coords[:,1:]>right).any() == True:
			labels[labels==region.label] = 0
	labels = measure.label(labels) #renumeracja etykiet
	#lisc wybierany jest na podstawie pola powierzchni i odległosci od środka. Wartości są normalizowane -
	#największe pole powierzchni i najmniejsza odległość od środka mają wartości 1, pozostałe wartości proporcjonalnie mniejsze
	#wygrywa etykieta o największej sumie pola powierzchni i odległości od środka
	center = np.array([img.shape[0]/2,img.shape[1]/2])
	leaf_label = leaf_label_identification(labels,center)
	labels[labels<leaf_label] = 0
	labels[labels>leaf_label] = 0
	labels[labels==leaf_label] = 255
	return labels
					

def leaf_feature_identification(path_to_dir,output):
	results = []
	for path, subdirs, files in os.walk(path_to_dir):
			for file_name in files:
				if file_name.lower().endswith('.jpg'):
					#identyfikacja liścia
					leaf = io.imread(os.path.join(path,file_name))
					only_leaf = leaf_detection(leaf)
					#liść po erozji
					only_leaf_erosed1 = morphology.erosion(only_leaf, morphology.square(5))
					only_leaf_erosed2 = morphology.erosion(only_leaf, morphology.square(7))
					only_leaf_erosed3 = morphology.erosion(only_leaf, morphology.square(11))
					#io.imsave(os.getcwd()+'/foty/'+file_name,only_leaf)  #zapis do pliku, może zostać
					#obliczanie cech 
					leaf_label = measure.label(only_leaf)
					leaf_label_erosed1,number_of_objects_1 = measure.label(only_leaf_erosed1,return_num=True)
					leaf_label_erosed2,number_of_objects_2 = measure.label(only_leaf_erosed2,return_num=True)
					leaf_label_erosed3,number_of_objects_3 = measure.label(only_leaf_erosed3,return_num=True)

					reg = measure.regionprops(leaf_label)
					reg_erosed1 = measure.regionprops(leaf_label_erosed1)
					reg_erosed2 = measure.regionprops(leaf_label_erosed2)
					reg_erosed3 = measure.regionprops(leaf_label_erosed3)

					#area
					area_leaf = reg[0].area
					#bbox_area
					bbox_area_leaf = reg[0].bbox_area
					#perimeter
					perimete_leaf = reg[0].perimeter
					#convex_area
					convex_area_leaf =reg[0].convex_area
					#eccentricity
					eccentricity_leaf = reg[0].eccentricity
					#equivalent_diameter
					equivalent_diameter_leaf = reg[0].equivalent_diameter
					#extent_number
					extent_leaf = reg[0].extent
					#major_axis_length
					major_axis_length_leaf = reg[0].major_axis_length
					#minor_axis_length
					minor_axis_length_leaf = reg[0].minor_axis_length
					#solidity
					solidity_leaf = reg[0].solidity
					#stosunek powierzchni liścia po erozji do przed erozją
					area_ratio_1 = reg_erosed1[0].area/area_leaf
					area_ratio_2 = reg_erosed2[0].area/area_leaf
					area_ratio_3 = reg_erosed3[0].area/area_leaf
					#stosunek obwodu liścia po erozji do przed erozją
					perimeter_ratio_1 = reg_erosed1[0].perimeter/perimete_leaf
					perimeter_ratio_2 = reg_erosed2[0].perimeter/perimete_leaf
					perimeter_ratio_3 = reg_erosed3[0].perimeter/perimete_leaf
					#zapis wyników w tablicy results
					results.append([file_name,area_leaf,bbox_area_leaf,perimete_leaf,convex_area_leaf,eccentricity_leaf,equivalent_diameter_leaf,extent_leaf,major_axis_length_leaf,minor_axis_length_leaf,solidity_leaf,area_ratio_1,perimeter_ratio_1,number_of_objects_1,area_ratio_2,perimeter_ratio_2,number_of_objects_2,area_ratio_3,perimeter_ratio_3,number_of_objects_3])
	#zapis wyników do pliku
	save_to_file(output,results)


if __name__ == "__main__":
	if len(sys.argv)<3:
		sys.argv.append('output.csv')
	leaf_feature_identification(sys.argv[1],sys.argv[2])